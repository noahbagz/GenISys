"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import json


import os


import time


from pathlib import Path


import numpy as np


import pandas as pd


import torch


import torch.nn.functional as F


from training_utils.checkpoints import load_complete_checkpoint, sampling_weight_context, save_complete_checkpoint, save_recovery_checkpoint, seed_everything


from training_utils.data import SUBSYSTEM_ORDER, PreparedFirstExperimentData, prepare_first_experiment_data


from models.generators import FirstExperimentGenerator, build_first_experiment_generator, train_or_load_guidance_classifiers


from training_utils.model_size import count_trainable_parameters


from training_utils.label_sampling import environment_sampling_probabilities


from training_utils.batches import _draw_batch_indices


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(
    table: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = None,
) -> None:
    """Install a complete CSV with one filesystem replacement."""

    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    """Install a deterministic JSON document with one filesystem replacement."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _prepare_data(
    project_root: Path,
    settings: dict,
    paths: dict[str, Path],
) -> PreparedFirstExperimentData:
    return prepare_first_experiment_data(
        unified_root=paths["unified"],
        artifact_root=paths["data"],
        split_seed=int(settings["split_seed"]),
    )


def _generator_directory(
    paths: dict[str, Path],
    population: str,
    architecture: str,
    training_seed: int,
) -> Path:
    return paths["generators"] / f"seed_{training_seed}" / population / architecture


def _generator_signature(settings: dict, population: str, architecture: str) -> dict:
    keys = (
        "split_seed",
        "training_seed",
        "device_name",
        "batch_size",
        "diffusion_timesteps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "cosine_s",
        "training_epochs",
        "batches_per_epoch",
        "learning_rate",
        "weight_decay",
        "timestep_embedding_dim",
        "network",
        "feasible_subsystem_network",
        "full_subsystem_network",
        "hcddpm_feasible_subsystem_network",
        "hcddpm_full_subsystem_network",
        "scddpm_feasible_subsystem_network",
        "scddpm_full_subsystem_network",
        "subsystem_timestep_embedding_dim",
        "normalization_type",
        "sampling_weights",
        "compute_mode",
        "training_parent_fraction",
        "data_subset_seed",
        "parent_balanced_sampling",
        "full_data_label_balanced_sampling",
        "full_data_unknown_sampling_mass",
        "unknown_label_policy",
        "matched_training_row_count",
        "ema_decay",
        "ema_update_cadence",
        "validation_every_epochs",
        "validation_row_limit",
        "generator_early_stopping_patience",
        "generator_early_stopping_min_delta",
        "raw_stability_sample_count",
        "classifier_guidance_mode",
        "guidance_probability_threshold",
        "guidance_relu_mode",
        "guidance_probability_floor",
        "guidance_gradient_floor",
        "guidance_max_displacement_rms",
        "guidance_max_displacement_to_noise_rms_ratio",
        "guidance_active_timestep_fraction",
        "guidance_active_timestep_window",
        "guidance_balance_combined_gradient_rms",
        "sampling_x0_clip",
    )
    signature = {
        "schema_version": 2,
        "population": population,
        "architecture": architecture,
        **{key: settings[key] for key in keys},
    }
    if population == "full_data":
        signature["feasibility_condition_encoding"] = "pass_fail"
    for recipe_key in (
        "normalization_fit_scope",
        "normalization_bound_scope",
        "rebatch_every",
        "training_report_every_epochs",
        "sampling_omit_t0",
        "safe_sampling_state",
    ):
        if recipe_key in settings:
            signature[recipe_key] = settings[recipe_key]
    if "learning_rate_schedule_epochs" in settings:
        signature["learning_rate_schedule_epochs"] = int(
            settings["learning_rate_schedule_epochs"]
        )

    unknown_mass = float(settings["full_data_unknown_sampling_mass"])
    known_mass_each = (1.0 - unknown_mass) / 2.0
    signature["full_data_label_balance_provenance"] = {
        "observed_unknown_row_numerator": settings.get(
            "full_data_unknown_sampling_observed_numerator"
        ),
        "observed_training_row_denominator": settings.get(
            "full_data_unknown_sampling_observed_denominator"
        ),
        "target_unknown_mass": unknown_mass,
        "target_known_pass_mass_when_both_present": known_mass_each,
        "target_known_fail_mass_when_both_present": known_mass_each,
        "absent_known_stratum_policy": settings.get(
            "full_data_absent_known_stratum_policy",
            "redistribute_known_mass_equally_across_present_known_strata",
        ),
    }
    return signature


def _training_batch_budgets(
    generator: FirstExperimentGenerator,
    settings: dict,
) -> list[int]:
    total = int(settings["batches_per_epoch"])
    count = len(generator.models)
    if settings["compute_mode"] == "per_model_cshipgen":
        return [total] * count
    if settings["compute_mode"] == "matched_per_coordinate_exposure":
        return [total] * count
    if settings["compute_mode"] != "matched_total_compute":
        raise ValueError(f"Unknown compute_mode {settings['compute_mode']!r}.")
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _configure_training_sampling_probabilities(
    generator: FirstExperimentGenerator,
    data: PreparedFirstExperimentData,
    positions: np.ndarray,
    settings: dict,
) -> None:
    """Assign reproducible parent-aware probabilities to each denoiser."""

    if not bool(settings["parent_balanced_sampling"]):
        return

    if (
        generator.population == "full_data"
        and bool(settings["full_data_label_balanced_sampling"])
    ):
        probability_sets = environment_sampling_probabilities(
            data,
            positions,
            architecture=generator.architecture,
            population=generator.population,
            unknown_mass=float(settings["full_data_unknown_sampling_mass"]),
        )
    else:
        common = data.parent_balancing_probabilities(positions)
        probability_sets = [common.copy() for _ in generator.models]

    for environment, probabilities in zip(
        generator.models,
        probability_sets,
        strict=True,
    ):
        environment.sampling_probabilities = probabilities


def _generator_data_sha256(
    data: PreparedFirstExperimentData,
    population: str,
    settings: dict,
) -> str:
    positions = data.generator_positions(
        population,
        parent_fraction=float(settings["training_parent_fraction"]),
        subset_seed=int(settings["data_subset_seed"]),
        unknown_label_policy=str(settings["unknown_label_policy"]),
        matched_row_count=settings.get("matched_training_row_count"),
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(data.model_values[positions], dtype="<f8").tobytes())
    if population == "full_data":
        digest.update(
            np.asarray(data.global_condition[positions], dtype="<f4").tobytes()
        )
        for name in SUBSYSTEM_ORDER:
            digest.update(
                np.asarray(
                    data.local_conditions[name][positions],
                    dtype="<f4",
                ).tobytes()
            )
    return digest.hexdigest()


def _generator_validation_positions(
    data: PreparedFirstExperimentData,
    population: str,
    settings: dict,
) -> np.ndarray:
    keep = data.row_table["Split"].eq("validation").to_numpy()
    if population in {"feasible_only", "locally_feasible_subsystems"}:
        keep &= data.feasible_mask & data.known_mask
    elif population == "full_data":
        if settings["unknown_label_policy"] == "exclude":
            keep &= data.known_mask
    else:
        raise ValueError(f"Unknown generator population {population!r}.")
    return np.flatnonzero(keep)


def _validation_payloads(
    generator: FirstExperimentGenerator,
    data: PreparedFirstExperimentData,
    settings: dict,
) -> list[tuple[torch.Tensor, list[torch.Tensor] | None]]:
    """Prepare frozen validation tensors in each environment's coordinates."""

    positions = _generator_validation_positions(
        data,
        generator.population,
        settings,
    )
    maximum = int(settings.get("validation_row_limit", 4096))
    positions = positions[:maximum]
    values = data.model_values[positions]
    payloads = []

    for level, (environment, subset) in enumerate(
        zip(generator.models, generator.parameter_subsets, strict=True)
    ):
        if generator.architecture == "monolithic":
            model_values = values
        else:
            model_values = values[:, subset]
        normalized = environment.data_norm.transform_Data(model_values)
        x = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=environment.device,
        )

        condition_values = []
        if generator.conditioned:
            if generator.architecture == "monolithic":
                condition_values.append(data.global_condition[positions])
            else:
                condition_values.append(
                    data.local_conditions[SUBSYSTEM_ORDER[level]][positions]
                )
        if generator.architecture == "hcddpm" and level:
            predecessor = np.concatenate(generator.parameter_subsets[:level])
            condition_values.append(values[:, predecessor])
        elif generator.architecture == "scddpm":
            complement = np.concatenate(
                generator.parameter_subsets[:level]
                + generator.parameter_subsets[level + 1 :]
            )
            condition_values.append(values[:, complement])

        conditions = None
        if condition_values:
            conditions = [
                torch.as_tensor(
                    environment.cond_norm[index].transform_Data(condition),
                    dtype=torch.float32,
                    device=environment.device,
                )
                for index, condition in enumerate(condition_values)
            ]
        payloads.append((x, conditions))
    return payloads


def _validation_denoising_loss(
    environment,
    payload: tuple[torch.Tensor, list[torch.Tensor] | None],
    *,
    seed: int,
) -> float:
    """Record live-weight denoising MSE without choosing a checkpoint."""

    x, conditions = payload
    batch_size = min(environment.batch_size, len(x))
    generator = torch.Generator(device=environment.device)
    generator.manual_seed(int(seed))
    losses = []
    batch_sizes = []

    environment.diffusion.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            clean = x[start : start + batch_size]
            condition_batch = (
                [value[start : start + batch_size] for value in conditions]
                if conditions is not None
                else None
            )
            time_index = torch.randint(
                0, environment.timesteps, (len(clean),),
                generator=generator, device=environment.device,
            )
            noise = torch.randn(
                clean.shape, generator=generator, device=environment.device,
            )
            noisy = environment.q_sample(clean, time_index, noise=noise)
            if condition_batch is None:
                prediction = environment.diffusion(noisy, time_index)
            else:
                prediction = environment.diffusion(
                    noisy, condition_batch, time_index,
                )
            losses.append(F.mse_loss(prediction, noise))
            batch_sizes.append(len(clean))
    environment.diffusion.train()

    batch_losses = torch.stack(losses).cpu().numpy()
    weighted_loss = sum(
        float(loss) * size
        for loss, size in zip(batch_losses, batch_sizes, strict=True)
    )
    return weighted_loss / sum(batch_sizes)


def _train_generator_with_validation(
    generator: FirstExperimentGenerator,
    data: PreparedFirstExperimentData,
    settings: dict,
    budgets: list[int],
    training_state: dict,
    checkpoint_callback,
) -> dict:
    """Train all configured epochs and retain only final live weights."""

    payloads = _validation_payloads(generator, data, settings)
    target_epochs = int(settings["training_epochs"])
    validation_every = int(settings.get("validation_every_epochs", 1))

    training_state.setdefault("stopped", [False] * len(generator.models))
    training_state.setdefault("final_validation_loss", [float("nan")] * len(generator.models))
    training_state.setdefault("validation_history", [[] for _ in generator.models])
    training_state.setdefault("training_diagnostics", [[] for _ in generator.models])

    for environment_index, (environment, budget, payload) in enumerate(
        zip(generator.models, budgets, payloads, strict=True)
    ):
        environment.loss_history = list(np.asarray(environment.loss_history).tolist())
        environment.validation_history = training_state["validation_history"][environment_index]
        environment.training_diagnostics = training_state["training_diagnostics"][environment_index]
        first_epoch = int(training_state["completed_epochs"][environment_index])
        environment.diffusion.train()
        environment.ema.mu = float(settings["ema_decay"])
        batch_indices_gpu = None
        reporting_cadence = max(
            1, int(settings.get("training_report_every_epochs", 1))
        )
        pending_losses = []

        for epoch_index in range(first_epoch, target_epochs):
            if epoch_index % max(1, int(environment.rebatch_every)) == 0:
                training_state["batch_indices"][environment_index] = (
                    _draw_batch_indices(environment, budget)
                )
                batch_indices_gpu = None
            batch_indices = np.asarray(
                training_state["batch_indices"][environment_index],
                dtype=np.int64,
            )
            # Reuse sampled rows across the rebatch interval. Moving indices
            # together avoids a CPU-to-GPU copy at every optimizer step.
            if batch_indices_gpu is None:
                batch_indices_gpu = torch.as_tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=environment.device,
                )
            losses = []
            for index_tensor in batch_indices_gpu:
                x_batch = environment.X.index_select(0, index_tensor)
                if hasattr(environment, "Cond"):
                    condition_batch = [
                        condition.index_select(0, index_tensor)
                        for condition in environment.Cond
                    ]
                    loss = environment.run_step(x_batch, condition_batch)
                else:
                    loss = environment.run_step(x_batch)
                # Update EMA after each optimizer step, as configured.
                # Sampling still uses the final live weights.
                environment.ema.update(environment.diffusion)
                # Keep reporting losses on the GPU until the epoch ends.
                losses.append(loss.detach())

            # Keep epoch losses on the GPU. A single host read per report
            # interval avoids synchronizing the 4090 on every update.
            pending_losses.extend(losses)
            completed = epoch_index + 1
            if completed % reporting_cadence == 0 or completed == target_epochs:
                gradient_norm = float(
                    torch.sqrt(
                        sum(
                            torch.sum(parameter.grad.detach() ** 2)
                            for parameter in environment.diffusion.parameters()
                            if parameter.grad is not None
                        )
                    ).cpu()
                )
                training_loss = float(
                    torch.stack(pending_losses).mean().cpu()
                )
                pending_losses.clear()
                environment.loss_history.append([completed, training_loss])
                environment.training_diagnostics.append(
                    {
                        "Epoch": completed,
                        "Learning_Rate": float(
                            environment.optimizer.param_groups[0]["lr"]
                        ),
                        "Gradient_Norm": gradient_norm,
                    }
                )
            environment._anneal_lr(epoch_index)
            training_state["completed_epochs"][environment_index] = epoch_index + 1

            if (epoch_index + 1) % validation_every == 0:
                validation_loss = _validation_denoising_loss(
                    environment,
                    payload,
                    seed=int(settings["training_seed"]) + environment_index,
                )
                environment.validation_history.append(
                    {"Epoch": epoch_index + 1, "Validation_Loss": validation_loss}
                )
                training_state["final_validation_loss"][environment_index] = validation_loss

            checkpoint_callback(training_state, environment_index)

        if target_epochs % validation_every:
            validation_loss = _validation_denoising_loss(
                environment, payload,
                seed=int(settings["training_seed"]) + environment_index,
            )
            environment.validation_history.append(
                {"Epoch": target_epochs, "Validation_Loss": validation_loss}
            )
            training_state["final_validation_loss"][environment_index] = validation_loss
        environment.loss_history = np.asarray(environment.loss_history, dtype=float)
    return training_state


def _generator_training_is_complete(
    training_state: dict,
    *,
    model_count: int,
    target_epochs: int,
) -> bool:
    """Require the last configured epoch for every denoiser."""

    completed = list(training_state.get("completed_epochs", []))
    return len(completed) == model_count and all(
        int(epoch_count) == int(target_epochs) for epoch_count in completed
    )


def _generator_history_table(
    generator: FirstExperimentGenerator,
    training_state: dict,
    architecture: str,
) -> pd.DataFrame:
    """Reconstruct the training history entirely from checkpointed state."""

    rows = []
    validation_histories = training_state.get(
        "validation_history", [[] for _ in generator.models]
    )
    diagnostic_histories = training_state.get(
        "training_diagnostics", [[] for _ in generator.models]
    )

    for level, environment in enumerate(generator.models):
        label = "All" if architecture == "monolithic" else SUBSYSTEM_ORDER[level]
        validation_by_epoch = {
            int(row["Epoch"]): float(row["Validation_Loss"])
            for row in validation_histories[level]
        }
        diagnostics_by_epoch = {
            int(row["Epoch"]): row for row in diagnostic_histories[level]
        }
        for epoch, loss in np.asarray(environment.loss_history):
            diagnostics = diagnostics_by_epoch.get(int(epoch), {})
            rows.append(
                {
                    "Subsystem": label,
                    "Epoch": int(epoch),
                    "Training_Loss": float(loss),
                    "Validation_Loss": validation_by_epoch.get(
                        int(epoch), np.nan
                    ),
                    "Learning_Rate": diagnostics.get("Learning_Rate", np.nan),
                    "Gradient_Norm": diagnostics.get("Gradient_Norm", np.nan),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "Subsystem",
            "Epoch",
            "Training_Loss",
            "Validation_Loss",
            "Learning_Rate",
            "Gradient_Norm",
        ],
    )


def _generator_summary_payload(
    generator: FirstExperimentGenerator,
    settings: dict,
    signature: dict,
    checkpoint: dict,
    final_path: Path,
    history_path: Path,
    budgets: list[int],
    positions: np.ndarray,
    data_sha256: str,
) -> dict:
    """Build the completion record installed last for a generator run."""

    metadata = checkpoint.get("metadata", {})
    training_state = checkpoint["training_state"]
    return {
        "finalization_schema_version": 1,
        "finalization_ready": True,
        "generator_signature": signature,
        **signature,
        "training_data_sha256": data_sha256,
        "training_row_count": int(len(positions)),
        "trainable_parameters": int(
            metadata["trainable_parameters"]
            if "trainable_parameters" in metadata
            else count_trainable_parameters(generator)
        ),
        "batch_budgets": [int(value) for value in budgets],
        "compute_mode": settings["compute_mode"],
        "optimizer_steps_per_epoch": int(sum(budgets)),
        "coordinate_exposures_per_epoch": int(
            sum(
                budget * len(subset)
                for budget, subset in zip(
                    budgets,
                    generator.parameter_subsets,
                    strict=True,
                )
            )
        ),
        "weight_selection": "final_epoch_live",
        "final_epoch_by_model": training_state.get("completed_epochs", []),
        "final_validation_loss_by_model": training_state.get(
            "final_validation_loss", []
        ),
        "early_stopped_by_model": training_state.get("stopped", []),
        "training_wall_seconds": metadata.get("training_wall_seconds"),
        "peak_gpu_memory_bytes": int(metadata.get("peak_gpu_memory_bytes", 0)),
        "raw_sampling_stability_gate": metadata.get(
            "raw_sampling_stability_gate"
        ),
        "checkpoint_sha256": _sha256_file(final_path),
        "training_history_sha256": _sha256_file(history_path),
    }


def _generator_companions_are_complete(
    *,
    summary_path: Path,
    history_path: Path,
    final_path: Path,
    signature: dict,
    data_sha256: str,
    expected_history_rows: int,
) -> bool:
    """Verify that checkpoint, history, and final completion record agree."""

    if not summary_path.exists() or not history_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        history = pd.read_csv(history_path)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False

    required_columns = {
        "Subsystem",
        "Epoch",
        "Training_Loss",
        "Validation_Loss",
        "Learning_Rate",
        "Gradient_Norm",
    }
    return bool(
        summary.get("finalization_ready") is True
        and summary.get("generator_signature") == signature
        and summary.get("training_data_sha256") == data_sha256
        and summary.get("checkpoint_sha256") == _sha256_file(final_path)
        and summary.get("training_history_sha256") == _sha256_file(history_path)
        and required_columns.issubset(history.columns)
        and len(history) == int(expected_history_rows)
    )


def _finalize_generator_companions(
    generator: FirstExperimentGenerator,
    settings: dict,
    *,
    architecture: str,
    signature: dict,
    checkpoint: dict,
    final_path: Path,
    history_path: Path,
    summary_path: Path,
    budgets: list[int],
    positions: np.ndarray,
    data_sha256: str,
) -> None:
    """Repair or install deterministic companions, with the summary last."""

    history = _generator_history_table(generator, checkpoint["training_state"], architecture)
    if _generator_companions_are_complete(
        summary_path=summary_path,
        history_path=history_path,
        final_path=final_path,
        signature=signature,
        data_sha256=data_sha256,
        expected_history_rows=len(history),
    ):
        return

    # The summary is the commit record.  Installing history first means an
    # interruption can only leave an obviously incomplete finalization, which
    # the next invocation reconstructs from the immutable checkpoint.
    _write_csv_atomic(history, history_path)
    summary = _generator_summary_payload(
        generator,
        settings,
        signature,
        checkpoint,
        final_path,
        history_path,
        budgets,
        positions,
        data_sha256,
    )
    _write_json_atomic(summary, summary_path)

    if not _generator_companions_are_complete(
        summary_path=summary_path,
        history_path=history_path,
        final_path=final_path,
        signature=signature,
        data_sha256=data_sha256,
        expected_history_rows=len(history),
    ):
        raise RuntimeError(
            f"Generator finalization could not be verified for {final_path.parent}."
        )


def _validate_generator_checkpoint(
    checkpoint: dict,
    *,
    signature: dict,
    data_sha256: str,
    row_ids: np.ndarray,
    model_count: int,
    target_epochs: int,
    path: Path,
    require_complete: bool,
) -> None:
    """Validate the immutable training contract recorded in one checkpoint."""

    if checkpoint.get("configuration") != signature:
        raise ValueError(f"Generator configuration changed for {path}.")
    if checkpoint.get("metadata", {}).get("training_data_sha256") != data_sha256:
        raise ValueError(f"Generator training data changed for {path}.")
    if not np.array_equal(
        np.asarray(checkpoint.get("training_indices"), dtype=np.int64),
        np.asarray(row_ids, dtype=np.int64),
    ):
        raise ValueError(f"Generator training rows changed for {path}.")
    if require_complete and not _generator_training_is_complete(
        checkpoint.get("training_state", {}),
        model_count=model_count,
        target_epochs=target_epochs,
    ):
        raise RuntimeError(f"Generator checkpoint is not training-complete: {path}.")


def _aio_latent_stability_summary(latent: np.ndarray | torch.Tensor) -> dict:
    """Measure the final AIO reverse state before quantile inversion."""

    values = np.asarray(latent, dtype=np.float64)
    all_finite = bool(np.isfinite(values).all())
    magnitude_limit = 10.0

    # Compute the square in float64: an unstable float32 latent can be around
    # 1e23, whose square would overflow in float32.
    rms = float(np.sqrt(np.mean(values**2))) if all_finite else None
    max_abs = float(np.max(np.abs(values))) if all_finite else None
    return {
        "latent_all_finite": all_finite,
        "final_latent_rms": rms,
        "final_latent_max_abs": max_abs,
        "latent_max_abs_threshold": magnitude_limit,
        "latent_magnitude_ok": bool(
            all_finite and max_abs <= magnitude_limit
        ),
    }


def _train_one_generator(
    data: PreparedFirstExperimentData,
    settings: dict,
    paths: dict[str, Path],
    *,
    population: str,
    architecture: str,
) -> Path:
    directory = _generator_directory(
        paths,
        population,
        architecture,
        int(settings["training_seed"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "checkpoint.pt"
    recovery_path = directory / "checkpoint.recovery.pt"
    summary_path = directory / "Training_Summary.json"
    history_path = directory / "Training_History.csv"
    signature = _generator_signature(settings, population, architecture)

    seed_everything(int(settings["training_seed"]))
    training_started = time.perf_counter()
    if str(settings["device_name"]).startswith("cuda"):
        # PyTorch requires an initialized CUDA context before resetting stats.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(settings["device_name"]).index)
    generator = build_first_experiment_generator(
        data,
        settings,
        architecture=architecture,
        population=population,
    )
    # A truncated run can retain the source run's learning-rate trajectory.
    schedule_epochs = int(settings.get("learning_rate_schedule_epochs", settings["training_epochs"]))
    for environment in generator.models:
        environment.num_epochs = schedule_epochs
    budgets = _training_batch_budgets(generator, settings)
    positions = data.generator_positions(
        population,
        parent_fraction=float(settings["training_parent_fraction"]),
        subset_seed=int(settings["data_subset_seed"]),
        unknown_label_policy=str(settings["unknown_label_policy"]),
        matched_row_count=settings.get("matched_training_row_count"),
    )
    _configure_training_sampling_probabilities(
        generator,
        data,
        positions,
        settings,
    )
    row_ids = positions.astype(np.int64)
    data_sha256 = _generator_data_sha256(data, population, settings)

    # A final checkpoint alone is not a completed generator job.  Load and
    # verify it, then reconstruct missing or stale companions from its embedded
    # training state before declaring the job reusable.
    if final_path.exists():
        checkpoint = load_complete_checkpoint(
            generator,
            final_path,
            restore_rng=False,
        )
        _validate_generator_checkpoint(
            checkpoint,
            signature=signature,
            data_sha256=data_sha256,
            row_ids=row_ids,
            model_count=len(generator.models),
            target_epochs=int(settings["training_epochs"]),
            path=final_path,
            require_complete=True,
        )
        _finalize_generator_companions(
            generator,
            settings,
            architecture=architecture,
            signature=signature,
            checkpoint=checkpoint,
            final_path=final_path,
            history_path=history_path,
            summary_path=summary_path,
            budgets=budgets,
            positions=positions,
            data_sha256=data_sha256,
        )
        recovery_path.unlink(missing_ok=True)
        return final_path

    training_state = None
    if recovery_path.exists():
        recovery = load_complete_checkpoint(
            generator,
            recovery_path,
            restore_rng=True,
        )
        _validate_generator_checkpoint(
            recovery,
            signature=signature,
            data_sha256=data_sha256,
            row_ids=row_ids,
            model_count=len(generator.models),
            target_epochs=int(settings["training_epochs"]),
            path=recovery_path,
            require_complete=False,
        )
        training_state = recovery.get("training_state")

    checkpoint_every = int(settings["checkpoint_every_epochs"])

    def save_progress(state: dict, environment_index: int) -> None:
        completed = int(sum(state["completed_epochs"]))
        stopped = state.get("stopped", [False] * len(generator.models))[
            environment_index
        ]
        if (
            completed % checkpoint_every
            and state["completed_epochs"][environment_index]
            != int(settings["training_epochs"])
            and not stopped
        ):
            return
        save_recovery_checkpoint(
            generator,
            recovery_path,
            configuration=signature,
            training_indices=row_ids,
            metadata={
                "trainable_parameters": count_trainable_parameters(generator),
                "population_row_count": int(len(positions)),
                "training_data_sha256": data_sha256,
            },
            training_state=state,
        )
        print(
            f"  checkpoint {population}/{architecture}: "
            f"epochs={state['completed_epochs']}",
            flush=True,
        )

    if training_state is None:
        # The generic loop creates the initial state. Saving immediately makes
        # interruption before the first epoch restartable as well.
        training_state = {
            "schema_version": 1,
            "target_epochs": int(settings["training_epochs"]),
            "completed_epochs": [0] * len(generator.models),
            "batch_budgets": budgets,
            "batch_indices": [None] * len(generator.models),
        }
        save_progress(training_state, 0)

    training_state = _train_generator_with_validation(
        generator,
        data,
        settings,
        budgets,
        training_state=training_state,
        checkpoint_callback=save_progress,
    )
    parameter_count = count_trainable_parameters(generator)
    training_seconds = time.perf_counter() - training_started
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(torch.device(settings["device_name"]).index))
        if str(settings["device_name"]).startswith("cuda")
        else 0
    )

    stability_count = int(settings.get("raw_stability_sample_count", 32))
    stability_started = time.perf_counter()
    seed_everything(int(settings["training_seed"]) + 900_000)
    # Capture AIO's final pre-inverse latent through the existing optional
    # sampler path, then restore the environment's previous capture setting.
    aio_environment = generator.models[0] if architecture == "monolithic" else None
    if aio_environment is not None:
        previous_capture = getattr(aio_environment, "capture_sampling_latent", False)
        aio_environment.capture_sampling_latent = True
    try:
        with sampling_weight_context(generator, "live"):
            stability_samples = generator.generate(
                stability_count,
                guidance="none",
            )
    finally:
        if aio_environment is not None:
            aio_environment.capture_sampling_latent = previous_capture
    stability_seconds = time.perf_counter() - stability_started
    if not np.isfinite(stability_samples).all():
        raise ValueError(
            f"Raw sampling stability gate failed for {population}/{architecture}: "
            "non-finite sample coordinates."
        )

    strict_outside = (
        (stability_samples < data.lower.reshape(1, -1))
        | (stability_samples > data.upper.reshape(1, -1))
    )
    coordinate_width = data.upper - data.lower
    bound_scale = np.maximum.reduce(
        (
            np.abs(data.lower),
            np.abs(data.upper),
            np.abs(coordinate_width),
            np.ones_like(data.lower),
        )
    )
    bound_tolerance = (
        8.0 * np.finfo(np.float32).eps * bound_scale
    ).reshape(1, -1)
    outside = (
        (stability_samples < data.lower.reshape(1, -1) - bound_tolerance)
        | (stability_samples > data.upper.reshape(1, -1) + bound_tolerance)
    )
    lower_excursion = np.maximum(
        data.lower.reshape(1, -1) - stability_samples,
        0.0,
    )
    upper_excursion = np.maximum(
        stability_samples - data.upper.reshape(1, -1),
        0.0,
    )
    maximum_excursion = float(
        np.maximum(lower_excursion, upper_excursion).max()
    )
    stability_summary = {
        "sample_count": stability_count,
        "all_finite": True,
        "coordinate_fraction_outside_training_range": float(outside.mean()),
        "coordinate_fraction_outside_training_range_strict": float(
            strict_outside.mean()
        ),
        "maximum_physical_bound_excursion": maximum_excursion,
        "maximum_float32_bound_tolerance": float(bound_tolerance.max()),
        "minimum": float(stability_samples.min()),
        "maximum": float(stability_samples.max()),
        "inference_seconds": stability_seconds,
    }
    if aio_environment is not None:
        stability_summary.update(
            _aio_latent_stability_summary(aio_environment.last_sampling_latent)
        )
    metadata = {
        "trainable_parameters": parameter_count,
        "population_row_count": int(len(positions)),
        "optimizer_steps_per_epoch": int(sum(budgets)),
        "training_data_sha256": data_sha256,
        "weight_selection": "final_epoch_live",
        "final_epoch_by_model": training_state["completed_epochs"],
        "final_validation_loss_by_model": training_state["final_validation_loss"],
        "training_wall_seconds": training_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "raw_sampling_stability_gate": stability_summary,
    }
    save_complete_checkpoint(
        generator,
        final_path,
        configuration=signature,
        training_indices=row_ids,
        metadata=metadata,
        training_state=training_state,
    )

    # Keep the rolling recovery file until history and the final completion
    # record are both installed and verified.
    checkpoint = {
        "configuration": signature,
        "training_indices": row_ids,
        "metadata": metadata,
        "training_state": training_state,
    }
    _finalize_generator_companions(
        generator,
        settings,
        architecture=architecture,
        signature=signature,
        checkpoint=checkpoint,
        final_path=final_path,
        history_path=history_path,
        summary_path=summary_path,
        budgets=budgets,
        positions=positions,
        data_sha256=data_sha256,
    )
    recovery_path.unlink(missing_ok=True)
    if settings.get("safe_sampling_state", False):
        # Keep a tensor-only copy of final live weights for safe diagnostic
        # loading without deserializing the optimizer or NumPy RNG state.
        safe_path = directory / "sampling_live_state.pt"
        safe_payload = {
            "model_states": [
                {name: value.detach().cpu() for name, value in model.diffusion.state_dict().items()}
                for model in generator.models
            ],
            "completed_epochs": [int(value) for value in training_state["completed_epochs"]],
            "generator_signature": signature,
            "training_data_sha256": data_sha256,
        }
        temporary = safe_path.with_name(f".{safe_path.name}.tmp")
        torch.save(safe_payload, temporary)
        os.replace(temporary, safe_path)
    return final_path


def _load_generator(
    data: PreparedFirstExperimentData,
    settings: dict,
    paths: dict[str, Path],
    *,
    population: str,
    architecture: str,
) -> FirstExperimentGenerator:
    generator = build_first_experiment_generator(
        data,
        settings,
        architecture=architecture,
        population=population,
    )
    directory = _generator_directory(
        paths,
        population,
        architecture,
        int(settings["training_seed"]),
    )
    checkpoint_path = directory / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing generator checkpoint {checkpoint_path}.")

    positions = data.generator_positions(
        population,
        parent_fraction=float(settings["training_parent_fraction"]),
        subset_seed=int(settings["data_subset_seed"]),
        unknown_label_policy=str(settings["unknown_label_policy"]),
        matched_row_count=settings.get("matched_training_row_count"),
    )
    data_sha256 = _generator_data_sha256(data, population, settings)
    expected = _generator_signature(settings, population, architecture)
    checkpoint = load_complete_checkpoint(
        generator,
        checkpoint_path,
        restore_rng=False,
    )
    _validate_generator_checkpoint(
        checkpoint,
        signature=expected,
        data_sha256=data_sha256,
        row_ids=positions.astype(np.int64),
        model_count=len(generator.models),
        target_epochs=int(settings["training_epochs"]),
        path=checkpoint_path,
        require_complete=True,
    )

    # Direct sampling receives the same completion guarantee as the full
    # training pipeline. Missing companions are deterministic and repairable
    # from the complete checkpoint; a stale checkpoint itself is rejected.
    _finalize_generator_companions(
        generator,
        settings,
        architecture=architecture,
        signature=expected,
        checkpoint=checkpoint,
        final_path=checkpoint_path,
        history_path=directory / "Training_History.csv",
        summary_path=directory / "Training_Summary.json",
        budgets=_training_batch_budgets(generator, settings),
        positions=positions,
        data_sha256=data_sha256,
    )
    (directory / "checkpoint.recovery.pt").unlink(missing_ok=True)
    return generator


