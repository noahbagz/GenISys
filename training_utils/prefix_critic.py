"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import json


import os


from dataclasses import asdict


from pathlib import Path


import numpy as np


import pandas as pd


import torch


from training_utils.checkpoints import seed_everything


from models.critics import CHECKPOINT_SCHEMA, PerformanceCriticConfig, PrefixPerformanceCritic, _gaussian_nll, _q_sample, _schedule


from training_utils.critic_data import PerformanceGuidanceData


def prefix_mask_bank(
    parameter_subsets: tuple[np.ndarray, ...],
    *,
    control_count: int,
) -> np.ndarray:
    """Return one cumulative predecessor/current visibility mask per level."""

    bank = np.zeros((len(parameter_subsets), int(control_count)), dtype=np.float32)
    observed: list[int] = []
    for level, subset in enumerate(parameter_subsets):
        observed.extend(np.asarray(subset, dtype=int).tolist())
        bank[level, observed] = 1.0
    return bank


def _build_prefix_model(
    data: PerformanceGuidanceData,
    config: PerformanceCriticConfig,
    *,
    level_count: int,
    device: torch.device,
) -> tuple[PrefixPerformanceCritic, dict]:
    model_config = {
        "control_dim": int(data.controls.shape[1]),
        "condition_dim": int(data.conditions.shape[1]),
        "target_dim": int(data.targets.shape[1]),
        "level_count": int(level_count),
        "timestep_embedding_dim": int(config.timestep_embedding_dim),
        "level_embedding_dim": 32,
        "hidden_layers": list(config.hidden_layers),
        "dropout": float(config.dropout),
    }
    model = PrefixPerformanceCritic(
        control_dim=model_config["control_dim"],
        condition_dim=model_config["condition_dim"],
        target_dim=model_config["target_dim"],
        level_count=model_config["level_count"],
        timestep_embedding_dim=model_config["timestep_embedding_dim"],
        level_embedding_dim=model_config["level_embedding_dim"],
        hidden_layers=tuple(model_config["hidden_layers"]),
        dropout=model_config["dropout"],
    ).to(device)
    return model, model_config


def train_or_load_prefix_performance_critic(
    data: PerformanceGuidanceData,
    parameter_subsets: tuple[np.ndarray, ...],
    *,
    config: PerformanceCriticConfig,
    output_directory: str | Path,
    prefix_context_mode: str = "all_noisy",
) -> PrefixPerformanceCritic:
    """Train one shared critic across all twenty HCDDPM prefix levels."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "best_prefix_performance_critic.pt"
    recovery_path = output_directory / "prefix_performance_critic.recovery.pt"
    history_path = output_directory / "Prefix_Critic_Training_History.csv"
    device = torch.device(config.device_name)
    model, model_config = _build_prefix_model(
        data,
        config,
        level_count=len(parameter_subsets),
        device=device,
    )
    masks = prefix_mask_bank(
        parameter_subsets,
        control_count=data.controls.shape[1],
    )

    random = np.random.default_rng(config.training_seed + 1)
    train_positions = data.positions("train")
    validation_positions = data.positions("validation")
    if config.maximum_training_rows is not None:
        count = min(int(config.maximum_training_rows), len(train_positions))
        train_positions = np.sort(random.choice(train_positions, count, replace=False))
    validation_positions = np.sort(
        random.choice(
            validation_positions,
            min(int(config.validation_row_limit), len(validation_positions)),
            replace=False,
        )
    )
    signature = {
        "checkpoint_schema": f"{CHECKPOINT_SCHEMA}.prefix",
        "model_config": model_config,
        "training_config": asdict(config),
        "prefix_context_mode": prefix_context_mode,
        "data_schema": data.manifest["schema"],
        "data_artifact_hash": hashlib.sha256(
            json.dumps(data.manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "parameter_subsets": [subset.tolist() for subset in parameter_subsets],
        "train_positions_hash": hashlib.sha256(
            train_positions.astype("<i8").tobytes()
        ).hexdigest(),
        "validation_positions_hash": hashlib.sha256(
            validation_positions.astype("<i8").tobytes()
        ).hexdigest(),
    }

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(f"Prefix critic contract changed for {checkpoint_path}.")
        model.load_state_dict(payload["model"])
        model.eval()
        return model

    seed_everything(config.training_seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    train_x = torch.as_tensor(
        data.controls[train_positions], dtype=torch.float32, device=device
    )
    train_c = torch.as_tensor(
        data.conditions[train_positions], dtype=torch.float32, device=device
    )
    train_k = torch.as_tensor(
        data.condition_known[train_positions], dtype=torch.float32, device=device
    )
    train_y = torch.as_tensor(
        data.targets[train_positions], dtype=torch.float32, device=device
    )
    validation_x = torch.as_tensor(
        data.controls[validation_positions], dtype=torch.float32, device=device
    )
    validation_c = torch.as_tensor(
        data.conditions[validation_positions], dtype=torch.float32, device=device
    )
    validation_k = torch.as_tensor(
        data.condition_known[validation_positions], dtype=torch.float32, device=device
    )
    validation_y = torch.as_tensor(
        data.targets[validation_positions], dtype=torch.float32, device=device
    )
    mask_bank = torch.as_tensor(masks, dtype=torch.float32, device=device)
    current_mask_bank = torch.zeros_like(mask_bank)
    for level, subset in enumerate(parameter_subsets):
        current_mask_bank[level, np.asarray(subset, dtype=int)] = 1.0
    betas = _schedule(config, device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(config.training_seed + 910_000)
    validation_levels = torch.arange(len(validation_x), device=device) % len(masks)
    validation_timesteps = torch.randint(
        0,
        int(config.diffusion_timesteps),
        (len(validation_x),),
        generator=validation_generator,
        device=device,
    )
    validation_noise = torch.randn(
        validation_x.shape,
        generator=validation_generator,
        device=device,
    )
    validation_mask = mask_bank[validation_levels]
    validation_noisy = _q_sample(
        validation_x,
        validation_timesteps,
        alpha_bar,
        noise=validation_noise,
    )
    if prefix_context_mode == "clean_predecessors":
        current_mask = current_mask_bank[validation_levels]
        validation_state = (
            validation_x * (validation_mask - current_mask)
            + validation_noisy * current_mask
        )
    else:
        validation_state = validation_noisy * validation_mask

    first_epoch = 0
    history: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model = None
    if recovery_path.exists():
        payload = torch.load(recovery_path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(f"Prefix critic contract changed for {recovery_path}.")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first_epoch = int(payload["completed_epochs"])
        history = list(payload["history"])
        best_validation_loss = float(payload["best_validation_loss"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        best_model = payload["best_model"]

    def validation_loss() -> float:
        model.eval()
        total = torch.zeros((), device=device)
        batch_size = min(int(config.batch_size), len(validation_x))
        with torch.no_grad():
            for start in range(0, len(validation_x), batch_size):
                stop = min(start + batch_size, len(validation_x))
                prediction, log_variance = model(
                    validation_state[start:stop],
                    validation_mask[start:stop],
                    validation_levels[start:stop],
                    validation_timesteps[start:stop],
                    validation_c[start:stop],
                    validation_k[start:stop],
                )
                loss = _gaussian_nll(
                    prediction,
                    log_variance,
                    validation_y[start:stop],
                )
                total += loss * (stop - start)
        model.train()
        return float((total / len(validation_x)).item())

    print(
        f"Prefix performance critic: {first_epoch}/{config.training_epochs} epochs complete",
        flush=True,
    )
    for epoch in range(first_epoch, int(config.training_epochs)):
        torch.manual_seed(config.training_seed + 1 + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training_seed + 1 + epoch)
        batch_size = min(int(config.batch_size), len(train_x))
        loss_total = torch.zeros((), device=device)
        model.train()
        for _ in range(int(config.batches_per_epoch)):
            index_tensor = torch.randint(
                0, len(train_x), (batch_size,), device=device,
            )
            level_tensor = torch.randint(
                0, len(masks), (batch_size,), device=device,
            )
            timestep = torch.randint(
                0,
                int(config.diffusion_timesteps),
                (batch_size,),
                device=device,
            )
            mask = mask_bank[level_tensor]
            clean = train_x[index_tensor]
            noisy = _q_sample(clean, timestep, alpha_bar)
            if prefix_context_mode == "clean_predecessors":
                current_mask = current_mask_bank[level_tensor]
                state = clean * (mask - current_mask) + noisy * current_mask
            else:
                state = noisy * mask

            optimizer.zero_grad(set_to_none=True)
            prediction, log_variance = model(
                state,
                mask,
                level_tensor,
                timestep,
                train_c[index_tensor],
                train_k[index_tensor],
            )
            loss = _gaussian_nll(
                prediction,
                log_variance,
                train_y[index_tensor],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            loss_total += loss.detach()

        training_loss = float((loss_total / int(config.batches_per_epoch)).item())
        held_out = validation_loss()
        history.append(
            {
                "Epoch": epoch + 1,
                "Training_NLL": training_loss,
                "Validation_NLL": held_out,
            }
        )
        improved = (
            held_out
            < best_validation_loss - float(config.early_stopping_min_delta)
        )
        if improved:
            best_validation_loss = held_out
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        print(
            f"  prefix epoch {epoch + 1}/{config.training_epochs}: "
            f"train_nll={history[-1]['Training_NLL']:.6f}, "
            f"validation_nll={held_out:.6f}",
            flush=True,
        )
        recovery_payload = {
            "schema": f"{CHECKPOINT_SCHEMA}.prefix",
            "signature": signature,
            "model_config": model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": epoch + 1,
            "history": history,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model": best_model,
            "epochs_without_improvement": epochs_without_improvement,
            "scalers": data.scalers,
        }
        temporary = recovery_path.with_name(f".{recovery_path.name}.tmp")
        torch.save(recovery_payload, temporary)
        os.replace(temporary, recovery_path)
        if epochs_without_improvement >= int(config.early_stopping_patience):
            break

    final = torch.load(recovery_path, map_location=device, weights_only=False)
    if final["best_model"] is None:
        raise RuntimeError("Prefix performance critic produced no best checkpoint.")
    final["model"] = final["best_model"]
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(final, temporary)
    os.replace(temporary, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    recovery_path.unlink()
    model.load_state_dict(final["model"])
    model.eval()
    return model


def load_prefix_performance_critic_checkpoint(
    checkpoint_path: str | Path,
    *,
    device_name: str,
) -> tuple[PrefixPerformanceCritic, dict]:
    """Load a level-aware HCDDPM critic for sampling."""

    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["model_config"]
    model = PrefixPerformanceCritic(
        control_dim=int(config["control_dim"]),
        condition_dim=int(config["condition_dim"]),
        target_dim=int(config["target_dim"]),
        level_count=int(config["level_count"]),
        timestep_embedding_dim=int(config["timestep_embedding_dim"]),
        level_embedding_dim=int(config["level_embedding_dim"]),
        hidden_layers=tuple(config["hidden_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


__all__ = [
    "load_prefix_performance_critic_checkpoint",
    "prefix_mask_bank",
    "train_or_load_prefix_performance_critic",
]


