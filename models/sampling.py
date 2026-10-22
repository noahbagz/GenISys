"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


from dataclasses import dataclass


import numpy as np


import torch


from models import diffusion as TDT


from training_utils.hull_guidance_data import HULL_SUBSYSTEMS


from training_utils.data import LOCAL_LABELS_BY_SUBSYSTEM, SUBSYSTEM_ORDER


from models.generators import FirstExperimentGenerator, GuidanceClassifiers


from models.critics import FullStatePerformanceCritic, PrefixPerformanceCritic


GUIDANCE_START_TIMESTEP = 799


@dataclass(frozen=True)
class SamplingReport:
    architecture: str
    requested: int
    guidance_start_timestep: int
    active_guided_steps: int
    editable_coordinates: int
    maximum_absolute_latent: float


def _guidance_active(timestep: int, performance_or_constraint: bool) -> bool:
    return bool(performance_or_constraint and 1 <= timestep <= GUIDANCE_START_TIMESTEP)


def _physical_to_latent(
    generator: FirstExperimentGenerator,
    physical: np.ndarray,
) -> np.ndarray:
    """Transform complete physical controls into each denoiser's coordinates."""

    physical = np.asarray(physical, dtype=np.float64)
    latent = np.empty_like(physical, dtype=np.float64)
    if generator.architecture == "monolithic":
        return generator.environments[0].data_norm.transform_Data(physical)
    for environment, subset in zip(
        generator.environments, generator.parameter_subsets, strict=True,
    ):
        latent[:, subset] = environment.data_norm.transform_Data(physical[:, subset])
    return latent


def _latent_to_physical(
    generator: FirstExperimentGenerator,
    latent: torch.Tensor,
) -> np.ndarray:
    values = latent.detach().cpu().numpy()
    physical = np.empty_like(values, dtype=np.float64)
    if generator.architecture == "monolithic":
        physical = generator.environments[0].data_norm.scale_X(values)
    else:
        for environment, subset in zip(
            generator.environments, generator.parameter_subsets, strict=True,
        ):
            physical[:, subset] = environment.data_norm.scale_X(values[:, subset])
    return generator._binary_physical_controls(physical)


def _locked_state(
    clean: torch.Tensor,
    fixed_noise: torch.Tensor,
    environment,
    timestep: int,
) -> torch.Tensor:
    """Return q(x_t | x_0) for locked coordinates along one fixed noise path."""

    if timestep <= 0:
        return clean
    alpha_bar = environment.alphas_cumprod[int(timestep)]
    return torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * fixed_noise


def _masked_unit_descent(
    values: torch.Tensor,
    predict,
    objective_index: int,
    editable_mask: torch.Tensor,
) -> torch.Tensor:
    """Return unit-L2 descent gradients restricted to editable coordinates."""

    with torch.enable_grad():
        differentiable = values.detach().requires_grad_(True)
        prediction, _ = predict(differentiable)
        gradient = torch.autograd.grad(
            prediction[:, int(objective_index)].sum(), differentiable,
        )[0]
    gradient = -torch.nan_to_num(gradient) * editable_mask
    return TDT.normalize_classifier_gradient_l2(gradient)


def _masked_constraint_gradient(
    classifier,
    values: torch.Tensor,
    timestep: torch.Tensor,
    target: np.ndarray,
    environment,
    editable_mask: torch.Tensor,
) -> torch.Tensor:
    gradient = TDT.classifier_probability_hinge_gradient(
        classifier,
        values,
        timestep,
        target,
        probability_threshold=environment.guidance_probability_threshold,
        relu_mode=environment.guidance_relu_mode,
        probability_floor=environment.guidance_probability_floor,
        gradient_floor=environment.guidance_gradient_floor,
        probability_objective=environment.guidance_probability_objective,
    )
    return torch.nan_to_num(gradient) * editable_mask


def _cap_move(move: torch.Tensor, beta: torch.Tensor, coordinate_count: int) -> torch.Tensor:
    limit = torch.sqrt(beta) * np.sqrt(max(int(coordinate_count), 1))
    return TDT.cap_guidance_displacement(move, limit)


@torch.no_grad()
def sample_masked_aio(
    generator: FirstExperimentGenerator,
    classifiers: GuidanceClassifiers,
    critic: FullStatePerformanceCritic,
    parent_controls: np.ndarray,
    editable_indices: np.ndarray,
    *,
    feasibility_alpha: float = 0.7,
    performance_alpha: float = 0.0,
    objective_index: int = 0,
    rng=None,
) -> tuple[np.ndarray, SamplingReport]:
    """Run full AiO diffusion while holding the complement to each parent."""

    if generator.architecture != "monolithic" or generator.population != "feasible_only":
        raise ValueError("Masked AiO sampling requires the feasible-only monolithic model.")
    parent_controls = np.asarray(parent_controls, dtype=np.float64)
    count = len(parent_controls)
    environment = generator.environments[0]
    environment.diffusion.eval()
    classifiers.eval()
    critic.eval()
    device = generator.device

    editable = torch.zeros((1, generator.xdim), dtype=torch.float32, device=device)
    editable[:, np.asarray(editable_indices, dtype=int)] = 1.0
    locked = 1.0 - editable
    clean = torch.as_tensor(
        _physical_to_latent(generator, parent_controls), dtype=torch.float32, device=device,
    )
    fixed_noise = rng.draw(clean, 'lock')
    x = rng.draw(clean, 'initial')
    x = x * editable + _locked_state(clean, fixed_noise, environment, 999) * locked
    hull_indices = torch.as_tensor(generator.hull_indices, dtype=torch.long, device=device)
    hull_mask = editable.index_select(1, hull_indices)
    empty = torch.empty((count, 0), dtype=torch.float32, device=device)
    maximum = torch.zeros((), device=device)
    active_steps = 0

    for timestep in environment.reverse_step_indices():
        time = torch.full((count,), timestep, dtype=torch.long, device=device)
        predicted_noise = environment.diffusion(x, time)
        mean = environment._reverse_mean_from_epsilon(x, timestep, predicted_noise)
        if timestep == 0:
            x = mean * editable + clean * locked
            maximum = torch.maximum(maximum, torch.amax(torch.abs(x)))
            continue
        variance = environment.extract(environment.posterior_variance, time, x.shape)
        standard_deviation = torch.sqrt(variance)
        noise = rng.draw(x, 'innovation')
        active = _guidance_active(
            timestep, feasibility_alpha > 0.0 or performance_alpha > 0.0,
        )
        if not active:
            proposed = mean + standard_deviation * noise
        else:
            global_gradient = _masked_constraint_gradient(
                classifiers.global_feasibility, x, time,
                np.asarray([1.0], dtype=np.float32), environment, editable,
            )
            hull_gradient = _masked_constraint_gradient(
                classifiers.hull_feasibility, x.index_select(1, hull_indices), time,
                np.asarray([1.0], dtype=np.float32), environment, hull_mask,
            )
            full_hull = torch.zeros_like(x)
            full_hull[:, hull_indices] = hull_gradient
            constraint_move = feasibility_alpha * standard_deviation * (
                global_gradient + full_hull
            )
            performance_move = torch.zeros_like(x)
            if performance_alpha > 0.0:
                direction = _masked_unit_descent(
                    x,
                    lambda state: critic(state, time, empty, empty),
                    objective_index,
                    editable,
                )
                performance_move = performance_alpha * standard_deviation * direction
            move = _cap_move(
                constraint_move + performance_move,
                environment.betas[timestep],
                len(editable_indices),
            )
            proposed = mean + (1.0 - feasibility_alpha) * standard_deviation * noise + move
            active_steps += 1
        next_locked = _locked_state(clean, fixed_noise, environment, timestep - 1)
        x = proposed * editable + next_locked * locked
        maximum = torch.maximum(maximum, torch.amax(torch.abs(x)))

    physical = _latent_to_physical(generator, x)
    physical[:, locked[0].bool().cpu().numpy()] = parent_controls[:, locked[0].bool().cpu().numpy()]
    return physical, SamplingReport(
        "monolithic", count, GUIDANCE_START_TIMESTEP, active_steps,
        len(editable_indices), float(maximum.cpu()),
    )


@torch.no_grad()
def sample_masked_hcddpm(
    generator: FirstExperimentGenerator,
    classifiers: GuidanceClassifiers,
    critic: PrefixPerformanceCritic,
    parent_controls: np.ndarray,
    editable_indices: np.ndarray,
    *,
    feasibility_alpha: float = 0.7,
    performance_alpha: float = 0.0,
    objective_index: int = 0,
    rng=None,
) -> tuple[np.ndarray, SamplingReport]:
    """Regenerate editable HCDDPM blocks and preserve all other parent controls."""

    if generator.architecture != "hcddpm" or generator.population != "feasible_only":
        raise ValueError("Masked HCDDPM sampling requires the feasible-only HCDDPM.")
    parent_controls = np.asarray(parent_controls, dtype=np.float64)
    count = len(parent_controls)
    device = generator.device
    generated = parent_controls.copy()
    completed_latent = torch.as_tensor(
        _physical_to_latent(generator, parent_controls), dtype=torch.float32, device=device,
    )
    requested = set(int(value) for value in np.asarray(editable_indices, dtype=int))
    local_classifiers = classifiers.local_for("hcddpm", "cshipgen_noise_blend_hinge")
    empty = torch.empty((count, 0), dtype=torch.float32, device=device)
    maximum = torch.zeros((), device=device)
    active_steps = 0

    for level, (environment, subset) in enumerate(
        zip(generator.environments, generator.parameter_subsets, strict=True),
    ):
        local_positions = [i for i, value in enumerate(subset) if int(value) in requested]
        if not local_positions:
            continue
        environment.diffusion.eval()
        subsystem = SUBSYSTEM_ORDER[level]
        classifier = local_classifiers[subsystem]
        classifier.eval()
        previous = np.concatenate(generator.parameter_subsets[:level]) if level else None
        conditions = [generated[:, previous]] if previous is not None else []
        cond_target, _ = environment._prepare_cfg_condition_tensors(
            conditions, count,
        ) if conditions else (None, None)
        clean_block = torch.as_tensor(
            environment.data_norm.transform_Data(parent_controls[:, subset]),
            dtype=torch.float32, device=device,
        )
        editable = torch.zeros((1, len(subset)), dtype=torch.float32, device=device)
        editable[:, local_positions] = 1.0
        locked = 1.0 - editable
        fixed_noise = rng.draw(clean_block, 'lock')
        x = rng.draw(clean_block, 'initial')
        x = x * editable + _locked_state(clean_block, fixed_noise, environment, 999) * locked
        target = np.ones(len(LOCAL_LABELS_BY_SUBSYSTEM[subsystem]), dtype=np.float32)
        observed = torch.zeros_like(completed_latent)
        prefix = np.concatenate(generator.parameter_subsets[: level + 1])
        observed[:, prefix] = 1.0
        level_tensor = torch.full((count,), level, dtype=torch.long, device=device)

        for timestep in environment.reverse_step_indices():
            time = torch.full((count,), timestep, dtype=torch.long, device=device)
            predicted = (
                environment.diffusion(x, cond_target, time)
                if cond_target is not None else environment.diffusion(x, time)
            )
            mean = environment._reverse_mean_from_epsilon(x, timestep, predicted)
            if timestep == 0:
                x = mean * editable + clean_block * locked
                continue
            variance = environment.extract(environment.posterior_variance, time, x.shape)
            standard_deviation = torch.sqrt(variance)
            noise = rng.draw(x, 'innovation')
            active = _guidance_active(
                timestep, feasibility_alpha > 0.0 or performance_alpha > 0.0,
            )
            if not active:
                proposed = mean + standard_deviation * noise
            else:
                classifier_values = (
                    torch.cat((x, cond_target[-1]), dim=1)
                    if cond_target is not None else x
                )
                classifier_mask = torch.zeros_like(classifier_values)
                classifier_mask[:, : len(subset)] = editable
                gradient = _masked_constraint_gradient(
                    classifier, classifier_values, time, target,
                    environment, classifier_mask,
                )[:, : len(subset)]
                constraint_move = feasibility_alpha * standard_deviation * gradient
                performance_move = torch.zeros_like(x)
                if performance_alpha > 0.0:
                    def predict(current: torch.Tensor):
                        state = completed_latent.detach().clone()
                        state[:, subset] = current
                        return critic(
                            state * observed, observed, level_tensor, time, empty, empty,
                        )
                    direction = _masked_unit_descent(
                        x, predict, objective_index, editable,
                    )
                    performance_move = performance_alpha * standard_deviation * direction
                move = _cap_move(
                    constraint_move + performance_move,
                    environment.betas[timestep], len(local_positions),
                )
                proposed = mean + (1.0 - feasibility_alpha) * standard_deviation * noise + move
                active_steps += 1
            next_locked = _locked_state(clean_block, fixed_noise, environment, timestep - 1)
            x = proposed * editable + next_locked * locked
            maximum = torch.maximum(maximum, torch.amax(torch.abs(x)))

        completed_latent[:, subset] = x
        block = environment.data_norm.scale_X(x.detach().cpu().numpy())
        generated[:, subset[local_positions]] = block[:, local_positions]
        if generator.boolean_indices is not None:
            block_booleans = np.intersect1d(generator.boolean_indices, subset)
            generated[:, block_booleans] = (generated[:, block_booleans] >= 0.5).astype(float)

    locked_indices = np.asarray(sorted(set(range(generator.xdim)) - requested), dtype=int)
    generated[:, locked_indices] = parent_controls[:, locked_indices]
    return generated, SamplingReport(
        "hcddpm", count, GUIDANCE_START_TIMESTEP, active_steps,
        len(editable_indices), float(maximum.cpu()),
    )


@torch.no_grad()
def sample_masked_scddpm(
    generator: FirstExperimentGenerator,
    classifiers: GuidanceClassifiers,
    critic: FullStatePerformanceCritic,
    parent_controls: np.ndarray,
    editable_indices: np.ndarray,
    *,
    feasibility_alpha: float = 0.7,
    performance_alpha: float = 0.0,
    objective_index: int = 0,
    rng=None,
) -> tuple[np.ndarray, SamplingReport]:
    """Run simultaneous block denoising with exact post-merge locks."""

    if generator.architecture != "scddpm" or generator.population != "feasible_only":
        raise ValueError("Masked SCDDPM sampling requires the feasible-only SCDDPM.")
    parent_controls = np.asarray(parent_controls, dtype=np.float64)
    count = len(parent_controls)
    device = generator.device
    reference = generator.environments[0]
    for environment in generator.environments:
        environment.diffusion.eval()
    classifiers.eval()
    critic.eval()
    editable = torch.zeros((1, generator.xdim), dtype=torch.float32, device=device)
    editable[:, np.asarray(editable_indices, dtype=int)] = 1.0
    locked = 1.0 - editable
    clean = torch.as_tensor(
        _physical_to_latent(generator, parent_controls), dtype=torch.float32, device=device,
    )
    fixed_noise = rng.draw(clean, 'lock')
    x = rng.draw(clean, 'initial')
    x = x * editable + _locked_state(clean, fixed_noise, reference, 999) * locked
    local_classifiers = classifiers.local_for("scddpm", "cshipgen_noise_blend_hinge")
    hull_indices = torch.as_tensor(generator.hull_indices, dtype=torch.long, device=device)
    hull_position = {int(index): position for position, index in enumerate(generator.hull_indices)}
    empty = torch.empty((count, 0), dtype=torch.float32, device=device)
    maximum = torch.zeros((), device=device)
    active_steps = 0

    for timestep in reference.reverse_step_indices():
        context = x.clone()
        time = torch.full((count,), timestep, dtype=torch.long, device=device)
        means, variances = [], []
        for level, (environment, subset) in enumerate(
            zip(generator.environments, generator.parameter_subsets, strict=True),
        ):
            complement = np.concatenate(
                generator.parameter_subsets[:level] + generator.parameter_subsets[level + 1 :]
            )
            predicted = environment.diffusion(x[:, subset], [context[:, complement]], time)
            means.append(environment._reverse_mean_from_epsilon(x[:, subset], timestep, predicted))
            variances.append(environment.extract(environment.posterior_variance, time, x[:, subset].shape))
        full_mean = torch.empty_like(x)
        for subset, mean in zip(generator.parameter_subsets, means, strict=True):
            full_mean[:, subset] = mean
        if timestep == 0:
            x = full_mean * editable + clean * locked
            continue
        full_variance = torch.empty_like(x)
        full_noise = torch.empty_like(x)
        next_x = torch.empty_like(x)
        active = _guidance_active(
            timestep, feasibility_alpha > 0.0 or performance_alpha > 0.0,
        )
        for level, (environment, subset, mean, variance) in enumerate(
            zip(generator.environments, generator.parameter_subsets, means, variances, strict=True),
        ):
            noise = rng.draw(mean, 'innovation')
            subset_mask = editable[:, subset]
            if not active or not bool(subset_mask.any()):
                proposed = mean + torch.sqrt(variance) * noise
            else:
                subsystem = SUBSYSTEM_ORDER[level]
                hull_local = generator.hull_local_guidance and subsystem in HULL_SUBSYSTEMS
                classifier_values = x.index_select(1, hull_indices) if hull_local else x
                classifier_mask = editable.index_select(1, hull_indices) if hull_local else editable
                gradient = _masked_constraint_gradient(
                    local_classifiers[subsystem], classifier_values, time,
                    np.ones(len(LOCAL_LABELS_BY_SUBSYSTEM[subsystem]), dtype=np.float32),
                    environment, classifier_mask,
                )
                gradient = (
                    gradient[:, [hull_position[int(index)] for index in subset]]
                    if hull_local else gradient[:, subset]
                )
                move = feasibility_alpha * torch.sqrt(variance) * gradient
                editable_count = int(subset_mask.sum().item())
                move = _cap_move(move, environment.betas[timestep], editable_count)
                proposed = mean + (1.0 - feasibility_alpha) * torch.sqrt(variance) * noise + move
            next_x[:, subset] = proposed
            full_variance[:, subset] = variance
            full_noise[:, subset] = noise
        if active:
            system_time = torch.full_like(time, timestep - 1)
            global_gradient = _masked_constraint_gradient(
                classifiers.global_feasibility, next_x, system_time,
                np.asarray([1.0], dtype=np.float32), reference, editable,
            )
            system_move = feasibility_alpha * torch.sqrt(full_variance) * global_gradient
            performance_move = torch.zeros_like(x)
            if performance_alpha > 0.0:
                direction = _masked_unit_descent(
                    next_x,
                    lambda state: critic(state, system_time, empty, empty),
                    objective_index,
                    editable,
                )
                performance_move = performance_alpha * torch.sqrt(full_variance) * direction
            full_count = len(editable_indices)
            system_move = _cap_move(system_move, reference.betas[timestep], full_count)
            combined_base = full_mean + (1.0 - feasibility_alpha) * torch.sqrt(full_variance) * full_noise
            combined = next_x + system_move + performance_move - combined_base
            combined = _cap_move(combined, reference.betas[timestep], full_count)
            proposed_full = combined_base + combined
            active_steps += 1
        else:
            proposed_full = next_x
        next_locked = _locked_state(clean, fixed_noise, reference, timestep - 1)
        x = proposed_full * editable + next_locked * locked
        maximum = torch.maximum(maximum, torch.amax(torch.abs(x)))

    physical = _latent_to_physical(generator, x)
    locked_numpy = locked[0].bool().cpu().numpy()
    physical[:, locked_numpy] = parent_controls[:, locked_numpy]
    return physical, SamplingReport(
        "scddpm", count, GUIDANCE_START_TIMESTEP, active_steps,
        len(editable_indices), float(maximum.cpu()),
    )


SAMPLERS = {
    "monolithic": sample_masked_aio,
    "hcddpm": sample_masked_hcddpm,
    "scddpm": sample_masked_scddpm,
}


__all__ = [
    "GUIDANCE_START_TIMESTEP",
    "SAMPLERS",
    "SamplingReport",
    "sample_masked_aio",
    "sample_masked_hcddpm",
    "sample_masked_scddpm",
]


