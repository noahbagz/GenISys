"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import json


import hashlib


import os


from dataclasses import dataclass


from pathlib import Path


import numpy as np


import pandas as pd


import torch


import torch.nn.functional as F


from sklearn.metrics import average_precision_score, balanced_accuracy_score


from tqdm import tqdm


from models import diffusion as TDT


from training_utils.hull_guidance_data import HULL_CONTROLS, HULL_SUBSYSTEMS


from training_utils.data import LOCAL_LABELS_BY_SUBSYSTEM, PreparedFirstExperimentData, SUBSYSTEM_ORDER, feasible_condition_target


from training_utils.schema import V1p1ParameterSchema


@dataclass(frozen=True)
class AIOBoundaryPull:
    """Optional clean-coordinate boundary pull for a sampling diagnostic."""

    continuous_indices: np.ndarray
    limit: float
    gamma: float
    final_steps: int


@dataclass
class FirstExperimentGenerator:
    """A uniform interface over the three retained generator architectures."""

    architecture: str
    population: str
    models: list
    parameter_subsets: tuple[np.ndarray, ...]
    conditioned: bool
    local_targets: tuple[np.ndarray, ...]
    global_target: np.ndarray
    device: torch.device
    system_guidance_target: str = "system_constraints"
    hull_indices: np.ndarray | None = None
    aio_hull_guidance: bool = False
    hull_local_guidance: bool = False
    boolean_indices: np.ndarray | None = None
    capture_sampling_latent: bool = False
    last_sampling_latent: np.ndarray | None = None
    capture_sampling_trace: bool = False
    last_sampling_trace: dict | None = None

    @property
    def environments(self) -> list:
        return self.models

    @property
    def xdim(self) -> int:
        return int(sum(len(values) for values in self.parameter_subsets))

    def _binary_physical_controls(self, values: np.ndarray) -> np.ndarray:
        """Represent physical Boolean design controls as exact zero or one."""

        physical = np.asarray(values, dtype=np.float64)
        if self.boolean_indices is None:
            return physical

        physical = physical.copy()
        physical[:, self.boolean_indices] = (
            physical[:, self.boolean_indices] >= 0.5
        ).astype(np.float64)
        return physical

    def generate(
        self,
        count: int,
        *,
        guidance: str,
        classifiers: "GuidanceClassifiers | None" = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Generate physical 133-control vectors for one matrix treatment."""

        if guidance not in {"none", "subsystem", "system", "combined"}:
            raise ValueError(f"Unknown guidance mode {guidance!r}.")
        if guidance != "none" and classifiers is None:
            raise ValueError("Classifier guidance requires loaded classifiers.")
        # Zero guidance must use the identical unguided reverse path. The
        # classifier sampler can otherwise consume a different random stream
        # or follow a slightly different calculation even at alpha zero.
        if float(alpha) == 0.0:
            guidance = "none"

        if self.architecture == "monolithic":
            physical = self._generate_monolithic(
                count,
                guidance=guidance,
                classifiers=classifiers,
                alpha=alpha,
            )
        elif self.architecture == "hcddpm":
            physical = self._generate_hierarchical(
                count,
                guidance=guidance,
                classifiers=classifiers,
                alpha=alpha,
            )
        elif self.architecture == "scddpm":
            physical = self._generate_simultaneous(
                count,
                guidance=guidance,
                classifiers=classifiers,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Unknown architecture {self.architecture!r}.")

        # The denoisers use continuous quantile coordinates. Their returned
        # physical controls must still honor the Boolean design vocabulary.
        return self._binary_physical_controls(physical)

    def _generate_monolithic(
        self,
        count: int,
        *,
        guidance: str,
        classifiers: "GuidanceClassifiers | None",
        alpha: float,
    ) -> np.ndarray:
        if guidance not in {"none", "system"}:
            raise ValueError("The all-in-one model supports system guidance only.")

        environment = self.models[0]
        condition = [self.global_target]
        if guidance == "none":
            if self.conditioned:
                result = environment.gen_samples(count, condition)
            else:
                result = environment.gen_samples(count)
        elif self.aio_hull_guidance:
            return self._generate_monolithic_dual_hull_guidance(
                count, classifiers=classifiers, alpha=alpha,
            )
        elif self.conditioned:
            result = environment.gen_conditional_guided_samples(
                count,
                condition,
                classifiers.global_feasibility,
                alpha=alpha,
                class_target=np.asarray([1.0], dtype=np.float32),
            )
        else:
            result = environment.generate_guided_samples(
                count,
                classifiers.global_feasibility,
                alpha=alpha,
            )
        if getattr(environment, "capture_sampling_latent", False):
            environment.last_sampling_latent = result.raw.detach().cpu().numpy()
        return np.asarray(result.scaled, dtype=np.float64)

    @torch.no_grad()
    def _generate_monolithic_dual_hull_guidance(
        self, count: int, *, classifiers: "GuidanceClassifiers", alpha: float,
        boundary_pull: AIOBoundaryPull | None = None,
    ) -> np.ndarray:
        """Blend two separately gated gradients into each AIO reverse step."""

        environment = self.models[0]
        if environment.classifier_guidance_mode != "cshipgen_noise_blend_hinge":
            raise ValueError("Dual AIO guidance requires the C_ShipGen noise blend.")
        if classifiers.hull_feasibility is None or self.hull_indices is None:
            raise RuntimeError("The AIO hull guidance classifier is missing.")

        environment.diffusion.eval()
        classifiers.eval()
        x = torch.randn((count, self.xdim), device=self.device)
        condition = None
        if self.conditioned:
            condition, _ = environment._prepare_cfg_condition_tensors(
                [self.global_target], count,
            )

        hull_indices = torch.as_tensor(
            self.hull_indices, dtype=torch.long, device=self.device,
        )
        continuous_indices = None
        if boundary_pull is not None:
            continuous_indices = torch.as_tensor(
                boundary_pull.continuous_indices,
                dtype=torch.long, device=self.device,
            )

        boundary_start_row_max = None
        trace_timesteps = []
        trace_max_abs = []
        for timestep in tqdm(environment.reverse_step_indices(), desc="AIO dual guidance"):
            if boundary_pull is not None and timestep == boundary_pull.final_steps:
                # Retain this diagnostic on the GPU until sampling completes.
                boundary_start_row_max = torch.amax(torch.abs(x), dim=1).detach()
            time = torch.full((count,), timestep, dtype=torch.int64, device=self.device)
            predicted = (
                environment.diffusion(x, condition, time)
                if self.conditioned else environment.diffusion(x, time)
            )
            mean = environment._reverse_mean_from_epsilon(x, timestep, predicted)
            if timestep == 0:
                x = mean
                if self.capture_sampling_trace:
                    trace_timesteps.append(timestep)
                    trace_max_abs.append(torch.amax(torch.abs(x)).detach())
                continue

            variance = environment.extract(environment.posterior_variance, time, x.shape)
            standard_deviation = torch.sqrt(variance)
            noise = torch.randn_like(x)

            # This optional diagnostic uses the same epsilon prediction to
            # estimate the clean vector. It leaves every discrete control and
            # every early reverse step untouched.
            boundary_direction = None
            if boundary_pull is not None and timestep <= boundary_pull.final_steps:
                alpha_bar = environment.alphas_cumprod[timestep]
                clean_estimate = (
                    x - torch.sqrt(1.0 - alpha_bar) * predicted
                ) / torch.sqrt(alpha_bar)
                clean_continuous = clean_estimate[:, continuous_indices]
                inward = (
                    torch.sigmoid(-boundary_pull.limit - clean_continuous)
                    - torch.sigmoid(clean_continuous - boundary_pull.limit)
                )
                boundary_direction = torch.zeros_like(x)
                boundary_direction[:, continuous_indices] = boundary_pull.gamma * inward

            if not environment._guidance_timestep_is_active(timestep):
                x = mean + standard_deviation * noise
                if boundary_direction is not None:
                    x = x + standard_deviation * boundary_direction
                if self.capture_sampling_trace:
                    trace_timesteps.append(timestep)
                    trace_max_abs.append(torch.amax(torch.abs(x)).detach())
                continue

            # Each classifier receives the same alpha and its own ReLU gate.
            # The hull gradient changes only the 42 hull controls.
            global_gradient = TDT.classifier_probability_hinge_gradient(
                classifiers.global_feasibility, x, time, np.asarray([1.0]),
                probability_threshold=environment.guidance_probability_threshold,
                relu_mode=environment.guidance_relu_mode,
                probability_floor=environment.guidance_probability_floor,
                gradient_floor=environment.guidance_gradient_floor,
                probability_objective=environment.guidance_probability_objective,
            )
            hull_gradient = TDT.classifier_probability_hinge_gradient(
                classifiers.hull_feasibility, x[:, hull_indices], time,
                np.asarray([1.0]),
                probability_threshold=environment.guidance_probability_threshold,
                relu_mode=environment.guidance_relu_mode,
                probability_floor=environment.guidance_probability_floor,
                gradient_floor=environment.guidance_gradient_floor,
                probability_objective=environment.guidance_probability_objective,
            )
            if getattr(environment, "guidance_normalize_l2", False):
                # Normalize each classifier in its own input space. Keep a
                # closed ReLU gate at zero rather than amplifying tiny values.
                global_gradient = TDT.normalize_classifier_gradient_l2(
                    global_gradient,
                )
                hull_gradient = TDT.normalize_classifier_gradient_l2(
                    hull_gradient,
                )
            full_hull_gradient = torch.zeros_like(x)
            full_hull_gradient[:, hull_indices] = hull_gradient
            if environment.guidance_step_cap:
                classifier_move = (
                    alpha * torch.sqrt(variance)
                    * (global_gradient + full_hull_gradient)
                )
                classifier_move = TDT.cap_guidance_displacement(
                    classifier_move, environment.guidance_step_l2_limits[timestep]
                )
                x = mean + standard_deviation * (1.0 - alpha) * noise + classifier_move
            else:
                x = mean + standard_deviation * (
                    (1.0 - alpha) * noise
                    + alpha * global_gradient
                    + alpha * full_hull_gradient
                )

            # Keep this pull separate from classifier alpha. It is a soft
            # variance-scaled displacement, not clipping or a new condition.
            if boundary_direction is not None:
                x = x + standard_deviation * boundary_direction
            if self.capture_sampling_trace:
                trace_timesteps.append(timestep)
                trace_max_abs.append(torch.amax(torch.abs(x)).detach())

        latent = x.detach().cpu().numpy()
        if self.capture_sampling_trace:
            # A single transfer after sampling avoids a GPU wait per step.
            self.last_sampling_trace = {
                "timestep": trace_timesteps,
                "max_abs": torch.stack(trace_max_abs).cpu().tolist(),
            }
        if boundary_pull is not None and boundary_start_row_max is not None:
            environment.last_boundary_start_row_max = boundary_start_row_max.cpu().numpy()
        if getattr(environment, "capture_sampling_latent", False):
            environment.last_sampling_latent = latent
        return environment.data_norm.scale_X(latent)

    def _generate_hierarchical(
        self,
        count: int,
        *,
        guidance: str,
        classifiers: "GuidanceClassifiers | None",
        alpha: float,
    ) -> np.ndarray:
        if guidance not in {"none", "subsystem"}:
            raise ValueError("HCDDPM supports local subsystem guidance only.")

        generated = np.zeros((count, self.xdim), dtype=np.float64)
        raw_generated = (
            torch.empty((count, self.xdim), device=self.device)
            if self.capture_sampling_latent else None
        )
        for level, (environment, subset) in enumerate(
            zip(self.models, self.parameter_subsets, strict=True)
        ):
            previous = np.concatenate(self.parameter_subsets[:level]) if level else None
            conditions = []
            if self.conditioned:
                conditions.append(self.local_targets[level])
            if previous is not None:
                conditions.append(generated[:, previous])

            if guidance == "subsystem":
                guidance_mode = environment.classifier_guidance_mode
                local_classifiers = classifiers.local_for(
                    "hcddpm", guidance_mode
                )
                classifier = local_classifiers[SUBSYSTEM_ORDER[level]]
                target = np.ones(
                    len(LOCAL_LABELS_BY_SUBSYSTEM[SUBSYSTEM_ORDER[level]]),
                    dtype=np.float32,
                )
                if conditions:
                    context_indices = None
                    if (
                        guidance_mode in {"posterior_mean_hinge", "cshipgen_noise_blend_hinge"}
                        and previous is not None
                    ):
                        # Predecessor context is the last condition for both
                        # feasible-only and label-conditioned HCDDPMs.
                        context_indices = [len(conditions) - 1]
                    result = environment.gen_conditional_guided_samples(
                        count,
                        conditions,
                        classifier,
                        alpha=alpha,
                        class_target=target,
                        guidance_context_indices=context_indices,
                    )
                else:
                    result = environment.generate_guided_samples(
                        count,
                        classifier,
                        alpha=alpha,
                    )
            elif conditions:
                result = environment.gen_samples(count, conditions)
            else:
                result = environment.gen_samples(count)

            generated[:, subset] = np.asarray(result.scaled, dtype=np.float64)
            if raw_generated is not None:
                # Keep final block states on the GPU until all 20 blocks finish.
                # The raw diagnostic needs only one transfer per generated set.
                subset_tensor = torch.as_tensor(
                    subset, dtype=torch.long, device=self.device,
                )
                raw_generated[:, subset_tensor] = result.raw.detach()

            # Later HCDDPM blocks read these completed physical controls.
            # Give them the same binary representation used in training.
            if self.boolean_indices is not None:
                block_booleans = np.intersect1d(
                    self.boolean_indices, subset, assume_unique=True,
                )
                generated[:, block_booleans] = (
                    generated[:, block_booleans] >= 0.5
                ).astype(np.float64)
        if raw_generated is not None:
            self.last_sampling_latent = raw_generated.cpu().numpy()
        return generated

    @torch.no_grad()
    def _generate_simultaneous(
        self,
        count: int,
        *,
        guidance: str,
        classifiers: "GuidanceClassifiers | None",
        alpha: float,
    ) -> np.ndarray:
        use_system = guidance in {"system", "combined"}
        use_local = guidance in {"subsystem", "combined"}
        x = torch.randn((count, self.xdim), device=self.device)

        for environment in self.models:
            environment.diffusion.eval()
            environment.guidance_diagnostics = []
        if classifiers is not None:
            classifiers.eval()

        reference_environment = self.models[0]
        guidance_mode = reference_environment.classifier_guidance_mode
        if guidance_mode not in {"legacy_noise_blend", "posterior_mean_hinge", "cshipgen_noise_blend_hinge"}:
            raise ValueError(f"Unknown classifier_guidance_mode {guidance_mode!r}.")

        local_classifiers = None
        if use_local:
            local_classifiers = classifiers.local_for("scddpm", guidance_mode)
        hull_position = {
            int(index): position
            for position, index in enumerate(self.hull_indices)
        } if self.hull_local_guidance else {}
        hull_tensor_indices = (
            torch.as_tensor(self.hull_indices, dtype=torch.long, device=self.device)
            if self.hull_local_guidance else None
        )

        full_step_l2_limits = (
            torch.sqrt(reference_environment.betas) * np.sqrt(self.xdim)
            if reference_environment.guidance_step_cap else None
        )
        trace_timesteps = []
        trace_max_abs = []
        for timestep in tqdm(
            reference_environment.reverse_step_indices(),
            desc="SCDDPM reverse diffusion",
        ):
            # All denoisers receive the same x_t, so subsystem order cannot
            # affect their proposed reverse means.
            context = x.clone()
            means = []
            variances = []
            time = torch.full(
                (count,), timestep, dtype=torch.int64, device=self.device
            )
            for level, (environment, subset) in enumerate(
                zip(self.models, self.parameter_subsets, strict=True)
            ):
                complement = np.concatenate(
                    self.parameter_subsets[:level]
                    + self.parameter_subsets[level + 1 :]
                )
                condition = []
                if self.conditioned:
                    target = torch.as_tensor(
                        self.local_targets[level],
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(1, -1).expand(count, -1)
                    condition.append(target)
                condition.append(context[:, complement])

                x_slice = x[:, subset]
                predicted_noise = environment.diffusion(x_slice, condition, time)
                means.append(
                    environment._reverse_mean_from_epsilon(
                        x_slice, timestep, predicted_noise
                    )
                )
                variances.append(
                    environment.extract(
                        environment.posterior_variance, time, x_slice.shape
                    )
                )

            full_mean = torch.empty_like(x)
            for subset, mean in zip(self.parameter_subsets, means, strict=True):
                full_mean[:, subset] = mean
            if timestep == 0:
                x = full_mean
                if self.capture_sampling_trace:
                    trace_timesteps.append(timestep)
                    trace_max_abs.append(torch.amax(torch.abs(x)).detach())
                continue

            guidance_active = (
                float(alpha) != 0.0
                and reference_environment._guidance_timestep_is_active(timestep)
            )
            next_x = torch.empty_like(x)
            full_variance = torch.empty_like(x)
            full_innovation = torch.empty_like(x)

            # Local classifiers inspect the complete noisy x_t. Each local
            # gradient changes only the parameters owned by its subsystem.
            for level, (environment, subset, mean, variance) in enumerate(
                zip(
                    self.models,
                    self.parameter_subsets,
                    means,
                    variances,
                    strict=True,
                )
            ):
                noise = torch.randn_like(mean)
                local_gradient = None
                if use_local and guidance_active:
                    name = SUBSYSTEM_ORDER[level]
                    local_target = np.ones(
                        len(LOCAL_LABELS_BY_SUBSYSTEM[name]), dtype=np.float32
                    )
                    if guidance_mode in {"posterior_mean_hinge", "cshipgen_noise_blend_hinge"}:
                        hull_local = self.hull_local_guidance and name in HULL_SUBSYSTEMS
                        classifier_values = (
                            x.index_select(1, hull_tensor_indices) if hull_local else x
                        )
                        local_result = TDT.classifier_probability_hinge_gradient(
                            local_classifiers[name],
                            classifier_values,
                            time,
                            local_target,
                            probability_threshold=(
                                environment.guidance_probability_threshold
                            ),
                            relu_mode=environment.guidance_relu_mode,
                            probability_floor=(
                                environment.guidance_probability_floor
                            ),
                            gradient_floor=environment.guidance_gradient_floor,
                            probability_objective=environment.guidance_probability_objective,
                            return_diagnostics=(
                                environment.guidance_collect_diagnostics
                            ),
                        )
                        if environment.guidance_collect_diagnostics:
                            local_gradient, diagnostics = local_result
                            environment._record_guidance_diagnostics(
                                timestep,
                                {"scope": f"local:{name}", **diagnostics},
                            )
                        else:
                            local_gradient = local_result
                        if hull_local:
                            local_gradient = local_gradient[:, [
                                hull_position[int(index)] for index in subset
                            ]]
                        else:
                            local_gradient = local_gradient[:, subset]
                    else:
                        local_gradient = _normalized_guidance_gradient(
                            local_classifiers[name], mean, local_target, time
                        )

                    if (
                        guidance_mode == "cshipgen_noise_blend_hinge"
                        and getattr(environment, "guidance_normalize_l2", False)
                    ):
                        # Normalize after selecting the owned subsystem block.
                        local_gradient = TDT.normalize_classifier_gradient_l2(
                            local_gradient,
                        )

                if local_gradient is None:
                    next_x[:, subset] = mean + torch.sqrt(variance) * noise
                elif guidance_mode == "cshipgen_noise_blend_hinge":
                    if environment.guidance_step_cap:
                        classifier_move = alpha * torch.sqrt(variance) * local_gradient
                        classifier_move = TDT.cap_guidance_displacement(
                            classifier_move, environment.guidance_step_l2_limits[timestep]
                        )
                        next_x[:, subset] = (
                            mean + torch.sqrt(variance) * (1.0 - alpha) * noise
                            + classifier_move
                        )
                    else:
                        # Keep the existing uncapped noise blend unchanged.
                        guided_noise = (1.0 - alpha) * noise + alpha * local_gradient
                        next_x[:, subset] = mean + torch.sqrt(variance) * guided_noise
                elif guidance_mode == "posterior_mean_hinge":
                    next_x[:, subset] = TDT.posterior_mean_guided_sample(
                        mean,
                        variance,
                        noise,
                        local_gradient,
                        alpha,
                        max_displacement_rms=(
                            environment.guidance_max_displacement_rms
                        ),
                        max_displacement_to_noise_rms_ratio=(
                            environment.guidance_max_displacement_to_noise_rms_ratio
                        ),
                    )
                else:
                    guided_noise = (1.0 - alpha) * noise + alpha * local_gradient
                    next_x[:, subset] = mean + torch.sqrt(variance) * guided_noise

                full_variance[:, subset] = variance
                full_innovation[:, subset] = noise

            # The system classifier sees the complete post-subsystem x_(t-1).
            # It then moves all coordinates once, using the same alpha as each
            # active local classifier and the variance of each owned block.
            if use_system and guidance_active:
                system_time = torch.full_like(time, timestep - 1)
                if guidance_mode in {"posterior_mean_hinge", "cshipgen_noise_blend_hinge"}:
                    system_result = TDT.classifier_probability_hinge_gradient(
                        (
                            classifiers.global_feasibility
                            if self.system_guidance_target == "global_feasibility"
                            else classifiers.system_level
                        ),
                        next_x,
                        system_time,
                        np.asarray([1.0], dtype=np.float32),
                        probability_threshold=(
                            reference_environment.guidance_probability_threshold
                        ),
                        relu_mode=reference_environment.guidance_relu_mode,
                        probability_floor=(
                            reference_environment.guidance_probability_floor
                        ),
                        gradient_floor=(
                            reference_environment.guidance_gradient_floor
                        ),
                        probability_objective=(
                            reference_environment.guidance_probability_objective
                        ),
                        return_diagnostics=(
                            reference_environment.guidance_collect_diagnostics
                        ),
                    )
                    if reference_environment.guidance_collect_diagnostics:
                        system_gradient, diagnostics = system_result
                        reference_environment._record_guidance_diagnostics(
                            timestep - 1,
                            {"scope": "system", **diagnostics},
                        )
                    else:
                        system_gradient = system_result
                    if guidance_mode == "cshipgen_noise_blend_hinge":
                        if getattr(
                            reference_environment, "guidance_normalize_l2", False,
                        ):
                            # The post-merge classifier acts in the full space.
                            system_gradient = TDT.normalize_classifier_gradient_l2(
                                system_gradient,
                            )
                        # Cap the post-merge system move in the full 133-space.
                        if reference_environment.guidance_step_cap:
                            system_move = (
                                alpha * torch.sqrt(full_variance) * system_gradient
                            )
                            system_move = TDT.cap_guidance_displacement(
                                system_move, full_step_l2_limits[timestep]
                            )
                            base = (
                                full_mean + (1.0 - alpha)
                                * torch.sqrt(full_variance) * full_innovation
                            )
                            if use_local:
                                # The final combined classifier contribution
                                # has one additional full-system budget.
                                combined_move = next_x + system_move - base
                                combined_move = TDT.cap_guidance_displacement(
                                    combined_move, full_step_l2_limits[timestep]
                                )
                                x = base + combined_move
                            else:
                                x = base + system_move
                        elif use_local:
                            x = (
                                next_x
                                + alpha * torch.sqrt(full_variance)
                                * system_gradient
                            )
                        else:
                            x = (
                                full_mean
                                + (1.0 - alpha) * (next_x - full_mean)
                                + alpha * torch.sqrt(full_variance)
                                * system_gradient
                            )
                    else:
                        full_stochastic = torch.sqrt(full_variance) * full_innovation
                        x = TDT.posterior_mean_guided_sample(
                            next_x - full_stochastic,
                            full_variance,
                            full_innovation,
                            system_gradient,
                            alpha,
                            max_displacement_rms=(
                                reference_environment.guidance_max_displacement_rms
                            ),
                            max_displacement_to_noise_rms_ratio=(
                                reference_environment.guidance_max_displacement_to_noise_rms_ratio
                            ),
                        )
                else:
                    system_gradient = _normalized_guidance_gradient(
                        (
                            classifiers.global_feasibility
                            if self.system_guidance_target == "global_feasibility"
                            else classifiers.system_level
                        ),
                        next_x,
                        np.asarray([1.0], dtype=np.float32),
                        system_time,
                    )
                    x = next_x + alpha * torch.sqrt(full_variance) * system_gradient
            else:
                x = next_x
            if self.capture_sampling_trace:
                trace_timesteps.append(timestep)
                trace_max_abs.append(torch.amax(torch.abs(x)).detach())

        if self.capture_sampling_trace:
            # Keep scalar maxima on the GPU until the reverse loop is done.
            self.last_sampling_trace = {
                "timestep": trace_timesteps,
                "max_abs": torch.stack(trace_max_abs).cpu().tolist(),
            }
        if self.capture_sampling_latent:
            # Capture the complete post-merge latent before blockwise inversion.
            self.last_sampling_latent = x.detach().cpu().numpy()

        physical = np.zeros((count, self.xdim), dtype=np.float64)
        for environment, subset in zip(
            self.models,
            self.parameter_subsets,
            strict=True,
        ):
            physical[:, subset] = environment.data_norm.scale_X(
                x[:, subset].detach().cpu().numpy()
            )
        return physical


@dataclass
class GuidanceClassifiers:
    """Global and architecture-specific local guidance classifiers."""

    global_feasibility: TDT.ConstraintClassifier
    system_level: TDT.ConstraintClassifier | None
    hull_feasibility: TDT.ConstraintClassifier | None = None
    # Block-only local classifiers are retained solely for explicit legacy
    # diagnostic reproduction.
    local: dict[str, TDT.ConstraintClassifier] | None = None
    hcddpm_local: dict[str, TDT.ConstraintClassifier] | None = None
    scddpm_local: dict[str, TDT.ConstraintClassifier] | None = None

    def local_for(
        self,
        architecture: str,
        guidance_mode: str,
    ) -> dict[str, TDT.ConstraintClassifier]:
        if guidance_mode == "legacy_noise_blend":
            if self.local is None:
                raise RuntimeError(
                    "Legacy block-only local classifiers were not loaded."
                )
            return self.local

        if guidance_mode not in {"posterior_mean_hinge", "cshipgen_noise_blend_hinge"}:
            raise ValueError(
                f"Unknown classifier_guidance_mode {guidance_mode!r}."
            )

        if architecture == "hcddpm":
            collection = self.hcddpm_local
        elif architecture == "scddpm":
            collection = self.scddpm_local
        else:
            raise ValueError(
                f"No local classifier collection for {architecture!r}."
            )
        if collection is None:
            raise RuntimeError(
                "Corrected guidance requires architecture-specific contextual "
                f"{architecture.upper()} local classifiers."
            )
        return collection

    def eval(self) -> None:
        self.global_feasibility.eval()
        if self.hull_feasibility is not None:
            self.hull_feasibility.eval()
        if self.system_level is not None:
            self.system_level.eval()
        for collection in (
            self.local,
            self.hcddpm_local,
            self.scddpm_local,
        ):
            if collection is not None:
                for classifier in collection.values():
                    classifier.eval()


def _normalized_guidance_gradient(
    classifier: TDT.ConstraintClassifier,
    values: torch.Tensor,
    target: np.ndarray,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Return a per-sample unit-RMS gradient toward the requested labels."""

    with torch.enable_grad():
        values_for_gradient = values.detach().requires_grad_(True)
        prediction = classifier(values_for_gradient, timestep)
        target_tensor = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=values.device,
        ).reshape(1, -1).expand(len(values), -1)
        error = F.binary_cross_entropy(
            prediction,
            target_tensor,
            reduction="sum",
        )
        gradient = -torch.autograd.grad(error, values_for_gradient)[0]
    rms = torch.sqrt(torch.mean(gradient**2, dim=1, keepdim=True))
    return gradient / (rms + 1.0e-8)


def _base_environment_config(
    settings: dict,
    *,
    xdim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    network: list[int],
) -> dict:
    return {
        "xdim": int(xdim),
        "datalength": 0,
        "batch_size": int(settings["batch_size"]),
        "rebatch_every": int(settings.get("rebatch_every", 100)),
        "sampling_omit_t0": bool(settings.get("sampling_omit_t0", False)),
        "net": list(network),
        "tdim": int(settings["timestep_embedding_dim"]),
        "Diffusion_Timesteps": int(settings["diffusion_timesteps"]),
        "Training_Epochs": int(settings["training_epochs"]),
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings.get("weight_decay", 0.0)),
        "device_name": str(settings["device_name"]),
        "X_LL": np.asarray(lower, dtype=float).tolist(),
        "X_UL": np.asarray(upper, dtype=float).tolist(),
        "data_normalizer_type": str(settings.get("normalization_type", "minmax")),
        "CFG_prob": 0.0,
        "beta_schedule": str(settings["beta_schedule"]),
        "beta_start": float(settings.get("beta_start", 0.0001)),
        "beta_end": float(settings.get("beta_end", 0.02)),
        "cosine_s": float(settings.get("cosine_s", 0.008)),
        # The experiment config selects either posterior-mean guidance or
        # the C_ShipGen reverse-step noise blend. The generic environment
        # keeps its named historical mode when this key is absent.
        "classifier_guidance_mode": str(
            settings.get(
                "classifier_guidance_mode",
                "posterior_mean_hinge",
            )
        ),
        "guidance_probability_threshold": float(
            settings.get("guidance_probability_threshold", 0.5)
        ),
        "guidance_relu_mode": str(
            settings.get("guidance_relu_mode", "hard")
        ),
        "guidance_probability_objective": str(
            settings.get("guidance_probability_objective", "log")
        ),
        "guidance_step_cap": bool(settings.get("guidance_step_cap", False)),
        "guidance_normalize_l2": bool(
            settings.get("guidance_normalize_l2", False)
        ),
        "guidance_probability_floor": float(
            settings.get("guidance_probability_floor", 1.0e-6)
        ),
        "guidance_gradient_floor": float(
            settings.get("guidance_gradient_floor", 1.0e-12)
        ),
        "guidance_max_displacement_rms": settings.get(
            "guidance_max_displacement_rms"
        ),
        "guidance_max_displacement_to_noise_rms_ratio": settings.get(
            "guidance_max_displacement_to_noise_rms_ratio"
        ),
        "guidance_active_timestep_fraction": float(
            settings.get("guidance_active_timestep_fraction", 1.0)
        ),
        "guidance_active_timestep_window": settings.get(
            "guidance_active_timestep_window"
        ),
        "guidance_balance_combined_gradient_rms": bool(
            settings.get("guidance_balance_combined_gradient_rms", False)
        ),
        "guidance_collect_diagnostics": bool(
            settings.get("guidance_collect_diagnostics", False)
        ),
        "sampling_x0_clip": bool(
            settings.get("sampling_x0_clip", False)
        ),
    }


def _conditional_environment(
    config: dict,
    x: np.ndarray,
    conditions: list[np.ndarray],
    condition_bounds: list[tuple[np.ndarray, np.ndarray]],
    *,
    noisy_condition_indices: list[int] | None = None,
    condition_normalizers: list | None = None,
) -> TDT.Conditional_DiffusionEnv:
    config = dict(config)
    config["cdim"] = [int(value.shape[1]) for value in conditions]
    config["Cond_LL"] = [lower.tolist() for lower, _ in condition_bounds]
    config["Cond_UL"] = [upper.tolist() for _, upper in condition_bounds]
    config["Cond_type"] = ["minmax"] * len(conditions)
    if condition_normalizers is not None:
        config["pre_fitted_condition_normalizers"] = condition_normalizers
    if noisy_condition_indices:
        config["context_training_mode"] = "time_matched_noisy"
        config["noise_condition_indices"] = list(noisy_condition_indices)
    return TDT.Conditional_DiffusionEnv(config, x, conditions)


def build_first_experiment_generator(
    data: PreparedFirstExperimentData,
    settings: dict,
    *,
    architecture: str,
    population: str,
) -> FirstExperimentGenerator:
    """Construct one of the six shared generator checkpoints."""

    locally_feasible = population == "locally_feasible_subsystems"
    if locally_feasible and architecture == "monolithic":
        raise ValueError(
            "The locally-feasible population is defined only for HCDDPM and SCDDPM."
        )

    # The side experiment deliberately preserves the original feasible-only
    # coordinate system.  Only the rows shown to each subsystem denoiser change.
    selection_population = "feasible_only" if locally_feasible else population
    positions = data.generator_positions(
        selection_population,
        parent_fraction=float(settings.get("training_parent_fraction", 1.0)),
        subset_seed=int(settings.get("data_subset_seed", settings["split_seed"])),
        unknown_label_policy=str(settings.get("unknown_label_policy", "include")),
        matched_row_count=settings.get("matched_training_row_count"),
    )
    x = data.model_values[positions]
    normalizer_population = (
        selection_population
        if settings.get("normalization_fit_scope") == "per_population"
        else None
    )
    label_policy = str(settings.get("unknown_label_policy", "include"))
    bounds_scope = str(settings.get("normalization_bound_scope", "population"))
    model_normalizer = data.model_normalizer(
        population=normalizer_population,
        unknown_label_policy=label_policy,
        bounds_scope=bounds_scope,
    )
    model_lower = model_normalizer.X_LL_Scaled
    model_upper = model_normalizer.X_UL_Scaled
    conditioned = population == "full_data"
    if conditioned and not bool(data.known_mask[positions].all()):
        raise ValueError(
            "Two-value pass/fail conditioning requires known-label rows only."
        )
    device = torch.device(settings["device_name"])
    subsets = tuple(
        np.asarray(data.parameter_subsets[name], dtype=int)
        for name in SUBSYSTEM_ORDER
    )
    global_target = feasible_condition_target(1)
    local_targets = tuple(
        feasible_condition_target(len(LOCAL_LABELS_BY_SUBSYSTEM[name]))
        for name in SUBSYSTEM_ORDER
    )

    if architecture == "monolithic":
        config = _base_environment_config(
            settings,
            xdim=x.shape[1],
            lower=model_lower,
            upper=model_upper,
            network=settings["network"],
        )
        if str(settings.get("normalization_type", "minmax")) == "quantile":
            config["pre_fitted_data_normalizer"] = model_normalizer
        if conditioned:
            condition = data.global_condition[positions]
            environment = _conditional_environment(
                config,
                x,
                [condition],
                [
                    (
                        np.zeros(condition.shape[1]),
                        np.ones(condition.shape[1]),
                    )
                ],
            )
        else:
            environment = TDT.DiffusionEnv(config, x)
        models = [environment]

    elif architecture in {"hcddpm", "scddpm"}:
        models = []
        population_prefix = (
            "feasible"
            if population in {"feasible_only", "locally_feasible_subsystems"}
            else "full"
        )
        architecture_key = f"{architecture}_{population_prefix}_subsystem_network"
        fallback_key = f"{population_prefix}_subsystem_network"
        subnetwork = settings.get(architecture_key, settings[fallback_key])
        subsystem_settings = dict(settings)
        subsystem_settings["timestep_embedding_dim"] = int(
            settings.get(
                "subsystem_timestep_embedding_dim",
                settings["timestep_embedding_dim"],
            )
        )
        for level, (name, subset) in enumerate(zip(SUBSYSTEM_ORDER, subsets, strict=True)):
            # Every local implicit denoiser receives all complete unified rows
            # on which its own aggregated subsystem label passes.  Context is
            # taken from the same physical designs and may contain failures in
            # other subsystems, as prescribed by the side experiment.
            subsystem_x = (
                data.model_values[data.local_feasible_positions(name)]
                if locally_feasible else x
            )
            config = _base_environment_config(
                subsystem_settings,
                xdim=len(subset),
                lower=model_lower[subset],
                upper=model_upper[subset],
                network=subnetwork,
            )
            if str(settings.get("normalization_type", "minmax")) == "quantile":
                config["pre_fitted_data_normalizer"] = data.model_normalizer(
                    subset,
                    population=normalizer_population,
                    unknown_label_policy=label_policy,
                    bounds_scope=bounds_scope,
                )
            conditions = []
            bounds = []
            condition_normalizers = []
            if conditioned:
                condition = data.local_conditions[name][positions]
                conditions.append(condition)
                condition_normalizers.append(None)
                bounds.append(
                    (
                        np.zeros(condition.shape[1]),
                        np.ones(condition.shape[1]),
                    )
                )

            if architecture == "hcddpm":
                context_indices = np.concatenate(subsets[:level]) if level else None
            else:
                context_indices = np.concatenate(subsets[:level] + subsets[level + 1 :])

            if context_indices is not None:
                conditions.append(subsystem_x[:, context_indices])
                condition_normalizers.append(
                    data.model_normalizer(
                        context_indices,
                        population=normalizer_population,
                        unknown_label_policy=label_policy,
                        bounds_scope=bounds_scope,
                    )
                    if str(settings.get("normalization_type", "minmax")) == "quantile"
                    else None
                )
                bounds.append((
                    model_lower[context_indices], model_upper[context_indices]
                ))

            if conditions:
                noisy = [len(conditions) - 1] if architecture == "scddpm" else None
                environment = _conditional_environment(
                    config,
                    subsystem_x[:, subset],
                    conditions,
                    bounds,
                    noisy_condition_indices=noisy,
                    condition_normalizers=condition_normalizers,
                )
            else:
                environment = TDT.DiffusionEnv(config, subsystem_x[:, subset])
            models.append(environment)
    else:
        raise ValueError(f"Unknown architecture {architecture!r}.")

    generator_subsets = (
        (np.arange(x.shape[1], dtype=int),)
        if architecture == "monolithic"
        else subsets
    )
    registry = V1p1ParameterSchema().registry
    boolean_indices = np.asarray(
        [
            index
            for index, name in enumerate(data.parameter_names)
            if name in registry.boolean_names
        ],
        dtype=int,
    )
    generator = FirstExperimentGenerator(
        architecture=architecture,
        population=population,
        models=models,
        parameter_subsets=generator_subsets,
        conditioned=conditioned,
        local_targets=local_targets,
        global_target=global_target,
        device=device,
        system_guidance_target=str(
            settings.get("system_guidance_target", "system_constraints")
        ),
        hull_indices=np.asarray(
            [data.parameter_names.index(name) for name in HULL_CONTROLS], dtype=int,
        ),
        aio_hull_guidance=bool(settings.get("aio_hull_guidance", False)),
        hull_local_guidance=bool(settings.get("hull_local_guidance", False)),
        boolean_indices=boolean_indices,
    )
    if bool(settings.get("parent_balanced_sampling", False)):
        probabilities = data.parent_balancing_probabilities(positions)
        for environment in generator.models:
            environment.sampling_probabilities = probabilities
    return generator


def _classifier_config(settings: dict, xdim: int, cdim: int) -> dict:
    return {
        "xdim": int(xdim),
        "tdim": int(settings["classifier_embedding_dim"]),
        "cdim": int(cdim),
        "net": list(settings["classifier_network"]),
        **(
            {"cshipgen_layers": True}
            if settings.get("classifier_training_recipe") == "cshipgen_final"
            else {}
        ),
        "time_conditioned": (
            str(settings.get("classifier_noise_mode", "noisy_timestep_conditioned"))
            == "noisy_timestep_conditioned"
        ),
    }


def _fit_guidance_temperatures(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Fit one validation-only logit temperature per binary output."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    temperatures = np.ones(probabilities.shape[1], dtype=np.float32)

    for output_index in range(probabilities.shape[1]):
        truth = labels[:, output_index]
        if np.unique(truth).size < 2:
            continue

        bounded = np.clip(
            probabilities[:, output_index],
            1.0e-6,
            1.0 - 1.0e-6,
        )
        logits = torch.as_tensor(
            np.log(bounded / (1.0 - bounded)),
            dtype=torch.float64,
        )
        target = torch.as_tensor(truth, dtype=torch.float64)
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.25,
            max_iter=80,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature.clamp(-3.0, 3.0))
            loss = F.binary_cross_entropy_with_logits(
                logits / temperature, target
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures[output_index] = float(
            torch.exp(log_temperature.detach().clamp(-3.0, 3.0))
        )

    return temperatures


def _apply_guidance_temperatures(
    probabilities: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Apply the differentiable logit-temperature mapping in NumPy."""

    bounded = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    logits = np.log(bounded / (1.0 - bounded))
    scaled = logits / np.asarray(temperatures, dtype=np.float64).reshape(1, -1)
    return 1.0 / (1.0 + np.exp(-scaled))


def _train_one_classifier(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    settings: dict,
    path: Path,
    seed: int,
    name: str,
    noisy_value_indices: np.ndarray | None = None,
) -> TDT.ConstraintClassifier:
    """Train one unified-data constraint classifier under the selected recipe."""

    cshipgen_recipe = settings.get("classifier_training_recipe") == "cshipgen_final"
    device = torch.device(settings["device_name"])
    config = _classifier_config(settings, values.shape[1], labels.shape[1])

    # A constant classifier target contains no decision boundary.  Preserve
    # the fitted model and its validation record, but freeze its sampling
    # contribution to zero through an explicit output mask.
    positive_counts = np.asarray(labels, dtype=np.float64).sum(axis=0)
    negative_counts = len(labels) - positive_counts
    guidance_output_mask = (
        (positive_counts > 0.0) & (negative_counts > 0.0)
    ).astype(np.float32)

    explicit_noise_columns = noisy_value_indices is not None
    if noisy_value_indices is None:
        noisy_value_indices_array = np.arange(values.shape[1], dtype=int)
    else:
        noisy_value_indices_array = np.asarray(
            noisy_value_indices, dtype=int
        )
    signature = {
        "config": config,
        "training_epochs": int(settings["classifier_training_epochs"]),
        "batches_per_epoch": int(settings["classifier_batches_per_epoch"]),
        "batch_size": int(settings["classifier_batch_size"]),
        "learning_rate": float(settings["classifier_learning_rate"]),
        "seed": int(seed),
        "beta_schedule": str(settings["beta_schedule"]),
        "beta_start": float(settings.get("beta_start", 0.0001)),
        "beta_end": float(settings.get("beta_end", 0.02)),
        "cosine_s": float(settings.get("cosine_s", 0.008)),
        "diffusion_timesteps": int(settings["diffusion_timesteps"]),
        "early_stopping_patience": int(
            settings.get("classifier_early_stopping_patience", 20)
        ),
        "classifier_noise_mode": str(
            settings.get("classifier_noise_mode", "noisy_timestep_conditioned")
        ),
        **(
            {"classifier_training_recipe": "cshipgen_final"}
            if cshipgen_recipe else {}
        ),
        "values_sha256": hashlib.sha256(
            np.asarray(values, dtype="<f4").tobytes()
        ).hexdigest(),
        "labels_sha256": hashlib.sha256(
            np.asarray(labels, dtype="<f4").tobytes()
        ).hexdigest(),
        "validation_values_sha256": hashlib.sha256(
            np.asarray(validation_values, dtype="<f4").tobytes()
        ).hexdigest(),
        "validation_labels_sha256": hashlib.sha256(
            np.asarray(validation_labels, dtype="<f4").tobytes()
        ).hexdigest(),
    }
    # Old all-feature checkpoints did not record a noise mask. Keep their
    # signatures loadable; contextual checkpoints state their mask explicitly.
    if explicit_noise_columns:
        signature["noisy_value_indices"] = (
            noisy_value_indices_array.tolist()
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization must depend only on this classifier's declared seed.
    # Otherwise resuming directly at the classifier stage can produce a
    # different model than reaching it after generator training.
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    classifier = TDT.ConstraintClassifier(config, device=device).to(device)
    classifier.guidance_output_mask = torch.as_tensor(
        guidance_output_mask,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(settings["classifier_learning_rate"]),
        weight_decay=float(settings.get("classifier_weight_decay", 1.0e-5)),
    )
    recovery_path = path.with_suffix(".recovery.pt")
    first_epoch = 0
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model = None

    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(f"Classifier training contract changed for {path}.")
        classifier.load_state_dict(payload["model"])
        recorded_mask = np.asarray(
            payload.get("guidance_output_mask", guidance_output_mask),
            dtype=np.float32,
        )
        if not np.array_equal(recorded_mask, guidance_output_mask):
            raise ValueError(
                f"Classifier guidance-output mask changed for {path}."
            )
        classifier.guidance_output_mask = torch.as_tensor(
            recorded_mask, dtype=torch.float32, device=device
        )
        classifier.guidance_temperature = torch.as_tensor(
            payload.get(
                "guidance_temperature",
                np.ones(labels.shape[1], dtype=np.float32),
            ),
            dtype=torch.float32,
            device=device,
        )
        classifier.eval()
        return classifier
    if recovery_path.exists():
        payload = torch.load(recovery_path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(
                f"Classifier recovery contract changed for {recovery_path}."
            )
        classifier.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first_epoch = int(payload["completed_epochs"])
        history = list(payload["history"])
        best_validation_loss = float(payload["best_validation_loss"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        best_model = payload["best_model"]
        if cshipgen_recipe:
            torch.set_rng_state(payload["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(payload["cuda_rng_states"])

    if cshipgen_recipe and not bool(guidance_output_mask.any()) and first_epoch == 0:
        # A constant label has no decision boundary. Its frozen guidance mask
        # makes its gradient exactly zero, so optimizer work cannot change any
        # generated sample. Record this exception instead of spending 150,000
        # updates on an uninformative classifier.
        classifier.guidance_temperature = torch.ones(
            labels.shape[1], dtype=torch.float32, device=device,
        )
        checkpoint = {
            "signature": signature,
            "model": classifier.state_dict(),
            "guidance_output_mask": guidance_output_mask.tolist(),
            "guidance_temperature": [1.0] * labels.shape[1],
            "completed_epochs": 0,
            "constant_label_no_training": True,
        }
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        pd.DataFrame([{
            "Classifier": name,
            "Output_Index": index,
            "Positive_Count": int(labels[:, index].sum()),
            "Negative_Count": int(len(labels) - labels[:, index].sum()),
            "PR_AUC": np.nan,
            "Balanced_Accuracy": np.nan,
            "Mean_Calibration_Error": np.nan,
            "Binary_Cross_Entropy": np.nan,
            "Temperature": 1.0,
            "Validation_Loss": np.nan,
            "Best_Epoch": 0,
            "Status": "constant_training_label_no_guidance_boundary",
        } for index in range(labels.shape[1])]).to_csv(
            path.with_suffix(".validation.csv"), index=False,
        )
        pd.DataFrame(columns=["epoch", "training_loss", "learning_rate"]).to_csv(
            path.with_suffix(".history.csv"), index=False,
        )
        classifier.eval()
        return classifier

    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(
        validation_values,
        dtype=torch.float32,
        device=device,
    )
    validation_y = torch.as_tensor(
        validation_labels,
        dtype=torch.float32,
        device=device,
    )
    batch_size = min(int(settings["classifier_batch_size"]), len(values))
    batches = int(settings["classifier_batches_per_epoch"])
    epochs = int(settings["classifier_training_epochs"])

    positives = np.maximum(labels.sum(axis=0), 1.0)
    negatives = np.maximum(len(labels) - positives, 1.0)
    positive_weight = torch.as_tensor(
        np.clip(negatives / positives, 0.25, 25.0),
        dtype=torch.float32,
        device=device,
    )

    schedule_config = {
        "Diffusion_Timesteps": int(settings["diffusion_timesteps"]),
        "beta_schedule": str(settings["beta_schedule"]),
        "beta_start": float(settings.get("beta_start", 0.0001)),
        "beta_end": float(settings.get("beta_end", 0.02)),
        "cosine_s": float(settings.get("cosine_s", 0.008)),
    }
    betas = TDT.make_beta_schedule(schedule_config, device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    patience = int(settings.get("classifier_early_stopping_patience", 20))
    noisy_classifier = config["time_conditioned"]

    noisy_columns = torch.as_tensor(
        noisy_value_indices_array,
        dtype=torch.long,
        device=device,
    )

    def noisy_inputs(
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sqrt_alpha = torch.sqrt(alpha_bar[timesteps]).reshape(-1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar[timesteps]).reshape(-1, 1)
        if noise is None:
            noise = torch.randn_like(clean)

        noised = clean.clone()
        noised[:, noisy_columns] = (
            sqrt_alpha * clean[:, noisy_columns]
            + sqrt_one_minus * noise[:, noisy_columns]
        )
        return noised

    def validate(epoch: int) -> tuple[float, np.ndarray]:
        classifier.eval()
        predictions = []
        losses = []
        batch_sizes = []
        validation_batch = min(batch_size, len(validation_x))
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(seed + 1_000_000 + epoch)
        with torch.no_grad():
            for start in range(0, len(validation_x), validation_batch):
                stop = min(start + validation_batch, len(validation_x))
                clean = validation_x[start:stop]
                target = validation_y[start:stop]
                time = torch.randint(
                    0,
                    int(settings["diffusion_timesteps"]),
                    (len(clean),),
                    generator=validation_generator,
                    device=device,
                )
                sqrt_alpha = torch.sqrt(alpha_bar[time]).reshape(-1, 1)
                sqrt_one_minus = torch.sqrt(1.0 - alpha_bar[time]).reshape(-1, 1)
                noise = torch.randn(
                    clean.shape,
                    generator=validation_generator,
                    device=device,
                )
                classifier_input = (
                    noisy_inputs(clean, time, noise=noise)
                    if noisy_classifier
                    else clean
                )
                prediction = classifier(
                    classifier_input,
                    time if noisy_classifier else None,
                )
                losses.append(F.binary_cross_entropy(prediction, target))
                batch_sizes.append(len(clean))
                predictions.append(prediction)
        classifier.train()

        # Collect validation results with one transfer per output type.
        batch_losses = torch.stack(losses).cpu().numpy()
        weighted_loss = sum(
            float(loss) * size
            for loss, size in zip(batch_losses, batch_sizes, strict=True)
        )
        return (
            weighted_loss / len(validation_x),
            torch.cat(predictions).cpu().numpy(),
        )

    classifier.train()
    print(f"Classifier: {name} ({first_epoch}/{epochs} epochs complete)", flush=True)
    for epoch in range(first_epoch, epochs):
        # Each C_ShipGen epoch draws one fresh uniform batch. Keep the
        # sampling operation on the GPU and derive its seed from the epoch so
        # a resumed fit receives the same batch.
        if cshipgen_recipe:
            batch_generator = torch.Generator(device=device)
            batch_generator.manual_seed(seed + epoch)
            index_batches = torch.randint(
                len(values), (batches, batch_size),
                generator=batch_generator, device=device,
            )
        else:
            generator = np.random.default_rng(seed + epoch)
            torch.manual_seed(seed + epoch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + epoch)
            index_batches = torch.as_tensor(
                generator.integers(0, len(values), size=(batches, batch_size)),
                dtype=torch.long,
                device=device,
            )
        losses = []
        for indices in index_batches:
            xb = x.index_select(0, indices)
            yb = y.index_select(0, indices)
            time = torch.randint(
                0,
                int(settings["diffusion_timesteps"]),
                (len(xb),),
                device=device,
            )
            if noisy_classifier:
                xb = noisy_inputs(xb, time)
            optimizer.zero_grad()
            prediction = classifier(xb, time if noisy_classifier else None)
            if cshipgen_recipe:
                loss = F.binary_cross_entropy(prediction, yb)
            else:
                weights = 1.0 + yb * (positive_weight - 1.0)
                loss = F.binary_cross_entropy(
                    prediction, yb, weight=weights
                )
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

        # The C_ShipGen recipe keeps final weights. Log and checkpoint at a
        # sparse cadence so 150,000 one-batch epochs do not validate or write
        # a checkpoint after every optimizer update.
        if cshipgen_recipe:
            completed = epoch + 1
            report_every = int(settings.get("classifier_report_every_epochs", 1000))
            checkpoint_every = int(
                settings.get("classifier_checkpoint_every_epochs", 1000)
            )
            if completed % report_every == 0 or completed == epochs:
                training_loss = float(torch.stack(losses).mean().cpu())
                history.append({
                    "epoch": completed,
                    "training_loss": training_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                })
                print(
                    f"  {name}: epoch {completed}/{epochs}, "
                    f"train={training_loss:.6f}",
                    flush=True,
                )
            if completed % checkpoint_every == 0 or completed == epochs:
                checkpoint = {
                    "name": name,
                    "config": config,
                    "signature": signature,
                    "model": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed_epochs": completed,
                    "history": history,
                    "best_validation_loss": float("inf"),
                    "best_epoch": completed,
                    "best_model": None,
                    "epochs_without_improvement": 0,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda" else None
                    ),
                }
                temporary = recovery_path.with_name(
                    f".{recovery_path.name}.tmp"
                )
                torch.save(checkpoint, temporary)
                os.replace(temporary, recovery_path)
            continue

        # Keep step losses on the GPU, then read one epoch of scalars at once.
        training_loss = float(np.mean(
            torch.stack(losses).cpu().numpy(),
            dtype=np.float64,
        ))
        validation_loss, validation_prediction = validate(epoch)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        improved = validation_loss < best_validation_loss - 1.0e-7
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        report_every = max(1, epochs // 20)
        if (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                f"  {name}: epoch {epoch + 1}/{epochs}, "
                f"train={history[-1]['training_loss']:.6f}, "
                f"validation={validation_loss:.6f}",
                flush=True,
            )
        checkpoint = {
            "name": name,
            "config": config,
            "signature": signature,
            "model": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": epoch + 1,
            "history": history,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model": best_model,
            "epochs_without_improvement": epochs_without_improvement,
        }
        temporary = recovery_path.with_name(f".{recovery_path.name}.tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, recovery_path)

        if epochs_without_improvement >= patience:
            break

    final = torch.load(recovery_path, map_location=device, weights_only=False)
    if not cshipgen_recipe:
        final["model"] = final["best_model"]
    else:
        best_epoch = epochs
    final["guidance_output_mask"] = guidance_output_mask.tolist()
    classifier.load_state_dict(final["model"])
    classifier.guidance_output_mask = torch.as_tensor(
        guidance_output_mask, dtype=torch.float32, device=device
    )

    validation_loss, validation_prediction = validate(max(0, best_epoch - 1))
    guidance_temperature = (
        np.ones(labels.shape[1], dtype=np.float32)
        if cshipgen_recipe
        else _fit_guidance_temperatures(validation_prediction, validation_labels)
    )
    validation_prediction = _apply_guidance_temperatures(
        validation_prediction, guidance_temperature
    )
    final["guidance_temperature"] = guidance_temperature.tolist()
    classifier.guidance_temperature = torch.as_tensor(
        guidance_temperature, dtype=torch.float32, device=device
    )
    metric_rows = []
    for output_index in range(labels.shape[1]):
        truth = validation_labels[:, output_index].astype(int)
        score = validation_prediction[:, output_index]
        prediction = score >= 0.5
        if len(np.unique(truth)) == 2:
            pr_auc = float(average_precision_score(truth, score))
            balanced_accuracy = float(balanced_accuracy_score(truth, prediction))
        else:
            pr_auc = float("nan")
            balanced_accuracy = float("nan")
        binary_cross_entropy = float(
            -(
                truth * np.log(np.clip(score, 1.0e-6, 1.0))
                + (1 - truth)
                * np.log(np.clip(1.0 - score, 1.0e-6, 1.0))
            ).mean()
        )
        calibration_error = 0.0
        bin_edges = np.linspace(0.0, 1.0, 11)
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            in_bin = (score >= lower) & (
                score <= upper if upper == 1.0 else score < upper
            )
            if in_bin.any():
                calibration_error += float(in_bin.mean()) * abs(
                    float(score[in_bin].mean()) - float(truth[in_bin].mean())
                )
        metric_rows.append(
            {
                "Classifier": name,
                "Output_Index": output_index,
                "Positive_Count": int(truth.sum()),
                "Negative_Count": int((1 - truth).sum()),
                "PR_AUC": pr_auc,
                "Balanced_Accuracy": balanced_accuracy,
                "Mean_Calibration_Error": calibration_error,
                "Binary_Cross_Entropy": binary_cross_entropy,
                "Temperature": float(guidance_temperature[output_index]),
                "Validation_Loss": validation_loss,
                "Best_Epoch": best_epoch,
            }
        )
    pd.DataFrame(metric_rows).to_csv(path.with_suffix(".validation.csv"), index=False)
    pd.DataFrame(history).to_csv(path.with_suffix(".history.csv"), index=False)

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(final, temporary)
    os.replace(temporary, path)
    recovery_path.unlink()
    classifier.eval()
    return classifier


def _load_hull_guidance_training(
    data: PreparedFirstExperimentData,
    settings: dict,
    population: str | None,
) -> dict | None:
    """Read the versioned hull-only release in the generator's quantile space."""

    relative = settings.get("hull_guidance_dataset_root")
    if relative is None:
        return None
    project_root = Path(__file__).resolve().parents[1]
    release = project_root / relative
    parameters = pd.read_csv(
        release / "Hull_Parameters.csv.gz",
        usecols=["Hull_ID", "Split", *HULL_CONTROLS],
    )
    labels = pd.read_csv(release / "Hull_Subsystem_Labels.csv.gz")
    rows = parameters.merge(
        labels.drop(columns="Split"), on="Hull_ID", validate="one_to_one",
    )
    if len(rows) != 50_000 or not rows["Hull_Constraints_Known"].all():
        raise ValueError("The hull guidance release must contain 50,000 known rows.")

    indices = np.asarray(
        [data.parameter_names.index(name) for name in HULL_CONTROLS], dtype=int,
    )
    normalizer = data.model_normalizer(
        indices,
        population=population if settings.get("normalization_fit_scope") == "per_population" else None,
        unknown_label_policy=str(settings.get("unknown_label_policy", "include")),
        bounds_scope=str(settings.get("normalization_bound_scope", "population")),
    )
    scaled = normalizer.transform_Data(
        rows.loc[:, HULL_CONTROLS].to_numpy(np.float64)
    ).astype(np.float32)
    return {
        "indices": indices,
        "positions": {int(index): offset for offset, index in enumerate(indices)},
        "train_values": scaled[rows["Split"].eq("train").to_numpy()],
        "validation_values": scaled[rows["Split"].eq("validation").to_numpy()],
        "train_labels": rows.loc[rows["Split"].eq("train")].reset_index(drop=True),
        "validation_labels": rows.loc[rows["Split"].eq("validation")].reset_index(drop=True),
        "source_manifest": str(release / "Hull_Dataset_Manifest.json"),
    }


def train_or_load_guidance_classifiers(
    data: PreparedFirstExperimentData,
    settings: dict,
    output_root: str | Path,
    *,
    population: str | None = None,
) -> GuidanceClassifiers:
    """Train guidance on unified rows in the generator population's coordinates."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train_positions = data.classifier_positions("train")
    validation_positions = data.classifier_positions("validation")
    scaled_values = (
        data.scaled_values_for_population(
            population,
            unknown_label_policy=str(settings.get("unknown_label_policy", "include")),
            bounds_scope=str(settings.get("normalization_bound_scope", "population")),
        )
        if settings.get("normalization_fit_scope") == "per_population"
        else data.scaled_values
    )
    train_values = scaled_values[train_positions]
    validation_values = scaled_values[validation_positions]

    global_feasibility = _train_one_classifier(
        values=train_values,
        labels=data.global_pass_labels[train_positions],
        validation_values=validation_values,
        validation_labels=data.global_pass_labels[validation_positions],
        settings=settings,
        path=output_root / "Global_Feasibility_Classifier.pt",
        seed=int(settings["training_seed"]) + 500,
        name="Global_Feasibility",
    )

    hull_training = _load_hull_guidance_training(data, settings, population)
    hull_feasibility = None
    if hull_training is not None and settings.get("aio_hull_guidance", False):
        original_train_hull = np.logical_and.reduce([
            data.local_pass_labels[name][train_positions].astype(bool)
            for name in HULL_SUBSYSTEMS
        ]).astype(np.float32)
        original_validation_hull = np.logical_and.reduce([
            data.local_pass_labels[name][validation_positions].astype(bool)
            for name in HULL_SUBSYSTEMS
        ]).astype(np.float32)
        hull_feasibility = _train_one_classifier(
            values=np.vstack((
                train_values[:, hull_training["indices"]],
                hull_training["train_values"],
            )),
            labels=np.vstack((
                original_train_hull.reshape(-1, 1),
                hull_training["train_labels"][["Feasible::Hull_Global"]].to_numpy(np.float32),
            )),
            validation_values=np.vstack((
                validation_values[:, hull_training["indices"]],
                hull_training["validation_values"],
            )),
            validation_labels=np.vstack((
                original_validation_hull.reshape(-1, 1),
                hull_training["validation_labels"][["Feasible::Hull_Global"]].to_numpy(np.float32),
            )),
            settings=settings,
            path=output_root / "Hull_Global_Classifier.pt",
            seed=int(settings["training_seed"]) + 502,
            name="Hull_Global",
        )

    # The one-shot study uses the AIO global classifier after SCDDPM merges
    # its blocks. Skip the unused cross-subsystem classifier in that recipe.
    system_level = None
    if settings.get("system_guidance_target", "system_constraints") != "global_feasibility":
        system_level = _train_one_classifier(
            values=train_values,
            labels=data.system_pass_labels[train_positions],
            validation_values=validation_values,
            validation_labels=data.system_pass_labels[validation_positions],
            settings=settings,
            path=output_root / "SCDDPM_System_Level_Classifier.pt",
            seed=int(settings["training_seed"]) + 501,
            name="SCDDPM_System_Level",
        )

    guidance_mode = str(
        settings.get(
            "classifier_guidance_mode",
            "posterior_mean_hinge",
        )
    )
    legacy_local = None
    hcddpm_local = None
    scddpm_local = None

    if guidance_mode == "legacy_noise_blend":
        # This collection reproduces old diagnostics only. It intentionally
        # lacks architecture context and is never accepted by corrected mode.
        legacy_local = {}
        for offset, name in enumerate(SUBSYSTEM_ORDER, start=2):
            subset = np.asarray(data.parameter_subsets[name], dtype=int)
            legacy_local[name] = _train_one_classifier(
                values=train_values[:, subset],
                labels=data.local_pass_labels[name][train_positions],
                validation_values=validation_values[:, subset],
                validation_labels=(
                    data.local_pass_labels[name][validation_positions]
                ),
                settings=settings,
                path=output_root / f"{name}_Classifier.pt",
                seed=int(settings["training_seed"]) + 500 + offset,
                name=f"Legacy_Block_Only_{name}",
            )
    elif guidance_mode in {"posterior_mean_hinge", "cshipgen_noise_blend_hinge"}:
        hcddpm_local = {}
        scddpm_local = {}
        ordered_subsets = tuple(
            np.asarray(data.parameter_subsets[name], dtype=int)
            for name in SUBSYSTEM_ORDER
        )

        for level, (name, owned) in enumerate(
            zip(SUBSYSTEM_ORDER, ordered_subsets, strict=True)
        ):
            predecessors = (
                np.concatenate(ordered_subsets[:level])
                if level
                else np.asarray([], dtype=int)
            )
            hcddpm_columns = np.concatenate((owned, predecessors))
            hull_local = (
                hull_training is not None
                and bool(settings.get("hull_local_guidance", False))
                and name in HULL_SUBSYSTEMS
            )
            hcddpm_train_x = train_values[:, hcddpm_columns]
            hcddpm_validation_x = validation_values[:, hcddpm_columns]
            scddpm_columns = hull_training["indices"] if hull_local else np.arange(train_values.shape[1])
            scddpm_train_x = train_values[:, scddpm_columns]
            scddpm_validation_x = validation_values[:, scddpm_columns]
            train_labels = data.local_pass_labels[name][train_positions]
            validation_labels = data.local_pass_labels[name][validation_positions]

            if hull_local:
                label_column = f"Feasible::Subsystem::{name}"
                synthetic_train_known = hull_training["train_labels"][label_column].notna().to_numpy()
                synthetic_validation_known = hull_training["validation_labels"][label_column].notna().to_numpy()
                synthetic_train = hull_training["train_values"][synthetic_train_known]
                synthetic_validation = hull_training["validation_values"][synthetic_validation_known]
                train_labels = np.vstack((
                    train_labels,
                    hull_training["train_labels"].loc[
                        synthetic_train_known, [label_column]
                    ].to_numpy(np.float32),
                ))
                validation_labels = np.vstack((
                    validation_labels,
                    hull_training["validation_labels"].loc[
                        synthetic_validation_known, [label_column]
                    ].to_numpy(np.float32),
                ))
                hcddpm_hull_columns = [
                    hull_training["positions"][int(index)] for index in hcddpm_columns
                ]
                hcddpm_train_x = np.vstack((
                    hcddpm_train_x, synthetic_train[:, hcddpm_hull_columns],
                ))
                hcddpm_validation_x = np.vstack((
                    hcddpm_validation_x,
                    synthetic_validation[:, hcddpm_hull_columns],
                ))
                scddpm_train_x = np.vstack((scddpm_train_x, synthetic_train))
                scddpm_validation_x = np.vstack((scddpm_validation_x, synthetic_validation))

            hcddpm_local[name] = _train_one_classifier(
                values=hcddpm_train_x,
                labels=train_labels,
                validation_values=hcddpm_validation_x,
                validation_labels=validation_labels,
                settings=settings,
                path=output_root / "hcddpm_local" / f"{name}_Classifier.pt",
                seed=int(settings["training_seed"]) + 600 + level,
                name=f"HCDDPM_Contextual_{name}",
                # HCDDPM sees the current owned block and fixed predecessors.
                noisy_value_indices=np.arange(len(owned), dtype=int),
            )

            scddpm_local[name] = _train_one_classifier(
                values=scddpm_train_x,
                labels=train_labels,
                validation_values=scddpm_validation_x,
                validation_labels=validation_labels,
                settings=settings,
                path=output_root / "scddpm_local" / f"{name}_Classifier.pt",
                seed=int(settings["training_seed"]) + 700 + level,
                name=f"SCDDPM_Contextual_{name}",
                # Hull constraints depend on all 42 hull controls. Other
                # subsystem classifiers keep their original full context.
                noisy_value_indices=np.arange(scddpm_train_x.shape[1], dtype=int),
            )
    else:
        raise ValueError(
            f"Unknown classifier_guidance_mode {guidance_mode!r}."
        )

    manifest = {
        "schema_version": 3,
        "guidance_mode": guidance_mode,
        "known_training_rows": int(len(train_positions)),
        "known_validation_rows": int(len(validation_positions)),
        "global_output": "fully feasible (one binary output)",
        "hull_guidance_dataset": (
            hull_training["source_manifest"] if hull_training is not None else None
        ),
        "hull_global_output": (
            "all 49 hull equations pass (one binary output)"
            if hull_feasibility is not None else None
        ),
        "system_level_output": (
            "all cross-subsystem/system constraints pass (one binary output)"
            if system_level is not None else None
        ),
        "scddpm_post_merge_target": str(
            settings.get("system_guidance_target", "system_constraints")
        ),
        "local_outputs": {
            name: LOCAL_LABELS_BY_SUBSYSTEM[name][0]
            for name in SUBSYSTEM_ORDER
        },
        "guidance_output_masks": {
            "global": global_feasibility.guidance_output_mask.detach().cpu().tolist(),
            "hull_global": (
                hull_feasibility.guidance_output_mask.detach().cpu().tolist()
                if hull_feasibility is not None else None
            ),
            "system_level": (
                system_level.guidance_output_mask.detach().cpu().tolist()
                if system_level is not None else None
            ),
            "hcddpm_local": (
                {
                    name: classifier.guidance_output_mask.detach().cpu().tolist()
                    for name, classifier in hcddpm_local.items()
                }
                if hcddpm_local is not None
                else None
            ),
            "scddpm_local": (
                {
                    name: classifier.guidance_output_mask.detach().cpu().tolist()
                    for name, classifier in scddpm_local.items()
                }
                if scddpm_local is not None
                else None
            ),
            "meaning": "1=learnable two-class output; 0=exact zero guidance",
        },
        "guidance_temperatures": {
            "global": global_feasibility.guidance_temperature.detach().cpu().tolist(),
            "hull_global": (
                hull_feasibility.guidance_temperature.detach().cpu().tolist()
                if hull_feasibility is not None else None
            ),
            "system_level": (
                system_level.guidance_temperature.detach().cpu().tolist()
                if system_level is not None else None
            ),
            "hcddpm_local": (
                {
                    name: classifier.guidance_temperature.detach().cpu().tolist()
                    for name, classifier in hcddpm_local.items()
                }
                if hcddpm_local is not None
                else None
            ),
            "scddpm_local": (
                {
                    name: classifier.guidance_temperature.detach().cpu().tolist()
                    for name, classifier in scddpm_local.items()
                }
                if scddpm_local is not None
                else None
            ),
        },
        "input_scaling": (
            f"{settings.get('normalization_fit_scope', 'shared')} "
            f"{settings.get('normalization_type', 'quantile')} coordinates"
        ),
        "noise_training": (
            "clean design coordinates"
            if settings.get("classifier_noise_mode") == "clean"
            else "random timestep with schedule-matched Gaussian noise"
        ),
        "contextual_collections": {
            "hcddpm_local": {
                "available": hcddpm_local is not None,
                "input_order": (
                    "owned subsystem block, then every predecessor block in "
                    "SUBSYSTEM_ORDER"
                ),
                "noised_at_t": "owned subsystem block only",
                "gradient_application": "owned subsystem block only",
            },
            "scddpm_local": {
                "available": scddpm_local is not None,
                "input_order": "complete 133-control canonical vector",
                "noised_at_t": "complete simultaneous state",
                "gradient_application": "owned subsystem block only",
            },
            "legacy_block_only": {
                "available": legacy_local is not None,
                "use": "explicit legacy_noise_blend diagnostics only",
            },
        },
        "validation_metrics": (
            "Per-output PR-AUC, balanced accuracy, mean calibration error, "
            "and BCE"
        ),
    }
    with (output_root / "Classifier_Manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return GuidanceClassifiers(
        global_feasibility=global_feasibility,
        system_level=system_level,
        hull_feasibility=hull_feasibility,
        local=legacy_local,
        hcddpm_local=hcddpm_local,
        scddpm_local=scddpm_local,
    )


__all__ = [
    "FirstExperimentGenerator",
    "GuidanceClassifiers",
    "build_first_experiment_generator",
    "train_or_load_guidance_classifiers",
]


