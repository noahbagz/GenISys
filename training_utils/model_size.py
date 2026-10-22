"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


def diffusion_modules(model) -> list:
    """Return all denoisers in stable subsystem order."""

    if hasattr(model, "models"):
        return [submodel.diffusion for submodel in model.models]
    return [model.diffusion]


def count_trainable_parameters(model) -> int:
    """Count total denoiser capacity across every subsystem block."""

    return int(
        sum(
            parameter.numel()
            for module in diffusion_modules(model)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


