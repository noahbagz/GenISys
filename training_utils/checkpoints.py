"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import os


import platform


import random


import tempfile


from contextlib import ExitStack, contextmanager, nullcontext


from pathlib import Path


import numpy as np


import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch from one recorded training seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_environments(model) -> list:
    if hasattr(model, "models"):
        return list(model.models)
    return [model]


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def hardware_manifest() -> dict:
    """Record the hardware/software fields required by the experiment plan."""

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return result


def _checkpoint_payload(
    model,
    *,
    configuration: dict,
    training_indices: np.ndarray,
    metadata: dict | None = None,
    training_state: dict | None = None,
) -> dict:
    """Build one complete, restartable checkpoint payload."""

    environment_states = []
    for environment in _training_environments(model):
        environment_states.append(
            {
                "diffusion": environment.diffusion.state_dict(),
                "optimizer": environment.optimizer.state_dict(),
                "ema": environment.ema.state_dict(),
                "ema_buffers": getattr(environment, "ema_buffers", None),
                "data_normalizer": environment.data_norm,
                "condition_normalizers": getattr(environment, "cond_norm", None),
                "loss_history": np.asarray(environment.loss_history),
            }
        )

    checkpoint = {
        "schema_version": 2,
        "configuration": configuration,
        "training_indices": np.asarray(training_indices, dtype=np.int64),
        "environments": environment_states,
        "rng_state": _rng_state(),
        "hardware": hardware_manifest(),
        "metadata": metadata or {},
        "training_state": training_state or {},
    }
    return checkpoint


def _write_checkpoint_atomically(
    checkpoint: dict,
    path: Path,
    *,
    replace: bool,
) -> None:
    """Install one fully written checkpoint with one atomic filesystem step."""

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)

        torch.save(checkpoint, temporary_path)

        # Flush the completed temporary file before making it visible under
        # the public checkpoint name.
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())

        if replace:
            os.replace(temporary_path, path)
        else:
            # A hard-link install is atomic and fails if the immutable final
            # path appeared concurrently.  Unlike rename/replace, it never
            # overwrites an existing final checkpoint.
            os.link(temporary_path, path)
            temporary_path.unlink()
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_complete_checkpoint(
    model,
    path: str | Path,
    *,
    configuration: dict,
    training_indices: np.ndarray,
    metadata: dict | None = None,
    training_state: dict | None = None,
) -> Path:
    """Save a final checkpoint without overwriting an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}.")

    checkpoint = _checkpoint_payload(
        model,
        configuration=configuration,
        training_indices=training_indices,
        metadata=metadata,
        training_state=training_state,
    )
    _write_checkpoint_atomically(checkpoint, path, replace=False)
    return path


def save_recovery_checkpoint(
    model,
    path: str | Path,
    *,
    configuration: dict,
    training_indices: np.ndarray,
    metadata: dict | None = None,
    training_state: dict,
) -> Path:
    """Atomically update the single rolling interrupted-training checkpoint.

    Final checkpoints remain immutable and use :func:`save_complete_checkpoint`.
    This separate recovery artifact is intentionally replaceable so a process
    interruption cannot leave a partially written file or require one full
    checkpoint per epoch.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_payload(
        model,
        configuration=configuration,
        training_indices=training_indices,
        metadata=metadata,
        training_state=training_state,
    )

    _write_checkpoint_atomically(checkpoint, path, replace=True)

    return path


def load_complete_checkpoint(
    model,
    path: str | Path,
    *,
    restore_rng: bool = True,
) -> dict:
    """Restore a model already reconstructed from the recorded data/config."""

    checkpoint = torch.load(
        Path(path),
        map_location=_training_environments(model)[0].device,
        weights_only=False,
    )
    environments = _training_environments(model)
    states = checkpoint["environments"]
    if len(environments) != len(states):
        raise ValueError("Checkpoint subsystem count does not match the model.")

    for environment, state in zip(environments, states, strict=True):
        environment.diffusion.load_state_dict(state["diffusion"])
        environment.optimizer.load_state_dict(state["optimizer"])
        environment.ema.load_state_dict(state["ema"])
        environment.ema_buffers = state.get("ema_buffers")
        environment.data_norm = state["data_normalizer"]
        if state["condition_normalizers"] is not None:
            environment.cond_norm = state["condition_normalizers"]
        environment.loss_history = np.asarray(state["loss_history"])

    if restore_rng:
        _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


@contextmanager
def sampling_weight_context(model, treatment: str):
    """Select live or EMA denoiser weights for one complete sampling call."""

    if treatment not in {"live", "ema"}:
        raise ValueError("Sampling treatment must be 'live' or 'ema'.")

    if treatment == "live":
        with nullcontext():
            yield
        return

    with ExitStack() as stack:
        for environment in _training_environments(model):
            stack.enter_context(
                environment.ema.average_parameters(environment.diffusion)
            )
            if getattr(environment, "ema_buffers", None) is not None:
                live_buffers = {
                    name: tensor.detach().clone()
                    for name, tensor in environment.diffusion.named_buffers()
                }
                for name, tensor in environment.diffusion.named_buffers():
                    tensor.copy_(environment.ema_buffers[name].to(tensor.device))
                stack.callback(
                    _restore_named_buffers,
                    environment.diffusion,
                    live_buffers,
                )
        yield


def _restore_named_buffers(module, buffers: dict) -> None:
    """Put live-weight normalization statistics back after EMA sampling."""

    for name, tensor in module.named_buffers():
        tensor.copy_(buffers[name].to(tensor.device))


__all__ = [
    "hardware_manifest",
    "load_complete_checkpoint",
    "sampling_weight_context",
    "save_complete_checkpoint",
    "save_recovery_checkpoint",
    "seed_everything",
]


