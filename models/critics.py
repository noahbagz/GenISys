"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import hashlib


import json


import os


from dataclasses import asdict, dataclass


from pathlib import Path


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.nn.functional as F


from models import diffusion as TDT


from training_utils.checkpoints import seed_everything


from training_utils.critic_data import PerformanceGuidanceData


CHECKPOINT_SCHEMA = "genisys.full_state_performance_critic.v1"


@dataclass(frozen=True)
class PerformanceCriticConfig:
    """Training and architecture settings recorded in every checkpoint."""

    device_name: str = "cuda:0"
    diffusion_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    cosine_s: float = 0.008
    timestep_embedding_dim: int = 128
    hidden_layers: tuple[int, ...] = (384, 384, 256)
    dropout: float = 0.05
    batch_size: int = 512
    training_epochs: int = 40
    batches_per_epoch: int = 300
    learning_rate: float = 0.0005
    weight_decay: float = 0.00001
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.00001
    validation_row_limit: int = 4096
    maximum_training_rows: int | None = None
    training_seed: int = 32001


class FullStatePerformanceCritic(nn.Module):
    """Timestep-conditioned heteroscedastic regression over full ship state."""

    def __init__(
        self,
        *,
        control_dim: int,
        condition_dim: int,
        target_dim: int,
        timestep_embedding_dim: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        self.control_dim = int(control_dim)
        self.condition_dim = int(condition_dim)
        self.target_dim = int(target_dim)
        self.timestep_embedding_dim = int(timestep_embedding_dim)

        input_dim = (
            self.control_dim
            + 2 * self.condition_dim
            + self.timestep_embedding_dim
        )
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                (
                    nn.Linear(previous, int(width)),
                    nn.SiLU(),
                    nn.LayerNorm(int(width)),
                    nn.Dropout(float(dropout)),
                )
            )
            previous = int(width)

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, self.target_dim)
        self.log_variance_head = nn.Linear(previous, self.target_dim)

    def forward(
        self,
        controls: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: torch.Tensor,
        condition_known: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = TDT.timestep_embedding(
            timesteps,
            self.timestep_embedding_dim,
            device=controls.device,
        )
        features = torch.cat(
            (controls, conditions, condition_known, time),
            dim=1,
        )
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 5.0)
        return mean, log_variance


class PrefixPerformanceCritic(nn.Module):
    """Level-aware value critic for one HCDDPM predecessor/current prefix."""

    def __init__(
        self,
        *,
        control_dim: int,
        condition_dim: int,
        target_dim: int,
        level_count: int,
        timestep_embedding_dim: int,
        level_embedding_dim: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        self.control_dim = int(control_dim)
        self.condition_dim = int(condition_dim)
        self.target_dim = int(target_dim)
        self.level_count = int(level_count)
        self.timestep_embedding_dim = int(timestep_embedding_dim)
        self.level_embedding = nn.Embedding(
            self.level_count,
            int(level_embedding_dim),
        )

        input_dim = (
            2 * self.control_dim
            + 2 * self.condition_dim
            + self.timestep_embedding_dim
            + int(level_embedding_dim)
        )
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layers:
            layers.extend(
                (
                    nn.Linear(previous, int(width)),
                    nn.SiLU(),
                    nn.LayerNorm(int(width)),
                    nn.Dropout(float(dropout)),
                )
            )
            previous = int(width)

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, self.target_dim)
        self.log_variance_head = nn.Linear(previous, self.target_dim)

    def forward(
        self,
        controls: torch.Tensor,
        observed_mask: torch.Tensor,
        levels: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: torch.Tensor,
        condition_known: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = TDT.timestep_embedding(
            timesteps,
            self.timestep_embedding_dim,
            device=controls.device,
        )
        level = self.level_embedding(levels)
        features = torch.cat(
            (
                controls,
                observed_mask,
                conditions,
                condition_known,
                time,
                level,
            ),
            dim=1,
        )
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(-8.0, 5.0)
        return mean, log_variance


def _schedule(config: PerformanceCriticConfig, device: torch.device) -> torch.Tensor:
    return TDT.make_beta_schedule(
        {
            "Diffusion_Timesteps": int(config.diffusion_timesteps),
            "beta_schedule": str(config.beta_schedule),
            "beta_start": float(config.beta_start),
            "beta_end": float(config.beta_end),
            "cosine_s": float(config.cosine_s),
        },
        device,
    )


def _q_sample(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(clean)
    sqrt_alpha = torch.sqrt(alpha_bar[timesteps]).reshape(-1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar[timesteps]).reshape(-1, 1)
    return sqrt_alpha * clean + sqrt_one_minus * noise


def _gaussian_nll(
    prediction: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    return 0.5 * (torch.exp(-log_variance) * squared_error + log_variance).mean()


def _model_config(
    data: PerformanceGuidanceData,
    config: PerformanceCriticConfig,
) -> dict:
    return {
        "control_dim": int(data.controls.shape[1]),
        "condition_dim": int(data.conditions.shape[1]),
        "target_dim": int(data.targets.shape[1]),
        "timestep_embedding_dim": int(config.timestep_embedding_dim),
        "hidden_layers": list(config.hidden_layers),
        "dropout": float(config.dropout),
    }


def _build_model(model_config: dict, device: torch.device) -> FullStatePerformanceCritic:
    return FullStatePerformanceCritic(
        control_dim=int(model_config["control_dim"]),
        condition_dim=int(model_config["condition_dim"]),
        target_dim=int(model_config["target_dim"]),
        timestep_embedding_dim=int(model_config["timestep_embedding_dim"]),
        hidden_layers=tuple(int(value) for value in model_config["hidden_layers"]),
        dropout=float(model_config["dropout"]),
    ).to(device)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _data_signature(data: PerformanceGuidanceData) -> str:
    encoded = json.dumps(data.manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_or_load_performance_critic(
    data: PerformanceGuidanceData,
    *,
    config: PerformanceCriticConfig,
    output_directory: str | Path,
) -> FullStatePerformanceCritic:
    """Train with fixed noisy validation, early stopping, and atomic resume."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "best_performance_critic.pt"
    recovery_path = output_directory / "performance_critic.recovery.pt"
    history_path = output_directory / "Performance_Critic_Training_History.csv"

    device = torch.device(config.device_name)
    model_config = _model_config(data, config)
    model = _build_model(model_config, device)

    train_positions = data.positions("train")
    validation_positions = data.positions("validation")
    random = np.random.default_rng(config.training_seed)
    if config.maximum_training_rows is not None:
        count = min(int(config.maximum_training_rows), len(train_positions))
        train_positions = np.sort(random.choice(train_positions, count, replace=False))
    validation_count = min(int(config.validation_row_limit), len(validation_positions))
    validation_positions = np.sort(
        random.choice(validation_positions, validation_count, replace=False)
    )

    signature = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_config": model_config,
        "training_config": asdict(config),
        "data_manifest_sha256": _data_signature(data),
        "train_positions_sha256": _array_sha256(train_positions.astype("<i8")),
        "validation_positions_sha256": _array_sha256(
            validation_positions.astype("<i8")
        ),
        "target_names": list(data.target_names),
        "condition_names": list(data.condition_names),
        "control_names": list(data.control_names),
    }

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(
                f"Performance critic contract changed for {checkpoint_path}."
            )
        model.load_state_dict(payload["model"])
        model.eval()
        return model

    seed_everything(config.training_seed)
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

    betas = _schedule(config, device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    # A fixed validation ledger ensures early stopping never compares models on
    # different noising draws.
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(config.training_seed + 900_000)
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
    validation_noisy = _q_sample(
        validation_x,
        validation_timesteps,
        alpha_bar,
        noise=validation_noise,
    )

    first_epoch = 0
    history: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model = None

    if recovery_path.exists():
        payload = torch.load(recovery_path, map_location=device, weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError(
                f"Performance critic contract changed for {recovery_path}."
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first_epoch = int(payload["completed_epochs"])
        history = list(payload["history"])
        best_validation_loss = float(payload["best_validation_loss"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        best_model = payload["best_model"]

    def validate() -> tuple[float, float]:
        model.eval()
        weighted_loss = torch.zeros((), device=device)
        weighted_mse = torch.zeros((), device=device)
        validation_batch = min(int(config.batch_size), len(validation_x))
        with torch.no_grad():
            for start in range(0, len(validation_x), validation_batch):
                stop = min(start + validation_batch, len(validation_x))
                prediction, log_variance = model(
                    validation_noisy[start:stop],
                    validation_timesteps[start:stop],
                    validation_c[start:stop],
                    validation_k[start:stop],
                )
                target = validation_y[start:stop]
                weighted_loss += (
                    _gaussian_nll(prediction, log_variance, target)
                    * len(target)
                )
                weighted_mse += F.mse_loss(prediction, target) * len(target)
        model.train()
        return (
            float((weighted_loss / len(validation_x)).item()),
            float((weighted_mse / len(validation_x)).item()),
        )

    print(
        "Performance critic: "
        f"{first_epoch}/{config.training_epochs} epochs complete",
        flush=True,
    )
    for epoch in range(first_epoch, int(config.training_epochs)):
        torch.manual_seed(config.training_seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training_seed + epoch)

        model.train()
        loss_total = torch.zeros((), device=device)
        batch_size = min(int(config.batch_size), len(train_x))
        for _ in range(int(config.batches_per_epoch)):
            index_tensor = torch.randint(
                0, len(train_x), (batch_size,), device=device,
            )
            clean = train_x[index_tensor]
            timestep = torch.randint(
                0,
                int(config.diffusion_timesteps),
                (batch_size,),
                device=device,
            )
            noisy = _q_sample(clean, timestep, alpha_bar)

            optimizer.zero_grad(set_to_none=True)
            prediction, log_variance = model(
                noisy,
                timestep,
                train_c[index_tensor],
                train_k[index_tensor],
            )
            loss = _gaussian_nll(prediction, log_variance, train_y[index_tensor])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            loss_total += loss.detach()

        training_loss = float((loss_total / int(config.batches_per_epoch)).item())
        validation_loss, validation_mse = validate()
        history.append(
            {
                "Epoch": epoch + 1,
                "Training_NLL": training_loss,
                "Validation_NLL": validation_loss,
                "Validation_MSE": validation_mse,
                "Learning_Rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = (
            validation_loss
            < best_validation_loss - float(config.early_stopping_min_delta)
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch {epoch + 1}/{config.training_epochs}: "
            f"train_nll={history[-1]['Training_NLL']:.6f}, "
            f"validation_nll={validation_loss:.6f}, "
            f"validation_mse={validation_mse:.6f}",
            flush=True,
        )

        recovery_payload = {
            "schema": CHECKPOINT_SCHEMA,
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
        raise RuntimeError("Performance critic training produced no best checkpoint.")
    final["model"] = final["best_model"]
    final["checkpoint_schema"] = CHECKPOINT_SCHEMA
    final["training_positions"] = train_positions
    final["validation_positions"] = validation_positions

    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(final, temporary)
    os.replace(temporary, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    recovery_path.unlink()

    model.load_state_dict(final["model"])
    model.eval()
    return model


def load_performance_critic_checkpoint(
    checkpoint_path: str | Path,
    *,
    device_name: str,
) -> tuple[FullStatePerformanceCritic, dict]:
    """Reconstruct a critic without requiring the training arrays."""

    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model(payload["model_config"], device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


