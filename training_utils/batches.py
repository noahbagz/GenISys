"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import numpy as np


import torch


def _draw_batch_indices(environment, batch_budget: int) -> np.ndarray:
    """Draw one rebatching interval using the environment's historical RNG."""

    probabilities = getattr(environment, "sampling_probabilities", None)
    if probabilities is not None:
        return np.asarray(
            [
                np.random.choice(
                    len(environment.X),
                    size=environment.batch_size,
                    replace=True,
                    p=np.asarray(probabilities, dtype=float),
                )
                for _ in range(batch_budget)
            ],
            dtype=np.int64,
        )

    rows = []
    if hasattr(environment, "Cond"):
        for _ in range(batch_budget):
            row = torch.randint(
                0,
                len(environment.X),
                (environment.batch_size,),
                device=environment.device,
            )
            rows.append(row.detach().cpu().numpy())
    else:
        for _ in range(batch_budget):
            rows.append(
                np.random.randint(
                    0,
                    len(environment.X),
                    environment.batch_size,
                )
            )
    return np.asarray(rows, dtype=np.int64)


