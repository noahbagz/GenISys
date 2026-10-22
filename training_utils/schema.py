"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


from dataclasses import asdict, dataclass


import numpy as np


import pandas as pd


from parametric_repair_tools.parameters import ParameterRegistry


EXPERIMENT_FIXED_VALUES = {
    "Beta": 0.0,
    "web frame w": 0.0,
}


class _UnclippedExperimentRegistry(ParameterRegistry):
    """Represent discrete controls without projecting design values to bounds."""

    def clip_to_repair_domain(self, name: str, value: float) -> float:
        # ParameterRegistry.canonicalize calls this for discrete controls.
        # Keep its dependency rules, but never use its repair-domain clipping.
        if name in self.integer_names:
            return float(np.rint(value))
        if name in self.boolean_names:
            return float(value >= 0.5)
        return float(value)


@dataclass(frozen=True)
class PostprocessChange:
    """One traceable change made while decoding a generated vector."""

    sample_index: int
    parameter: str
    old_value: float
    new_value: float
    reason: str


class V1p1ParameterSchema:
    """Map between 133 learned controls and the complete 189-value interface."""

    def __init__(self):
        self.registry = _UnclippedExperimentRegistry()
        self.full_names = tuple(self.registry.names)
        self.full_index = dict(self.registry.index)

        self.model_names = tuple(
            name
            for name in self.full_names
            if self.registry.role(name).startswith("control_")
            and name not in EXPERIMENT_FIXED_VALUES
        )
        self.model_indices = np.asarray(
            [self.full_index[name] for name in self.model_names],
            dtype=int,
        )

        if len(self.full_names) != 189 or len(self.model_names) != 133:
            raise ValueError(
                "The frozen V1.1 schema must contain 189 complete and 133 "
                "genuinely variable model coordinates."
            )

    def schema_table(self) -> pd.DataFrame:
        """Return the machine-readable coordinate ownership table."""

        rows = []
        model_position = {name: i for i, name in enumerate(self.model_names)}
        for full_position, name in enumerate(self.full_names):
            role = self.registry.role(name)
            rows.append(
                {
                    "Full_Index": full_position,
                    "Model_Index": model_position.get(name, pd.NA),
                    "Name": name,
                    "Role": role,
                    "Modeled": name in model_position,
                    "Experiment_Fixed_Value": EXPERIMENT_FIXED_VALUES.get(
                        name,
                        np.nan,
                    ),
                    "Sampling_LL": self.registry.sampling_bounds(name)[0],
                    "Sampling_UL": self.registry.sampling_bounds(name)[1],
                }
            )
        return pd.DataFrame(rows)

    def encode(self, full_vectors: np.ndarray) -> np.ndarray:
        """Select independent, nonconstant coordinates from complete vectors."""

        full_vectors = np.asarray(full_vectors, dtype=np.float64)
        if full_vectors.ndim != 2 or full_vectors.shape[1] != 189:
            raise ValueError("encode expects an [n, 189] parameter matrix.")
        return full_vectors[:, self.model_indices].copy()

    def decode(
        self,
        model_vectors: np.ndarray,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Restore, discretize, couple, and derive the complete design vector.

        No stochastic decision occurs here.  The exact same model vector always
        produces the same 189-value evaluator input.
        """

        model_vectors = np.asarray(model_vectors, dtype=np.float64)
        if model_vectors.ndim == 1:
            model_vectors = model_vectors.reshape(1, -1)
        if model_vectors.ndim != 2 or model_vectors.shape[1] != 133:
            raise ValueError("decode expects an [n, 133] model-coordinate matrix.")

        decoded = np.zeros((len(model_vectors), 189), dtype=np.float64)
        change_rows: list[dict] = []

        for sample_index, model_vector in enumerate(model_vectors):
            proposed = np.zeros(189, dtype=np.float64)

            # Preserve every continuous model output, including values outside
            # the training and random-sampling ranges. Only representational
            # rounding is applied to integer and Boolean controls.
            for model_position, name in enumerate(self.model_names):
                raw_value = float(model_vector[model_position])
                cleaned_value = (
                    self.registry.clip_to_repair_domain(name, raw_value)
                    if name in self.registry.integer_names
                    or name in self.registry.boolean_names
                    else raw_value
                )
                proposed[self.full_index[name]] = cleaned_value

                if not np.isclose(raw_value, cleaned_value, atol=1.0e-12, rtol=0.0):
                    change_rows.append(
                        asdict(
                            PostprocessChange(
                                sample_index=sample_index,
                                parameter=name,
                                old_value=raw_value,
                                new_value=cleaned_value,
                                reason="discrete representation",
                            )
                        )
                    )

            for name, value in EXPERIMENT_FIXED_VALUES.items():
                proposed[self.full_index[name]] = value

            try:
                canonical, canonical_changes = self.registry.canonicalize(
                    proposed,
                    ensure_bilge_fit=False,
                )
            except ValueError as error:
                # A model may generate controls outside the representable
                # 189-parameter interface (for example, more than 15 cargo
                # holds). Keep its controls and mark missing dependent values
                # so the exact evaluation records a failed requested design.
                canonical = proposed.copy()
                for name in self.full_names:
                    if self.registry.role(name) == "derived":
                        canonical[self.full_index[name]] = np.nan
                canonical_changes = []
                change_rows.append(
                    asdict(
                        PostprocessChange(
                            sample_index=sample_index,
                            parameter="Decode_Status",
                            old_value=np.nan,
                            new_value=np.nan,
                            reason=f"unrepresentable model controls: {error}",
                        )
                    )
                )
            decoded[sample_index] = canonical

            for change in canonical_changes:
                change_rows.append(
                    asdict(
                        PostprocessChange(
                            sample_index=sample_index,
                            parameter=change.parameter,
                            old_value=change.old_value,
                            new_value=change.new_value,
                            reason=change.reason,
                        )
                    )
                )

        changes = pd.DataFrame(
            change_rows,
            columns=(
                "sample_index",
                "parameter",
                "old_value",
                "new_value",
                "reason",
            ),
        )
        return decoded, changes

    def model_subsets(self, full_catalog: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return nonoverlapping model positions for the four subsystem blocks."""

        class_by_name = dict(
            zip(
                full_catalog["Name"].astype(str),
                full_catalog["Class"].astype(str),
                strict=True,
            )
        )
        subsets: dict[str, list[int]] = {}
        for model_index, name in enumerate(self.model_names):
            subsets.setdefault(class_by_name[name], []).append(model_index)
        return {
            key: np.asarray(values, dtype=int)
            for key, values in subsets.items()
        }


__all__ = [
    "EXPERIMENT_FIXED_VALUES",
    "PostprocessChange",
    "V1p1ParameterSchema",
]


