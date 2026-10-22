"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


import numpy as np


from training_utils.data import PreparedFirstExperimentData, SUBSYSTEM_ORDER


UNKNOWN = 0


KNOWN_FAIL = 1


KNOWN_PASS = 2


def label_strata(
    pass_labels: np.ndarray,
    known_mask: np.ndarray,
) -> np.ndarray:
    """Encode unknown, known-fail, and known-pass rows as integer strata."""

    passed = np.asarray(pass_labels, dtype=bool).reshape(-1)
    known = np.asarray(known_mask, dtype=bool).reshape(-1)

    strata = np.full(len(passed), UNKNOWN, dtype=np.int8)
    strata[known & ~passed] = KNOWN_FAIL
    strata[known & passed] = KNOWN_PASS
    return strata


def parent_label_balanced_probabilities(
    parent_ids: np.ndarray,
    pass_labels: np.ndarray,
    known_mask: np.ndarray,
    *,
    unknown_mass: float = 1.0 / 3.0,
) -> np.ndarray:
    """Balance label states while retaining parent balancing within each state.

    When all three states occur, ``unknown_mass`` is reserved for unknown rows
    and the remaining mass is divided equally between known pass and known
    fail.  If one known outcome is absent, its mass is reassigned to the known
    outcome that exists.  Within each state, the original inverse parent-row
    weighting is preserved before the state is normalized to its target mass.
    """

    parents = np.asarray(parent_ids).reshape(-1)
    strata = label_strata(pass_labels, known_mask)

    # Count rows separately for each (label state, parent) pair. A parent
    # can contribute an original row and repaired variants with different
    # outcomes, so one global parent count would not balance that parent
    # within each state.
    parent_weights = np.zeros(len(parents), dtype=np.float64)
    for value in (UNKNOWN, KNOWN_FAIL, KNOWN_PASS):
        selected = strata == value
        if not np.any(selected):
            continue
        _, inverse, counts = np.unique(
            parents[selected],
            return_inverse=True,
            return_counts=True,
        )
        parent_weights[selected] = 1.0 / counts[inverse].astype(np.float64)

    present_known = [
        value
        for value in (KNOWN_FAIL, KNOWN_PASS)
        if np.any(strata == value)
    ]
    has_unknown = bool(np.any(strata == UNKNOWN))

    targets: dict[int, float] = {}
    if has_unknown:
        targets[UNKNOWN] = float(unknown_mass)
        known_mass = 1.0 - float(unknown_mass)
    else:
        known_mass = 1.0

    for value in present_known:
        targets[value] = known_mass / len(present_known)

    probabilities = np.zeros(len(parents), dtype=np.float64)
    for value, target_mass in targets.items():
        selected = strata == value
        probabilities[selected] = (
            target_mass
            * parent_weights[selected]
            / parent_weights[selected].sum()
        )

    return probabilities / probabilities.sum()


def environment_sampling_probabilities(
    data: PreparedFirstExperimentData,
    positions: np.ndarray,
    *,
    architecture: str,
    population: str,
    unknown_mass: float = 1.0 / 3.0,
) -> list[np.ndarray]:
    """Return the training probabilities for every denoising environment.

    Feasible-only models retain ordinary parent balancing.  The full-data
    monolithic model balances the global label, while each HCDDPM or SCDDPM
    environment balances the local label owned by its subsystem.
    """

    positions = np.asarray(positions, dtype=np.int64)
    parent_ids = data.row_table.iloc[positions][
        "Source_Structure_Index"
    ].to_numpy(dtype=np.int64)

    if population == "feasible_only":
        probabilities = data.parent_balancing_probabilities(positions)
        count = 1 if architecture == "monolithic" else len(SUBSYSTEM_ORDER)
        return [probabilities.copy() for _ in range(count)]

    known = data.known_mask[positions]
    if architecture == "monolithic":
        labels = data.global_pass_labels[positions]
        return [
            parent_label_balanced_probabilities(
                parent_ids,
                labels,
                known,
                unknown_mass=unknown_mass,
            )
        ]

    if architecture in {"hcddpm", "scddpm"}:
        return [
            parent_label_balanced_probabilities(
                parent_ids,
                data.local_pass_labels[name][positions],
                known,
                unknown_mass=unknown_mass,
            )
            for name in SUBSYSTEM_ORDER
        ]

    raise ValueError(f"Unknown generator architecture {architecture!r}.")


__all__ = [
    "KNOWN_FAIL",
    "KNOWN_PASS",
    "UNKNOWN",
    "environment_sampling_probabilities",
    "label_strata",
    "parent_label_balanced_probabilities",
]


