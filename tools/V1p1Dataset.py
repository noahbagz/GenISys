"""Shared sampling and identifier rules for the GenISys V1.1 dataset.

The V1.1 population deliberately keeps the design relationships used by the
working legacy generator.  This module moves those relationships behind named
functions, repository-relative paths, deterministic random streams, and
explicit identifiers so the same design can be regenerated on any cluster
worker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cad_query_tools.parameters import ParametricStructureDefinition
from tools.HullParameterization import Hull_Parameterization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_PARAMETER_SCHEMA = PROJECT_ROOT / "tools" / "ShipSystemDesignParameters.csv"

NUMBER_OF_STRUCTURES = 5_000
HULLS_PER_STRUCTURE = 4
OPERATING_CONDITIONS_PER_HULL = 5
SYSTEMS_PER_STRUCTURE = HULLS_PER_STRUCTURE * OPERATING_CONDITIONS_PER_HULL
HULL_CANDIDATES_PER_STRUCTURE = 500

# One public base seed makes the complete population reproducible.  Separate
# stream tags below prevent changes to structure sampling from shifting the
# hull or operating-condition samples for the same Structure_Index.
DEFAULT_BASE_SEED = 1_104_2026
STRUCTURE_STREAM_TAG = 11
HULL_STREAM_TAG = 23
OPERATING_STREAM_TAG = 37

# These hull coordinates are overwritten after the structure and cargo-region
# layout determine the dimensional LOA, beam, depth, and bilge radius.  They
# therefore need a second bounds check *after* coupling; checking only the
# original random proposal does not describe the design that is actually
# written to the dataset.
POST_COUPLING_HULL_PARAMETER_NAMES = (
    "LOA",
    "Bd",
    "Dd",
    "Bc",
    "Rc",
)


def load_parameter_catalog() -> pd.DataFrame:
    """Return the authoritative ordered 189-coordinate system schema."""

    return pd.read_csv(FULL_PARAMETER_SCHEMA, encoding="utf-8-sig")


def parameter_names() -> list[str]:
    """Return all system parameter names in evaluator-vector order."""

    return load_parameter_catalog()["Name"].tolist()


def structural_parameter_catalog() -> pd.DataFrame:
    """Return the 120 structure rows with local zero-based indices."""

    catalog = load_parameter_catalog()
    return catalog.loc[catalog["Class"] == "Structure"].reset_index(drop=True)


def catalog_bound_violations(
    values: np.ndarray,
    catalog: pd.DataFrame,
    parameter_names_to_check: tuple[str, ...] | list[str] | None = None,
    parameter_names_to_ignore: tuple[str, ...] | list[str] = (),
) -> list[dict[str, float | str]]:
    """Return explicit lower/upper-bound violations for one parameter row.

    ``values`` must follow the row order of ``catalog``.  This helper is used
    both while generating designs and while compiling a completed dataset, so
    the acceptance test and the generation filter apply exactly the same
    comparison.  A very small scale-aware tolerance prevents CSV roundoff from
    turning a value written at a bound into a false violation.
    """

    values = np.asarray(values, dtype=float)
    ordered_catalog = catalog.reset_index(drop=True)

    if len(values) != len(ordered_catalog):
        raise ValueError(
            "The parameter row and catalog must contain the same number of "
            "ordered values."
        )

    names_to_ignore = set(parameter_names_to_ignore)

    if parameter_names_to_check is None:
        names_to_check = set(ordered_catalog["Name"].astype(str))
    else:
        names_to_check = set(parameter_names_to_check)

    violations = []

    for local_index, definition in ordered_catalog.iterrows():
        name = str(definition["Name"])

        if name not in names_to_check or name in names_to_ignore:
            continue

        value = float(values[local_index])
        lower = float(definition["LL"])
        upper = float(definition["UL"])
        tolerance = 1.0e-12*max(1.0, abs(lower), abs(upper))

        if value < lower - tolerance:
            violations.append(
                {
                    "Parameter": name,
                    "Value": value,
                    "Lower_Bound": lower,
                    "Upper_Bound": upper,
                    "Violation": "below_lower_bound",
                }
            )
        elif value > upper + tolerance:
            violations.append(
                {
                    "Parameter": name,
                    "Value": value,
                    "Lower_Bound": lower,
                    "Upper_Bound": upper,
                    "Violation": "above_upper_bound",
                }
            )

    return violations


def cleaned_structural_bound_violations(
    values: np.ndarray,
) -> list[dict[str, float | str]]:
    """Validate the final structural vector and its coupled member geometry."""

    catalog = structural_parameter_catalog()
    errors = catalog_bound_violations(
        values,
        catalog,
        parameter_names_to_ignore=("T", "web frame w"),
    )

    inner_side_index = int(
        catalog.index[catalog["Name"] == "Inner Side Shell Thickness"][0]
    )
    inner_side_value = float(values[inner_side_index])
    is_active = inner_side_value >= 10.0

    # The required inner-side shell closes the web-frame section, so the
    # separate web-frame flange width remains inactive in every V1.1 design.
    web_frame_width_index = int(
        catalog.index[catalog["Name"] == "web frame w"][0]
    )
    web_frame_width = float(values[web_frame_width_index])
    web_frame_width_lower = float(
        catalog.loc[web_frame_width_index, "LL"]
    )
    web_frame_width_upper = float(
        catalog.loc[web_frame_width_index, "UL"]
    )
    web_frame_width_is_valid = is_active and abs(web_frame_width) <= 1.0e-12

    if not web_frame_width_is_valid:
        errors.append(
            {
                "Parameter": "web frame w",
                "Value": web_frame_width,
                "Lower_Bound": web_frame_width_lower,
                "Upper_Bound": web_frame_width_upper,
                "Violation": "inconsistent_with_inner_side_shell",
            }
        )

    return errors


def rng_for(
    base_seed: int,
    structure_index: int,
    stream_tag: int,
    attempt: int = 0,
) -> np.random.Generator:
    """Create one deterministic random stream for a structure-level action."""

    seed = np.random.SeedSequence(
        [int(base_seed), int(structure_index), int(stream_tag), int(attempt)]
    )
    return np.random.default_rng(seed)


def system_index(
    structure_index: int,
    hull_variant_index: int,
    operating_condition_index: int,
) -> int:
    """Return the stable V1.1 system identifier for one factorial record."""

    return (
        int(structure_index) * SYSTEMS_PER_STRUCTURE
        + int(hull_variant_index) * OPERATING_CONDITIONS_PER_HULL
        + int(operating_condition_index)
    )


def derive_constraint_draft_m(system_parameters: np.ndarray) -> float:
    """Derive the dimensional constraint draft from the active system row.

    The source variable is ``Draft_frac``.  ``Dd`` is the depth-to-LOA ratio,
    and the structure's dimensional ``D`` is coupled to ``Dd*LOA`` by the hull
    generator.  Calculating from the ratio form keeps the data lineage clear.
    """

    catalog = load_parameter_catalog()
    index_by_name = dict(zip(catalog["Name"], catalog["idx"], strict=True))

    return float(
        system_parameters[index_by_name["Draft_frac"]]
        * system_parameters[index_by_name["Dd"]]
        * system_parameters[index_by_name["LOA"]]
    )


def derive_structural_bilge_radius_m(
    beam_m: float | np.ndarray,
    hull_rc: float | np.ndarray,
) -> float | np.ndarray:
    """Scale the hull bilge-radius ratio into the structure's radius.

    The hull parameterization defines ``Rc`` as the full bilge diameter
    divided by beam, or ``Rc = 2*R_b/B``.  The structural model requires the
    corresponding dimensional radius in metres.  Keeping this conversion in
    one named function makes the hull-to-structure design-data direction
    explicit and prevents the dimensionless ratio from being used as metres.
    """

    radius_m = (
        0.5
        * np.asarray(beam_m, dtype=float)
        * np.asarray(hull_rc, dtype=float)
    )

    if radius_m.ndim == 0:
        return float(radius_m)

    return radius_m


def sample_structural_parameter_row(
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample and clean one legacy-style bulk-carrier structure vector."""

    system_catalog = load_parameter_catalog()
    structure_catalog = structural_parameter_catalog()
    hull_catalog = system_catalog.loc[system_catalog["Class"] == "Hull"]

    lower = structure_catalog["LL"].to_numpy(dtype=float)
    upper = structure_catalog["UL"].to_numpy(dtype=float)
    params = np.zeros(len(structure_catalog), dtype=float)

    # Preserve the old generator's coupled principal-dimension sampling.  The
    # temporary LOA is used only to put beam in the intended ship-scale range.
    params[0] = rng.uniform(lower[0], upper[0])
    sampled_loa = rng.uniform(
        float(hull_catalog.loc[hull_catalog["Name"] == "LOA", "LL"].iloc[0]),
        float(hull_catalog.loc[hull_catalog["Name"] == "LOA", "UL"].iloc[0]),
    )
    beam_ratio_lower = float(
        hull_catalog.loc[hull_catalog["Name"] == "Bd", "LL"].iloc[0]
    )
    beam_ratio_upper = float(
        hull_catalog.loc[hull_catalog["Name"] == "Bd", "UL"].iloc[0]
    )
    bilge_ratio_lower = float(
        hull_catalog.loc[hull_catalog["Name"] == "Rc", "LL"].iloc[0]
    )
    bilge_ratio_upper = float(
        hull_catalog.loc[hull_catalog["Name"] == "Rc", "UL"].iloc[0]
    )

    params[1] = sampled_loa * rng.uniform(beam_ratio_lower, beam_ratio_upper)
    params[2] = 0.0
    params[3] = params[1] * rng.uniform(0.5, 0.75)
    params[5] = rng.uniform(lower[5], upper[5])
    params[6] = 1_000.0 * params[1] * rng.uniform(0.05, 0.0675)

    # Rc belongs to the hull design module.  The structural geometry also
    # requires R_b <= Db.  Sample Rc only over the compatible portion of its
    # hull-module interval, then convert it once to dimensional R_b.  This
    # preserves the direction Rc -> R_b without allowing structure cleanup to
    # clip or replace the hull-derived radius.
    compatible_rc_upper = min(
        bilge_ratio_upper,
        2.0*(params[6]/1_000.0)/params[1],
    )
    sampled_hull_rc = rng.uniform(
        bilge_ratio_lower,
        compatible_rc_upper,
    )
    params[4] = derive_structural_bilge_radius_m(
        params[1],
        sampled_hull_rc,
    )

    # The remaining values begin as uniform samples.  The established cleanup
    # routine then rounds discrete values, fixes class-specific inactive blocks,
    # and aligns coupled structural dimensions.
    params[7:] = rng.uniform(lower[7:], upper[7:])

    data_types = structure_catalog["data type"].fillna("").str.lower()
    boolean_indices = structure_catalog.index[data_types == "bool"].to_numpy()
    integer_indices = structure_catalog.index[data_types == "integer"].to_numpy()
    params[boolean_indices] = params[boolean_indices] >= 0.5
    params[integer_indices] = np.floor(params[integer_indices] + 0.5)

    # GenISys V1.1 is a bulk-carrier-only population.
    params[57:60] = 0.0
    params[59] = 1.0

    # Preserve the legacy random bracket-angle construction before the common
    # structure cleanup snaps bracket ends to nearby supporting members.
    bracket_slope_lower = np.tan(np.radians(35.0))
    bracket_slope_upper = np.tan(np.radians(50.0))
    params[61] = params[60] * rng.uniform(
        bracket_slope_lower,
        bracket_slope_upper,
    )
    params[64] = params[63] * rng.uniform(
        bracket_slope_lower,
        bracket_slope_upper,
    )

    cleaned = ParametricStructureDefinition.clean_struct_params(
        params.reshape(1, -1)
    )
    return cleaned[0].copy()


def _sample_hull_candidate(
    structural_parameters: np.ndarray,
    rng: np.random.Generator,
    catalog: pd.DataFrame,
) -> np.ndarray:
    """Sample one complete system vector around a fixed structure."""

    lower = catalog["LL"].to_numpy(dtype=float)
    upper = catalog["UL"].to_numpy(dtype=float)
    system_parameters = np.zeros(len(catalog), dtype=float)

    # The last 120 coordinates are the corrected structural design.  Their
    # beam, depth, and bilge radius define the compatible hull dimensions.
    system_parameters[69:] = structural_parameters
    beam_m = structural_parameters[1]
    depth_m = structural_parameters[3]
    bilge_radius_m = structural_parameters[4]
    bulkhead_spacing_m = (
        structural_parameters[0]
        * (1.0 - 2.0 * structural_parameters[5])
        / 3.0
    )

    # Sample the five non-cargo end-region fractions and the number of holds.
    system_parameters[48:53] = rng.uniform(lower[48:53], upper[48:53])
    system_parameters[53] = rng.integers(
        int(lower[53]),
        int(upper[53]) + 1,
    )
    cargo_region_fraction = 1.0 - np.sum(system_parameters[48:53])
    system_parameters[0] = (
        bulkhead_spacing_m
        * system_parameters[53]
        / cargo_region_fraction
    )

    # Preserve the old uniform hull, draft, Froude-number, and voyage-range
    # proposal.  Cargo fractions start at zero and are optimized by the system
    # evaluator after the available full-load cargo weight is calculated.
    system_parameters[1:48] = rng.uniform(lower[1:48], upper[1:48])

    loa_m = system_parameters[0]
    system_parameters[3] = beam_m / loa_m
    system_parameters[4] = depth_m / loa_m
    system_parameters[7] = beam_m / loa_m
    system_parameters[9] = 2.0 * bilge_radius_m / beam_m

    # ``WL`` is a hull-form coordinate.  It is scaled once by the first sampled
    # draft fraction and then held fixed when operating copies are made, exactly
    # as in the archived generator.
    system_parameters[6] *= system_parameters[45]

    data_types = catalog["data type"].fillna("").str.lower()
    boolean_indices = catalog.index[
        (data_types == "bool") & (catalog["idx"] < 69)
    ].to_numpy()
    system_parameters[boolean_indices] = np.floor(
        system_parameters[boolean_indices] + 0.5
    )
    system_parameters[21] = system_parameters[20]

    return system_parameters


def generate_valid_hull_variants(
    structural_parameters: np.ndarray,
    rng: np.random.Generator,
    candidate_count: int = HULL_CANDIDATES_PER_STRUCTURE,
) -> list[tuple[int, np.ndarray]]:
    """Return the first four legacy-style candidates passing hull constraints."""

    catalog = load_parameter_catalog()
    valid_hulls = []

    for candidate_index in range(int(candidate_count)):
        candidate = _sample_hull_candidate(
            structural_parameters,
            rng,
            catalog,
        )

        # Reject a candidate whose *derived* principal ratios leave the
        # published V1.1 design space.  Previously, the structural beam/depth
        # and cargo-region length could overwrite these fields after sampling
        # and produce accepted LOA, B/L, D/L, Bc/L, or bilge-radius ratios
        # outside their declared bounds.
        if catalog_bound_violations(
            candidate,
            catalog,
            parameter_names_to_check=POST_COUPLING_HULL_PARAMETER_NAMES,
        ):
            continue

        try:
            hull = Hull_Parameterization(candidate[:45])
            violations = hull.input_Constraints() > 0
        except Exception:
            # A geometric construction failure is an invalid hull proposal, not
            # a fabricated system row.  The structure-level batch status still
            # records failure if 500 proposals cannot supply four valid hulls.
            continue

        if np.sum(violations) == 0:
            valid_hulls.append((candidate_index, candidate.copy()))

        if len(valid_hulls) == HULLS_PER_STRUCTURE:
            break

    return valid_hulls


def expand_operating_conditions(
    structure_index: int,
    valid_hulls: list[tuple[int, np.ndarray]],
    rng: np.random.Generator,
    structure_sample_attempt: int,
) -> list[dict[str, float | int]]:
    """Expand four valid hulls into five legacy-style operating rows each."""

    catalog = load_parameter_catalog()
    names = catalog["Name"].tolist()
    lower = catalog["LL"].to_numpy(dtype=float)
    upper = catalog["UL"].to_numpy(dtype=float)
    index_by_name = dict(zip(names, catalog["idx"], strict=True))
    rows = []

    for hull_variant_index, (candidate_index, base_hull) in enumerate(valid_hulls):
        for operating_condition_index in range(OPERATING_CONDITIONS_PER_HULL):
            system_parameters = base_hull.copy()

            if operating_condition_index > 0:
                operating_indices = np.array([45, 46, 47], dtype=int)
                system_parameters[operating_indices] = rng.uniform(
                    lower[operating_indices],
                    upper[operating_indices],
                )

            # The archived table left structural T equal to zero.  V1.1 makes
            # the active system draft explicit so every regulatory constraint
            # receives the correct operating/loading-condition value.
            system_parameters[index_by_name["T"]] = derive_constraint_draft_m(
                system_parameters
            )

            row = {
                "Index": system_index(
                    structure_index,
                    hull_variant_index,
                    operating_condition_index,
                ),
                "Structure_Index": int(structure_index),
                "Hull_Variant_Index": int(hull_variant_index),
                "Operating_Condition_Index": int(operating_condition_index),
                "Hull_Candidate_Index": int(candidate_index),
                "Structure_Sample_Attempt": int(structure_sample_attempt),
            }
            row.update(dict(zip(names, system_parameters, strict=True)))
            rows.append(row)

    return rows
