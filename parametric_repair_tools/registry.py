"""Explicit constraint-to-control registry for GenISys V1.1 repair.

Rules are selected from the full ``Requirement::Name`` identity.  Pattern
matching is used only for physically repeated instances such as numbered
cargo or ballast tanks; the selected engineering relationship is still
written into the exported registry and every repair-action row.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from .hull_domain_repairs import (
    HOLTROP_REPAIR_CONTROLS,
    HULL_REPAIR_CONTROLS,
)


@dataclass(frozen=True)
class RepairRule:
    """Describe one reviewed constraint-to-parameter relationship."""

    method: str
    family: str
    parameters: tuple[str, ...]
    priority: int
    status: str = "repairable"
    note: str = ""
    offered_scale: float = 1.0


PLATE_PARAMETER_BY_NAME = {
    "Min Inner Side Shell Thickness": "Inner Side Shell Thickness",
    "Min Floor Thickness": "floor t",
    "Min Bottom Floor Thickness": "floor t",
    "MVR solid-floor plate thickness": "floor t",
    "Double-bottom floor web thickness in deep tank": "floor t",
    "Min Bottom Girder Thickness": "bottom girder thickness",
    "Double-bottom girder web thickness in deep tank": (
        "bottom girder thickness"
    ),
    "Min Inner Bottom Grab-Service Thickness": "Inner Bottom Thickness",
    "Inner bottom plating - general Rule thickness": "Inner Bottom Thickness",
    "Inner bottom plating - cargo pressure": "Inner Bottom Thickness",
    "Inner bottom plating - grab loading": "Inner Bottom Thickness",
    "Inner bottom plating - excavator wheel loading": "Inner Bottom Thickness",
    "Min Lower Hopper Grab-Service Thickness": "bulk car bottom hopper t",
    "Lower hopper plating - grab loading": "bulk car bottom hopper t",
    "Strength deck plating - absolute minimum": "Top Deck Thickness",
    "Strength deck plating - unloaded exposed-deck pressure": (
        "Top Deck Thickness"
    ),
    "Strength deck plating - excavator wheel loading": "Top Deck Thickness",
    "Strength deck plating - grab loading": "Top Deck Thickness",
    "Bottom-shell absolute minimum thickness": "bottom shell thickness",
    "MVR bottom-shell pressure thickness": "bottom shell thickness",
    "MVR longitudinally framed minimum bottom-shell thickness": (
        "bottom shell thickness"
    ),
    "MVR side-shell pressure thickness": "side shell thickness",
    "MVR side-shell amidship end-rule minimum": "side shell thickness",
    "Side-shell end thickness at actual spacing": "side shell thickness",
    "Watertight bulkhead plating - absolute minimum": (
        "Trans Bulkhead Thickness"
    ),
    "Watertight bulkhead plating - pressure equation": (
        "Trans Bulkhead Thickness"
    ),
    "Watertight bulkhead plating - spacing minimum": (
        "Trans Bulkhead Thickness"
    ),
    "Min Web Frame Thickness": "web frame t",
    "Min Deck Beam Thickness": "trans deck beam t",
    "Min Deck Girder Thickness": "deck girder t",
    "Min Lower Hopper Bracket Thickness": "floor web bracket t",
    "Min Upper Hopper Bracket Thickness": "deck web bracke t",
    "Min Lower Hopper Plate Thickness": "bulk car bottom hopper t",
    "Min Upper Hopper Plate Thickness": "bulk car top hopper t",
}

DEPTH_PARAMETER_BY_NAME = {
    "Min Web Frame Height": "web frame h",
    "Min Side Stringer Height": "side shell Long stiff h",
    "Min Bottom Floor Height": "Db",
    "Min Bottom Girder Height": "Db",
    "Min Deck Beam Height": "trans deck beam h",
    "Min Deck Girder Height": "deck girder h",
    "MVR deck-girder minimum depth": "deck girder h",
    "MVR deck-transverse minimum depth": "trans deck beam h",
}

DOUBLE_BOTTOM_HEIGHT_NAMES = {
    "Min Double Bottom Height",
    "Double-bottom height - absolute minimum",
    "Double-bottom height - breadth twentieth",
    "Double-bottom height - draft and breadth equation",
    "SOLAS double-bottom height",
}

SPACING_CONTROL_BY_NAME = {
    "Max Bottom Girder Spacing": "num bottom girders",
    "Max Bottom Stiffener Spacing": "num intermediate bottom stiffeners",
    "Max Floor Spacing": "web frames per hold",
    "Max Frame Spacing": "num intermediate trans frames",
    "MVR maximum transverse-frame spacing": "num intermediate trans frames",
}

ASPECT_CONTROL_BY_PREFIX = {
    "Shear_Strake": ("Shear Strake thickness", "Shear Strake Height"),
    "Bottom_Girder": ("bottom girder thickness", "Db"),
    "Bottom_Floor": ("floor t", "Db"),
    "Bottom_Stringer": (
        "intermediate bottom stiffener t",
        "intermediate bottom stiffener h",
    ),
    "Inner_Bottom_Stringer": (
        "intermediate bottom stiffener t",
        "intermediate bottom stiffener h",
    ),
    "Bottom_Frame": (
        "intermediate bottom frame t",
        "intermediate bottom frame h",
    ),
    "Inner_Bottom_Frame": (
        "intermediate bottom frame t",
        "intermediate bottom frame h",
    ),
    "Deck_Beam": ("trans deck beam t", "trans deck beam h"),
    "Deck_Girder": ("deck girder t", "deck girder h"),
    "Deck_Stringer": ("deck stiff t", "deck stiff h"),
    "Intermediate_Deck_Beam": (
        "intermediate deck frame t",
        "intermediate deck frame h",
    ),
    "Horizontal_Bulkhead_Stringer": (
        "blkhd trans stiff t",
        "blkhd trans stiff h",
    ),
    "Vertical_Bulkhead_Stringer": (
        "blkhd vert stiff t",
        "blkhd vert stiff h",
    ),
    "WebFrame": ("web frame t", "web frame h"),
    "Side_Stringer": (
        "side shell long stiff t",
        "side shell Long stiff h",
    ),
    "Lower_Hopper_Stiffener": (
        "bulk car bottom hopper stiff t",
        "bulk car bottom hopper stiff h",
    ),
    "Upper_Hopper_Stiffener": (
        "bulk car top hopper stiff t",
        "bulk car top hopper stiff h",
    ),
}


SECTION_FAMILIES = {
    "bottom_stringer": (
        (
            "Min Bottom Stringer SM",
            "Min Relative Bottom Stringer Section Modulus",
            "Min Relative Inner Bottom Stringer Section Modulus",
            "Inner-bottom longitudinal - bottom-strength minimum",
            "Inner-bottom longitudinal - optimized cargo loading",
        ),
        (
            "intermediate bottom stiffener h",
            "intermediate bottom stiffener t",
            "intermediate bottom stiffener w",
            "num intermediate bottom stiffeners",
            "num bottom girders",
        ),
    ),
    "bottom_frame": (
        (
            "Min Bottom Frame SM",
            "Min Relative Bottom Frame Section Modulus",
            "Min Relative Inner Bottom Frame Section Modulus",
        ),
        (
            "intermediate bottom frame h",
            "intermediate bottom frame t",
            "intermediate bottom frame w",
            "num intermediate trans frames",
        ),
    ),
    "bottom_girder": (
        (
            "Min Bottom Girder Section Modulus",
            "Min Relative Bottom Girder Section Modulus Lower",
            "Min Relative Bottom Girder Section Modulus Upper",
        ),
        ("Db", "bottom girder thickness", "num bottom girders"),
    ),
    "bottom_floor": (
        (
            "Min Bottom Floor Section Modulus",
            "Min Relative Floor Section Modulus Lower",
            "Min Relative Floor Section Modulus Upper",
        ),
        ("Db", "floor t", "web frames per hold"),
    ),
    "deck_stringer": (
        (
            "Min Relative Deck Stringer Section Modulus",
            "Strength-deck longitudinal section modulus",
        ),
        (
            "deck stiff h",
            "deck stiff t",
            "deck stiff w",
            "num intermediate bottom stiffeners",
            "num bottom girders",
        ),
    ),
    "deck_frame": (
        ("Min Relative Deck Frame Section Modulus",),
        (
            "intermediate deck frame h",
            "intermediate deck frame t",
            "intermediate deck frame w",
            "num intermediate trans frames",
        ),
    ),
    "deck_beam": (
        (
            "Min Relative Deck Beam Section Modulus",
            "Ordinary strength-deck transverse-beam section modulus",
            "Strength-deck transverse section modulus clear of tanks",
        ),
        (
            "trans deck beam h",
            "trans deck beam t",
            "trans deck beam w",
            "web frames per hold",
            "num bottom girders",
        ),
    ),
    "deck_girder": (
        (
            "Min Relative Deck Girder Section Modulus",
            "Strength-deck girder section modulus clear of tanks",
        ),
        (
            "deck girder h",
            "deck girder t",
            "deck girder w",
            "num bottom girders",
            "num intermediate bottom stiffeners",
        ),
    ),
    "side_stringer": (
        (
            "Min Relative Side Stringer Section Modulus Lower",
            "Min Relative Side Stringer Section Modulus Upper",
            "Side stringer - ordinary side-shell loading",
        ),
        (
            "side shell Long stiff h",
            "side shell long stiff t",
            "side shell long stiff w",
            "num side shell stiffeners",
        ),
    ),
    "side_frame": (
        ("MVR transverse side-frame section modulus",),
        (
            "intermediate side frame h",
            "intermediate side frame t",
            "intermediate side frame w",
            "num intermediate trans frames",
        ),
    ),
    "web_frame": (
        ("MVR web-frame section modulus",),
        ("web frame h", "web frame t", "web frame w", "web frames per hold"),
    ),
    "bulkhead_horizontal": (
        (
            "Min Relative Transverse Bulkhead Stiffener Section Modulus Upper",
            "Min Relative Transverse Bulkhead Stiffener Section Modulus Lower",
            "Horizontal bulkhead stiffener - watertight loading",
        ),
        (
            "blkhd trans stiff h",
            "blkhd trans stiff t",
            "blkhd trans stiff w",
            "num transverse bulkhead stiffeners",
        ),
    ),
    "bulkhead_vertical": (
        ("Vertical bulkhead stiffener - watertight loading",),
        (
            "blkhd vert stiff h",
            "blkhd vert stiff t",
            "blkhd vert stiff w",
            "num vertical bulkhead stiffeners",
        ),
    ),
    "hopper_lower": (
        ("Min Relative Lower Hopper Stiffener Section Modulus",),
        (
            "bulk car bottom hopper stiff h",
            "bulk car bottom hopper stiff t",
            "bulk car bottom hopper stiff w",
            "num bulk car bottom hopper stiff",
        ),
    ),
    "hopper_upper": (
        ("Min Relative Upper Hopper Stiffener Section Modulus",),
        (
            "bulk car top hopper stiff h",
            "bulk car top hopper stiff t",
            "bulk car top hopper stiff w",
            "num bulk car top hopper stiff",
        ),
    ),
    "shear_strake": (
        ("Min Relative Shear Strake Section Modulus",),
        ("Shear Strake Height", "Shear Strake thickness"),
    ),
}

SECTION_BY_NAME = {
    name: (family, parameters)
    for family, (names, parameters) in SECTION_FAMILIES.items()
    for name in names
}

HULL_GIRDER_NAMES = {
    "Hull-girder total-load section modulus - bottom",
    "Hull-girder total-load section modulus - deck",
    "Hull-girder absolute minimum section modulus - bottom",
    "Hull-girder absolute minimum section modulus - deck",
    "Hull-girder minimum moment of inertia",
    "Hull-girder side-shell nominal shear-stress limit",
}


# These are the two legacy labels whose corrected equations and generated-
# geometry inputs are identical to the supported atomic ABS MVR 3-2-7/3.1
# rows.  They remain active traceability aliases and therefore fall through to
# ``SECTION_BY_NAME``.  The other historical ``Min Relative ...`` rows are the
# seventeen unsupported member applications removed under CRF-009.
SUPPORTED_RELATIVE_SECTION_ALIASES = {
    "Min Relative Deck Stringer Section Modulus",
    "Min Relative Deck Frame Section Modulus",
}


class ConstraintRepairRegistry:
    """Resolve runtime rows to reviewed repair or exclusion policies."""

    def classify(self, row: pd.Series | dict) -> RepairRule:
        """Return the applicable rule without relying on row position."""

        name = str(row.get("Name", ""))
        requirement = str(row.get("Requirement", ""))
        result_class = str(row.get("Class", ""))

        # The long-form frozen source tables predate the approved evaluator
        # dispositions.  Preserve those historical rows, but prevent a row
        # that is now explicitly N/A or superseded from driving a first-pass
        # mutation before the exact evaluator writes the corrected contract.
        if name.endswith("_Aspect_Ratio"):
            return RepairRule(
                "none",
                "member_aspect_ratio",
                (),
                999,
                status="out_of_scope_not_counted",
                note=(
                    "Project disposition: ABS MVR 3-1-2/13.5.2 aspect-ratio "
                    "evaluations are excluded from GenISys feasibility."
                ),
            )

        if (
            name.startswith("Min Relative ")
            and name not in SUPPORTED_RELATIVE_SECTION_ALIASES
        ):
            return RepairRule(
                "none",
                "relative_section_modulus_application",
                (),
                999,
                status="superseded_rule_application_not_counted",
                note=(
                    "CRF-009 disposition: these legacy spellings over-apply "
                    "MVR 3-2-7/3.1 and are removed from active V1.1 output. "
                    "The two supported deck members use separate atomic rows."
                ),
            )

        if name in PLATE_PARAMETER_BY_NAME:
            return RepairRule(
                "direct_minimum",
                "plate_or_web_thickness",
                (PLATE_PARAMETER_BY_NAME[name],),
                20,
            )

        if name == "Min Center Bottom Girder Thickness":
            return RepairRule(
                "direct_minimum",
                "center_girder_double_thickness",
                ("bottom girder thickness",),
                20,
                offered_scale=2.0,
                note="Center girder is twice the ordinary girder thickness.",
            )

        if name == "MVR center-girder plate thickness amidships":
            return RepairRule(
                "direct_minimum",
                "center_girder_double_thickness",
                ("bottom girder thickness",),
                20,
                offered_scale=2.0,
                note="Center girder is twice the ordinary girder thickness.",
            )

        if name in DEPTH_PARAMETER_BY_NAME:
            return RepairRule(
                "direct_minimum",
                "member_depth",
                (DEPTH_PARAMETER_BY_NAME[name],),
                25,
            )

        if name in DOUBLE_BOTTOM_HEIGHT_NAMES:
            return RepairRule(
                "direct_minimum",
                "double_bottom_height",
                ("Db",),
                15,
            )

        if name in SPACING_CONTROL_BY_NAME:
            return RepairRule(
                "maximum_spacing_count",
                "member_spacing",
                (SPACING_CONTROL_BY_NAME[name],),
                10,
            )

        if name in SECTION_BY_NAME:
            family, parameters = SECTION_BY_NAME[name]
            return RepairRule(
                "section_modulus",
                family,
                tuple(parameters),
                40,
            )

        if name in HULL_GIRDER_NAMES:
            return RepairRule(
                "global_hull_girder",
                "hull_girder",
                (
                    "bottom shell thickness",
                    "side shell thickness",
                    "Top Deck Thickness",
                    "Inner Bottom Thickness",
                    "Inner Side Shell Thickness",
                    "Shear Strake thickness",
                ),
                50,
                note=(
                    "Only longitudinally effective shell/deck zones are "
                    "scaled. Transverse bulkhead plating is repaired by its "
                    "own pressure/minimum rules and is not credited to the "
                    "hull-girder effective section."
                ),
            )

        if name in {
            "Min Lower Hopper Bracket Angle",
            "Max Lower Hopper Bracket Angle",
        }:
            return RepairRule(
                "bounded_bracket_angle",
                "lower_hopper_bracket_angle",
                ("floor web bracket L1", "floor web bracket L2"),
                12,
            )

        if name in {
            "Min Upper Hopper Bracket Angle",
            "Max Upper Hopper Bracket Angle",
        }:
            return RepairRule(
                "bounded_bracket_angle",
                "upper_hopper_bracket_angle",
                ("deck web bracket L1", "deck web bracket L2"),
                12,
            )

        if name == "MVR strength-deck hatch-corner radius":
            return RepairRule(
                "direct_minimum",
                "hatch_corner_radius",
                ("Hatch Opening R",),
                18,
            )

        if name == "Min Hatch Area":
            return RepairRule(
                "hatch_area",
                "hatch_opening_area",
                ("Hatch Opening L", "Hatch Opening W", "Hatch Opening R"),
                35,
            )

        if name == "Min Engine Room Length":
            return RepairRule(
                "engine_room_length",
                "machinery_arrangement",
                ("frac_engineRoom_L", "LOA", "Fn_ship"),
                8,
            )

        if result_class == "Fuel Tank Volume" or name == "Fuel Tank Not Overfilled":
            return RepairRule(
                "tank_capacity",
                "fuel_tank_capacity",
                ("Fn_ship", "Voyage_Range"),
                60,
            )

        if result_class == "Ballast Tank Volume" and re.match(
            r"Ballast Tank \d+ Not Overfilled", name
        ):
            # Cargo-replacement ballast capacity is retained only as a legacy
            # diagnostic under HDF-ASM-015. Frozen source batches may still
            # contain these rows, so classify them explicitly as non-targets.
            return RepairRule(
                "none",
                "ballast_tank_capacity",
                (),
                999,
                status="diagnostic_not_counted",
                note=(
                    "HDF-ASM-015 removes cargo-replacement ballast overfill "
                    "from active feasibility and repair targeting."
                ),
            )

        if result_class == "Cargo Hold Volume" and re.match(
            r"Cargo Hold \d+ Not Overfilled", name
        ):
            return RepairRule(
                "tank_capacity",
                "cargo_hold_capacity",
                ("Bd", "Dd", "Num_Cargo_Holds"),
                4,
            )

        if name == "Total cargo-hold mass capacity":
            return RepairRule(
                "system_capacity",
                "cargo_hold_capacity",
                ("Bd", "Dd", "Draft_frac", "Num_Cargo_Holds"),
                4,
            )

        if name in {
            "Positive available cargo payload",
            "Modeled lightship fits selected displacement",
        }:
            return RepairRule(
                "system_displacement",
                "mass_displacement",
                ("Draft_frac", "Bd", "Dd", "LOA", "Fn_ship", "Voyage_Range"),
                5,
            )

        if name == (
            "Capacity-feasible cargo allocation reaches longitudinal equilibrium"
        ):
            return RepairRule(
                "cargo_equilibrium",
                "cargo_allocation",
                (
                    "Draft_frac",
                    "Lb",
                    "frac_CrashBlkhd",
                    "frac_AftPeakBlkhd",
                    "frac_lazarette_L",
                    "Num_Cargo_Holds",
                ),
                6,
            )

        if (
            result_class == "Loading Condition Trim"
            and name == "Full-cargo arrival one-percent-draft trim"
        ):
            return RepairRule(
                "none",
                "arrival_trim_superseded",
                (),
                999,
                status="not_applicable_under_approved_arrival_policy",
                note=(
                    "HDF-ASM-038 removes a separate arrival trim constraint; "
                    "arrival will use compensating ballast to reproduce the "
                    "departure displacement and trim."
                ),
            )

        if result_class == "Loading Condition Trim":
            return RepairRule(
                "trim",
                "full_load_trim",
                (
                    "Voyage_Range",
                    "Fn_ship",
                    "frac_fuel_tank_L",
                    "frac_engineRoom_L",
                    "frac_lazarette_L",
                    "Draft_frac",
                ),
                70,
            )

        if result_class == "Flooding Diagnostic":
            return RepairRule(
                "none",
                "floodable_length",
                (),
                999,
                status="diagnostic_not_counted",
                note=(
                    "CRF-005 disposition: HDF-083 remains a retained binary-"
                    "search diagnostic and does not enter feasibility."
                ),
            )

        if name == "SOLAS double-bottom aft extent":
            return RepairRule(
                "none",
                "solas_double_bottom_extent",
                (),
                999,
                status="project_basis_evidence_not_counted",
                note=(
                    "CRF-001 disposition: the invalid local/whole-ship "
                    "comparison is replaced by documented continuous-double-"
                    "bottom project-basis evidence, not parameter repair."
                ),
            )

        if name in {
            "SOLAS double-bottom forward extent",
            "SOLAS double-bottom transverse protection to bilge",
            "SOLAS double-bottom without wells or omitted regions",
        }:
            return RepairRule(
                "none",
                "solas_double_bottom_geometry",
                (),
                999,
                status="project_basis_evidence_not_counted",
                note=(
                    "CRF-001 disposition: continuous-double-bottom project "
                    "basis is retained as zero-valued N/A/evidence rather "
                    "than a local-to-whole-ship geometric constraint."
                ),
            )

        if result_class == "Powering Method Domain":
            controls = HOLTROP_REPAIR_CONTROLS.get(name)

            if controls is None:
                return RepairRule(
                    "none",
                    "holtrop_domain",
                    (),
                    999,
                    status="unreviewed_holtrop_domain_identity",
                    note=(
                        "Only the five active, equation-traced Holtrop "
                        "domain identities have approved repairs."
                    ),
                )

            return RepairRule(
                "hull_domain",
                "holtrop_domain",
                controls,
                2,
                note=(
                    "Analytical or directed bounded update from the active "
                    "Holtrop domain equation; complete reevaluation remains "
                    "authoritative."
                ),
            )

        if result_class == "Hull":
            controls = HULL_REPAIR_CONTROLS.get(name)

            if controls is None:
                return RepairRule(
                    "none",
                    "hull_geometry_domain",
                    (),
                    999,
                    status="unreviewed_hull_domain_identity",
                    note=(
                        "This hull identity is not one of the 49 equations "
                        "traced to HullParameterization.input_Constraints."
                    ),
                )

            return RepairRule(
                "hull_domain",
                "hull_geometry_domain",
                controls,
                1,
                note=(
                    "Constraint-specific controls traced to the active hull "
                    "algebra; direct projection and bounded equation search "
                    "never replace the exact integrated evaluator."
                ),
            )

        if result_class in {
            "Hydrostatic Numerical Foundation",
            "Loading Condition Numerical Quality",
            "Longitudinal Load Numerical Check",
            "Flooding Numerical Check",
        }:
            return RepairRule(
                "none",
                "numerical_quality",
                (),
                999,
                status="diagnostic_not_directly_repaired",
            )

        if result_class in {
            "Regulatory Applicability",
            "Load Line Applicability",
            "Load Line Input",
            "Structural Material",
            "Structure Rule Method",
        }:
            return RepairRule(
                "none",
                "fixed_project_basis",
                (),
                999,
                status="fixed_context_not_repair_target",
            )

        if name == "CSR uniform cargo-region bulkhead spacing":
            return RepairRule(
                "subdivision",
                "uniform_hold_spacing",
                ("Num_Cargo_Holds", "LOA"),
                4,
            )

        if name == "CSR minimum bulk-carrier double-side width":
            return RepairRule(
                "direct_minimum",
                "double_side_width",
                ("web frame h",),
                25,
                # The audited constraint is reported in metres, whereas the
                # structural web-frame depth control is stored in millimetres.
                offered_scale=0.001,
                note=(
                    "Generated double-side width is controlled by web-frame "
                    "depth; convert metres of width deficit to millimetres."
                ),
            )

        if name in {
            "MVR maximum Rule length",
            "MVR maximum breadth from length",
            "MVR maximum breadth from depth",
        }:
            return RepairRule(
                "principal_dimensions",
                "mvr_principal_dimension_domain",
                ("LOA", "Bd", "Dd"),
                3,
            )

        if name in {
            "Max Allowable Uniform Bottom Loading",
            "Max Allowable Uniform Deck Loading",
        }:
            return RepairRule(
                "uniform_bulkhead_capacity",
                "transverse_bulkhead_vertical_support",
                (
                    "Trans Bulkhead Thickness",
                    "num vertical bulkhead stiffeners",
                    "blkhd vert stiff h",
                    "blkhd vert stiff t",
                    "blkhd vert stiff w",
                ),
                45,
                note=(
                    "Both runtime rows call the same ABS 3-2-8/3.3 "
                    "transverse-bulkhead vertical-support equation.  The "
                    "repair therefore changes the bulkhead plate/stiffener "
                    "composite, not deck-beam or bottom-floor scantlings."
                ),
            )

        if result_class == "Evaluation Failure":
            return RepairRule(
                "none",
                "evaluator_failure",
                (),
                999,
                status="implementation_or_domain_failure_requires_diagnosis",
            )

        return RepairRule(
            "none",
            "unmapped",
            (),
            999,
            status="unmapped",
            note="No engineering repair relationship has been approved.",
        )

    def export_runtime_registry(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Classify every observed runtime key into a machine-readable table."""

        rows = []

        for _, source in catalog.iterrows():
            rule = self.classify(source)
            rows.append(
                {
                    **source.to_dict(),
                    "Repair_Method": rule.method,
                    "Repair_Family": rule.family,
                    "Controlling_Parameters": "; ".join(rule.parameters),
                    "Repair_Priority": rule.priority,
                    "Repair_Status": rule.status,
                    "Repair_Note": rule.note,
                }
            )

        return pd.DataFrame(rows)


__all__ = [
    "ConstraintRepairRegistry",
    "RepairRule",
]
