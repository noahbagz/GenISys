"""Atomic structural constraint evaluation for GenISys V1.1.

The original :mod:`tools.Parametric_Structure_Eval` implementation often
calculated several independent ABS requirements as ``A``, ``B``, ``C`` and
``D`` and then saved only ``max(A, B, C, D)``.  That envelope was useful for
sizing a member, but it discarded the pass/fail result for each cited rule.

This module keeps the familiar eight-column legacy CSV contract and emits one
row for every independently evaluated rule/member pair.  Dimensions, draft,
    cargo depth, member geometry, and the two full-load still-water curves come
from the active GenISys system design.  The old fixed 250 m length, assumed
draft, and 0.667*Depth tank head are deliberately not used.  Deep-tank pressure
rows are withheld until the Rule tank/vent/overflow heads are modeled; a
calculated fill surface alone is not silently treated as that design head.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.MVR_Longitudinal_Shear import (
    actual_longitudinal_bulkhead_count,
)

from tools.ConstraintIntegration import (
    AUDITED_RESULT_COLUMNS,
    build_additional_structural_results,
    build_verified_structural_results,
)


CONSTRAINT_COLUMNS = [
    'Class',
    'Name',
    'Requirement',
    'Threshold Value',
    'Constraint Value',
    'Units',
    'Constraint Eval',
    'Constraint Bit',
]


# Metadata for atomic V1.1 rows that are not already selected by
# ``build_verified_structural_results`` or produced by
# ``build_additional_structural_results``.  The explicit map prevents a
# legacy/self-imposed row from entering the status-aware audited table merely
# because it happens to have the eight common CSV columns.
ATOMIC_STRUCTURAL_AUDIT_METADATA = {
    'Wave sagging bending moment amidships': (
        'MVR Part 3, Ch 2, Sec 1/3.5.1-a',
        'calculated_rule_load',
        'ASM-002; ASM-004',
    ),
    'Wave hogging bending moment amidships': (
        'MVR Part 3, Ch 2, Sec 1/3.5.1-b',
        'calculated_rule_load',
        'ASM-002; ASM-004',
    ),
    'Positive wave shear-force envelope maximum': (
        'MVR Part 3, Ch 2, Sec 1/3.5.3-a',
        'calculated_rule_load',
        'ASM-002; ASM-004',
    ),
    'Negative wave shear-force envelope minimum': (
        'MVR Part 3, Ch 2, Sec 1/3.5.3-b',
        'calculated_rule_load',
        'ASM-002; ASM-004',
    ),
    'Hull-girder section modulus to deck': (
        'MVR Part 3, Ch 2, Sec 1/9.5.1',
        'calculated_three_hold_gross_section',
        'ASM-002; ASM-005; ASM-016; ASM-017; ASM-018',
    ),
    'Hull-girder section modulus to bottom': (
        'MVR Part 3, Ch 2, Sec 1/9.5.1',
        'calculated_three_hold_gross_section',
        'ASM-002; ASM-005; ASM-016; ASM-017; ASM-018',
    ),
    'Continuous effective longitudinal members included': (
        'MVR Part 3, Ch 2, Sec 1/9.1',
        'calculated_three_hold_gross_section',
        'ASM-002; ASM-005; ASM-016; ASM-017',
    ),
    'Structural members excluded from hull-girder effective section': (
        'MVR Part 3, Ch 2, Sec 1/9.1',
        'calculated_three_hold_gross_section',
        'ASM-016; ASM-017',
    ),
    'Principal hatch deductions in representative section': (
        'MVR Part 3, Ch 2, Sec 1/9.3; 1/9.5.1',
        'calculated_three_hold_gross_section',
        'ASM-017; ASM-018',
    ),
    'Effective hatch-coaming members credited': (
        'MVR Part 3, Ch 2, Sec 1/9.3; 1/9.5.1',
        'calculated_three_hold_gross_section',
        'ASM-018',
    ),
    'Minor-opening deductions assumed': (
        'MVR Part 3, Ch 2, Sec 1/9.3',
        'calculated_three_hold_gross_section',
        'ASM-018',
    ),
    'Hull-girder absolute minimum section modulus - deck': (
        'MVR Part 3, Ch 2, Sec 1/3.7.1(b)',
        'calculated_three_hold_gross_section',
        'ASM-002; ASM-004; ASM-005; ASM-016; ASM-017; ASM-018',
    ),
    'Hull-girder absolute minimum section modulus - bottom': (
        'MVR Part 3, Ch 2, Sec 1/3.7.1(b)',
        'calculated_three_hold_gross_section',
        'ASM-002; ASM-004; ASM-005; ASM-016; ASM-017; ASM-018',
    ),
    # MVR 3-2-1/3.5.2 is a source-defined stationwise load calculation.  The
    # two extrema are independent output metrics for the same selected audit
    # requirement, so they deliberately carry the same Audit ID.
    'Minimum stationwise still-water plus wave bending': (
        'MVR Part 3, Ch 2, Sec 1/3.5.2',
        'calculated_named_condition_envelope',
        'ASM-002; ASM-004; ASM-014',
    ),
    'Maximum stationwise still-water plus wave bending': (
        'MVR Part 3, Ch 2, Sec 1/3.5.2',
        'calculated_named_condition_envelope',
        'ASM-002; ASM-004; ASM-014',
    ),
    'Minimum stationwise still-water plus wave shear': (
        'MVR Part 3, Ch 2, Sec 1/3.5.3-b',
        'calculated_named_condition_envelope',
        'ASM-002; ASM-004; ASM-014',
    ),
    'Maximum stationwise still-water plus wave shear': (
        'MVR Part 3, Ch 2, Sec 1/3.5.3-a',
        'calculated_named_condition_envelope',
        'ASM-002; ASM-004; ASM-014',
    ),
    # USER-015 and USER-029 close the GenISys section-property basis: ordinary
    # strength gross MVR scantlings, an explicit effective-member mapping, and
    # one representative maximum-hatch section from the generated three-hold
    # model.  USER-016 limits the load envelope to the two full-load states.
    # The status therefore describes the approved project scope; it does not
    # claim a full classification loading-manual envelope.
    'Hull-girder total-load section modulus - deck': (
        'MVR Part 3, Ch 2, Sec 1/3.7.1(a)',
        'calculated_full_load_scope',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-016; ASM-017; ASM-018',
    ),
    'Hull-girder total-load section modulus - bottom': (
        'MVR Part 3, Ch 2, Sec 1/3.7.1(a)',
        'calculated_full_load_scope',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-016; ASM-017; ASM-018',
    ),
    'Hull-girder minimum moment of inertia': (
        'MVR Part 3, Ch 2, Sec 1/3.7.2',
        'calculated_full_load_scope',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-016; ASM-017; ASM-018',
    ),
    'Nominal side-shell shear stress calculation': (
        'MVR Part 3, Ch 2, Sec 1/3.9.2',
        'calculated_conditional_branch',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-016; ASM-017; ASM-018',
    ),
    'Hull-girder side-shell nominal shear-stress limit': (
        'MVR Part 3, Ch 2, Sec 1/3.9.1',
        'calculated_full_load_scope',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-019',
    ),
    'Inner bottom plating - general Rule thickness': (
        'MVR Part 3, Ch 2, Sec 4/9.1',
        'calculated_atomic_partial',
        'ASM-002; ASM-005',
    ),
    'Inner bottom plating - grab loading': (
        'MVR Part 3, Ch 2, Sec 4/9.5',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-009',
    ),
    'Lower hopper plating - grab loading': (
        'MVR Part 3, Ch 2, Sec 4/9.5',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-009',
    ),
    'Strength deck plating - grab loading': (
        'MVR Part 3, Ch 2, Sec 4/9.5 project extension',
        'calculated_project_extension',
        'ASM-002; ASM-005; ASM-009',
    ),
    'Inner-bottom longitudinal - bottom-strength minimum': (
        'MVR Part 3, Ch 2, Sec 4/11.5-a',
        'calculated_verified',
        'ASM-002; ASM-004; ASM-005',
    ),
    'Inner-bottom longitudinal - optimized cargo loading': (
        'MVR Part 3, Ch 2, Sec 4/11.5-b',
        'calculated_full_load_scope',
        'ASM-002; ASM-004; ASM-005; ASM-014; ASM-016; ASM-017; ASM-018',
    ),
    'Side stringer - ordinary side-shell loading': (
        'MVR Part 3, Ch 2, Sec 6/5.1',
        'calculated_verified',
        'ASM-002; ASM-004; ASM-005',
    ),
    'Strength-deck longitudinal section modulus': (
        'MVR Part 3, Ch 2, Sec 7/3.1',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-013; ASM-022',
    ),
    'Ordinary strength-deck transverse-beam section modulus': (
        'MVR Part 3, Ch 2, Sec 7/3.1',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-013; ASM-022',
    ),
    'Strength-deck girder section modulus clear of tanks': (
        'MVR Part 3, Ch 2, Sec 8/5.3',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-013',
    ),
    'Strength-deck transverse section modulus clear of tanks': (
        'MVR Part 3, Ch 2, Sec 8/5.5',
        'calculated_verified',
        'ASM-002; ASM-005; ASM-013',
    ),
    'Deck primary members at modeled deep-tank tops': (
        'MVR Part 3, Ch 2, Sec 8/5.13-b',
        'not_applicable_by_geometry',
        'ASM-013',
    ),
    'Hatch-end beams supported by centerline pillars': (
        'MVR Part 3, Ch 2, Sec 8/7.1.1',
        'not_applicable_by_geometry',
        'ASM-013',
    ),
    'Double-bottom girder web thickness in deep tank': (
        'MVR Part 3, Ch 2, Sec 10/3.7.2.iii',
        'calculated_verified',
        'ASM-005; ASM-006',
    ),
    'Double-bottom floor web thickness in deep tank': (
        'MVR Part 3, Ch 2, Sec 10/3.7.2.iii',
        'calculated_verified',
        'ASM-005; ASM-006',
    ),
    'Bulk-carrier cargo-hatch rule-family routing selected': (
        (
            'MVR Part 3, Ch 2, Sec 15/2 Table 1A, '
            'bulk-carrier cargo hatchways'
        ),
        'calculated_routing_only',
        'ASM-001; ASM-003',
    ),
    'Double-bottom shear-force peak reduction credited': (
        'MVR Part 3, Ch 2, Sec 1/3.9.3',
        'optional_credit_not_taken',
        'ASM-002; ASM-005',
    ),
}


# The two verified-table aliases below predate the member-specific V1.1 grab
# names.  The historical constant name is retained for import compatibility,
# but the rows are now preserved for traceability.  ``ConstraintIdentity``
# prevents their spellings from inflating unique feasibility counts.
AUDITED_STRUCTURAL_ALIASES_REPLACED_BY_V1P1 = {
    'Min Inner Bottom Grab-Service Thickness',
    'Min Lower Hopper Grab-Service Thickness',
}


def _minimum(name, requirement, required, offered, units, result_class='Structure'):
    """Return one minimum-required constraint in the legacy row format."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': float(required),
        'Constraint Value': float(offered),
        'Units': units,
        'Constraint Eval': float(required) - float(offered),
        'Constraint Bit': bool(float(offered) >= float(required)),
    }


def _maximum(name, requirement, maximum, offered, units, result_class='Structure'):
    """Return one maximum-allowed constraint in the legacy row format."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': float(maximum),
        'Constraint Value': float(offered),
        'Units': units,
        'Constraint Eval': float(offered) - float(maximum),
        'Constraint Bit': bool(float(offered) <= float(maximum)),
    }


def _metric(name, requirement, value, units,
            result_class='Structure Rule Metric'):
    """Return one rule-derived calculation that is not itself pass/fail.

    Some MVR clauses define the load that a later strength rule must consume.
    Giving those load definitions a fabricated pass Boolean would overstate
    compliance.  They remain independent rows in every V1.1 file, while the
    null comparison fields keep the dataset compiler from counting them as
    feasibility constraints.
    """

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': np.nan,
        'Constraint Value': float(value),
        'Units': units,
        'Constraint Eval': np.nan,
        'Constraint Bit': np.nan,
    }


def _not_applicable(name, requirement, result_class='Structure'):
    """Return one explicit pass for a requirement outside this design."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': 0.0,
        'Constraint Value': 0.0,
        'Units': 'N/A',
        'Constraint Eval': 0.0,
        'Constraint Bit': True,
    }


def _mark_not_applicable(rows, mask, explanation):
    """Apply the project-wide N/A encoding without deleting traceability.

    The legacy eight-column schema has no dedicated applicability column.
    GenISys therefore retains the original named row, sets both numeric sides
    and the residual to zero, and sets ``Constraint Bit`` true.  This is an
    explicit pass by non-applicability, not a calculated member property.
    """

    if not bool(mask.any()):
        return rows

    rows.loc[mask, 'Threshold Value'] = 0.0
    rows.loc[mask, 'Constraint Value'] = 0.0
    rows.loc[mask, 'Units'] = 'N/A'
    rows.loc[mask, 'Constraint Eval'] = 0.0
    rows['Constraint Bit'] = rows['Constraint Bit'].astype(object)
    rows.loc[mask, 'Constraint Bit'] = True

    return rows


class StructureConstraintsV1p1:
    """Build the expanded V1.1 structural constraint table for one system."""

    # These legacy rows are governing envelopes or older duplicates.  Their
    # independent source-rule rows are generated below or by the audited MVR
    # builders, so retaining them would count the same physical requirement
    # twice.
    _SUPERSEDED_LEGACY_NAMES = {
        'Max_LOA',
        'Max_B',
        'Max_Beam_from_Draft',
        'Min Side Shell Thickness',
        'Min Bottom Shell Thickness',
        'Min Inner Bottom as Deck Thickness',
        'Min Inner Bottom Thickness',
        'Min Deck Shell Thickness',
        'Min Double Bottom Height',
        'Max Bottom Girder Spacing',
        'Max Floor Spacing',
        'Min Bottom Stringer SM',
        'Min Inner Bottom Stringer SM',
        'Min Frame Section Modulus',
        'Min Web Frame Height',
        'Min Web Frame Thickness',
        'Min Web Frame Section Modulus',
        'Min Side Stringer Section Modulus',
        'Min Bulkhead Plating Thickness',
        'Min Vertical Bulkhead Stiffener Section Modulus',
        'Min Horizontal Bulkhead Stiffener Section Modulus',
        # These legacy rows use c=1.5 even though the modeled strength deck is
        # clear of tanks.  Correct c=1.0 member-specific rows are emitted by
        # _atomic_deck_member_rows below.
        'Min Deck Girder Section Modulus',
        'Min Deck Beam Section Modulus',
        # The legacy hatch-end beam row assumes a centerline pillar that does
        # not exist in the generated structural element table.
        'Min Hatch Support Beam Section Modulus',
        # The legacy evaluator labeled nineteen members as relative MVR
        # 3-2-7/3.1 checks.  That clause governs deck beams and deck
        # longitudinals only.  The two applicable physical comparisons are
        # recalculated with exact generated spacing by
        # ``_atomic_deck_member_rows``.  Preserve the two corrected, exact
        # legacy aliases for traceability and filter the seventeen unrelated
        # member spellings so they cannot enter V1.1 feasibility under this
        # rule.  ``ConstraintIdentity`` counts each retained alias pair once.
        'Min Relative Deck Girder Section Modulus',
        'Min Relative Shear Strake Section Modulus',
        'Min Relative Side Stringer Section Modulus Upper',
        'Min Relative Upper Hopper Stiffener Section Modulus',
        'Min Relative Deck Beam Section Modulus',
        'Min Relative Transverse Bulkhead Stiffener Section Modulus Upper',
        'Min Relative Bottom Girder Section Modulus Upper',
        'Min Relative Bottom Stringer Section Modulus',
        'Min Relative Floor Section Modulus Upper',
        'Min Relative Bottom Frame Section Modulus',
        'Min Relative Transverse Bulkhead Stiffener Section Modulus Lower',
        'Min Relative Inner Bottom Stringer Section Modulus',
        'Min Relative Bottom Girder Section Modulus Lower',
        'Min Relative Side Stringer Section Modulus Lower',
        'Min Relative Floor Section Modulus Lower',
        'Min Relative Inner Bottom Frame Section Modulus',
        'Min Relative Lower Hopper Stiffener Section Modulus',
        # These three connected legacy rows use incomplete hull-girder load or
        # section-property combinations.  Valid atomic replacements are kept
        # below; unsupported combinations are not presented as constraints.
        'Min Hull Girder Longitudinal Section Modulus for Wave Bending',
        'Min Hull Girder Moment of Inertia',
        'Max Hull Girder Shear Stress',
    }

    def __init__(
        self,
        hull_dict,
        structure_calcs,
        cargo_holds,
        named_condition_load_arrays=None,
    ):
        self.hull = hull_dict
        self.structure = structure_calcs
        self.cargo = cargo_holds
        self.rules = structure_calcs.ABS
        self.named_condition_load_arrays = (
            named_condition_load_arrays
            if named_condition_load_arrays is not None
            else {}
        )

    def _value(self, name):
        """Read one calculated offered property from ``StructureCalcs``."""

        return float(
            self.structure.Struct_Dict.loc[
                self.structure.Struct_Dict['Parameter'] == name,
                'Value',
            ].iloc[0]
        )

    @staticmethod
    def _wheel_thickness(panel_aspect_ratio):
        """Return the approved legacy 13.8 t excavator plate requirement.

        ABS MVR 3-2-3/5.17 is evaluated with the project-approved legacy
        factors: k1=8.05, k2=0.20, c1=1.5, and a strength-deck factor of 1.1.
        The panel aspect ratio is calculated from the active generated member
        spacings rather than supplied as a fixed input.
        """

        k1 = 8.05
        k2 = 0.20
        c1 = 1.5
        excavator_weight_kN = 13_800.0*9.81/1000.0
        n = min(1.0, 0.85 + 0.15*(float(panel_aspect_ratio) - 1.0))

        return 1.1*k1*k2*n*np.sqrt(c1*excavator_weight_kN)

    def _maximum_loaded_cargo_depth(self):
        """Return the greatest actual cargo depth above the inner bottom."""

        inner_bottom_z_m = self._value('Double_Bottom_Height')/1000.0
        cargo_surface_m = self.cargo.CargoHolds[
            'Cargo_Fill_Height_m'
        ].astype(float)

        return float(np.maximum(cargo_surface_m - inner_bottom_z_m, 0.0).max())

    def _maximum_cargo_equivalent_head(self):
        """Return the governing 7.01 kN/m^3-equivalent cargo head.

        ABS MVR 3-2-4/9.2 and 11.5 use the same proportional adjustment when
        the cargo load differs from the Rule reference specific weight.  The
        project cargo density and gravity are fixed by ASM-014.
        """

        cargo_depth_m = self._maximum_loaded_cargo_depth()
        cargo_specific_weight_kN_m3 = 2400.0*9.81/1000.0

        return cargo_depth_m*cargo_specific_weight_kN_m3/7.01

    def _stationwise_total_load_envelope(self):
        """Combine Rule wave loads with every named still-water curve.

        ABS MVR 3-2-1/3.5.2, 3.5.3 and 3.7 require the still-water and wave
        values to be added at the same longitudinal station.  The union of
        the hydrostatic grid and every published Rule break point is used so
        neither a curve corner nor the edge of the amidship 0.4L region can
        be missed by grid spacing.
        """

        if not self.named_condition_load_arrays:
            return None

        L = float(self.hull['L'])
        bending_breaks = np.array([0.30, 0.40, 0.65, 0.70])
        shear_breaks = np.array([
            0.0, 0.20, 0.30, 0.40, 0.60, 0.70, 0.85, 1.0,
        ])
        envelope = {
            'minimum_total_bending_kNm': np.inf,
            'maximum_total_bending_kNm': -np.inf,
            'minimum_total_shear_kN': np.inf,
            'maximum_total_shear_kN': -np.inf,
        }
        wave_sagging_kNm, wave_hogging_kNm = (
            self.rules.calc_WaveBending_Moments()
        )

        for load_case in self.named_condition_load_arrays.values():
            source_x_m = np.asarray(load_case['X_m'], dtype=float)

            # Hydrostatics publishes X relative to the forward perpendicular:
            # X=0 at FP and X increases aft to AP.  ABS Figures 2 through 4
            # use the opposite direction, x/L=0 at AP and x/L=1 at FP.  Map
            # every station before applying the asymmetric F1/F2 diagrams.
            # The numerator is LBP when that integrated hull value is present;
            # the maximum source station is the exact equivalent for isolated
            # unit use whose array ends at AP.
            aft_perpendicular_relative_m = float(
                self.hull.get(
                    'Length_Between_Perpendiculars_m',
                    np.max(source_x_m),
                )
            )
            rule_x_over_L = (
                aft_perpendicular_relative_m-source_x_m
            )/L

            # Retain only stations that lie inside the published Rule domain.
            # In particular, do not extrapolate the still-water curve to an
            # LOA-based Rule breakpoint when the hydrostatic grid ends short
            # of that location.  Add Figure breakpoints only inside the
            # actually supported coordinate interval.
            supported = (
                (rule_x_over_L >= 0.0)
                & (rule_x_over_L <= 1.0)
            )
            rule_x_over_L = rule_x_over_L[supported]
            still_bending_source_kNm = (
                np.asarray(
                    load_case['Bending_Moment_Nm'],
                    dtype=float,
                )[supported]/1000.0
            )
            still_shear_source_kN = (
                np.asarray(
                    load_case['Shear_Force_N'],
                    dtype=float,
                )[supported]/1000.0
            )

            order = np.argsort(rule_x_over_L)
            rule_x_over_L = rule_x_over_L[order]
            still_bending_kNm = (
                still_bending_source_kNm[order]
            )
            still_shear_kN = (
                still_shear_source_kN[order]
            )
            supported_min = float(rule_x_over_L[0])
            supported_max = float(rule_x_over_L[-1])

            # MVR 3-2-1/3.7.1 sizes the common section throughout 0.4L
            # amidships, from 0.30L through 0.70L in the AP-to-FP coordinate.
            bending_x_over_L = np.unique(np.concatenate([
                rule_x_over_L[
                    (rule_x_over_L >= 0.30)
                    & (rule_x_over_L <= 0.70)
                ],
                bending_breaks[
                    (bending_breaks >= supported_min)
                    & (bending_breaks <= supported_max)
                ],
            ]))
            bending_still_kNm = np.interp(
                bending_x_over_L,
                rule_x_over_L,
                still_bending_kNm,
            )
            bending_factor = (
                self.rules.calc_wave_bending_distribution_factor(
                    bending_x_over_L
                )
            )
            sagging_total_kNm = (
                bending_still_kNm + wave_sagging_kNm*bending_factor
            )
            hogging_total_kNm = (
                bending_still_kNm + wave_hogging_kNm*bending_factor
            )
            envelope['minimum_total_bending_kNm'] = min(
                envelope['minimum_total_bending_kNm'],
                float(np.min(sagging_total_kNm)),
                float(np.min(hogging_total_kNm)),
            )
            envelope['maximum_total_bending_kNm'] = max(
                envelope['maximum_total_bending_kNm'],
                float(np.max(sagging_total_kNm)),
                float(np.max(hogging_total_kNm)),
            )

            # MVR 3-2-1/3.9 checks nominal shear over the full Rule length.
            shear_x_over_L = np.unique(np.concatenate([
                rule_x_over_L,
                shear_breaks[
                    (shear_breaks >= supported_min)
                    & (shear_breaks <= supported_max)
                ],
            ]))
            shear_still_kN = np.interp(
                shear_x_over_L,
                rule_x_over_L,
                still_shear_kN,
            )
            positive_wave_kN, negative_wave_kN = (
                self.rules.calc_Wave_Shear_Force_Envelopes(
                    shear_x_over_L
                )
            )
            positive_total_kN = shear_still_kN + positive_wave_kN
            negative_total_kN = shear_still_kN + negative_wave_kN
            envelope['minimum_total_shear_kN'] = min(
                envelope['minimum_total_shear_kN'],
                float(np.min(positive_total_kN)),
                float(np.min(negative_total_kN)),
            )
            envelope['maximum_total_shear_kN'] = max(
                envelope['maximum_total_shear_kN'],
                float(np.max(positive_total_kN)),
                float(np.max(negative_total_kN)),
            )

        envelope['condition_count'] = len(
            self.named_condition_load_arrays
        )

        return envelope

    def _atomic_longitudinal_strength_rows(self):
        """Emit the independent MVR hull-girder load/property predicates.

        MVR 3-2-1/3.5.1 and 3.5.3 define loads rather than a direct pass/fail
        comparison.  Those four values are intentionally emitted as metrics.
        MVR 3-2-1/3.7.1(b), by contrast, is a true minimum section-modulus
        constraint and is checked separately at the deck and bottom fibers.
        """

        rows = []

        wave_sagging_kNm, wave_hogging_kNm = (
            self.rules.calc_WaveBending_Moments()
        )
        positive_wave_shear_kN, negative_wave_shear_kN = (
            self.rules.calc_Wave_Shear_Forces()
        )

        rows.extend(
            [
                _metric(
                    'Wave sagging bending moment amidships',
                    'ABS MVR 3-2-1/3.5.1-a',
                    wave_sagging_kNm,
                    'kN-m',
                ),
                _metric(
                    'Wave hogging bending moment amidships',
                    'ABS MVR 3-2-1/3.5.1-b',
                    wave_hogging_kNm,
                    'kN-m',
                ),
                _metric(
                    'Positive wave shear-force envelope maximum',
                    'ABS MVR 3-2-1/3.5.3-a; Figure 3',
                    positive_wave_shear_kN,
                    'kN',
                ),
                _metric(
                    'Negative wave shear-force envelope minimum',
                    'ABS MVR 3-2-1/3.5.3-b; Figure 4',
                    negative_wave_shear_kN,
                    'kN',
                ),
            ]
        )

        # MVR 3-2-1/9.5.1 requires I/y to be evaluated independently to the
        # molded deck line at side and to the baseline.  StructureCalcs now
        # retains both values instead of only the smaller governing modulus.
        deck_sm_cm3 = self._value('Hull_Girder_Deck_SM')
        bottom_sm_cm3 = self._value('Hull_Girder_Bottom_SM')
        effective_member_count = self._value(
            'Hull_Girder_Effective_Member_Count'
        )
        excluded_member_count = self._value(
            'Hull_Girder_Excluded_Member_Count'
        )
        hatch_deduction_count = self._value(
            'Hull_Girder_Hatch_Deduction_Count'
        )
        coaming_credit_count = self._value(
            'Hull_Girder_Coaming_Credit_Count'
        )
        minor_opening_count = self._value(
            'Hull_Girder_Minor_Opening_Deduction_Count'
        )

        rows.extend(
            [
                _metric(
                    'Hull-girder section modulus to deck',
                    'ABS MVR 3-2-1/9.5.1; SM_deck=I/y_deck',
                    deck_sm_cm3,
                    'cm^3',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Hull-girder section modulus to bottom',
                    'ABS MVR 3-2-1/9.5.1; SM_bottom=I/y_bottom',
                    bottom_sm_cm3,
                    'cm^3',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Continuous effective longitudinal members included',
                    'ABS MVR 3-2-1/9.1',
                    effective_member_count,
                    'members',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Structural members excluded from hull-girder effective section',
                    'ABS MVR 3-2-1/9.1; ASM-017',
                    excluded_member_count,
                    'members',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Principal hatch deductions in representative section',
                    'ABS MVR 3-2-1/9.3; 9.5.1; ASM-018',
                    hatch_deduction_count,
                    'openings',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Effective hatch-coaming members credited',
                    'ABS MVR 3-2-1/9.3; 9.5.1; ASM-018',
                    coaming_credit_count,
                    'members',
                    result_class='Structure Property Metric',
                ),
                _metric(
                    'Minor-opening deductions assumed',
                    'ABS MVR 3-2-1/9.3; ASM-018',
                    minor_opening_count,
                    'openings',
                    result_class='Structure Property Metric',
                ),
            ]
        )

        absolute_minimum_cm2_m = (
            self.rules.calc_Absolute_Minimum_Hull_Girder_SM()
        )
        absolute_minimum_cm3 = 100.0*absolute_minimum_cm2_m
        rows.extend(
            [
                _minimum(
                    'Hull-girder absolute minimum section modulus - deck',
                    'ABS MVR 3-2-1/3.7.1(b)',
                    absolute_minimum_cm3,
                    deck_sm_cm3,
                    'cm^3',
                ),
                _minimum(
                    'Hull-girder absolute minimum section modulus - bottom',
                    'ABS MVR 3-2-1/3.7.1(b)',
                    absolute_minimum_cm3,
                    bottom_sm_cm3,
                    'cm^3',
                ),
            ]
        )

        # Preserve a mathematically defined local-member coefficient when the
        # evaluator is used without integrated loading-condition arrays.  The
        # normal GenISys path supplies those arrays and replaces this fallback
        # with the governing total-load requirement below.
        self.rules.min_Hull_Girder_SM = absolute_minimum_cm2_m
        total_loads = self._stationwise_total_load_envelope()

        if total_loads is None:
            return rows

        rows.extend(
            [
                _metric(
                    'Minimum stationwise still-water plus wave bending',
                    'ABS MVR 3-2-1/3.5.2; 3.7.1(a)',
                    total_loads['minimum_total_bending_kNm'],
                    'kN-m',
                ),
                _metric(
                    'Maximum stationwise still-water plus wave bending',
                    'ABS MVR 3-2-1/3.5.2; 3.7.1(a)',
                    total_loads['maximum_total_bending_kNm'],
                    'kN-m',
                ),
                _metric(
                    'Minimum stationwise still-water plus wave shear',
                    'ABS MVR 3-2-1/3.5.3; 3.9',
                    total_loads['minimum_total_shear_kN'],
                    'kN',
                ),
                _metric(
                    'Maximum stationwise still-water plus wave shear',
                    'ABS MVR 3-2-1/3.5.3; 3.9',
                    total_loads['maximum_total_shear_kN'],
                    'kN',
                ),
            ]
        )

        total_moment_kNm = max(
            abs(total_loads['minimum_total_bending_kNm']),
            abs(total_loads['maximum_total_bending_kNm']),
        )
        total_moment_required_cm2_m = total_moment_kNm/17.5
        governing_required_cm2_m = max(
            total_moment_required_cm2_m,
            absolute_minimum_cm2_m,
        )
        self.rules.min_Hull_Girder_SM = governing_required_cm2_m

        rows.extend(
            [
                _minimum(
                    'Hull-girder total-load section modulus - deck',
                    'ABS MVR 3-2-1/3.7.1(a)',
                    100.0*total_moment_required_cm2_m,
                    deck_sm_cm3,
                    'cm^3',
                ),
                _minimum(
                    'Hull-girder total-load section modulus - bottom',
                    'ABS MVR 3-2-1/3.7.1(a)',
                    100.0*total_moment_required_cm2_m,
                    bottom_sm_cm3,
                    'cm^3',
                ),
                _minimum(
                    'Hull-girder minimum moment of inertia',
                    'ABS MVR 3-2-1/3.7.2',
                    (
                        float(self.hull['L'])
                        * governing_required_cm2_m/33.3
                    ),
                    self._value('Hull_Girder_I')*10000.0,
                    'cm^2-m^2',
                ),
            ]
        )

        structural_elements = getattr(
            self.structure,
            'Structural_Elements',
            pd.DataFrame(),
        )
        longitudinal_bulkhead_count = actual_longitudinal_bulkhead_count(
            structural_elements
        )
        total_shear_kN = max(
            abs(total_loads['minimum_total_shear_kN']),
            abs(total_loads['maximum_total_shear_kN']),
        )

        # The current GenISys project fixes the actual longitudinal-bulkhead
        # block off.  Inner_Side_Shell remains a separate inner-hull boundary
        # and does not count as a longitudinal bulkhead for MVR 3.9 routing.
        # Reject an externally supplied out-of-scope geometry instead of
        # silently evaluating it under an unapproved branch.
        if longitudinal_bulkhead_count != 0:
            raise ValueError(
                'Actual longitudinal bulkheads are outside the current '
                'GenISys project scope; the active MVR shear route requires '
                'zero Longitudinal_Bulkhead elements.'
            )

        # ABS MVR 3-2-1/3.9.2: nominal side-shell shear stress for a vessel
        # without an effective longitudinal bulkhead.  This is the only active
        # GenISys shear-distribution branch under ASM-019.
        first_area_moment_cm3 = self._value(
            'Hull_Girder_First_Area_Moment'
        )
        side_shell_thickness_cm = (
            self._value('Side_Shell_Thickness')/10.0
        )
        hull_girder_i_cm4 = self._value('Hull_Girder_I')*100.0**4.0
        nominal_shear_kN_cm2 = (
            total_shear_kN*first_area_moment_cm3
            / (2.0*side_shell_thickness_cm*hull_girder_i_cm4)
        )
        rows.extend(
            [
                _metric(
                    'Nominal side-shell shear stress calculation',
                    'ABS MVR 3-2-1/3.9.2',
                    nominal_shear_kN_cm2,
                    'kN/cm^2',
                ),
                _maximum(
                    'Hull-girder side-shell nominal shear-stress limit',
                    'ABS MVR 3-2-1/3.9.1',
                    11.0,
                    nominal_shear_kN_cm2,
                    'kN/cm^2',
                ),
            ]
        )

        return rows

    def _atomic_plate_rows(self):
        """Evaluate the formerly combined inner-bottom/deck plate rules."""

        rows = []

        L = float(self.hull['L'])
        frame_spacing_mm = 1000.0*self._value('Frame_Spacing')
        bottom_spacing_mm = max(
            frame_spacing_mm,
            1000.0*self._value('Bottom_Stringer_Spacing'),
        )
        deck_spacing_mm = max(
            frame_spacing_mm,
            1000.0*self._value('Deck_Stringer_Spacing'),
        )

        inner_bottom_mm = self._value('Inner_Bottom_Thickness')
        lower_hopper_mm = self._value('Lower_Hopper_Plate_Thickness')
        strength_deck_mm = self._value('Top_Deck_Thickness')

        # ABS MVR 3-2-4/9.1: GenISys uses longitudinal framing and all
        # generated vessels in the active bulk-carrier space are L >= 90 m,
        # so the source coefficient is c2 = 1.5 mm.  Deep-tank pressure is
        # emitted below as a separate rule row; the separate buckling branch
        # remains outside the present gross-scantling model.
        inner_bottom_base_mm = self.rules.round_rule_thickness(
            0.037*L + 0.009*bottom_spacing_mm - 1.5
        )
        rows.append(
            _minimum(
                'Inner bottom plating - general Rule thickness',
                'ABS MVR 3-2-4/9.1 base equation; longitudinal framing',
                inner_bottom_base_mm,
                inner_bottom_mm,
                'mm',
            )
        )

        # ABS MVR 3-2-4/9.2: use the greatest optimized cargo surface from
        # the current operating/loading condition.
        cargo_depth_m = self._maximum_loaded_cargo_depth()
        cargo_required_mm = self.rules.calc_inner_bottom_cargo_pressure_thickness(
            bottom_spacing_mm,
            cargo_depth_m,
        )
        rows.append(
            _minimum(
                'Inner bottom plating - cargo pressure',
                'ABS MVR 3-2-4/9.2',
                cargo_required_mm,
                inner_bottom_mm,
                'mm',
            )
        )

        # ABS MVR 3-2-4/9.5: the user selected the grab-service equation for
        # the inner bottom, lower hopper, and strength deck.  Each physical
        # plate is an independent constraint with its own actual spacing.
        grab_members = [
            (
                'Inner bottom plating - grab loading',
                'ABS MVR 3-2-4/9.5',
                bottom_spacing_mm,
                inner_bottom_mm,
            ),
            (
                'Lower hopper plating - grab loading',
                'ABS MVR 3-2-4/9.5',
                self._value('Lower_Hopper_Stiffener_Spacing'),
                lower_hopper_mm,
            ),
            (
                'Strength deck plating - grab loading',
                # ASM-009 deliberately extends the 9.5 impact threshold to
                # the strength deck.  The source clause directly governs the
                # inner bottom/lower hopper, so identify this as a project
                # extension rather than overstating ABS applicability.
                'Project extension of ABS MVR 3-2-4/9.5 under ASM-009',
                deck_spacing_mm,
                strength_deck_mm,
            ),
        ]

        for name, requirement, spacing_mm, offered_mm in grab_members:
            required_mm = self.rules.round_rule_thickness(
                0.037*L + 0.009*spacing_mm + 5.5
            )
            rows.append(
                _minimum(
                    name,
                    requirement,
                    required_mm,
                    offered_mm,
                    'mm',
                )
            )

        # Do not turn the current liquid fill surface into an MVR 3-2-10/3.1
        # design head.  That Rule also requires the tank-top-to-overflow or
        # vent height, its Table-2(e) deck-height branch, load-line height,
        # panel aspect factor, and two independent plate minima.  Those tank
        # boundary/vent inputs are not yet modeled, so the former preliminary
        # deep-tank rows are deliberately omitted rather than counted as
        # classification constraints.

        # The excavator wheel-load calculation is applied independently to
        # the inner bottom and strength deck per the user's scope decision.
        inner_panel_ratio = max(
            self._value('WebFrame_Spacing'),
            self._value('Bottom_Girder_Spacing'),
        )/min(
            self._value('WebFrame_Spacing'),
            self._value('Bottom_Girder_Spacing'),
        )
        deck_panel_ratio = max(
            self._value('WebFrame_Spacing'),
            self._value('Deck_Girder_Spacing'),
        )/min(
            self._value('WebFrame_Spacing'),
            self._value('Deck_Girder_Spacing'),
        )

        rows.append(
            _minimum(
                'Inner bottom plating - excavator wheel loading',
                'ABS MVR 3-2-4/9.7; 3-2-3/5.17',
                self.rules.round_rule_thickness(
                    self._wheel_thickness(inner_panel_ratio)
                ),
                inner_bottom_mm,
                'mm',
            )
        )
        rows.append(
            _minimum(
                'Strength deck plating - excavator wheel loading',
                'ABS MVR 3-2-3/5.17',
                self.rules.round_rule_thickness(
                    self._wheel_thickness(deck_panel_ratio)
                ),
                strength_deck_mm,
                'mm',
            )
        )

        # ABS MVR 3-2-3/5.1.  The pressure equation and absolute minimum are
        # deliberately separate instead of one max(A, 5 mm) envelope.
        unloaded_deck_required_mm = self.rules.round_rule_thickness(
            deck_spacing_mm*np.sqrt(3.66)/254.0 + 1.5
        )
        rows.append(
            _minimum(
                'Strength deck plating - unloaded exposed-deck pressure',
                'ABS MVR 3-2-3/5.1',
                unloaded_deck_required_mm,
                strength_deck_mm,
                'mm',
            )
        )
        rows.append(
            _minimum(
                'Strength deck plating - absolute minimum',
                'ABS MVR 3-2-3/5.1',
                5.0,
                strength_deck_mm,
                'mm',
            )
        )

        return rows

    def _atomic_inner_bottom_longitudinal_rows(self):
        """Evaluate both independent MVR 3-2-4/11.5 alternatives."""

        rows = []

        D = float(self.hull['D'])
        draft = float(self.hull['d'])
        spacing_m = self._value('Bottom_Stringer_Spacing')
        # MVR 3-2-4/11.3 applies a 1.83 m lower bound to the no-strut bottom
        # longitudinal span.  The separate 11.5 cargo equation defines l as
        # the actual distance between supports and does not repeat that floor.
        bottom_span_m = max(self._value('WebFrame_Spacing'), 1.83)
        cargo_span_m = self._value('WebFrame_Spacing')
        offered_cm3 = self._value('Inner_Bottom_Stringer_SM')

        # ABS MVR 3-2-4/11.5 first paragraph: the inner-bottom member must
        # retain at least 85% of the *required* MVR 11.3 bottom-longitudinal
        # modulus.  Do not compare against 85% of the offered bottom member.
        bottom_required_cm3 = (
            7.8*1.3*max(draft, 0.67*D)*spacing_m*bottom_span_m**2.0
        )
        rows.append(
            _minimum(
                'Inner-bottom longitudinal - bottom-strength minimum',
                'ABS MVR 3-2-4/11.5-a; 0.85 of 3-2-4/11.3',
                0.85*bottom_required_cm3,
                offered_cm3,
                'cm^3',
            )
        )

        # ABS MVR 3-2-4/11.5 cargo branch.  The hull-girder factor k uses the
        # required amidship modulus in cm^2-m, the positive neutral-axis
        # distance to the inner bottom in metres, and I_A in cm^2-m^2.
        hull_girder_I_cm2_m2 = self._value('Hull_Girder_I')*100.0**2.0
        neutral_axis_m = self._value('Hull_Girder_Z_Centroid')
        inner_bottom_m = self._value('Double_Bottom_Height')/1000.0
        y_m = abs(inner_bottom_m-neutral_axis_m)

        # ``min_Hull_Girder_SM`` is set by the active longitudinal-strength
        # routine before the V1.1 table is built.  The absolute Rule minimum
        # is retained as a safe mathematical fallback for isolated unit use.
        required_hull_sm_cm2_m = float(getattr(
            self.rules,
            'min_Hull_Girder_SM',
            self.rules.calc_Absolute_Minimum_Hull_Girder_SM(),
        ))
        k = required_hull_sm_cm2_m*y_m/hull_girder_I_cm2_m2
        c = 1.0/(1.709 - 0.651*k)

        cargo_required_cm3 = (
            7.8*c*self._maximum_cargo_equivalent_head()
            * spacing_m*cargo_span_m**2.0
        )
        rows.append(
            _minimum(
                'Inner-bottom longitudinal - optimized cargo loading',
                'ABS MVR 3-2-4/11.5-b',
                cargo_required_cm3,
                offered_cm3,
                'cm^3',
            )
        )

        return rows

    def _atomic_deck_member_rows(self):
        """Evaluate each modeled clear-of-tank deck member independently."""

        rows = []
        L = float(self.hull['L'])
        D = float(self.hull['D'])
        deck_head_m = self.rules.calc_freeboard_deck_head()
        deck_distance_from_neutral_axis_m = abs(
            D-self._value('Hull_Girder_Z_Centroid')
        )
        hull_girder_i_cm2_m2 = self._value('Hull_Girder_I')*10000.0
        required_hull_sm_cm2_m = float(
            self.rules.min_Hull_Girder_SM
        )
        hull_girder_factor = (
            required_hull_sm_cm2_m
            * deck_distance_from_neutral_axis_m
            / hull_girder_i_cm2_m2
        )
        longitudinal_coefficient = (
            1.0/(1.709-0.651*hull_girder_factor)
        )
        transverse_coefficient = 0.585 if L >= 90.0 else 0.60

        # CRF-009 / ABS MVR 3-2-7/3.1.  The Deck_Stringer property is the ordinary
        # longitudinal at the freeboard/strength deck; its support spacing is
        # the web-frame spacing.  The Intermediate_Deck_Beam property is the
        # ordinary transverse beam at each intermediate frame and spans
        # between adjacent deck girders.
        deck_longitudinal_required_cm3 = (
            7.8*longitudinal_coefficient*deck_head_m
            * self._value('Deck_Stringer_Spacing')
            * self._value('WebFrame_Spacing')**2.0
        )
        ordinary_deck_beam_required_cm3 = (
            7.8*transverse_coefficient*deck_head_m
            * self._value('Frame_Spacing')
            * self._value('Deck_Girder_Spacing')**2.0
        )
        rows.extend(
            [
                _minimum(
                    'Strength-deck longitudinal section modulus',
                    'ABS MVR 3-2-7/3.1; longitudinal strength deck',
                    deck_longitudinal_required_cm3,
                    self._value('Deck_Stringer_SM'),
                    'cm^3',
                ),
                _minimum(
                    'Ordinary strength-deck transverse-beam section modulus',
                    'ABS MVR 3-2-7/3.1; beam between deck girders',
                    ordinary_deck_beam_required_cm3,
                    self._value('Intermediate_Deck_Beam_SM'),
                    'cm^3',
                ),
            ]
        )

        # ABS MVR 3-2-8/5.3 and /5.5.  The top deck is clear of tanks, so c
        # is 1.0.  The primary longitudinal girder spans between transverse
        # bulkheads; the primary deck transverse spans between deck girders.
        deck_girder_required_cm3 = (
            4.74*self._value('Deck_Girder_Spacing')*deck_head_m
            * self._value('Bulkhead_Spacing')**2.0
        )
        deck_transverse_required_cm3 = (
            4.74*self._value('WebFrame_Spacing')*deck_head_m
            * self._value('Deck_Girder_Spacing')**2.0
        )
        rows.extend(
            [
                _minimum(
                    'Strength-deck girder section modulus clear of tanks',
                    'ABS MVR 3-2-8/5.3',
                    deck_girder_required_cm3,
                    self._value('Deck_Girder_SM'),
                    'cm^3',
                ),
                _minimum(
                    'Strength-deck transverse section modulus clear of tanks',
                    'ABS MVR 3-2-8/5.5',
                    deck_transverse_required_cm3,
                    self._value('Deck_Beam_SM'),
                    'cm^3',
                ),
            ]
        )

        # The modeled freeboard deck is explicitly clear of tanks.  Keep the
        # selected tank-top branch visible as an explicit pass by
        # non-applicability instead of misapplying its 0.0833*l requirement.
        rows.append(
            _not_applicable(
                'Deck primary members at modeled deep-tank tops',
                'ABS MVR 3-2-8/5.13-b applicability',
                result_class='Structure Applicability Metric',
            )
        )

        # MVR 3-2-8/7.1.1 is a special hatch-end-beam branch for a centerline
        # pillar arrangement.  GenISys has no pillars; ordinary hatch-end
        # beams remain governed by the clear-of-tank transverse row above.
        structural_elements = getattr(
            self.structure,
            'Structural_Elements',
            pd.DataFrame(),
        )
        pillar_count = (
            int(
                structural_elements['Class'].astype(str).str.contains(
                    'Pillar',
                    case=False,
                    regex=False,
                ).sum()
            )
            if not structural_elements.empty
            else 0
        )
        rows.append(
            _not_applicable(
                'Hatch-end beams supported by centerline pillars',
                'ABS MVR 3-2-8/7.1.1 applicability',
                result_class='Structure Applicability Metric',
            )
        )

        return rows

    def _atomic_deep_tank_web_rows(self):
        """Check webs forming the continuous modeled double-bottom tanks."""

        depth_mm = self._value('Double_Bottom_Height')
        required_web_mm = self.rules.round_rule_thickness(
            min(0.01*depth_mm+3.0, 11.5)
        )

        # MVR 3-2-10/3.7.2(iii) applies independently to the longitudinal
        # girders and transverse floors inside the continuous ballastable
        # double bottom.  No unmodeled tank-head value enters this geometric
        # web-proportion rule.
        return [
            _minimum(
                'Double-bottom girder web thickness in deep tank',
                'ABS MVR 3-2-10/3.7.2(iii)',
                required_web_mm,
                self._value('Bottom_Girder_Plate_Thickness'),
                'mm',
            ),
            _minimum(
                'Double-bottom floor web thickness in deep tank',
                'ABS MVR 3-2-10/3.7.2(iii)',
                required_web_mm,
                self._value('Bottom_Floor_Plate_Thickness'),
                'mm',
            ),
        ]

    @staticmethod
    def _rule_routing_rows():
        """Expose non-numerical MVR bulk-carrier routing decisions."""

        return [
            _metric(
                'Bulk-carrier cargo-hatch rule-family routing selected',
                'ABS MVR 3-2-15/2 Table 1A',
                1.0,
                'Boolean branch selected',
                result_class='Structure Applicability Metric',
            ),
            # MVR 3-2-1/3.9.3 permits, but does not require, a local reduction
            # of the double-bottom shear peak.  GenISys conservatively takes
            # no optional credit, so no adjacent-hold load transfer needs to
            # be subtracted from the stationwise total shear envelope.
            _metric(
                'Double-bottom shear-force peak reduction credited',
                'ABS MVR 3-2-1/3.9.3; optional reduction not taken',
                0.0,
                'kN',
                result_class='Structure Rule Method',
            ),
        ]

    def _atomic_double_bottom_rows(self):
        """Split the three legacy double-bottom height alternatives."""

        B = float(self.hull['B'])
        draft = float(self.hull['d'])
        offered_mm = self._value('Double_Bottom_Height')

        return [
            _minimum(
                'Double-bottom height - draft and breadth equation',
                'ABS MVR 3-2-4/3.1.1(c)',
                32.0*B + 190.0*np.sqrt(draft),
                offered_mm,
                'mm',
            ),
            _minimum(
                'Double-bottom height - breadth twentieth',
                'ABS MVR 3-2-4/1.2.ii',
                min(1000.0*B/20.0, 2000.0),
                offered_mm,
                'mm',
            ),
            _minimum(
                'Double-bottom height - absolute minimum',
                'ABS MVR 3-2-4/1.2.ii',
                760.0,
                offered_mm,
                'mm',
            ),
        ]

    def _atomic_shell_rows(self):
        """Add source-rule shell branches hidden by legacy max envelopes."""

        rows = []
        L = float(self.hull['L'])
        frame_spacing_mm = 1000.0*self._value('Frame_Spacing')
        side_spacing_mm = max(
            frame_spacing_mm,
            1000.0*self._value('Side_Stringer_Spacing'),
        )
        side_offered_mm = self._value('Side_Shell_Thickness')

        if L <= 427.0:
            rows.append(
                _minimum(
                    'Side-shell end thickness at actual spacing',
                    'ABS MVR 3-2-2/5.1',
                    self.rules.calc_shell_end_thickness(side_spacing_mm),
                    side_offered_mm,
                    'mm',
                )
            )
        else:
            # ``build_additional_structural_results`` owns the one
            # audit-annotated method-domain failure above 427 m.  Do not emit
            # a second spelling/class of the same physical rejection here.
            # Keeping the domain row in that shared builder also guarantees
            # it reaches both StructureConstraintsV1p1 and AuditedConstraints.
            pass
        rows.append(
            _minimum(
                'Bottom-shell absolute minimum thickness',
                'ABS MVR 3-2-2/3.15',
                8.5,
                self._value('Bottom_Shell_Thickness'),
                'mm',
            )
        )

        return rows

    def _atomic_side_and_bulkhead_rows(self):
        """Split the supported ordinary side/bulkhead rule envelopes.

        The reference evaluator stored ``max(A, B)`` for the side stringer,
        bulkhead plate, and bulkhead stiffeners.  These rows restore only the
        ordinary MVR requirements supported by current inputs.  Deep-tank
        pressure branches remain absent until their complete Rule heads and
        tank-boundary mappings exist.  The calculation retains the approved
        unsupported-end coefficient ``c = 0.60`` for bulkhead stiffeners.
        """

        rows = []
        D = float(self.hull['D'])
        draft = float(self.hull['d'])
        web_spacing_m = self._value('WebFrame_Spacing')
        side_spacing_m = self._value('Side_Stringer_Spacing')
        structural_elements = getattr(
            self.structure,
            'Structural_Elements',
            pd.DataFrame(),
        )
        side_stringers = structural_elements.loc[
            structural_elements['Class'].astype(str).eq(
                'Side_Shell_Longitudinal_Stiffener'
            )
        ] if not structural_elements.empty else pd.DataFrame()

        # ABS MVR 3-2-6/5.1 uses h at each actual stringer elevation.  The
        # lowest generated stringer produces the greatest draft-line branch;
        # the independent 2D/3 and 1.8 m floors are then applied exactly.
        lowest_side_stringer_z_m = (
            float(side_stringers['z_loc'].astype(float).min())
            if not side_stringers.empty
            else side_spacing_m
        )
        side_stringer_head_m = max(
            draft-lowest_side_stringer_z_m,
            2.0*D/3.0,
            1.8,
        )
        ordinary_side_stringer_sm = (
            4.74*1.50*side_stringer_head_m
            * side_spacing_m*web_spacing_m**2.0
        )
        rows.append(
            _minimum(
                'Side stringer - ordinary side-shell loading',
                'ABS MVR 3-2-6/5.1',
                ordinary_side_stringer_sm,
                self._value('Side_Stringer_SM'),
                'cm^3',
            )
        )

        vertical_spacing_mm = (
            1000.0*self._value('VertBulkhead_Stringer_Spacing')
        )
        horizontal_spacing_mm = (
            1000.0*self._value('HorizBulkhead_Stringer_Spacing')
        )
        governing_spacing_mm = max(
            vertical_spacing_mm,
            horizontal_spacing_mm,
        )
        bulkhead_offered_mm = self._value('Bulkhead_Thickness')

        # ABS MVR 3-2-9/5.1 contains three simultaneous plate minima.  These
        # generated cargo-region bulkheads are ordinary watertight bulkheads,
        # not the collision bulkhead, so the equation divisor is c=290 and
        # the stiffener factor below is k=1.00.  The two orthogonal stiffener
        # spacings provide the actual rectangular plate-panel aspect ratio.
        panel_aspect_ratio = (
            max(vertical_spacing_mm, horizontal_spacing_mm)
            / min(vertical_spacing_mm, horizontal_spacing_mm)
        )
        panel_factor = (
            (3.075*np.sqrt(panel_aspect_ratio)-2.077)
            / (panel_aspect_ratio+0.272)
            if panel_aspect_ratio <= 2.0
            else 1.0
        )
        bulkhead_pressure_mm = self.rules.round_rule_thickness(
            governing_spacing_mm*panel_factor*np.sqrt(D)/290.0 + 1.5
        )
        bulkhead_spacing_minimum_mm = self.rules.round_rule_thickness(
            governing_spacing_mm/200.0 + 2.5
        )
        rows.extend(
            [
                _minimum(
                    'Watertight bulkhead plating - pressure equation',
                    'ABS MVR 3-2-9/5.1 pressure equation',
                    bulkhead_pressure_mm,
                    bulkhead_offered_mm,
                    'mm',
                ),
                _minimum(
                    'Watertight bulkhead plating - absolute minimum',
                    'ABS MVR 3-2-9/5.1 absolute minimum',
                    6.0,
                    bulkhead_offered_mm,
                    'mm',
                ),
                _minimum(
                    'Watertight bulkhead plating - spacing minimum',
                    'ABS MVR 3-2-9/5.1 spacing minimum',
                    bulkhead_spacing_minimum_mm,
                    bulkhead_offered_mm,
                    'mm',
                ),
            ]
        )

        # MVR 3-2-10/3.1 is not repeated here.  A calculated liquid surface
        # is not the complete Rule design head; vent/overflow elevations and
        # an explicit tank-boundary mapping are required before that separate
        # deep-tank load case can be represented as a compliance row.

        # MVR 3-2-9/5.3: evaluate each physical stiffener orientation with its
        # own spacing, unsupported span, and center-of-span head.  The prior
        # legacy envelope incorrectly used the horizontal stiffener *spacing*
        # as the span of both orientations and the collision-bulkhead factor.
        k_watertight = 1.00
        c_unsupported = 0.60
        vertical_span_m = D
        vertical_head_m = 0.5*D

        if vertical_head_m < 6.1:
            vertical_head_m = 0.8*vertical_head_m + 1.22

        horizontal_span_m = float(self.hull['B'])
        horizontal_stiffeners = structural_elements.loc[
            structural_elements['Class'].astype(str).eq(
                'Transverse_Bulkhead_Transverse_Stiffener'
            )
        ] if not structural_elements.empty else pd.DataFrame()
        lowest_horizontal_z_m = (
            float(horizontal_stiffeners['z_loc'].astype(float).min())
            if not horizontal_stiffeners.empty
            else horizontal_spacing_mm/1000.0
        )
        horizontal_head_m = D-lowest_horizontal_z_m

        if horizontal_head_m < 6.1:
            horizontal_head_m = 0.8*horizontal_head_m + 1.22

        vertical_bulkhead_sm = (
            7.8*k_watertight*c_unsupported*vertical_head_m
            * (vertical_spacing_mm/1000.0)*vertical_span_m**2.0
        )
        horizontal_bulkhead_sm = (
            7.8*k_watertight*c_unsupported*horizontal_head_m
            * (horizontal_spacing_mm/1000.0)*horizontal_span_m**2.0
        )
        rows.append(
            _minimum(
                'Vertical bulkhead stiffener - watertight loading',
                'ABS MVR 3-2-9/5.3; unsupported ends c=0.60',
                vertical_bulkhead_sm,
                self._value('Vertical_Bulkhead_Stringer_SM'),
                'cm^3',
            )
        )
        rows.append(
            _minimum(
                'Horizontal bulkhead stiffener - watertight loading',
                'ABS MVR 3-2-9/5.3; unsupported ends c=0.60',
                horizontal_bulkhead_sm,
                self._value('Horizontal_Bulkhead_Stringer_SM'),
                'cm^3',
            )
        )

        return rows

    def evaluate(self):
        """Return the complete expanded eight-column structure table."""

        legacy = self.structure.StructureConstraints.loc[
            ~self.structure.StructureConstraints['Name'].isin(
                self._SUPERSEDED_LEGACY_NAMES
            ),
            CONSTRAINT_COLUMNS,
        ].copy()

        # ABS MVR 3-1-2/13.5.2 aspect-ratio checks are excluded from the
        # GenISys project by user decision.  Remove any such rows supplied by
        # historical or external StructureCalcs objects as well as relying on
        # the current StructureCalcs source not to emit them.
        legacy = legacy.loc[
            ~legacy['Name'].astype(str).str.endswith('_Aspect_Ratio')
        ].copy()

        # Current V1.1 designs require this member. Keep generated geometry as
        # the applicability authority so historical or externally supplied
        # zero-sentinel vectors are still reported honestly as N/A.
        structural_elements = getattr(
            self.structure,
            'Structural_Elements',
            pd.DataFrame(),
        )
        has_inner_side_shell = (
            isinstance(structural_elements, pd.DataFrame)
            and not structural_elements.empty
            and 'Class' in structural_elements.columns
            and structural_elements['Class'].astype(str).eq(
                'Inner_Side_Shell'
            ).any()
        )
        legacy = _mark_not_applicable(
            legacy,
            (
                legacy['Name'].astype(str).eq(
                    'Min Inner Side Shell Thickness'
                )
                & (not has_inner_side_shell)
            ),
            'no generated Inner_Side_Shell member',
        )

        # Reuse the already validated active MVR equations and geometry
        # mappings, but drop their four audit-only metadata columns so every
        # per-design file retains the requested legacy CSV schema.
        verified = build_verified_structural_results(
            self.structure.StructureConstraints
        )[CONSTRAINT_COLUMNS]
        additional = build_additional_structural_results(
            self.hull,
            self.structure.Struct_Dict,
            self.structure.Structural_Elements,
            self.rules,
            self.cargo.CargoHolds,
        )

        # Retain geometry-inapplicable audit rows in the eight-column output.
        # Their status-aware builder supplies the approved zero/blank N/A
        # encoding, so removing the metadata columns cannot turn them into
        # apparent passes or failures.

        # Preserve compatibility aliases in the long-form output.  Their
        # different labels remain useful provenance; ``ConstraintIdentity``
        # supplies a separate canonical identity so aggregate pass/fail totals
        # count each physical equation/member pair once.

        # Replace the combined B/20-with-760-mm envelope by its two atomic
        # source checks below.  The 3.1.1(c) row is similarly supplied by the
        # explicit three-row double-bottom group.
        verified = verified.loc[
            verified['Name'] != 'Min Double Bottom Height'
        ]
        additional = additional.loc[
            additional['Name'] != 'MVR breadth-based double-bottom height',
            CONSTRAINT_COLUMNS,
        ]

        atomic_rows = (
            self._atomic_longitudinal_strength_rows()
            + self._atomic_double_bottom_rows()
            + self._atomic_plate_rows()
            + self._atomic_inner_bottom_longitudinal_rows()
            + self._atomic_shell_rows()
            + self._atomic_deck_member_rows()
            + self._atomic_deep_tank_web_rows()
            + self._atomic_side_and_bulkhead_rows()
            + self._rule_routing_rows()
        )
        atomic = pd.DataFrame(atomic_rows, columns=CONSTRAINT_COLUMNS)

        result_frames = [
            frame
            for frame in (legacy, verified, additional, atomic)
            if not frame.empty
        ]
        result = pd.concat(
            result_frames,
            ignore_index=True,
        )

        # Collapse only literal duplicate copies of one row.  Differently
        # named compatibility aliases are intentionally retained and are
        # handled at the count/view boundary by ``ConstraintIdentity``.
        result = result.drop_duplicates(
            subset=CONSTRAINT_COLUMNS,
            keep='first',
        )

        return result[CONSTRAINT_COLUMNS].reset_index(drop=True)


def evaluate_structure_constraints_v1p1(
    hull_dict,
    structure_calcs,
    cargo_holds,
    named_condition_load_arrays=None,
):
    """Functional entry point used by the integrated system evaluator."""

    return StructureConstraintsV1p1(
        hull_dict,
        structure_calcs,
        cargo_holds,
        named_condition_load_arrays=named_condition_load_arrays,
    ).evaluate()


def build_atomic_structural_audit_results(structure_constraints_v1p1):
    """Annotate the newly completed atomic rows for ``AuditedConstraints``.

    ``build_verified_structural_results`` and
    ``build_additional_structural_results`` continue to own their existing
    rows.  This adapter returns only the additional V1.1 atomic rules, so the
    caller can append it without duplicating a lower-level legacy envelope.
    """

    selected = structure_constraints_v1p1.loc[
        structure_constraints_v1p1['Name'].isin(
            ATOMIC_STRUCTURAL_AUDIT_METADATA
        ),
        CONSTRAINT_COLUMNS,
    ].copy()

    selected['Audit ID'] = selected['Name'].map(
        lambda name: ATOMIC_STRUCTURAL_AUDIT_METADATA[name][0]
    )
    selected['Implementation Status'] = selected['Name'].map(
        lambda name: ATOMIC_STRUCTURAL_AUDIT_METADATA[name][1]
    )
    selected['Assumption IDs'] = selected['Name'].map(
        lambda name: ATOMIC_STRUCTURAL_AUDIT_METADATA[name][2]
    )
    selected['Result Basis'] = selected.apply(
        lambda row: (
            'Atomic V1.1 structural result calculated from generated member '
            'geometry and active design/load data. Status '
            f"'{row['Implementation Status']}' defines whether the row is a "
            'complete comparison, a source-defined metric, an early-design '
            'partial, or an inapplicable geometry branch.'
        ),
        axis=1,
    )

    # Apply the same machine-readable N/A convention used by the shared
    # audited-row builder.  These statuses are attached here, after the
    # eight-column atomic rows were created, so they must be normalized at
    # this adapter boundary rather than in ``_metric`` (which also serves
    # legitimate calculated metrics whose values must remain available).
    not_applicable = selected['Implementation Status'].astype(str).str.startswith(
        'not_applicable'
    )
    selected.loc[not_applicable, 'Threshold Value'] = 0.0
    selected.loc[not_applicable, 'Constraint Value'] = 0.0
    selected.loc[not_applicable, 'Units'] = 'N/A'
    selected.loc[not_applicable, 'Constraint Eval'] = 0.0
    selected['Constraint Bit'] = selected['Constraint Bit'].astype(object)
    selected.loc[not_applicable, 'Constraint Bit'] = True

    # The strength-deck grab row is an explicit GenISys design extension of
    # the inner-bottom/lower-hopper impact rule.  Preserve it in the audited
    # result set, but never present it as direct ABS certification evidence.
    strength_deck_grab = (
        selected['Name'] == 'Strength deck plating - grab loading'
    )
    selected.loc[strength_deck_grab, 'Result Basis'] = (
        'Project-selected application of the MVR 3-2-4/9.5 GRAB equation '
        'to the modeled strength deck under ASM-009. This is an early-design '
        'impact/scantling objective, not a direct applicability claim for '
        'the ABS inner-bottom and lower-hopper clause.'
    )

    return selected[AUDITED_RESULT_COLUMNS].drop_duplicates(
        subset=['Name', 'Requirement', 'Audit ID'],
        keep='first',
    ).reset_index(drop=True)


def merge_atomic_structural_audit_results(
    existing_audited_constraints,
    structure_constraints_v1p1,
):
    """Retain compatibility aliases and append atomic V1.1 audit rows.

    The integrated evaluator already builds its verified/additional audited
    rows before the expanded V1.1 table exists.  This adapter gives that call
    site one explicit, duplicate-safe merge operation after V1.1 evaluation.
    """

    retained = existing_audited_constraints[
        AUDITED_RESULT_COLUMNS
    ].copy()
    atomic = build_atomic_structural_audit_results(
        structure_constraints_v1p1
    )

    return pd.concat(
        [retained, atomic],
        ignore_index=True,
    ).drop_duplicates(
        subset=['Name', 'Requirement', 'Audit ID'],
        keep='first',
    ).reset_index(drop=True)
