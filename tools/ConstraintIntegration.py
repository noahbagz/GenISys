"""Create audit-traceable constraint results for integrated ship designs.

The existing GenISys evaluators contain a mixture of validated rules, legacy
rules from an earlier project, self-imposed optimization limits, and partially
implemented calculations.  This module does not hide those legacy results.
Instead, it creates a second, deliberately smaller result table containing only
the structural checks already validated by the audit plus the hydrostatic
foundation and arrangement checks implemented in the current integration step.
"""

import numpy as np
import pandas as pd


AUDITED_RESULT_COLUMNS = [
    'Class',
    'Name',
    'Requirement',
    'Threshold Value',
    'Constraint Value',
    'Units',
    'Constraint Eval',
    'Constraint Bit',
    'Audit ID',
    'Implementation Status',
    'Assumption IDs',
    'Result Basis',
]


# These are the eight active structural result names whose formulas and
# StructureCalcs mappings were classified as implemented in the structural
# audit.  Keeping the map explicit prevents the much larger legacy constraint
# table from being accidentally represented as audited/verified.
VERIFIED_STRUCTURAL_RULES = {
    'Min Double Bottom Height': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/3.1.1(c)',
        'assumptions': 'ASM-002; ASM-004; ASM-005',
    },
    'Max Bottom Girder Spacing': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/3.7',
        'assumptions': 'ASM-002; ASM-005',
    },
    'Max Floor Spacing': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/5.1-a',
        'assumptions': 'ASM-002; ASM-006',
    },
    'Min Inner Bottom Grab-Service Thickness': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/9.5',
        'assumptions': 'ASM-002; ASM-005; ASM-009',
    },
    'Min Lower Hopper Grab-Service Thickness': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/9.5',
        'assumptions': 'ASM-002; ASM-005; ASM-009',
    },
    'Min Bottom Stringer SM': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 4/11.3',
        'assumptions': 'ASM-002; ASM-004; ASM-005',
    },
    'Min Web Frame Height': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 6/3.5.i',
        'assumptions': 'ASM-002; ASM-004; ASM-005',
    },
    'Min Web Frame Thickness': {
        'audit_id': 'MVR Part 3, Ch 2, Sec 6/3.5.iii',
        'assumptions': 'ASM-002; ASM-005',
    },
}


def build_verified_structural_results(structure_constraints):
    """Select and annotate only audit-verified MVR structural checks."""

    rows = structure_constraints[
        structure_constraints['Name'].isin(VERIFIED_STRUCTURAL_RULES)
    ].copy()

    rows['Audit ID'] = rows['Name'].map(
        lambda name: VERIFIED_STRUCTURAL_RULES[name]['audit_id']
    )
    rows['Implementation Status'] = 'calculated_verified'
    rows['Assumption IDs'] = rows['Name'].map(
        lambda name: VERIFIED_STRUCTURAL_RULES[name]['assumptions']
    )
    rows['Result Basis'] = (
        'Calculated from the generated structural-element definition and '
        'active loading/design parameters; formula and mapping covered by '
        'the ABS structural audit.'
    )

    return rows[AUDITED_RESULT_COLUMNS].reset_index(drop=True)


def _structure_value(structure_evaluations, parameter):
    """Return one scalar from the generated structure-evaluation table."""

    return float(
        structure_evaluations.loc[
            structure_evaluations['Parameter'] == parameter,
            'Value',
        ].iloc[0]
    )


def _minimum_result(
    name,
    requirement,
    required,
    offered,
    units,
    audit_id,
    assumptions,
    result_basis,
    status='calculated_verified',
):
    """Construct a conventional minimum-required structural result."""

    return _result_row(
        'Structure',
        name,
        requirement,
        required,
        offered,
        units,
        required - offered,
        offered >= required,
        audit_id,
        status,
        assumptions,
        result_basis,
    )


def _maximum_result(
    name,
    requirement,
    maximum,
    offered,
    units,
    audit_id,
    assumptions,
    result_basis,
    status='calculated_verified',
):
    """Construct a conventional maximum-allowed structural result."""

    return _result_row(
        'Structure',
        name,
        requirement,
        maximum,
        offered,
        units,
        offered - maximum,
        offered <= maximum,
        audit_id,
        status,
        assumptions,
        result_basis,
    )


def build_additional_structural_results(
    hull_dict,
    structure_evaluations,
    structural_elements,
    abs_rules,
    cargo_holds,
):
    """Build complete atomic MVR/CSR checks enabled by current geometry.

    This function intentionally does not promote every formula in the legacy
    structural table.  Each result below has a complete source equation,
    current-design input mapping, comparison direction, and focused test.
    Rules requiring an unmodelled physical state remain in the blocker log.
    """

    rows = []

    L = float(hull_dict['L'])
    B = float(hull_dict['B'])
    D = float(hull_dict['D'])
    draft = float(hull_dict['d'])

    frame_spacing_mm = 1000.0*_structure_value(
        structure_evaluations,
        'Frame_Spacing',
    )
    side_longitudinal_spacing_mm = 1000.0*_structure_value(
        structure_evaluations,
        'Side_Stringer_Spacing',
    )
    bottom_longitudinal_spacing_mm = 1000.0*_structure_value(
        structure_evaluations,
        'Bottom_Stringer_Spacing',
    )

    side_plate_mm = _structure_value(
        structure_evaluations,
        'Side_Shell_Thickness',
    )
    bottom_plate_mm = _structure_value(
        structure_evaluations,
        'Bottom_Shell_Thickness',
    )

    # ABS MVR 3-1-2/7.  ASM-002 explicitly supplies LOA as the Rule length.
    rows.append(
        _maximum_result(
            'MVR maximum Rule length',
            'ABS MVR 3-1-2/7.i',
            500.0,
            L,
            'm',
            'MVR Part 3, Ch 1, Sec 2/7.i',
            'ASM-002',
            'Non-OSV steel-vessel limit; GenISys LOA is the approved Rule-length proxy.',
        )
    )
    rows.append(
        _maximum_result(
            'MVR maximum breadth from length',
            'ABS MVR 3-1-2/7.ii-a',
            0.2*L,
            B,
            'm',
            'MVR Part 3, Ch 1, Sec 2/7.ii-a',
            'ASM-002',
            'Full molded breadth compared with 0.2 times the approved LOA proxy.',
        )
    )

    breadth_depth_factor = 2.0 if L < 90.0 else 2.5
    rows.append(
        _maximum_result(
            'MVR maximum breadth from depth',
            'ABS MVR 3-1-2/7.ii-b',
            breadth_depth_factor*D,
            B,
            'm',
            'MVR Part 3, Ch 1, Sec 2/7.ii-b',
            'ASM-002',
            'Uses the modeled molded depth to the strength deck.',
        )
    )

    # ABS MVR 3-2-2/3.9.3 and /3.15.1 publish empirical pressure-thickness
    # branches only through L=427 m.  A design beyond that formula domain is
    # an ordinary failed applicability/method row, not an exception that
    # discards every other useful result for the design.
    rounded_thresholds = []
    side_spacing_mm = max(
        frame_spacing_mm,
        side_longitudinal_spacing_mm,
    )
    bottom_spacing_mm = max(
        frame_spacing_mm,
        bottom_longitudinal_spacing_mm,
    )

    if L <= 427.0:
        side_pressure_mm = abs_rules.calc_side_shell_pressure_thickness(
            side_spacing_mm
        )
        side_pressure_audit = (
            'MVR Part 3, Ch 2, Sec 2/3.9.3-a'
            if L <= 305.0
            else 'MVR Part 3, Ch 2, Sec 2/3.9.3-b'
        )
        rows.append(
            _minimum_result(
                'MVR side-shell pressure thickness',
                'ABS MVR 3-2-2/3.9.3',
                side_pressure_mm,
                side_plate_mm,
                'mm',
                side_pressure_audit,
                'ASM-002; ASM-004; ASM-005',
                f'Governing modeled spacing is {side_spacing_mm:.6g} mm.',
            )
        )

        side_end_minimum_mm = abs_rules.calc_shell_end_thickness(610.0)
        rows.append(
            _minimum_result(
                'MVR side-shell amidship end-rule minimum',
                'ABS MVR 3-2-2/3.9.3-c; 3-2-2/5.1',
                side_end_minimum_mm,
                side_plate_mm,
                'mm',
                'MVR Part 3, Ch 2, Sec 2/3.9.3-c',
                'ASM-002; ASM-005',
                'The end-shell equation uses the Rule-prescribed 610 mm spacing.',
            )
        )

        bottom_pressure_mm = abs_rules.calc_bottom_shell_pressure_thickness(
            bottom_spacing_mm
        )

        if L <= 122.0:
            bottom_pressure_audit = 'MVR Part 3, Ch 2, Sec 2/3.15.1-a'
        elif L <= 305.0:
            bottom_pressure_audit = 'MVR Part 3, Ch 2, Sec 2/3.15.1-b'
        else:
            bottom_pressure_audit = 'MVR Part 3, Ch 2, Sec 2/3.15.1-c'

        rows.append(
            _minimum_result(
                'MVR bottom-shell pressure thickness',
                'ABS MVR 3-2-2/3.15.1',
                bottom_pressure_mm,
                bottom_plate_mm,
                'mm',
                bottom_pressure_audit,
                'ASM-002; ASM-004; ASM-005',
                f'Governing modeled spacing is {bottom_spacing_mm:.6g} mm.',
            )
        )
        rounded_thresholds.extend([
            side_pressure_mm,
            side_end_minimum_mm,
            bottom_pressure_mm,
        ])
    else:
        rows.append(
            _maximum_result(
                'MVR empirical shell-thickness formula length domain',
                'ABS MVR 3-2-2/3.9.3; 3.15.1; 5.1',
                427.0,
                L,
                'm',
                'MVR Part 3, Ch 2, Sec 2/3.9.3',
                'ASM-002',
                (
                    'No published empirical side-, bottom-, or end-shell '
                    'thickness branch exists above L=427 m; direct/special '
                    'consideration is required.'
                ),
                status='rule_formula_domain_failed',
            )
        )

    # ABS MVR 3-2-2/3.15.2 remains an independent bottom-shell minimum.  The
    # current hull is longitudinally framed and no optional Rb reduction is
    # credited.

    standard_frame_spacing_mm = (
        2.08*L + 438.0 if L <= 270.0 else 1000.0
    )
    bottom_minimum_mm = abs_rules.calc_bottom_shell_minimum_thickness(
        bottom_spacing_mm,
        standard_frame_spacing_mm,
    )
    rows.append(
        _minimum_result(
            'MVR longitudinally framed minimum bottom-shell thickness',
            'ABS MVR 3-2-2/3.15.2',
            bottom_minimum_mm,
            bottom_plate_mm,
            'mm',
            'MVR Part 3, Ch 2, Sec 2/3.15.2',
            'ASM-002; ASM-005',
            'Uses the Rule spacing floor and takes no optional Rb reduction credit.',
        )
    )

    # ABS MVR 3-2-3/5.1-b.  The top deck is the strength/freeboard deck and
    # carries no deck cargo, so the separately audited 5.0 mm minimum applies.
    deck_plate_mm = _structure_value(
        structure_evaluations,
        'Top_Deck_Thickness',
    )
    rows.append(
        _minimum_result(
            'MVR general strength-deck minimum thickness',
            'ABS MVR 3-2-3/5.1-b',
            5.0,
            deck_plate_mm,
            'mm',
            'MVR Part 3, Ch 2, Sec 3/5.1-b',
            'ASM-005; ASM-013',
            'Topmost modeled deck is the exposed strength deck; no deck cargo is carried.',
        )
    )

    # ABS MVR 3-2-3/5.5.1.  The ordinary 0.125-width threshold already
    # exceeds the 150 mm narrow-ligament minimum for current hatch widths.
    hatch_width_m = _structure_value(
        structure_evaluations,
        'Hatch_Width',
    )
    hatch_radius_mm = _structure_value(
        structure_evaluations,
        'Hatch_Corner_Radius',
    )
    minimum_hatch_radius_mm = min(0.125*hatch_width_m*1000.0, 600.0)
    rows.append(
        _minimum_result(
            'MVR strength-deck hatch-corner radius',
            'ABS MVR 3-2-3/5.5.1',
            minimum_hatch_radius_mm,
            hatch_radius_mm,
            'mm',
            'MVR Part 3, Ch 2, Sec 3/5.5.1',
            'ASM-002',
            f'Modeled hatch width is {hatch_width_m:.6g} m.',
        )
    )

    # ABS MVR 3-2-4/1.2.ii is independent of the existing 3.1.1(c) depth
    # equation; both are retained and the design must satisfy each.
    double_bottom_height_m = _structure_value(
        structure_evaluations,
        'Double_Bottom_Height',
    )/1000.0
    rule_double_bottom_height_m = max(min(B/20.0, 2.0), 0.76)
    rows.append(
        _minimum_result(
            'MVR breadth-based double-bottom height',
            'ABS MVR 3-2-4/1.2.ii',
            rule_double_bottom_height_m,
            double_bottom_height_m,
            'm',
            'MVR Part 3, Ch 2, Sec 4/1.2.ii',
            'ASM-002; ASM-005',
            'Bulk-carrier B/20 rule bounded by 0.76 m and 2.0 m.',
        )
    )

    # ABS MVR 3-2-4/3.1.1(a).  ASM-015 resolves the half-model convention:
    # the centerline assembly consists of two coincident plate halves, so its
    # physical offered thickness is twice the ordinary bottom-girder variable.
    center_girder_required_mm = abs_rules.round_rule_thickness(
        0.056*L + 5.5
    )
    center_girder_offered_mm = 2.0*_structure_value(
        structure_evaluations,
        'Bottom_Girder_Plate_Thickness',
    )
    rows.append(
        _minimum_result(
            'MVR center-girder plate thickness amidships',
            'ABS MVR 3-2-4/3.1.1(a)',
            center_girder_required_mm,
            center_girder_offered_mm,
            'mm',
            'MVR Part 3, Ch 2, Sec 4/3.1.1(a)',
            'ASM-002; ASM-005; ASM-015',
            (
                'Physical center-girder thickness is twice the modeled '
                'ordinary bottom-girder plate-thickness variable.'
            ),
        )
    )

    # ABS MVR 3-2-4/5.1-b.  The generated double bottom is longitudinally
    # framed and its solid floors are the transverse floors at web-frame/deck-
    # beam stations under ASM-006.
    floor_required_mm = abs_rules.round_rule_thickness(
        0.036*L + 4.7 + 1.5
    )
    floor_offered_mm = _structure_value(
        structure_evaluations,
        'Bottom_Floor_Plate_Thickness',
    )
    rows.append(
        _minimum_result(
            'MVR solid-floor plate thickness',
            'ABS MVR 3-2-4/5.1-b',
            floor_required_mm,
            floor_offered_mm,
            'mm',
            'MVR Part 3, Ch 2, Sec 4/5.1-b',
            'ASM-002; ASM-005; ASM-006',
            'Longitudinal bottom and inner-bottom framing uses c = 1.5 mm.',
        )
    )

    # ABS MVR 3-2-4/9.2.  The active cargo optimizer produces a fill surface
    # for every hold.  The common inner-bottom scantling must satisfy the
    # greatest pressure threshold produced by any optimized full-load hold.
    inner_bottom_spacing_mm = max(
        frame_spacing_mm,
        bottom_longitudinal_spacing_mm,
    )
    double_bottom_top_m = double_bottom_height_m
    cargo_depths_m = np.maximum(
        cargo_holds['Cargo_Fill_Height_m'].astype(float).to_numpy()
        - double_bottom_top_m,
        0.0,
    )
    inner_bottom_cargo_thresholds_mm = [
        abs_rules.calc_inner_bottom_cargo_pressure_thickness(
            inner_bottom_spacing_mm,
            cargo_depth_m,
        )
        for cargo_depth_m in cargo_depths_m
    ]
    governing_inner_bottom_mm = max(inner_bottom_cargo_thresholds_mm)
    rows.append(
        _minimum_result(
            'MVR inner-bottom optimized-cargo pressure thickness',
            'ABS MVR 3-2-4/9.2',
            governing_inner_bottom_mm,
            _structure_value(structure_evaluations, 'Inner_Bottom_Thickness'),
            'mm',
            'MVR Part 3, Ch 2, Sec 4/9.2',
            'ASM-002; ASM-004; ASM-005; ASM-014',
            (
                f'Governing optimized cargo depth is '
                f'{max(cargo_depths_m):.6g} m at 2400 kg/m^3; '
                f'plate spacing is {inner_bottom_spacing_mm:.6g} mm.'
            ),
        )
    )

    # ABS MVR 3-2-5/1.7 and 3.1.1.  These reuse the corrected active rule
    # method so the legacy and audit-qualified tables cannot diverge.
    frame_span_m = _structure_value(structure_evaluations, 'l_frames')
    frame_head_m = _structure_value(structure_evaluations, 'h_frames')
    first_deck_support_m = _structure_value(
        structure_evaluations,
        'WebFrame_Height',
    )/1000.0
    maximum_frame_spacing_mm, required_frame_sm_cm3 = (
        abs_rules.calc_Frame_Constraints(
            frame_spacing_mm/1000.0,
            frame_span_m,
            frame_head_m,
            first_deck_support_m,
        )
    )
    if L <= 427.0:
        rows.append(
            _maximum_result(
                'MVR maximum transverse-frame spacing',
                'ABS MVR 3-2-5/1.7',
                maximum_frame_spacing_mm,
                frame_spacing_mm,
                'mm',
                'MVR Part 3, Ch 2, Sec 5/1.7',
                'ASM-002',
                'Actual generated ordinary-frame spacing is compared with the applicable length branch.',
            )
        )
    else:
        rows.append(
            _maximum_result(
                'MVR transverse-frame-spacing formula length domain',
                'ABS MVR 3-2-5/1.7',
                427.0,
                L,
                'm',
                'MVR Part 3, Ch 2, Sec 5/1.7',
                'ASM-002',
                (
                    'No published empirical transverse-frame spacing branch '
                    'exists above L=427 m; special consideration is required.'
                ),
                status='rule_formula_domain_failed',
            )
        )
    rows.append(
        _minimum_result(
            'MVR transverse side-frame section modulus',
            'ABS MVR 3-2-5/3.1.1',
            required_frame_sm_cm3,
            _structure_value(structure_evaluations, 'Side_Frame_SM'),
            'cm^3',
            'MVR Part 3, Ch 2, Sec 5/3.1.1',
            'ASM-002; ASM-004; ASM-005; ASM-013',
            'No-deck-cargo branch uses the Rule minimum h1 = 2.44 m.',
        )
    )

    # CRF-008 / ABS MVR 3-2-6/3.1.1.  A longitudinal deck with a transverse
    # deck beam at every web frame gives K = 1 for the generated arrangement.
    # ``WebFrame_SM`` is the weakest actual generated web/face-plate section
    # with the validated legacy attached-shell breadth, not a synthesized box.
    web_spacing_m = _structure_value(
        structure_evaluations,
        'WebFrame_Spacing',
    )
    web_span_m = D - double_bottom_height_m
    web_head_m = max(
        0.5*web_span_m,
        abs(0.5*web_span_m + double_bottom_height_m - draft),
    )
    required_web_sm_cm3 = abs_rules.calc_WebframeAndSideStringer_Constraints(
        web_spacing_m,
        web_span_m,
        web_head_m,
        first_deck_support_m,
        1.0,
        side_longitudinal_spacing_mm/1000.0,
    )[0]
    rows.append(
        _minimum_result(
            'MVR web-frame section modulus',
            'ABS MVR 3-2-6/3.1.1',
            required_web_sm_cm3,
            _structure_value(structure_evaluations, 'WebFrame_SM'),
            'cm^3',
            'MVR Part 3, Ch 2, Sec 6/3.1.1',
            'ASM-002; ASM-004; ASM-005; ASM-006; ASM-013; ASM-022; ASM-023',
            (
                'Longitudinal deck with a transverse deck beam at every web '
                'frame uses K = 1. Offered modulus is calculated from the '
                'weakest actual generated web-frame section.'
            ),
        )
    )

    # ABS MVR 3-2-8/5.7.i.  Both top-deck primary-member types are clear of
    # tanks in the current arrangement.  They are separate physical members,
    # so retain separate threshold/value/pass rows instead of a hidden minimum
    # utilization envelope.
    deck_girder_span_m = _structure_value(
        structure_evaluations,
        'Bulkhead_Spacing',
    )
    deck_transverse_span_m = _structure_value(
        structure_evaluations,
        'Deck_Girder_Spacing',
    )
    required_deck_girder_depth_mm = 1000.0*0.0583*deck_girder_span_m
    required_deck_transverse_depth_mm = (
        1000.0*0.0583*deck_transverse_span_m
    )
    rows.extend(
        [
            _minimum_result(
                'MVR deck-girder minimum depth',
                'ABS MVR 3-2-8/5.7.i',
                required_deck_girder_depth_mm,
                _structure_value(
                    structure_evaluations,
                    'Deck_Girder_Height',
                ),
                'mm',
                'MVR Part 3, Ch 2, Sec 8/5.7.i',
                'ASM-002; ASM-013',
                'Girder span is the generated transverse-bulkhead spacing.',
            ),
            _minimum_result(
                'MVR deck-transverse minimum depth',
                'ABS MVR 3-2-8/5.7.i',
                required_deck_transverse_depth_mm,
                _structure_value(
                    structure_evaluations,
                    'Deck_Beam_Height',
                ),
                'mm',
                'MVR Part 3, Ch 2, Sec 8/5.7.i',
                'ASM-002; ASM-013',
                'Transverse span is the generated deck-girder spacing.',
            ),
        ]
    )

    # CSR-H Pt 1, Ch 2, Sec 2 [1.1.5].  The generated cargo-region
    # transverse bulkheads are explicitly recoverable from the CAD element
    # table, so evaluate their actual coordinate intervals.
    bulkhead_x = np.sort(
        structural_elements.loc[
            structural_elements['Class'] == 'Transverse_Bulkhead',
            'x_loc',
        ].astype(float).unique()
    )
    bulkhead_intervals = np.diff(bulkhead_x)
    interval_deviation_m = float(
        np.max(np.abs(bulkhead_intervals - np.mean(bulkhead_intervals)))
    )
    coordinate_tolerance_m = 1.0e-8
    rows.append(
        _maximum_result(
            'CSR uniform cargo-region bulkhead spacing',
            'CSR-H Pt 1 Ch 2 Sec 2 [1.1.5]',
            coordinate_tolerance_m,
            interval_deviation_m,
            'm interval deviation',
            'CSR-H Pt 1 Ch 2 Sec 2 [1.1.5]',
            'ASM-003',
            f'Evaluated {len(bulkhead_intervals)} actual cargo-region intervals.',
        )
    )

    # CSR-H Pt 1, Ch 2, Sec 3 [3.1.2].  Both modeled boundaries are vertical,
    # so their transverse coordinate difference is the perpendicular width.
    inner_side_shells = structural_elements.loc[
        structural_elements['Class'] == 'Inner_Side_Shell'
    ]

    if inner_side_shells.empty:
        # CSR-H Pt 1, Ch 2, Sec 3 [3.1.2] governs the width of a double
        # side when that arrangement exists.  A generated single-side design
        # has no width to compare and must be reported as not applicable,
        # rather than failed or silently supplied with invented geometry.
        rows.append(
            _result_row(
                'Structure Arrangement',
                'CSR minimum bulk-carrier double-side width',
                'CSR-H Pt 1 Ch 2 Sec 3 [3.1.2]',
                0.0,
                0.0,
                'N/A',
                0.0,
                None,
                'CSR-H Pt 1 Ch 2 Sec 3 [3.1.2]',
                'not_applicable_by_geometry',
                'ASM-003',
                (
                    'No Inner_Side_Shell element exists; the generated '
                    'design is classified as single-side for this rule.'
                ),
            )
        )
    else:
        side_y = float(
            structural_elements.loc[
                structural_elements['Class'] == 'Side_Shell',
                'y_loc',
            ].iloc[0]
        )
        inner_side_y = float(inner_side_shells['y_loc'].iloc[0])
        double_side_width_m = abs(side_y - inner_side_y)
        rows.append(
            _minimum_result(
                'CSR minimum bulk-carrier double-side width',
                'CSR-H Pt 1 Ch 2 Sec 3 [3.1.2]',
                1.0,
                double_side_width_m,
                'm',
                'CSR-H Pt 1 Ch 2 Sec 3 [3.1.2]',
                'ASM-003',
                (
                    'Perpendicular separation of the modeled vertical side '
                    'shell and inner hull.'
                ),
            )
        )

    # CSR-H Pt 1, Ch 3, Sec 1 [2.1.1].  This is a material-definition
    # predicate rather than a scantling inequality, so both required constants
    # are recorded together as the one audited source rule.
    material_matches = (
        abs(abs_rules.E_steel - 206000.0) <= 1.0e-12
        and abs(abs_rules.nu_steel - 0.30) <= 1.0e-12
    )
    rows.append(
        _result_row(
            'Structural Material',
            'CSR carbon-steel elastic constants',
            'CSR-H Pt 1 Ch 3 Sec 1 [2.1.1]',
            True,
            material_matches,
            'Boolean',
            int(not material_matches),
            material_matches,
            'CSR-H Pt 1 Ch 3 Sec 1 [2.1.1]',
            'calculated_verified',
            'ASM-005',
            f'E={abs_rules.E_steel:.0f} N/mm^2; nu={abs_rules.nu_steel:.2f}.',
        )
    )

    # MVR 3-1-2/5.3 is represented once as an implementation-method check.
    # Every plate threshold generated above and by the existing grab/web
    # methods must lie on a half-millimetre increment.
    rounded_thresholds.extend([
        bottom_minimum_mm,
        floor_required_mm,
    ])
    all_half_millimetres = all(
        abs(2.0*value - round(2.0*value)) <= 1.0e-12
        for value in rounded_thresholds
    )
    rows.append(
        _result_row(
            'Structure Rule Method',
            'MVR plate-thickness rounding method',
            'ABS MVR 3-1-2/5.3',
            True,
            all_half_millimetres,
            'Boolean',
            int(not all_half_millimetres),
            all_half_millimetres,
            'MVR Part 3, Ch 1, Sec 2/5.3',
            'calculated_verified',
            'ASM-002; ASM-005',
            'All active audited plate thresholds use Rule half-up rounding to 0.5 mm.',
        )
    )

    return pd.DataFrame(rows, columns=AUDITED_RESULT_COLUMNS)


def _result_row(
    result_class,
    name,
    requirement,
    threshold,
    value,
    units,
    constraint_eval,
    passes,
    audit_id,
    status,
    assumptions,
    result_basis,
):
    """Construct one result using the shared GenISys constraint schema."""

    # A status-aware not-applicable row has one project-wide output encoding:
    # retain its name and basis, publish zero magnitudes, and count the row as
    # passed.  This makes optional arrangements comparable in fixed-width
    # dataset views without treating an absent component as a violation.
    if str(status).startswith('not_applicable'):
        threshold = 0.0
        value = 0.0
        units = 'N/A'
        constraint_eval = 0.0
        passes = True

    # ``None`` remains reserved for calculations that do not yet have a valid
    # pass/fail interpretation, including partial and diagnostic metrics.
    constraint_bit = np.nan if passes is None else bool(passes)

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': threshold,
        'Constraint Value': value,
        'Units': units,
        'Constraint Eval': constraint_eval,
        'Constraint Bit': constraint_bit,
        'Audit ID': audit_id,
        'Implementation Status': status,
        'Assumption IDs': assumptions,
        'Result Basis': result_basis,
    }


def build_hydrostatic_foundation_results(
    hull_dict,
    structure_evaluations,
    hydrostatics,
    weights,
):
    """Build the hydrostatic checks supportable at current design fidelity.

    Rows marked ``calculated_verified`` are numerical foundation checks.
    Rows marked ``assumption_based`` encode approved arrangement metadata.
    ``calculated_atomic_partial`` is an explicitly atomic portion of a larger
    rule and must not be read as compliance with the complete parent rule.
    """

    rows = []

    full_beam = float(hull_dict['B'])
    draft = float(hull_dict['d'])
    loa = float(hull_dict['L'])
    displacement_volume = float(hull_dict['Vol_Disp'])

    double_bottom_height_mm = float(
        structure_evaluations.loc[
            structure_evaluations['Parameter'] == 'Double_Bottom_Height',
            'Value',
        ].iloc[0]
    )
    double_bottom_height_m = double_bottom_height_mm / 1000.0

    # HDF-061/HDF-062 and HDF-ASM-032.  The 1969 Convention tonnage formula
    # uses the volume of all enclosed spaces.  GenISys does not yet model
    # every above-deck enclosed space, so the generated molded hull volume up
    # to the freeboard deck is deliberately treated as a lower bound.  When
    # this lower bound is already at least 500 GT, omitted enclosed spaces can
    # only increase formal Convention tonnage and applicability is proven.
    intact_volumes, _, _ = hydrostatics.calc_Volumes()
    enclosed_hull_volume_m3 = float(intact_volumes[-1])
    tonnage_coefficient = (
        0.2 + 0.02*np.log10(enclosed_hull_volume_m3)
    )
    convention_gt_lower_bound = (
        tonnage_coefficient*enclosed_hull_volume_m3
    )
    part_174_applies = convention_gt_lower_bound >= 500.0

    # HDF-049 / 46 CFR 170.001 and 93.01-1.  This is an explicit project
    # applicability decision, not a dimension-derived calculation: GenISys is
    # evaluating a new U.S.-flagged inspected cargo vessel under Subchapters I
    # and S.  Keeping it in the runtime table makes the legal route visible to
    # every downstream result.  HDF-ASM-004 intentionally treats this as a
    # not-yet-planned new build, so GenISys does not invent lifecycle dates.
    rows.append(
        _result_row(
            'Regulatory Applicability',
            'U.S. Subchapter-S cargo-vessel route',
            '46 CFR 170.001; 46 CFR 93.01-1',
            True,
            True,
            'Boolean',
            0,
            True,
            'HDF-049',
            'assumption_based_applicability',
            'HDF-ASM-003; HDF-ASM-004',
            (
                'Approved new U.S.-flagged inspected cargo-vessel design '
                'basis; current new-build provisions govern the preliminary '
                'design evaluation.'
            ),
        )
    )

    rows.append(
        _result_row(
            'Regulatory Applicability',
            '46 CFR Part 174 dry-cargo tonnage gate',
            '46 CFR 174.005; 46 CFR 69.57',
            500.0,
            convention_gt_lower_bound,
            '1969 Convention GT',
            500.0 - convention_gt_lower_bound,
            part_174_applies,
            'HDF-061',
            'calculated_conservative_applicability',
            'HDF-ASM-003; HDF-ASM-004; HDF-ASM-032',
            (
                f'Generated below-freeboard-deck enclosed hull volume is '
                f'{enclosed_hull_volume_m3:.6g} m^3; K1 is '
                f'{tonnage_coefficient:.8g}. Above-deck enclosed spaces are '
                f'omitted, so {convention_gt_lower_bound:.6g} GT is a lower '
                'bound rather than certificate tonnage.'
            ),
        )
    )

    rows.append(
        _result_row(
            'Regulatory Applicability',
            '46 CFR Subpart-J new dry-cargo tonnage gate',
            '46 CFR 174.350; 174.355; 46 CFR 69.57',
            500.0,
            convention_gt_lower_bound,
            '1969 Convention GT',
            500.0 - convention_gt_lower_bound,
            part_174_applies,
            'HDF-062',
            'calculated_conservative_applicability',
            'HDF-ASM-003; HDF-ASM-004; HDF-ASM-032',
            (
                'Current new-build and dry-cargo branches are approved. The '
                'generated enclosed-hull-volume lower bound independently '
                'resolves the remaining 500-GT applicability gate.'
            ),
        )
    )

    # HDF-050 / 46 CFR 170.165.  The intended International Load Line and
    # SOLAS Cargo Ship Safety Construction certificates select the 2008 IS
    # Code route.  The row selects the governing route; individual IS Code
    # numerical criteria remain separate constraints.
    rows.append(
        _result_row(
            'Regulatory Applicability',
            '2008 IS Code intact-stability route',
            '46 CFR 170.165; SOLAS II-1/5.1; 2008 IS Code Part A',
            True,
            True,
            'Boolean',
            0,
            True,
            'HDF-050',
            'assumption_based_applicability',
            'HDF-ASM-002; HDF-ASM-005',
            (
                'The approved intended certificates invoke the incorporated '
                '2008 IS Code route; this row does not satisfy its criteria.'
            ),
        )
    )

    # HDF-072 / 46 CFR Part 42.  The project expressly designs for an
    # International Load Line Certificate and unrestricted ocean service.
    # Formal Load Line dimensions are still required by the later freeboard
    # calculation, but the project route itself is no longer open.  For this
    # new international design above 79 ft, Part 42 does not use a GT gate.
    rows.append(
        _result_row(
            'Regulatory Applicability',
            'U.S. international-load-line design route',
            '46 CFR 42.03-5; 42.03-10; 42.03-30',
            True,
            True,
            'Boolean',
            0,
            True,
            'HDF-072',
            'assumption_based_applicability',
            'HDF-ASM-003; HDF-ASM-005',
            (
                'International Load Line certification is an approved design '
                'objective. Formal statutory dimensions remain required for '
                'the numerical freeboard package.'
            ),
        )
    )

    # HDF-023 / SOLAS II-1/9.2: h = B/20, bounded to [0.76, 2.0] m.
    # The result is atomic because complete statutory extent from the
    # collision bulkhead to the afterpeak still needs explicit arrangement
    # endpoints, although HDF-ASM-020 assumes a continuous conventional DB.
    required_double_bottom_height = min(max(full_beam / 20.0, 0.76), 2.0)
    rows.append(
        _result_row(
            'Subdivision Arrangement',
            'SOLAS double-bottom height',
            'SOLAS II-1/9.2',
            required_double_bottom_height,
            double_bottom_height_m,
            'm',
            required_double_bottom_height - double_bottom_height_m,
            double_bottom_height_m >= required_double_bottom_height,
            'HDF-023',
            'calculated_atomic_partial',
            'HDF-ASM-020; HDF-ASM-028',
            (
                'Height is calculated; the complete HDF-023 extent rule is '
                'not claimed until collision/afterpeak endpoints are modeled.'
            ),
        )
    )

    # HDF-095 / 46 CFR 42.09-5: a dry-bulk carrier is not a Type-A liquid
    # bulk carrier, so the approved load-line classification is Type B.
    rows.append(
        _result_row(
            'Load Line Applicability',
            'Standard Type-B load-line classification',
            '46 CFR 42.09-5',
            True,
            True,
            'Boolean',
            0,
            True,
            'HDF-095',
            'assumption_based_verified',
            'HDF-ASM-003 through HDF-ASM-006',
            (
                'Approved US-flagged dry-bulk cargo-vessel service basis; '
                'this is classification, not a completed freeboard result.'
            ),
        )
    )

    # HDF-097 branch selector: approved qualifying hatch-cover existence lets
    # the later Type-B freeboard calculation use the qualifying-cover branch.
    # It does not validate cover structure or the rest of the freeboard chain.
    rows.append(
        _result_row(
            'Load Line Input',
            'Qualifying position-1 hatch-cover branch',
            '46 CFR 42.20-5(b); 42.15-25(d) or 42.15-30',
            True,
            True,
            'Boolean',
            0,
            True,
            'HDF-097',
            'assumption_based_branch_only',
            'HDF-ASM-021 through HDF-ASM-027',
            (
                'Qualifying closed steel weathertight covers are assumed. '
                'No hatch-cover scantling certification or complete Type-B '
                'freeboard calculation is claimed.'
            ),
        )
    )

    # HDF-082: exercise the corrected full-waterplane calculation at the
    # selected design draft.  Positivity/finite checks catch failed numerical
    # integration without inventing a regulatory threshold.
    waterplane_area, lcf, ixx, iyy = hydrostatics.calc_WaterPlane(draft)
    waterplane_values = np.array(
        [waterplane_area, lcf, ixx, iyy],
        dtype=float,
    )
    waterplane_valid = (
        np.all(np.isfinite(waterplane_values))
        and waterplane_area > 0.0
        and ixx > 0.0
        and iyy > 0.0
    )
    rows.append(
        _result_row(
            'Hydrostatic Numerical Foundation',
            'Full-waterplane integration at design draft',
            'HDF-082 internal numerical method',
            True,
            waterplane_valid,
            'Boolean',
            int(not waterplane_valid),
            waterplane_valid,
            'HDF-082',
            'calculated_verified',
            'HDF-ASM-028',
            (
                f'Full area={waterplane_area:.6g} m^2; '
                f'LCF={lcf:.6g} m; Ixx={ixx:.6g} m^4; '
                f'Iyy={iyy:.6g} m^4.'
            ),
        )
    )

    # HDF-103: recompute Cb independently using the authoritative full-beam
    # geometry convention and compare it with the Weights module value.
    expected_cb = displacement_volume / (loa * full_beam * draft)
    cb_error = abs(float(weights.Cb) - expected_cb)
    cb_tolerance = 1.0e-12
    rows.append(
        _result_row(
            'Hydrostatic Numerical Foundation',
            'Consistent full-beam block coefficient',
            'HDF-103 internal numerical method',
            cb_tolerance,
            cb_error,
            'absolute Cb error',
            cb_error - cb_tolerance,
            cb_error <= cb_tolerance,
            'HDF-103',
            'calculated_verified',
            'HDF-ASM-028',
            (
                f'Cb={weights.Cb:.12g}; independently recomputed with LOA, '
                'full molded beam, and design draft.'
            ),
        )
    )

    return pd.DataFrame(rows, columns=AUDITED_RESULT_COLUMNS)
