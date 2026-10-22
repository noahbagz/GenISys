"""Hydrostatic, stability, and flooding constraint rows for GenISys V1.1.

The legacy :mod:`tools.Hydrostatics` solver remains responsible for the
trapezoidal hydrostatic integrations and the binary floodable-length search.
This module reformats those calculated results, combines them with the newer
hydrostatic foundation checks, and exposes one eight-column constraint table
that can be concatenated with the other per-design results.

Floodable-length results remain explicitly named as whole-slice diagnostics.
The binary search is implemented and tested, but the present damage geometry
does not establish full CFR/SOLAS damage-stability compliance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.ConstraintIntegration import (
    AUDITED_RESULT_COLUMNS,
    build_hydrostatic_foundation_results,
)
from tools.Hydrostatics import floating_point_length_tolerance
from tools.SOLASSubdivisionAndDetectors import (
    cargo_hold_water_level_alarm_requirements,
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


# These rows remain in ``Hydrostatics.StabilityConstraints`` solely for the
# explicitly requested legacy-output compatibility path.  They must not enter
# the active V1.1 table:
#
# * the weather-GM row calls the old CFR 170.170 arithmetic with an unapproved
#   20 m x 20 m deckhouse proxy;
# * the LCB/LCG row is an older self-imposed screen, not a selected rule; and
# * the upright-MCT trim row duplicates the two named-condition constraints,
#   which now use simultaneous displacement and longitudinal-moment
#   equilibrium to obtain physical forward and aft drafts (HDF-ASM-033).
#
# Filtering at the adapter boundary preserves old files without allowing a
# guessed deckhouse or a contradictory trim method to become an ML target.
LEGACY_ROWS_EXCLUDED_FROM_V1P1 = frozenset(
    {
        'Legacy weather-GM diagnostic (not selected)',
        'LCB within 2.5 percent of LCG',
        'Trim within 1 percent of Draft',
    }
)


def derive_generated_side_construction(structural_elements):
    """Derive the current V1.1 single-/double-side arrangement from CAD rows.

    ``Inner_Side_Shell`` is a dedicated generated structural type.  Its
    perpendicular distance from the vertical ``Side_Shell`` plate is the
    modeled double-side width.  Actual ``Longitudinal_Bulkhead`` plates are
    independent cargo-space structure and do not affect this classification.
    """

    classes = structural_elements['Class'].astype(str)
    side_shells = structural_elements.loc[classes == 'Side_Shell']
    inner_side_shells = structural_elements.loc[
        classes == 'Inner_Side_Shell'
    ]

    if side_shells.empty:
        return {
            'Side_Construction': 'unresolved',
            'Double_Side_Width_m': np.nan,
            'Classification_Resolved': False,
            'Classification_Basis': (
                'no generated Side_Shell plate is available for '
                'classification'
            ),
        }

    if inner_side_shells.empty:
        return {
            'Side_Construction': 'single_side',
            'Double_Side_Width_m': np.nan,
            'Classification_Resolved': True,
            'Classification_Basis': (
                'no generated Inner_Side_Shell plate'
            ),
        }

    inner_side_ordinates_m = np.sort(
        inner_side_shells['y_loc'].astype(float).unique()
    )

    if len(inner_side_ordinates_m) != 1 or side_shells.empty:
        return {
            'Side_Construction': 'unresolved',
            'Double_Side_Width_m': np.nan,
            'Classification_Resolved': False,
            'Classification_Basis': (
                'requires one side-shell ordinate and one generated '
                'inner-side ordinate under the current V1.1 contract'
            ),
        }

    side_shell_ordinates_m = side_shells['y_loc'].astype(float).unique()

    if len(side_shell_ordinates_m) != 1:
        return {
            'Side_Construction': 'unresolved',
            'Double_Side_Width_m': np.nan,
            'Classification_Resolved': False,
            'Classification_Basis': (
                'multiple side-shell ordinates require explicit side-role '
                'metadata'
            ),
        }

    double_side_width_m = abs(
        float(side_shell_ordinates_m[0])
        - float(inner_side_ordinates_m[0])
    )

    return {
        'Side_Construction': 'double_side',
        'Double_Side_Width_m': double_side_width_m,
        'Classification_Resolved': True,
        'Classification_Basis': (
            'one generated Inner_Side_Shell plate; width is '
            'its transverse separation from Side_Shell'
        ),
    }


def _minimum(
    name,
    requirement,
    required,
    offered,
    units,
    result_class,
    numerical_tolerance=0.0,
):
    """Return one minimum-required hydrostatic/flooding result row."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': float(required),
        'Constraint Value': float(offered),
        'Units': units,
        'Constraint Eval': float(required) - float(offered),
        'Constraint Bit': bool(
            float(offered)
            >= float(required) - float(numerical_tolerance)
        ),
    }


def _maximum(
    name,
    requirement,
    allowed,
    offered,
    units,
    result_class,
    numerical_tolerance=0.0,
):
    """Return one maximum-permitted arrangement result row."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': float(allowed),
        'Constraint Value': float(offered),
        'Units': units,
        'Constraint Eval': float(offered) - float(allowed),
        'Constraint Bit': bool(
            float(offered)
            <= float(allowed) + float(numerical_tolerance)
        ),
    }


def _boolean(name, requirement, offered, result_class):
    """Return one Boolean numerical-method or arrangement check."""

    offered = bool(offered)

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': True,
        'Constraint Value': offered,
        'Units': 'Boolean',
        'Constraint Eval': int(not offered),
        'Constraint Bit': offered,
    }


def _metric(name, requirement, value, units, result_class):
    """Return a calculated metric without treating it as a pass/fail rule."""

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': np.nan,
        'Constraint Value': value,
        'Units': units,
        'Constraint Eval': np.nan,
        'Constraint Bit': np.nan,
    }


def _non_counted_diagnostic(name, requirement, result_class):
    """Return the approved zero-valued encoding for an inactive check.

    A blank bit is the machine-readable signal that the row must not enter
    feasibility counting.  Zero values avoid presenting a diagnostic result
    as an offered regulatory margin.  The actual flooding calculation remains
    available in ``FloodingDiagnosticSummary`` and ``FloodableLengthCurve``.
    """

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': 0.0,
        'Constraint Value': 0.0,
        'Units': 'not applicable to feasibility',
        'Constraint Eval': 0.0,
        'Constraint Bit': np.nan,
    }


def _project_basis_result(name, requirement, result_class):
    """Return a passed arrangement atom supported by an approved design basis.

    The generated three-hold model is not a whole-ship longitudinal structure.
    HDF-ASM-020 nevertheless fixes a continuous conventional double bottom as
    part of every GenISys design.  Zero is used on both sides because this row
    records categorical project-basis evidence, not a fabricated coordinate.
    """

    return {
        'Class': result_class,
        'Name': name,
        'Requirement': requirement,
        'Threshold Value': 0.0,
        'Constraint Value': 0.0,
        'Units': 'project-basis evidence',
        'Constraint Eval': 0.0,
        'Constraint Bit': True,
    }


class HydrostaticsV1p1:
    """Assemble the V1.1 hydrostatic and flooding table for one system."""

    def __init__(
        self,
        hull_dict,
        structure_evaluations,
        hydrostatics,
        weights,
        cargo_holds=None,
        flooding_summary=None,
    ):
        self.hull = hull_dict
        self.structure_evaluations = structure_evaluations
        self.hydrostatics = hydrostatics
        self.weights = weights
        self.cargo_holds = cargo_holds
        self.flooding_summary = (
            pd.DataFrame()
            if flooding_summary is None
            else flooding_summary
        )

    def _cargo_hold_water_level_requirement_metrics(self):
        """Publish the geometry-derived portions of SOLAS XII/12.1.

        HDF-033 contains two different questions which must not be collapsed:

        * the Rule defines alarm heights from the inner-bottom elevation and
          cargo-hold depth; and
        * an actual design must establish applicability, choose detector
          setpoints/coordinates, and demonstrate the approved installation.

        GenISys already supplies the first set of inputs for every generated
        cargo hold.  It does not model water-level detectors and the audit has
        not resolved the SOLAS XII/9.2 single-alarm exception or the source-
        wording conflict that occurs when 15 percent of hold depth exceeds
        2 m.  Consequently these values are emitted as *metrics* with a blank
        ``Constraint Bit``.  They are useful design requirements, but cannot
        be mistaken for detector-compliance passes.
        """

        if self.cargo_holds is None:
            return []

        cargo_hold_table = getattr(self.cargo_holds, 'CargoHolds', None)

        if (
            not isinstance(cargo_hold_table, pd.DataFrame)
            or cargo_hold_table.empty
        ):
            return []

        inner_bottom_elevation_m = (
            self._structure_value('Double_Bottom_Height')/1000.0
        )
        cargo_hold_depth_m = (
            float(self.hull['D']) - inner_bottom_elevation_m
        )
        rows = []

        for hold_index, hold in cargo_hold_table.reset_index(drop=True).iterrows():
            hold_number = int(hold.get('Hold_Number', hold_index))
            hold_label = f'Cargo hold {hold_number + 1:03d}'

            requirement = cargo_hold_water_level_alarm_requirements(
                inner_bottom_elevation_m=inner_bottom_elevation_m,
                cargo_hold_depth_m=cargo_hold_depth_m,
                # Do not infer the formal SOLAS XII applicability/exception
                # decisions from the generic ``bulk carrier`` project label.
                solas_xii_12_applies=None,
                solas_xii_9_2_single_alarm_exception_applies=None,
            )

            common_requirement = (
                'SOLAS XII/12.1; HDF-033 numeric requirement only; '
                'applicability and detector installation not evaluated'
            )

            rows.extend(
                [
                    _metric(
                        hold_label+' modeled depth',
                        common_requirement,
                        cargo_hold_depth_m,
                        'm',
                        'Water-Level Detector Requirement Metric',
                    ),
                    _metric(
                        hold_label+' first alarm height above inner bottom',
                        common_requirement,
                        requirement[
                            'First_Alarm_Height_Above_Inner_Bottom_m'
                        ],
                        'm',
                        'Water-Level Detector Requirement Metric',
                    ),
                    _metric(
                        hold_label+' first alarm elevation',
                        common_requirement,
                        requirement['First_Alarm_Elevation_m'],
                        'm above modeled baseline',
                        'Water-Level Detector Requirement Metric',
                    ),
                    _metric(
                        hold_label+' main alarm minimum height',
                        common_requirement,
                        requirement[
                            'Main_Alarm_Minimum_Height_Above_Inner_Bottom_m'
                        ],
                        'm above inner bottom',
                        'Water-Level Detector Requirement Metric',
                    ),
                    _metric(
                        hold_label+' main alarm maximum height',
                        common_requirement,
                        requirement[
                            'Main_Alarm_Maximum_Height_Above_Inner_Bottom_m'
                        ],
                        'm above inner bottom',
                        'Water-Level Detector Requirement Metric',
                    ),
                    _metric(
                        hold_label+' main alarm source bounds overlap',
                        common_requirement,
                        requirement['Main_Alarm_Source_Bounds_Overlap'],
                        'Boolean metric',
                        'Water-Level Detector Requirement Metric',
                    ),
                ]
            )

        return rows

    def _bulk_carrier_side_construction_metrics(self):
        """Publish generated side-arrangement inputs used by SOLAS XII/4.

        HDF-030 applies different construction-date branches to single-side
        and double-side bulk carriers.  The user expressly directed GenISys
        to derive that classification from the generated structure.  In the
        current generator, ``Inner_Side_Shell`` is the optional inner hull,
        while ``Side_Shell`` is the outer vertical side plate.  Actual
        ``Longitudinal_Bulkhead`` rows are a separate structural role.

        These rows resolve that *geometry input* and, for a double side, the
        SOLAS XII/4.2 offset gate ``min(B/5, 11.5 m)``.  They intentionally
        remain metrics with blank constraint bits.  The formal SOLAS length,
        summer-load-line condition, cargo-hold damage model, residual
        equilibrium, and opening-dependent survival proof are not present,
        so this must not look like a flooded-hold-compliance pass.
        """

        structural_elements = getattr(
            self.weights,
            'structural_elements',
            None,
        )

        if (
            not isinstance(structural_elements, pd.DataFrame)
            or structural_elements.empty
        ):
            return []

        classification = derive_generated_side_construction(
            structural_elements
        )
        offset_limit_m = min(float(self.hull['B'])/5.0, 11.5)

        if classification['Side_Construction'] == 'single_side':
            within_offset_gate = np.nan
        elif classification['Classification_Resolved']:
            within_offset_gate = bool(
                classification['Double_Side_Width_m'] < offset_limit_m
            )
        else:
            within_offset_gate = np.nan

        requirement = (
            'SOLAS XII/4.1-4.2; HDF-030 applicability geometry only; '
            'flooded-hold survival not evaluated'
        )

        return [
            _metric(
                'Generated bulk-carrier side construction',
                requirement,
                classification['Side_Construction'],
                'classification',
                'Flooded-Hold Applicability Metric',
            ),
            _metric(
                'Generated double-side width',
                requirement,
                classification['Double_Side_Width_m'],
                'm',
                'Flooded-Hold Applicability Metric',
            ),
            _metric(
                'SOLAS XII/4.2 double-side offset limit',
                requirement,
                offset_limit_m,
                'm',
                'Flooded-Hold Applicability Metric',
            ),
            _metric(
                'Generated inner side lies within SOLAS XII/4.2 offset',
                requirement,
                within_offset_gate,
                'Boolean metric',
                'Flooded-Hold Applicability Metric',
            ),
        ]

    def bulk_carrier_side_construction_audit_results(self):
        """Return status-aware HDF-030 applicability-input rows.

        The HDF audit row covers the complete one-hold flooding-survival rule.
        Only the generated side-construction inputs are currently calculated,
        so every row is explicitly partial and carries no constraint bit.
        """

        results = pd.DataFrame(
            self._bulk_carrier_side_construction_metrics(),
            columns=CONSTRAINT_COLUMNS,
        )

        if results.empty:
            return pd.DataFrame(columns=AUDITED_RESULT_COLUMNS)

        classification = derive_generated_side_construction(
            self.weights.structural_elements
        )
        results['Audit ID'] = 'HDF-030'
        results['Implementation Status'] = np.where(
            classification['Classification_Resolved'],
            'calculated_applicability_geometry_partial',
            'unresolved_generated_geometry',
        )
        results['Assumption IDs'] = 'HDF-ASM-004; HDF-ASM-008'
        results['Result Basis'] = (
            classification['Classification_Basis']
            + '. The side-construction and offset inputs are derived from '
              'generated structural elements. Formal SOLAS length, assigned '
              'summer load line, compartment-specific flooding, damaged '
              'equilibrium, and opening-dependent survival remain inactive.'
        )

        return results[AUDITED_RESULT_COLUMNS]

    def cargo_hold_water_level_audit_results(self):
        """Return status-aware HDF-033 rows for ``AuditedConstraints``.

        The status-aware table is the authoritative place to say that only a
        numeric portion of an audited rule is active.  A blank constraint bit
        is essential here: GenISys has calculated required alarm geometry but
        has not evaluated a detector design against it.
        """

        results = pd.DataFrame(
            self._cargo_hold_water_level_requirement_metrics(),
            columns=CONSTRAINT_COLUMNS,
        )

        if results.empty:
            return pd.DataFrame(columns=AUDITED_RESULT_COLUMNS)

        results['Audit ID'] = 'HDF-033'
        results['Implementation Status'] = (
            'calculated_numeric_requirement_partial'
        )
        results['Assumption IDs'] = 'HDF-ASM-012'
        results['Result Basis'] = (
            'Modeled inner-bottom elevation and molded hold depth determine '
            'the source alarm-height arithmetic. Applicability, exception, '
            'setpoint, detector coordinates, system approval, and the '
            'deep-hold source-bound interpretation remain unresolved.'
        )

        return results[AUDITED_RESULT_COLUMNS]

    def _structure_value(self, parameter_name):
        """Return one named structural evaluation value as a float."""

        return float(
            self.structure_evaluations.loc[
                self.structure_evaluations['Parameter'] == parameter_name,
                'Value',
            ].iloc[0]
        )

    def _double_bottom_arrangement_rows(self):
        """Evaluate the remaining geometric portions of SOLAS II-1/9.

        HDF-023 and HDF-ASM-020 select a continuous conventional double
        bottom with no wells, omitted regions, or bottom openings.  The
        existing foundation table already evaluates the II-1/9.2 height.
        The three-hold structural segment cannot supply whole-ship forward and
        aft endpoints.  The longitudinal rows therefore record the approved
        continuous-double-bottom project basis without comparing incompatible
        local and whole-ship coordinates.  Transverse and opening coverage are
        still checked from the generated element metadata.

        A missing geometry source retains the two project-basis rows while
        withholding the two checks that actually require CAD evidence.
        """

        # Do not emit forward/aft extent rows into design feasibility.  The
        # three-hold structural model has no whole-ship longitudinal endpoints,
        # and HDF-ASM-020 already records the approved continuous-double-bottom
        # project basis.  The former ``_project_basis_result`` rows are kept
        # out of the evaluated table instead of reporting assumptions as
        # satisfied numerical constraints.
        #
        # _project_basis_result('SOLAS double-bottom forward extent', ...)
        # _project_basis_result('SOLAS double-bottom aft extent', ...)
        project_basis_rows = []

        structural_elements = getattr(
            self.weights,
            'structural_elements',
            None,
        )
        bulkhead_indices = getattr(self.weights, 'bulkhead_locs', None)
        x_grid = getattr(self.weights, 'X', None)

        if (
            not isinstance(structural_elements, pd.DataFrame)
            or structural_elements.empty
            or bulkhead_indices is None
            or x_grid is None
        ):
            return project_basis_rows

        bulkhead_indices = np.asarray(bulkhead_indices, dtype=int)
        x_grid = np.asarray(x_grid, dtype=float)

        if len(bulkhead_indices) < 2 or len(x_grid) < 2:
            return project_basis_rows

        double_bottom_height_m = (
            self._structure_value('Double_Bottom_Height')/1000.0
        )

        # CadQuery stores both the top deck and inner bottom as ``Deck``.
        # Select the plate at the evaluated double-bottom elevation rather
        # than relying on row order or an object-ID spelling convention.
        deck_rows = structural_elements.loc[
            structural_elements['Class'].astype(str) == 'Deck'
        ].copy()

        if deck_rows.empty or 'z_loc' not in deck_rows:
            return project_basis_rows

        z_distance = np.abs(
            pd.to_numeric(deck_rows['z_loc'], errors='coerce')
            - double_bottom_height_m
        ).dropna()

        # Do not let a malformed/non-numeric deck ordinate make ``idxmin``
        # select a substitute row.  With no usable CAD ordinate there is no
        # geometric evidence on which to evaluate SOLAS II-1/9 (HDF-023).
        if z_distance.empty:
            return project_basis_rows

        inner_bottom = deck_rows.loc[[z_distance.idxmin()]].iloc[0]

        geometry_scale_m = max(
            1.0,
            float(np.max(np.abs(x_grid))),
            float(self.hull['B']),
            float(self.hull.get('D', double_bottom_height_m)),
        )
        tolerance_m = floating_point_length_tolerance(
            geometry_scale_m,
            operation_count=max(len(x_grid), len(structural_elements)),
        )

        if abs(float(inner_bottom['z_loc']) - double_bottom_height_m) > tolerance_m:
            # The structural table is present but contains no plate at the
            # declared double-bottom elevation.  Return explicit failed rows
            # instead of selecting the top deck as a false substitute.
            inner_bottom_found = False
        else:
            inner_bottom_found = True

        collision_index = int(bulkhead_indices[0])
        afterpeak_index = int(bulkhead_indices[-1])

        if (
            collision_index < 0
            or afterpeak_index >= len(x_grid)
            or collision_index >= afterpeak_index
        ):
            return project_basis_rows

        plate_length_m = abs(float(inner_bottom.get('L1', np.nan)))

        # Hydrostatics.Y stores positive-side half-breadths.  Interpolate the
        # shell breadth at the inner-bottom elevation at every station in the
        # required longitudinal region, then use the largest breadth so the
        # plate must protect the entire modeled bottom-to-bilge envelope.
        required_half_breadth_m = float(self.hull['B'])/2.0

        if (
            hasattr(self.hydrostatics, 'Y')
            and hasattr(self.hydrostatics, 'Z')
        ):
            y_grid = np.asarray(self.hydrostatics.Y, dtype=float)
            z_grid = np.asarray(self.hydrostatics.Z, dtype=float)

            if y_grid.shape == (len(x_grid), len(z_grid)):
                station_half_breadths = np.array(
                    [
                        np.interp(
                            double_bottom_height_m,
                            z_grid,
                            y_grid[station_index],
                        )
                        for station_index in range(
                            collision_index,
                            afterpeak_index + 1,
                        )
                    ],
                    dtype=float,
                )
                required_half_breadth_m = float(
                    np.max(station_half_breadths)
                )

        plate_area_m2 = float(inner_bottom.get('area_pannel', np.nan))
        provided_half_breadth_m = (
            plate_area_m2/plate_length_m
            if plate_length_m > 0.0
            else np.nan
        )

        opening_value = inner_bottom.get('Lightening_Holes', np.nan)
        opening_text = str(opening_value).strip().lower()
        no_modeled_openings = (
            pd.isna(opening_value)
            or opening_text in {'', 'none', 'nan', '[]'}
        )

        # If no plate exists at the required elevation, make every geometric
        # component fail without fabricating substitute dimensions.
        if not inner_bottom_found:
            provided_half_breadth_m = -np.inf
            no_modeled_openings = False

        return project_basis_rows + [
            _minimum(
                'SOLAS double-bottom transverse protection to bilge',
                'SOLAS II-1/9.1; HDF-023',
                required_half_breadth_m,
                provided_half_breadth_m,
                'm half-breadth in mirrored hull model',
                'Subdivision Arrangement',
                numerical_tolerance=tolerance_m,
            ),
            _boolean(
                'SOLAS double-bottom without wells or omitted regions',
                'SOLAS II-1/9.1; HDF-023; HDF-ASM-020',
                no_modeled_openings,
                'Subdivision Arrangement',
            ),
        ]

    def _hydrostatic_metrics(self):
        """Expose primary design-derived hydrostatic values as metric rows."""

        draft_m = float(self.hull['d'])
        waterplane_area_m2, lcf_m, ixx_m4, iyy_m4 = (
            self.hydrostatics.calc_WaterPlane(draft_m)
        )

        return [
            _metric(
                'Design draft',
                'GenISys evaluated loading condition',
                draft_m,
                'm',
                'Hydrostatic Metric',
            ),
            _metric(
                'Displacement volume at design draft',
                'Composite trapezoidal integration',
                float(self.hull['Vol_Disp']),
                'm^3',
                'Hydrostatic Metric',
            ),
            _metric(
                'Waterplane area at design draft',
                'Composite trapezoidal integration',
                waterplane_area_m2,
                'm^2',
                'Hydrostatic Metric',
            ),
            _metric(
                'Longitudinal center of flotation',
                'Composite trapezoidal integration',
                lcf_m,
                'm',
                'Hydrostatic Metric',
            ),
            _metric(
                'Transverse waterplane second moment',
                'Composite trapezoidal integration',
                ixx_m4,
                'm^4',
                'Hydrostatic Metric',
            ),
            _metric(
                'Longitudinal waterplane second moment',
                'Composite trapezoidal integration',
                iyy_m4,
                'm^4',
                'Hydrostatic Metric',
            ),
            _metric(
                'Forward perpendicular location',
                'Bow intersection with evaluated design waterline',
                float(self.hull['Forward_Perpendicular_X_m']),
                'm',
                'Hydrostatic Metric',
            ),
            _metric(
                'Aft perpendicular location',
                'Modeled stern/transom plane',
                float(self.hull['Aft_Perpendicular_X_m']),
                'm',
                'Hydrostatic Metric',
            ),
            _metric(
                'Length between perpendiculars',
                'Derived forward and aft perpendicular locations',
                float(self.hull['Length_Between_Perpendiculars_m']),
                'm',
                'Hydrostatic Metric',
            ),
        ]

    def _longitudinal_load_equilibrium_rows(self):
        """Publish the numerical proof behind still-water load curves.

        These are internal numerical-quality rows, not class-rule
        requirements.  They make it impossible for a shear-force or bending-
        moment result to appear without the force-and-moment equilibrium state
        from which it was integrated.
        """

        diagnostics = getattr(
            self.weights,
            'LoadClosureDiagnostics',
            {},
        )

        if not diagnostics:
            return []

        rows = [
            _boolean(
                'Longitudinal load equilibrium solved',
                (
                    'GenISys simultaneous displacement-volume and '
                    'longitudinal-first-moment equilibrium'
                ),
                diagnostics['Longitudinal_Equilibrium_Solved'],
                'Longitudinal Load Numerical Check',
            )
        ]

        metric_definitions = [
            (
                'Load-equilibrium forward draft',
                'Load_Equilibrium_Forward_Draft_m',
                'm',
            ),
            (
                'Load-equilibrium aft draft',
                'Load_Equilibrium_Aft_Draft_m',
                'm',
            ),
            (
                'Load-equilibrium physical trim',
                'Load_Equilibrium_Trim_m',
                'm, aft minus forward',
            ),
            (
                'Normalized displacement-volume residual',
                'Normalized_Displacement_Volume_Residual',
                'dimensionless',
            ),
            (
                'Normalized displacement first-moment residual',
                'Normalized_Displacement_First_Moment_Residual',
                'dimensionless',
            ),
            (
                'Distributed mass residual',
                'Mass_Distribution_Residual_kg',
                'kg',
            ),
            (
                'Distributed first-moment residual',
                'Distribution_First_Moment_Residual_kg_m',
                'kg*m',
            ),
            (
                'Net vertical-force residual',
                'Net_Vertical_Force_Residual_N',
                'N',
            ),
            (
                'Net vertical first-moment residual',
                'Net_Vertical_First_Moment_Residual_Nm',
                'N*m',
            ),
            (
                'Aft shear-force endpoint residual',
                'Aft_Shear_Force_Residual_N',
                'N',
            ),
            (
                'Aft bending-moment endpoint residual',
                'Aft_Bending_Moment_Residual_Nm',
                'N*m',
            ),
        ]

        for name, key, units in metric_definitions:
            if key not in diagnostics:
                continue

            rows.append(
                _metric(
                    name,
                    'GenISys composite-trapezoidal load closure',
                    float(diagnostics[key]),
                    units,
                    'Longitudinal Load Metric',
                )
            )

        return rows

    def _flooding_rows(self):
        """Publish HDF-083 as a non-counted numerical diagnostic.

        HDF-083 does not establish regulatory flooding compliance.  Its raw
        solver values remain in the flooding summary and curve outputs.  The
        three former pass/fail rows stay visible by name in the constraint
        table, but use the approved N/A encoding so they cannot create design
        failures or be mistaken for passed regulatory requirements.
        """

        rows = []

        for _, summary in self.flooding_summary.iterrows():
            permeability = float(summary['Permeability'])
            suffix = f' at permeability {permeability:.2f}'

            # Retain the calculated tolerance as a metric below.  It is not a
            # feasibility allowance and therefore does not restore a bit on
            # any of these diagnostic rows.
            numerical_tolerance_m = float(
                summary.get('Numerical Feasibility Tolerance', 0.0)
            )

            rows.append(
                _non_counted_diagnostic(
                    'Floodable-length monotonic boundary'+suffix,
                    (
                        'GenISys HDF-083 binary-search diagnostic; '
                        'not applicable to design feasibility'
                    ),
                    'Flooding Numerical Check',
                )
            )
            rows.append(
                _non_counted_diagnostic(
                    'Minimum single-compartment floodable-length margin'+suffix,
                    (
                        'GenISys HDF-083 whole-slice diagnostic; '
                        'not applicable to design feasibility'
                    ),
                    'Flooding Diagnostic',
                )
            )
            rows.append(
                _non_counted_diagnostic(
                    'Minimum two-compartment floodable-length margin'+suffix,
                    (
                        'GenISys HDF-083 whole-slice diagnostic; '
                        'not applicable to design feasibility'
                    ),
                    'Flooding Diagnostic',
                )
            )

            # These calculated values make the numerical resolution and
            # boundary used by a particular design recoverable directly from
            # its constraint CSV without changing the legacy column schema.
            rows.extend(
                [
                    _metric(
                        'Flooding numerical feasibility tolerance'+suffix,
                        'HDF-083 / HDF-ASM-028 / HDF-ASM-029',
                        numerical_tolerance_m,
                        'm',
                        'Flooding Metric',
                    ),
                    _metric(
                        'Flooding critical draft'+suffix,
                        'GenISys HDF-083 whole-slice diagnostic',
                        float(summary['Critical Draft']),
                        'm',
                        'Flooding Metric',
                    ),
                    _metric(
                        'Flooding longitudinal evaluation step'+suffix,
                        'GenISys HDF-083 binary-search resolution',
                        int(summary['Evaluation Step']),
                        'hydrostatic-grid indices',
                        'Flooding Metric',
                    ),
                    _metric(
                        'Flooding evaluated station count'+suffix,
                        'GenISys HDF-083 binary-search resolution',
                        int(summary['Sample Count']),
                        'stations',
                        'Flooding Metric',
                    ),
                    _metric(
                        'Flooding monotonicity candidate evaluations'+suffix,
                        'GenISys HDF-083 binary-search validation',
                        int(summary['Monotonicity Candidate Evaluations']),
                        'candidate damage extents',
                        'Flooding Metric',
                    ),
                ]
            )

        return rows

    def evaluate(self):
        """Return legacy, updated, and diagnostic hydrostatic rows."""

        # The legacy table currently contains three results that must remain
        # available only through ``ShipSystemDesign.Constraints``:
        #
        # * HDF-051 is a weather-GM route expressly not selected for this
        #   new unrestricted-ocean bulk carrier;
        # * the 2.5-percent LCB/LCG band is a superseded project heuristic;
        # * the single full-cargo trim row uses the old upright linear model.
        #
        # Publishing any of them here with a Boolean bit would change global
        # V1.1 feasibility.  The active replacements are two-condition
        # simultaneous force/moment equilibrium and the physical departure
        # +/-1-percent-draft trim row in
        # Loading_Condition_Constraints_v1p1.py.  Arrival has no independent
        # trim predicate under HDF-ASM-038.
        legacy = self.hydrostatics.StabilityConstraints[
            CONSTRAINT_COLUMNS
        ].copy()

        # Preserve these diagnostics in ``ShipSystemDesign.Constraints`` but
        # exclude them from ``ConstraintsV1p1`` for the reasons documented by
        # ``LEGACY_ROWS_EXCLUDED_FROM_V1P1`` above.  Selection by exact stable
        # name avoids suppressing unrelated legacy results accidentally.
        legacy = legacy.loc[
            ~legacy['Name'].astype(str).isin(
                LEGACY_ROWS_EXCLUDED_FROM_V1P1
            )
        ].copy()

        foundation = build_hydrostatic_foundation_results(
            self.hull,
            self.structure_evaluations,
            self.hydrostatics,
            self.weights,
        )

        # Keep N/A foundation rows in the individual eight-column output.  The
        # status-aware builder encodes them as zero threshold/value/residual
        # with a blank bit, which is unambiguously excluded from feasibility.
        foundation = foundation[CONSTRAINT_COLUMNS]

        calculated = pd.DataFrame(
            self._hydrostatic_metrics()
            + self._longitudinal_load_equilibrium_rows()
            + self._double_bottom_arrangement_rows()
            + self._bulk_carrier_side_construction_metrics()
            + self._cargo_hold_water_level_requirement_metrics()
            # HDF-083 remains available in FloodingDiagnosticSummary and
            # FloodableLengthCurve.  Its former rows are intentionally not
            # inserted into the constraint-evaluation CSV because the
            # whole-slice calculation is a diagnostic, not a regulatory
            # feasibility requirement.
            # + self._flooding_rows()
            ,
            columns=CONSTRAINT_COLUMNS,
        )

        # Exclude empty frames so pandas does not infer result dtypes from
        # all-NA placeholders.  This also keeps targeted smoke tests warning
        # free without adding schema-changing validation columns.
        result_frames = [
            frame
            for frame in (legacy, foundation, calculated)
            if not frame.empty
        ]
        result = pd.concat(result_frames, ignore_index=True)

        return result[CONSTRAINT_COLUMNS].drop_duplicates(
            subset=['Class', 'Name', 'Requirement'],
            keep='first',
        ).reset_index(drop=True)


def evaluate_hydrostatic_constraints_v1p1(
    hull_dict,
    structure_evaluations,
    hydrostatics,
    weights,
    cargo_holds=None,
    flooding_summary=None,
):
    """Functional entry point used by the integrated system evaluator."""

    return HydrostaticsV1p1(
        hull_dict,
        structure_evaluations,
        hydrostatics,
        weights,
        cargo_holds=cargo_holds,
        flooding_summary=flooding_summary,
    ).evaluate()


def build_hdf033_detector_requirement_results(
    hull_dict,
    structure_evaluations,
    cargo_holds,
):
    """Build audited partial-result rows without running hydrostatics twice."""

    adapter = HydrostaticsV1p1(
        hull_dict,
        structure_evaluations,
        hydrostatics=None,
        weights=None,
        cargo_holds=cargo_holds,
    )

    return adapter.cargo_hold_water_level_audit_results()


def build_hdf030_side_construction_results(hull_dict, structural_elements):
    """Build audited HDF-030 applicability inputs without rerunning hydro."""

    weights = type(
        '_GeneratedStructureAdapter',
        (),
        {'structural_elements': structural_elements},
    )()
    adapter = HydrostaticsV1p1(
        hull_dict,
        structure_evaluations=None,
        hydrostatics=None,
        weights=weights,
    )

    return adapter.bulk_carrier_side_construction_audit_results()
