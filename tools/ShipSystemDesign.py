#This Script combines all tools togther to create a ship system design
#It will be used in the optimization framework to evaluate different ship designs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.HullDesign import HullDesign
from tools.Hydrostatics import Hydrostatics, evaluate_trim_constraint
from tools.StructureCalcs import StructureCalcs
from tools.CargoHolds import CargoHolds
from tools.Weights import LongitudinalEquilibriumError, Weights
from tools.CargoAllocation import (
    CargoAllocationSearchError,
    capacity_extreme_allocation,
    direct_required_lcg_allocation,
    deterministic_trim_search,
)
from tools.HatchCovers import build_hatch_cover_components
from tools.EarlyDesignMasses import build_simple_deckhouse_component
from tools.LoadingConditions import (
    build_condition_specific_fuel_tank_states,
    build_loading_condition_component_states,
    summarize_loading_conditions,
)
from tools.UprightStability import evaluate_upright_loading_conditions
from tools.Loading_Condition_Constraints_v1p1 import (
    evaluate_loading_condition_constraints_v1p1,
)
from tools.LoadingConditionLoads import (
    DEPARTURE_LOADING_CONDITION_IDS,
    build_named_condition_still_water_envelope,
    evaluate_named_condition_longitudinal_loads,
)
from tools.ConstraintIntegration import (
    build_additional_structural_results,
    build_hydrostatic_foundation_results,
    build_verified_structural_results,
)
from tools.Structure_Constraints_v1p1 import (
    CONSTRAINT_COLUMNS as V1P1_CONSTRAINT_COLUMNS,
    evaluate_structure_constraints_v1p1,
    merge_atomic_structural_audit_results,
)
from tools.Hydrostatics_v1p1 import (
    build_hdf030_side_construction_results,
    build_hdf033_detector_requirement_results,
    evaluate_hydrostatic_constraints_v1p1,
)
from tools.Powering_Constraints_v1p1 import (
    evaluate_powering_constraints_v1p1,
)
from tools.EconomicInputs import (
    DEFAULT_ECONOMIC_INPUTS,
    resolve_economic_inputs,
)
from tools.EvaluationErrors import PhysicalConstraintFailure
from tools.System_Feasibility_Constraints_v1p1 import (
    evaluate_cargo_equilibrium_constraint,
    evaluate_total_cargo_capacity_constraint,
)

from scipy.optimize import minimize





class ShipSystemDesign:
    '''
    This class combines all of the ship design tools to create a ship system design

    Key Modules: 
    -Hull Design
    -Hydrostatics and Stability
    -Structure Design
    -ABS Part 3 Constraints
    -Cargo Hold Design
    -Weights Calculation


    Evans Design Spiral 1959 Sequence: 

    1) Hull Form
    2) Primary Hydrostatics
    3) Flooding 
    4) Arrangements
    5) Structural Strength Eval 
    6) Resistance and Powering
    7) Distributed Weights and Centers
    8) Trim/Intact Stability
    9) Damage Stability
    10) cost and performance eval
    Repeat

    Sequence based on the information uncovered in the modules:

    1) Hull Form Generation
    2) Powering/Engine/Fuel Costs Inferred
    3) Structural Design -> Weights, SM, Spacing, and Constraint Calculations
    4) Bulkhead Assignments
    5) Arrangements -> Cargo Hold Assignments
    5a) Sectional Area Calculations
    5b) Lightship and Cargo Weights Calculations
    6) Distributed Weights and Centers
    7) Strutural Constraints Evaluation
    8) Hydrostatics and Stability
    9) Trim/Intact Stability Constraint Evaluation
    10) Flooding
    11) Damage Stability
    12) Cost and Performance Eval 

    '''

    def __init__(
        self,
        system_params,
        structural_elements: pd.DataFrame,
        file_id: str = 'Test',
        structure_id: str = 'rndStruct9',
        save_path: str = './Ship_System_Designs/Test/',
        run_flooding_diagnostic: bool = False,
        flooding_step: int = 15,
        flooding_permeabilities=(0.95,),
        powering_inputs=None,
        early_design_mass_inputs=None,
        economic_inputs=None,
    ):
        
        
        self.system_params = system_params
        self.Constraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])

        # Keep audit-qualified results and coarse flooding diagnostics separate
        # from the legacy aggregate table.  The legacy table remains available
        # for compatibility, but it contains rules with mixed audit status.
        self.AuditedConstraints = pd.DataFrame()
        self.StructureConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.SystemArrangementConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.SystemFeasibilityConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.HydrostaticConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.PoweringConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.ConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )
        self.FloodingDiagnosticSummary = pd.DataFrame()
        self.FloodableLengthCurve = pd.DataFrame()
        self.CargoOptimizationSummary = pd.DataFrame()
        self.EconomicPerformanceSummary = pd.DataFrame()
        self.HatchCoverComponents = pd.DataFrame()
        self.DeckhouseComponents = pd.DataFrame()
        self.LightshipMarginComponent = pd.DataFrame()
        self.EarlyDesignMassInputs = (
            {} if early_design_mass_inputs is None
            else dict(early_design_mass_inputs)
        )

        # Economic inputs are scenario assumptions, not geometric design
        # coordinates.  Resolve them once per ship so every performance value
        # and its provenance can be retained with the evaluated system while
        # the fixed 189-coordinate design vector remains unchanged.
        self.Economic_Inputs = resolve_economic_inputs(economic_inputs)

        unknown_mass_inputs = sorted(
            set(self.EarlyDesignMassInputs)
            - {"deckhouse", "lightship_margin"}
        )

        if unknown_mass_inputs:
            raise ValueError(
                "Unknown early-design mass input group(s): "
                + ", ".join(unknown_mass_inputs)
            )
        self.LoadingConditionComponentStates = pd.DataFrame()
        self.LoadingConditionFuelTankStates = pd.DataFrame()
        self.LoadingConditions = pd.DataFrame()
        self.UprightLoadingConditionEvaluations = pd.DataFrame()
        self.NamedConditionLongitudinalLoadSummary = pd.DataFrame()
        self.NamedConditionLongitudinalLoadArrays = {}
        self.StructuralLoadEnvelope = {}
        self.LoadingConditionConstraintsV1p1 = pd.DataFrame(
            columns=V1P1_CONSTRAINT_COLUMNS
        )

        self.rho_cargo = 2400.0 # Density of cargo in kg/m^3 (assumed to be iron ore for bulk carrier)
        
        #import system parameter labels
        hull_param_labels = pd.read_csv('./tools/ShipSystemDesignParameters.csv')
        
        self.path = save_path
        self.file_name = file_id + '_struct_' + structure_id

        #Extract Hull Parameters
        hull_idx = hull_param_labels[hull_param_labels['Class'] == 'Hull']['idx'].to_numpy()
        #print('hull_idx: ', hull_idx.shape)
        self.hull_params = self.system_params[hull_idx]
        #print('hull_params: ', self.hull_params.shape)
        draft_idx = hull_param_labels[hull_param_labels['Name'] == 'Draft_frac']['idx'].values[0]
        self.draft_frac = self.system_params[draft_idx] 
        Fn_idx = hull_param_labels[hull_param_labels['Name'] == 'Fn_ship']['idx'].values[0]
        self.Fn_ship = self.system_params[Fn_idx]
        Voyage_Range_idx = hull_param_labels[hull_param_labels['Name'] == 'Voyage_Range']['idx'].values[0]
        self.Voyage_Range = self.system_params[Voyage_Range_idx]

        
        

        #Run Hull Design Module
        # ``powering_inputs`` is an optional dataset-scenario mapping.  When it
        # is absent, HullDesign retains the existing documented global
        # assumptions.  Supplying it parameterizes only values already used by
        # the resistance/power/fuel equations; no propeller or certified
        # engine characteristics are synthesized.
        self.HD = HullDesign(
            self.hull_params,
            self.Fn_ship,
            self.draft_frac,
            self.Voyage_Range,
            powering_inputs=powering_inputs,
        )
        if self.HD.Hull_Dict is None:
            #Hull Design Constraints not met, end evaluation
            self.Weights = None
            self.Hydrostatics = None
            self.SC = None
            self.CargoHolds = None
            
            #copy constraints from hull design to self.Constraints
            self.Constraints = self.HD.HullConstraints.copy()
            self.ConstraintsV1p1 = self.HD.HullConstraints[
                V1P1_CONSTRAINT_COLUMNS
            ].copy()
            print("Warning: Hull Design Constraints Not Met, Ending Evaluation")
            return
        self.Hull_Dict = self.HD.Hull_Dict

        #print('Hull Dict: ', self.Hull_Dict)
        self.Constraints = self.HD.HullConstraints.copy()
        
        # Extract Structure Parameters
        structure_idx = hull_param_labels[hull_param_labels['Class'] == 'Structure']['idx'].to_numpy()
        self.structure_params = self.system_params[structure_idx]
        self.structural_elements = structural_elements

        #print('struct_params: ', self.structure_params.shape)

        #Run Structure Design Module
        self.SC = StructureCalcs(self.HD.Hull_Dict, self.structural_elements, self.structure_params)
        self.SC.calc_Structure_Values()

        #Save Structure_Dict
        #self.SC.Struct_Dict.to_csv(self.path + self.file_name + '_Structure_Evaluations.csv', index=False)


        #Extract Bulkhead Spacing and Structural Parameters
        bulkhead_idx = hull_param_labels[hull_param_labels['Class'] == 'Bulkhead_Spacing']['idx'].to_numpy()
      
        self.bulkhead_params = self.system_params[bulkhead_idx]
    
        #Evaluate Bulkheads and Spacing
        self.CargoHolds = CargoHolds(self.structure_params, self.rho_cargo, self.bulkhead_params, fuel_weight = self.Hull_Dict['Fuel_Weight_kg'])

        #5a) Sectional Area Calculations
        self.HS = Hydrostatics(self.HD.hull, self.Hull_Dict)

        # HDF-ASM-030/HDF-ASM-031: publish both perpendiculars and their
        # distance without overwriting Hull_Dict['L'], which remains the LOA
        # proxy approved specifically for structural-rule calculations.
        self.Hull_Dict['Forward_Perpendicular_X_m'] = self.HS.X_FP
        self.Hull_Dict['Aft_Perpendicular_X_m'] = self.HS.X_AP
        self.Hull_Dict['Length_Between_Perpendiculars_m'] = self.HS.LBP

        self.CargoHolds.upload_PC_SA_values(self.HS.X, self.HS.Z, self.HS.Y, self.HS.sectional_Areas, self.HS.SA_z_centroid)
        
        self.CargoHolds.calc_Bulkhead_Locations()
        self.CargoHolds.calc_CargoHold_Offsets()
        self.CargoHolds.calc_CargoAndBallast_SectionalAreas()

        # HDF-ASM-021 through HDF-ASM-025: represent one preliminary cover over
        # every generated cargo hold.  This supplies weight and CG fidelity for
        # the systems model; it is not a hatch-cover structural certification.
        self.HatchCoverComponents = build_hatch_cover_components(
            self.SC.Struct_Dict,
            self.CargoHolds,
            self.Hull_Dict['D'],
        )

        # Deckhouse dimensions, placement, and equivalent areal mass remain
        # project decisions.  If a dataset scenario supplies every required
        # value, build its mass row; otherwise keep the feature inactive and
        # do not invent a geometry.  Spatial fit is deliberately deferred.
        deckhouse_inputs = self.EarlyDesignMassInputs.get("deckhouse")

        if deckhouse_inputs is not None:
            # HDF-ASM-013 approves YCG = 0 only for symmetric intact loading.
            # The downstream condition solver does not yet carry transverse
            # component moments, so reject an asymmetric optional component
            # instead of silently publishing a centerline ship CG.
            if float(deckhouse_inputs.get("centerline_y_m", 0.0)) != 0.0:
                raise ValueError(
                    "The integrated deckhouse requires centerline_y_m = 0 "
                    "until transverse loading conditions are implemented."
                )

            self.DeckhouseComponents = build_simple_deckhouse_component(
                self.HS.X,
                **dict(deckhouse_inputs),
            )

        margin_inputs = self.EarlyDesignMassInputs.get("lightship_margin")

        if (
            margin_inputs is not None
            and float(margin_inputs.get("margin_tcg_m", 0.0)) != 0.0
        ):
            raise ValueError(
                "The integrated lightship margin requires margin_tcg_m = 0 "
                "until transverse loading conditions are implemented."
            )

        active_additional_lightship_tables = [
            table
            for table in (
                self.HatchCoverComponents,
                self.DeckhouseComponents,
            )
            if not table.empty
        ]
        additional_lightship_components = (
            pd.concat(
                active_additional_lightship_tables,
                ignore_index=True,
            )
            if active_additional_lightship_tables
            else pd.DataFrame()
        )

        #5b) Lightship and Cargo Weights Calculations
        # Need this from lightship weights
        self.Weights = Weights(
            self.HD.hull,
            self.structural_elements,
            self.Hull_Dict['Vol_Disp'],
            self.Hull_Dict['d'],
            self.Hull_Dict['Engine_Power'],
            self.Hull_Dict['Days_at_Sea'],
            # Use exactly the voyage-plus-reserve fuel mass calculated by the
            # active powering scenario.  Recalculating it here with the old
            # constants would make fuel-tank loading and available cargo
            # disagree whenever the JSON scenario overrides SFOC, auxiliary
            # demand, or reserve days.
            fuel_weight_kg=self.Hull_Dict['Fuel_Weight_kg'],
        )
        self.Weights.upload_Bulkhead_info(self.CargoHolds.Bulkhead_Locations)
        self.Weights.upload_PC_SA_values(self.HS.X, self.HS.Z, self.HS.Y, self.HS.sectional_Areas, self.HS.SA_z_centroid)
        self.Weights.calc_unit_Bulkhead_Weight()
        self.Weights.calc_Bulkhead_Weights()
        self.Weights.calc_unit_Long_Struct_Weight()
        self.Weights.calc_distributed_Long_Struct_Weight()
        try:
            self.Weights.calc_NonStruct_Weights(
                additional_lightship_components=(
                    additional_lightship_components
                ),
                lightship_margin_inputs=margin_inputs,
            )
        except PhysicalConstraintFailure as error:
            # Preserve both the already-passed hull checks and every physical
            # mass predicate evaluated before this dependency stop.  The
            # batch runner can then write useful partial constraint data for
            # an infeasible sample instead of reducing it to one error row.
            physical_rows = error.constraint_rows

            if physical_rows is None:
                physical_rows = pd.DataFrame(
                    columns=V1P1_CONSTRAINT_COLUMNS
                )

            error.constraint_rows = pd.concat(
                [
                    self.HD.HullConstraints[V1P1_CONSTRAINT_COLUMNS],
                    physical_rows[V1P1_CONSTRAINT_COLUMNS],
                ],
                ignore_index=True,
            )
            raise

        self.SystemFeasibilityConstraintsV1p1 = (
            self.Weights.SystemFeasibilityConstraintsV1p1.copy()
        )
        self.LightshipMarginComponent = (
            self.Weights.LightshipMarginComponent.copy()
        )

        self.CargoHolds.upload_CargoWeight(self.Weights.cargo_weight)

        #Optimize Cargo Distribution
        optimized_cargo_fractions = self.Optimize_Cargo_Distribution()

        self.CargoHolds.upload_CargoWeightFractions(optimized_cargo_fractions)

        self.system_params[bulkhead_idx[6:6+len(optimized_cargo_fractions)]] = optimized_cargo_fractions

        self.CargoHolds.calc_CargoHoldAndBallast_Weights()
        self.CargoHolds.calc_FuelTank_Weights()

        self.CargoHolds.calc_Bulkhead_Constraints(self.Hull_Dict)
        self.Constraints = pd.concat([self.Constraints, self.CargoHolds.BulkheadConstraints], ignore_index=True)

        # Retain the physical arrangement checks while excluding only the
        # audited-out passenger-vessel subdivision formula in 46 CFR 171.065(e).
        # Engine-room fit and cargo/ballast/fuel capacity are necessary system
        # feasibility constraints even though they are not ABS scantling rules.
        self.SystemArrangementConstraintsV1p1 = (
            self.CargoHolds.BulkheadConstraints.loc[
                self.CargoHolds.BulkheadConstraints['Requirement']
                != 'CFR 46 Part 171.065e',
                V1P1_CONSTRAINT_COLUMNS,
            ].copy()
        )


        #6) Distributed Weights and Centers
        self.Weights.upload_CargoBallastFuel_Weights_values(self.CargoHolds.CargoHolds, self.CargoHolds.BallastTanks, self.CargoHolds.Fuel_Tank, self.CargoHolds.rho_cargo)

        # HDF-ASM-034: preserve the approved full-load departure and arrival
        # conditions independently of the 4 x 5 dataset operating points.  The
        # component-state table makes every applied cargo and fuel fraction
        # auditable before hydrostatic condition evaluation.
        self.LoadingConditionFuelTankStates = (
            build_condition_specific_fuel_tank_states(self.CargoHolds)
        )
        self.LoadingConditionComponentStates = (
            build_loading_condition_component_states(
                self.Weights.Weights,
                condition_specific_fuel_tanks=(
                    self.LoadingConditionFuelTankStates
                ),
            )
        )
        self.LoadingConditions = summarize_loading_conditions(
            self.LoadingConditionComponentStates
        )

        # Propagate both full-load conditions through the same simultaneous
        # force-and-longitudinal-moment equilibrium used by the primary
        # still-water structural load case.  Scalar extrema and closure
        # evidence are retained for every condition; the complete arrays stay
        # available in memory without multiplying the production NPZ payload.
        (
            self.NamedConditionLongitudinalLoadSummary,
            self.NamedConditionLongitudinalLoadArrays,
        ) = evaluate_named_condition_longitudinal_loads(
            self.Weights,
            self.LoadingConditionComponentStates,
            condition_specific_fuel_tanks=(
                self.LoadingConditionFuelTankStates
            ),
        )

        # Evaluate the hydrostatic equilibrium, uncorrected GM, and physical
        # trim for every approved named condition.  The result deliberately
        # leaves corrected GM unevaluated until individual tank free-surface
        # groups are available; it therefore cannot be mistaken for complete
        # 46 CFR 170.285 or 2008 IS Code compliance.
        self.UprightLoadingConditionEvaluations = (
            evaluate_upright_loading_conditions(
                self.HS,
                self.LoadingConditions,
                free_surface_corrections=None,
            )
        )

        # HDF-ASM-033/HDF-ASM-038: evaluate the approved one-percent-of-draft
        # trim tolerance for full-cargo departure only.  Arrival remains a
        # named equilibrium/load state but has no independent trim constraint;
        # its approved compensating-ballast allocation awaits an explicit aft
        # trim-tank designation.  The corrected-GM row is intentionally absent
        # until explicit tank-group free-surface geometry is available, so
        # missing stability evidence cannot be recorded as a regulatory pass.
        self.LoadingConditionConstraintsV1p1 = (
            evaluate_loading_condition_constraints_v1p1(
                self.UprightLoadingConditionEvaluations,
                longitudinal_load_summary=(
                    self.NamedConditionLongitudinalLoadSummary
                ),
            )
        )

        #Save Weights_Dict
        #self.Weights.Weights.to_csv(self.path + self.file_name + '_Weights_Evaluations.csv', index=False)

        self.Weights.calc_Distributed_Load('Full Cargo')
        self.Weights.calc_ShearForce()
        self.Weights.calc_BendingMoment()

        # Keep both named-condition calculations as hydrostatic/numerical
        # evidence, but apply only departure to active hull-girder loading.
        # The user explicitly deferred arrival load bending moment until the
        # compensating ballast distribution has a physically modeled aft tank.
        # Filtering both the scalar envelope and station-by-station arrays at
        # this boundary prevents the incomplete unballasted arrival state from
        # governing a structural rule by accident.
        self.ActiveStructuralLoadingConditionIDs = tuple(
            sorted(DEPARTURE_LOADING_CONDITION_IDS)
        )
        active_structural_load_summary = (
            self.NamedConditionLongitudinalLoadSummary.loc[
                self.NamedConditionLongitudinalLoadSummary[
                    'Loading_Condition_ID'
                ].astype(str).isin(DEPARTURE_LOADING_CONDITION_IDS)
            ].copy()
        )
        self.ActiveStructuralNamedConditionLoadArrays = {
            condition_id: arrays
            for condition_id, arrays
            in self.NamedConditionLongitudinalLoadArrays.items()
            if str(condition_id) in DEPARTURE_LOADING_CONDITION_IDS
        }
        self.StructuralLoadEnvelope = (
            build_named_condition_still_water_envelope(
                active_structural_load_summary,
                required_condition_ids=DEPARTURE_LOADING_CONDITION_IDS,
            )
        )
        self.distributed_loading_dict = {
            'Fnsw': self.StructuralLoadEnvelope['Fnsw'], # in kN
            'Fpsw': self.StructuralLoadEnvelope['Fpsw'], # in kN
            'Mnsw': self.StructuralLoadEnvelope['Mnsw'], # in kNm
            'Mpsw': self.StructuralLoadEnvelope['Mpsw'], # in kNm
        }
        #print('Distributed Loading Dict: ', self.distributed_loading_dict)
        

        
        CG_df = self.Weights.calc_CG()
        CG_full_cargo = [CG_df[CG_df['Condition'] == 'Full Cargo']['LCG'].values[0], 0.0, CG_df[CG_df['Condition'] == 'Full Cargo']['VCG'].values[0]]
        self.HS.update_CG(CG_full_cargo)


        #7) Strutural Constraints Evaluation
        self.SC.calc_Structure_Values()
        self.SC.eval_Structure_Constraints(self.distributed_loading_dict)# Need to calculate Weights First. 
        self.Constraints = pd.concat([self.Constraints, self.SC.StructureConstraints], ignore_index=True)

        #8) Hydrostatics and Stability 
        #9) Trim/Intact Stability Constraint Evaluation
        #10) Flooding
        #11) Damage Stability

        self.HS.calc_Stability_Constraints(bulkhead_indices = self.CargoHolds.Bulkhead_Locations)
        print('Hydrostatics and Stability Constraints Evaluated')
        self.Constraints = pd.concat([self.Constraints, self.HS.StabilityConstraints], ignore_index=True)

        # Build the smaller audit-traceable result set after structure,
        # weights, and hydrostatics have all been evaluated.  This is the
        # first integrated constraint output intended for downstream use.
        verified_structure = build_verified_structural_results(
            self.SC.StructureConstraints
        )
        additional_structure = build_additional_structural_results(
            self.Hull_Dict,
            self.SC.Struct_Dict,
            self.SC.Structural_Elements,
            self.SC.ABS,
            self.CargoHolds.CargoHolds,
        )
        hydrostatic_foundation = build_hydrostatic_foundation_results(
            self.Hull_Dict,
            self.SC.Struct_Dict,
            self.HS,
            self.Weights,
        )
        self.AuditedConstraints = pd.concat(
            [verified_structure,
             additional_structure,
             hydrostatic_foundation],
            ignore_index=True,
        )

        if run_flooding_diagnostic:
            # HDF-083/HDF-ASM-029: this calls the production binary-search
            # path and saves the result separately from regulatory/audited
            # constraints because the current whole-slice damage model is a
            # diagnostic rather than CFR/SOLAS compliance evidence.
            (
                self.FloodingDiagnosticSummary,
                self.FloodableLengthCurve,
            ) = self.HS.calc_Flooding_Diagnostic(
                bulkhead_indices=self.CargoHolds.Bulkhead_Locations,
                permeabilities=flooding_permeabilities,
                step=flooding_step,
            )

        # Build the expanded V1.1 table only after cargo allocation, tank
        # fill-height calculation, structural evaluation, stability, and the
        # optional flooding binary search have all finished.  This ordering is
        # important: rule inputs come from the evaluated GenISys design, not
        # the fixed length/draft/head assumptions in the legacy evaluator.
        self.StructureConstraintsV1p1 = (
            evaluate_structure_constraints_v1p1(
                self.Hull_Dict,
                self.SC,
                self.CargoHolds,
                # ABS MVR 3-2-1/3.5 and /3.7 require still-water and wave
                # loads to be combined at the same longitudinal station.
                # Arrival bending is intentionally deferred until arrival
                # ballast has a physical aft-tank allocation; the approved
                # departure array remains an exact station-by-station case.
                named_condition_load_arrays=(
                    self.ActiveStructuralNamedConditionLoadArrays
                ),
            )
        )

        # Add the newly independent ABS rows to the status-aware audit table.
        # Compatibility aliases remain visible here for provenance.  The
        # separate canonical-identity boundary counts each reviewed physical
        # equation/member pair once, while the user-selected strength-deck
        # grab extension remains an independent project constraint.
        self.AuditedConstraints = merge_atomic_structural_audit_results(
            self.AuditedConstraints,
            self.StructureConstraintsV1p1,
        )

        self.HydrostaticConstraintsV1p1 = (
            evaluate_hydrostatic_constraints_v1p1(
                self.Hull_Dict,
                self.SC.Struct_Dict,
                self.HS,
                self.Weights,
                cargo_holds=self.CargoHolds,
                flooding_summary=self.FloodingDiagnosticSummary,
            )
        )

        # HDF-030 and HDF-033 are only partially evaluable at the present
        # design detail.  Preserve the calculated applicability geometry and
        # alarm requirements in the status-aware audited table as well as the
        # legacy-format V1.1 metrics.  Blank constraint bits prevent either
        # partial calculation from being counted as a regulatory pass.
        hdf030_side_construction = (
            build_hdf030_side_construction_results(
                self.Hull_Dict,
                self.SC.Structural_Elements,
            )
        )
        hdf033_detector_requirements = (
            build_hdf033_detector_requirement_results(
                self.Hull_Dict,
                self.SC.Struct_Dict,
                self.CargoHolds,
            )
        )
        self.AuditedConstraints = pd.concat(
            [
                self.AuditedConstraints,
                hdf030_side_construction,
                hdf033_detector_requirements,
            ],
            ignore_index=True,
        )

        self.PoweringConstraintsV1p1 = evaluate_powering_constraints_v1p1(
            self.Hull_Dict,
            deadweight_kg=self.Weights.deadweight,
        )

        # Keep exactly the legacy eight-column output contract requested for
        # every system design.  Hull design checks remain useful early-design
        # constraints.  Only the CargoHolds Part-171 bulkhead-spacing rows are
        # omitted because the audit established that they are an incorrect
        # passenger-vessel rule for this ordinary dry bulk carrier; physical
        # tank/hold capacity and machinery-space fit remain in the table.
        self.ConstraintsV1p1 = pd.concat(
            [
                self.HD.HullConstraints[V1P1_CONSTRAINT_COLUMNS],
                self.SystemFeasibilityConstraintsV1p1,
                self.SystemArrangementConstraintsV1p1,
                self.StructureConstraintsV1p1,
                self.HydrostaticConstraintsV1p1,
                self.PoweringConstraintsV1p1,
                self.LoadingConditionConstraintsV1p1,
            ],
            ignore_index=True,
        )

        #Save updated parameters

        

    #12) Cost and Performance Eval 
    @staticmethod
    def _CalculateTripFuelCost(
        fuel_weight_kg,
        fuel_price_per_tonne=(
            DEFAULT_ECONOMIC_INPUTS['Fuel_Price_pounds_per_tonne']
        ),
        fuel_cost_factor=DEFAULT_ECONOMIC_INPUTS['Fuel_Cost_Factor'],
    ):
        """Calculate one trip's fuel cost from its already-sized fuel mass.

        ``HullDesign.calc_Fuel_Consumption`` has already multiplied daily fuel
        use by voyage plus reserve days.  The economic model must therefore
        apply price once to that trip mass; multiplying by days and consumption
        again would square the fuel-demand terms dimensionally.
        """

        fuel_weight_kg = float(fuel_weight_kg)
        fuel_price_per_tonne = float(fuel_price_per_tonne)
        fuel_cost_factor = float(fuel_cost_factor)

        numeric_inputs = np.array(
            [
                fuel_weight_kg,
                fuel_price_per_tonne,
                fuel_cost_factor,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(numeric_inputs)):
            raise ValueError("Fuel-cost inputs must be finite.")

        if fuel_weight_kg < 0.0:
            raise ValueError("Trip fuel weight cannot be negative.")

        if fuel_price_per_tonne <= 0.0 or fuel_cost_factor <= 0.0:
            raise ValueError(
                "Fuel price and fuel cost factor must be positive."
            )

        fuel_weight_tonnes = fuel_weight_kg/1000.0

        return (
            fuel_cost_factor
            * fuel_weight_tonnes
            * fuel_price_per_tonne
        )

    def calc_Ship_System_Performance(self): 
        '''
        
        This function calculates the performance of the ship system design
        Annual Cargo,
        Required Freight Rate
        Ship Cost

        based on Hart and Vlahopoulos 2008 https://link.springer.com/article/10.1007/s00158-009-0414-0
        '''
        # Resolve the legacy baseline for small kernel tests and older callers
        # that construct this class without running ``__init__``.  Normal V1.1
        # evaluations resolve this mapping once in the constructor.
        if not hasattr(self, 'Economic_Inputs'):
            self.Economic_Inputs = resolve_economic_inputs()

        economic = self.Economic_Inputs

        Steel_Weight = self.Weights.structure_weight/1000.0 # in metric tons
        Outfit_Weight = self.Weights.outfit_weight/1000.0 # in metric tons
        Deadweight = self.Weights.deadweight/1000.0 # in metric tons
        Cargo_Weight = self.Weights.cargo_weight/1000.0 # in metric tons

        performance_inputs = np.array(
            [
                Steel_Weight,
                self.Weights.outfit_weight,
                self.Weights.deadweight,
                self.Weights.cargo_weight,
                self.Hull_Dict['Engine_Power'],
                self.Hull_Dict['Days_at_Sea'],
                self.Hull_Dict['Fuel_Weight_kg'],
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(performance_inputs)):
            raise ValueError(
                "Ship cost and performance inputs must all be finite."
            )

        if np.any(performance_inputs[:6] <= 0.0):
            raise ValueError(
                "Steel, outfit, deadweight, cargo, engine power, and days at "
                "sea must be positive for the performance calculation."
            )

        if self.Hull_Dict['Fuel_Weight_kg'] < 0.0:
            raise ValueError("Trip fuel weight cannot be negative.")

        # Only denominator and calendar values receive additional domain
        # checks here.  All remaining coefficients are exposed unchanged so a
        # future project scenario can be supplied without inventing bounds in
        # the implementation.
        cargo_handling_rate = float(
            economic['Cargo_Handling_Rate_t_per_day']
        )
        annual_operating_days = float(economic['Annual_Operating_Days'])

        if cargo_handling_rate <= 0.0:
            raise ValueError("Cargo handling rate must be positive.")

        if annual_operating_days <= 0.0:
            raise ValueError("Annual operating days must be positive.")

        # Original steel-weight alternative retained for traceability:
        # Ws = 0.34*(L**1.7)*(B**0.7)*(D**0.4)*(Cb**0.5), metric tonnes.
        # The active Hart/Vlahopoulos-style acquisition-cost correlation uses
        # the generated structural steel, outfit, and installed-power values.
        Ship_Cost = float(economic['Acquisition_Cost_Multiplier'])*(
            float(economic['Steel_Cost_Factor'])
            * Steel_Weight**float(economic['Steel_Cost_Exponent'])
            + float(economic['Outfit_Cost_Factor'])*Outfit_Weight
            + float(economic['Engine_Cost_Factor'])
            * self.Hull_Dict['Engine_Power']
            ** float(economic['Engine_Cost_Exponent'])
        ) # in pounds

        # Preserve the existing two-call voyage convention.  Each call has
        # cargo mass divided by handling rate plus the fixed port allowance.
        port_days = float(economic['Port_Call_Count_Per_Voyage'])*(
            Cargo_Weight/cargo_handling_rate
            + float(economic['Fixed_Port_Allowance_days_per_call'])
        )
        voyage_cycle_days = self.Hull_Dict['Days_at_Sea'] + port_days

        if voyage_cycle_days <= 0.0:
            raise ValueError("Voyage-cycle duration must be positive.")

        trips_per_year = annual_operating_days/voyage_cycle_days
        Annual_Cargo = Cargo_Weight*trips_per_year # metric tons per year

        # The trip fuel mass already contains voyage duration and the reserve
        # selected by the powering scenario. Preserve the legacy choice to
        # price that full loaded mass, but apply the economic-scenario price
        # exactly once instead of multiplying by duration and daily use again.
        Cost_fuel = self._CalculateTripFuelCost(
            self.Hull_Dict['Fuel_Weight_kg'],
            fuel_price_per_tonne=float(
                economic['Fuel_Price_pounds_per_tonne']
            ),
            fuel_cost_factor=float(economic['Fuel_Cost_Factor']),
        ) # in pounds per trip
        Cost_port = float(economic['Port_Cost_Factor'])*Deadweight**float(
            economic['Port_Cost_Deadweight_Exponent']
        ) # in pounds per trip
        Costs_voyage = (Cost_fuel + Cost_port) * trips_per_year # in pounds per year
        Costs_capital = (
            float(economic['Capital_Annual_Factor'])*Ship_Cost
        ) # in pounds per year
        Costs_running = float(economic['Running_Cost_Factor'])*(
            Deadweight
            ** float(economic['Running_Cost_Deadweight_Exponent'])
        ) # in pounds per year
        
        Costs_Annual = Costs_voyage + Costs_capital + Costs_running # in pounds per year
        if not np.isfinite(Annual_Cargo) or Annual_Cargo <= 0.0:
            raise ValueError(
                "Annual cargo must be finite and positive before calculating "
                "required freight rate."
            )

        Required_Freight_Rate = Costs_Annual/Annual_Cargo # in pounds per metric ton

        if not np.all(
            np.isfinite(
                [Ship_Cost, Annual_Cargo, Required_Freight_Rate]
            )
        ):
            raise ValueError(
                "Ship cost and performance calculation produced a nonfinite "
                "result."
            )

        # Retain both the complete scenario and the intermediate cost terms.
        # This makes an RFR value reproducible without reverse-engineering the
        # code and gives the batch writer a one-row economic detail artifact.
        self.EconomicPerformanceSummary = pd.DataFrame(
            [
                {
                    **economic,
                    'Economic_Model_Status': (
                        'preliminary_comparative_cost_model_not_a_quote'
                    ),
                    'Steel_Weight_t': Steel_Weight,
                    'Outfit_Weight_t': Outfit_Weight,
                    'Deadweight_t': Deadweight,
                    'Cargo_Weight_t': Cargo_Weight,
                    'Port_Days_per_Voyage': port_days,
                    'Voyage_Cycle_Days': voyage_cycle_days,
                    'Trips_per_Year': trips_per_year,
                    'Trip_Fuel_Cost_pounds': Cost_fuel,
                    'Trip_Port_Cost_pounds': Cost_port,
                    'Annual_Voyage_Cost_pounds': Costs_voyage,
                    'Annual_Capital_Cost_pounds': Costs_capital,
                    'Annual_Running_Cost_pounds': Costs_running,
                    'Annual_Total_Cost_pounds': Costs_Annual,
                    'Ship_Cost_pounds': Ship_Cost,
                    'Annual_Cargo_t': Annual_Cargo,
                    'Required_Freight_Rate_pounds_per_t': (
                        Required_Freight_Rate
                    ),
                }
            ]
        )

        return Ship_Cost, Annual_Cargo, Required_Freight_Rate
    
    def _Optimize_Cargo_Distribution_Legacy_Disabled(self):
        """Retain the prior unconstrained cargo search for traceability only."""

        '''
        This function optimizes the cargo distribution to minimize (distributed weight versus distributed gravity)^2 
        Constraints: 
        1) No cargo hold can be overfilled
        2) LCB and LCG cannot cause trim to exceed 1 cm. 
        3) sum of cargo fractions = 1.0 (automatic correction)
        '''
        self.CargoHolds.upload_CargoWeight(self.Weights.cargo_weight)
        self.CargoHolds.calc_FuelTank_Weights()

        def objective(x): 
            '''
            inputs: x: array of cargo fractions for each hold - sized N for N cargo holds

            Objective function to minimize: 
            Minimize net difference in distributed weight vs distributed volume

            constrained to: 
            1) No cargo hold can be overfilled
            2) LCB and LCG cannot cause trim to exceed 1 cm.
            3) sum of cargo fractions = 1.0 (automatic correction)
            '''

            # Need to write function to update cargo fractions to x 
            #print('Current Cargo Distribution Fractions: ', x)
            sum_x = np.sum(x)
            x = x/sum_x # normalize to sum to 1.0

            #print('Current Cargo Distribution Fractions: ', x)


            self.CargoHolds.upload_CargoWeightFractions(x)

            self.CargoHolds.calc_CargoHoldAndBallast_Weights()
            self.Weights.upload_CargoBallastFuel_Weights_values(self.CargoHolds.CargoHolds, self.CargoHolds.BallastTanks, self.CargoHolds.Fuel_Tank, self.CargoHolds.rho_cargo)
            

            self.Weights.calc_Distributed_Load('Full Cargo')
            CG_df= self.Weights.calc_CG()

            LCG = CG_df[CG_df['Condition'] == 'Full Cargo']['LCG'].values[0]
            Disp = self.Hull_Dict['Vol_Disp']
 
            Trim_obj = (self.Hull_Dict['LCB'] - LCG)**2.0 # minimize difference in LCB vs LCG

            
            
            DistLoad = np.sum(((self.Weights.Distributed_Load - self.Weights.buoyancy_dist)/(1000.0*Disp))**2.0)
            # As Trim_Obj approaches 1 and gets smaller, we can put more weight on DistLoad, but if Trim_Obj is large, we want to prioritize minimizing trim error.
            #DistLoad is also huge compared to Trim_obj, so we need to scale it down significantly.
            


            #print('X Cargo Distribution Fractions: ', x)
            #print('Trim Objective: ', Trim_obj, 'Dist Load Objective: ', DistLoad)
            return Trim_obj #+ 1e-6*DistLoad # minimize difference in distributed load vs buoyancy distribution
        
       
        def trim_constraint_func(x):
            # Need to write function to update cargo fractions to x 
            sum_x = np.sum(x)
            x = x/sum_x # normalize to sum to 1.0
            
            self.CargoHolds.upload_CargoWeightFractions(x)

            self.CargoHolds.calc_CargoHoldAndBallast_Weights()
    
            self.Weights.upload_CargoBallastFuel_Weights_values(self.CargoHolds.CargoHolds, self.CargoHolds.BallastTanks, self.CargoHolds.Fuel_Tank, self.CargoHolds.rho_cargo)

            self.Weights.calc_Distributed_Load('Full Cargo')
            CG_df = self.Weights.calc_CG()
            CG_full_cargo = [CG_df[CG_df['Condition'] == 'Full Cargo']['LCG'].values[0], 0.0, CG_df[CG_df['Condition'] == 'Full Cargo']['VCG'].values[0]]
            
            self.HS.update_CG(CG_full_cargo)
            Volumes, draft_cond, LCB_cond, VCB_cond = self.HS.calc_Damaged_Displacement_Condition()
            GM_T, GM_L, MTC, trim, draft_fwd, draft_aft = self.HS.calc_GM_TrimMoments(Volumes, draft_cond, LCB_cond, VCB_cond)

            #Assumption: Cargo holds will never be overfilled because cargo is more dense than water
            #Constraint 1: No cargo hold can be overfilled - IGNORED
            #Constraint 2: LCB and LCG cannot cause trim to exceed 1 cm.
            #trim_constraint = trim_limit - abs(trim) # trim must be less than trim limit
            
            return abs(trim)
        
        #Initial guess - even distribution
        x0 = np.ones((int(self.CargoHolds.bulkhead_params[5])),) / int(self.CargoHolds.bulkhead_params[5])
        #print('Initial Cargo Distribution Fractions: ', x0)
        bounds = [(1e-5, 1.0) for i in range(len(x0))]
        #print('Initial Cargo Distribution Fractions: ', x0)
        #print('Bounds: ', bounds)
        #cons = NonlinearConstraint(trim_constraint_func, -0.5, 0.5) #Trim +/- 1 cm
        
        #res = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=[cons], options={'maxiter': 32})
        print('OPTIMIZE CARGO DIST ERROR MAX ITER REDUCED TO 32. OG=128')
        res = minimize(trim_constraint_func, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter':32})

        #print('Optimization Result: ', res)

        cargo_dist = res.x

        #print('Optimized Cargo Distribution Fractions: ', cargo_dist/cargo_dist.sum())

        return cargo_dist/cargo_dist.sum()


    def Optimize_Cargo_Distribution(self):
        """Allocate full-load cargo to minimize physical departure trim.

        HDF-ASM-019 and HDF-ASM-033 require physical aft-minus-forward trim to
        remain within plus/minus one percent of equilibrium draft.  The direct
        required-LCG estimate is useful as an initial allocation, but generated
        hold shape and fill height make geometric cargo centers approximate.
        The active hybrid method therefore solves physical trim along the
        capacity-extreme segment first, then uses bounded all-hold SLSQP only
        when that first solution does not pass the trim constraint.
        """

        self.CargoHolds.upload_CargoWeight(self.Weights.cargo_weight)
        self.CargoHolds.calc_FuelTank_Weights()

        number_of_holds = int(self.CargoHolds.bulkhead_params[5])

        # Calculate each hold's full geometric capacity once.  A temporary
        # equal allocation only triggers the compartment-property calculation;
        # capacity itself does not depend on that trial distribution.
        equal_fractions = np.ones(number_of_holds)/number_of_holds
        self.CargoHolds.upload_CargoWeightFractions(equal_fractions)
        self.CargoHolds.calc_CargoHoldAndBallast_Weights()

        hold_capacity_fractions = (
            self.CargoHolds.CargoHolds['Total_Hold_Volume_m3'].to_numpy(
                dtype=float
            )
            * self.CargoHolds.rho_cargo
            / self.Weights.cargo_weight
        )

        hold_centers_m = 0.5*(
            np.asarray(self.CargoHolds.X_bulkheads[:number_of_holds], dtype=float)
            + np.asarray(
                self.CargoHolds.X_bulkheads[1:number_of_holds + 1],
                dtype=float,
            )
        )

        available_cargo_mass_kg = float(
            np.sum(
                self.CargoHolds.CargoHolds[
                    'Total_Hold_Volume_m3'
                ].to_numpy(dtype=float)
            )*self.CargoHolds.rho_cargo
        )
        cargo_capacity_constraint = (
            evaluate_total_cargo_capacity_constraint(
                self.Weights.cargo_weight,
                available_cargo_mass_kg,
            )
        )
        self.SystemFeasibilityConstraintsV1p1 = pd.concat(
            [
                self.SystemFeasibilityConstraintsV1p1,
                cargo_capacity_constraint,
            ],
            ignore_index=True,
        )

        try:
            if not bool(cargo_capacity_constraint.iloc[0]['Constraint Bit']):
                raise ValueError(
                    'Combined cargo-hold capacity is smaller than the '
                    'required full-load cargo mass.'
                )

            forwardmost_allocation = capacity_extreme_allocation(
                hold_capacity_fractions,
                hold_centers_m,
                maximize_longitudinal_moment=False,
            )
            aftmost_allocation = capacity_extreme_allocation(
                hold_capacity_fractions,
                hold_centers_m,
                maximize_longitudinal_moment=True,
            )
        except ValueError as error:
            failed_row = cargo_capacity_constraint.iloc[0]

            # A shape, coordinate, or nonfinite-input error is not a physical
            # capacity rejection when the calculated total-capacity predicate
            # itself passed.  Leave such an error in the numerical/data path
            # instead of attaching it to a passing physical row.
            if bool(failed_row['Constraint Bit']):
                raise

            raise PhysicalConstraintFailure(
                'Physical cargo-capacity failure: required cargo mass is '
                f'{self.Weights.cargo_weight:.6g} kg, while the generated '
                f'holds provide {available_cargo_mass_kg:.6g} kg at the '
                f'approved {self.CargoHolds.rho_cargo:.6g} kg/m^3 density.',
                name='Total cargo-hold mass capacity',
                requirement=(
                    'Generated cargo-hold capacity shall be at least the '
                    'selected full-load cargo mass'
                ),
                threshold_value=failed_row['Threshold Value'],
                constraint_value=failed_row['Constraint Value'],
                units='kg',
                comparison='minimum',
                result_class='System Capacity',
                constraint_rows=pd.concat(
                    [
                        self.HD.HullConstraints[
                            V1P1_CONSTRAINT_COLUMNS
                        ],
                        self.SystemFeasibilityConstraintsV1p1,
                    ],
                    ignore_index=True,
                ),
                diagnostics={
                    'Required_Cargo_Mass_kg': self.Weights.cargo_weight,
                    'Available_Cargo_Hold_Capacity_kg': (
                        available_cargo_mass_kg
                    ),
                    'Cargo_Hold_Capacity_Margin_kg': (
                        available_cargo_mass_kg
                        - self.Weights.cargo_weight
                    ),
                },
            ) from error

        cached_x = None
        cached_state = None

        def evaluate_distribution(x):
            """Return the physical equilibrium state for one feasible split."""

            nonlocal cached_x, cached_state

            x = np.asarray(x, dtype=float)

            if cached_x is not None and np.array_equal(x, cached_x):
                return cached_state

            self.CargoHolds.upload_CargoWeightFractions(x)
            self.CargoHolds.calc_CargoHoldAndBallast_Weights()
            self.Weights.upload_CargoBallastFuel_Weights_values(
                self.CargoHolds.CargoHolds,
                self.CargoHolds.BallastTanks,
                self.CargoHolds.Fuel_Tank,
                self.CargoHolds.rho_cargo,
            )

            self.Weights.calc_Distributed_Load('Full Cargo')

            # HDF-ASM-019/HDF-ASM-033: use the same two-endpoint equilibrium
            # that owns the still-water load curve.  The former implementation
            # calculated this physical equilibrium and then discarded it in
            # favor of the older linearized MCT trim estimate.  Selecting the
            # cargo split with the exact endpoint drafts prevents the optimizer
            # and the reported longitudinal-load constraint from disagreeing.
            load_equilibrium = self.Weights.LoadClosureDiagnostics
            draft_fwd = float(
                load_equilibrium['Load_Equilibrium_Forward_Draft_m']
            )
            draft_aft = float(
                load_equilibrium['Load_Equilibrium_Aft_Draft_m']
            )
            draft_cond = float(
                load_equilibrium['Load_Equilibrium_Draft_m']
            )
            trim_m, _, trim_limit_m, _ = evaluate_trim_constraint(
                draft_fwd,
                draft_aft,
                draft_cond,
            )

            cached_x = x.copy()
            cached_state = {
                'draft_m': float(draft_cond),
                'draft_forward_m': float(draft_fwd),
                'draft_aft_m': float(draft_aft),
                'trim_m': float(trim_m),
                'trim_limit_m': float(trim_limit_m),
            }

            return cached_state

        # Calculate the cargo longitudinal center needed for zero departure
        # trim.  Every non-cargo row already has a calculated mass and LCG;
        # fuel is calculated separately above because it has not yet been
        # inserted into the working weight table.  The selected design draft
        # supplies both displacement and LCB through the hull module.
        noncargo_rows = self.Weights.Weights.loc[
            ~self.Weights.Weights['Name'].isin(
                ['Cargo_Hold', 'Ballast_Tank', 'Fuel_Tank']
            )
        ]
        noncargo_moment_kg_m = float(
            np.sum(
                noncargo_rows['Total_Weight_kg'].to_numpy(dtype=float)
                * noncargo_rows['LCG'].to_numpy(dtype=float)
            )
        )
        fuel_moment_kg_m = float(
            np.sum(
                self.CargoHolds.Fuel_Tank[
                    'Total_Weight_kg'
                ].to_numpy(dtype=float)
                * self.CargoHolds.Fuel_Tank['LCG'].to_numpy(dtype=float)
            )
        )
        displacement_mass_kg = float(self.Weights.rho*self.Weights.V_disp)
        required_cargo_lcg_m = (
            displacement_mass_kg*float(self.Hull_Dict['LCB'])
            - noncargo_moment_kg_m
            - fuel_moment_kg_m
        )/float(self.Weights.cargo_weight)

        # The convex combination returned here always sums to one and obeys
        # every hold-capacity bound.  When the required cargo LCG is outside
        # the attainable interval, the nearest endpoint is returned and its
        # signed LCG deficit is retained for design repair diagnostics.
        direct_allocation = direct_required_lcg_allocation(
            forwardmost_allocation,
            aftmost_allocation,
            hold_centers_m,
            required_cargo_lcg_m,
        )

        try:
            # First solve the inexpensive one-dimensional problem.  Unlike
            # the geometric LCG estimate, each trial point is evaluated using
            # the generated cargo fill and complete longitudinal equilibrium.
            search = deterministic_trim_search(
                forwardmost_allocation,
                aftmost_allocation,
                evaluate_distribution,
                recoverable_exceptions=(LongitudinalEquilibriumError,),
            )
        except CargoAllocationSearchError as error:
            # Every sampled allocation was outside the buoyancy-equilibrium
            # domain.  This is a physical design outcome, not an optimizer
            # crash or an unrelated-system failure.
            equilibrium_constraint = evaluate_cargo_equilibrium_constraint(
                error.valid_evaluation_count
            )
            self.SystemFeasibilityConstraintsV1p1 = pd.concat(
                [
                    self.SystemFeasibilityConstraintsV1p1,
                    equilibrium_constraint,
                ],
                ignore_index=True,
            )
            failed_row = equilibrium_constraint.iloc[0]
            diagnostics = {
                'Cargo_Optimization_Iterations': (
                    error.evaluation_count
                ),
                'Cargo_Allocation_Invalid_Equilibrium_Evaluation_Count': (
                    error.invalid_evaluation_count
                ),
                'Cargo_Allocation_Valid_Equilibrium_Evaluation_Count': (
                    error.valid_evaluation_count
                ),
                **error.closest_diagnostics,
            }
            raise PhysicalConstraintFailure(
                str(error),
                name=(
                    'Capacity-feasible cargo allocation reaches '
                    'longitudinal equilibrium'
                ),
                requirement=(
                    'At least one capacity-feasible full-load cargo '
                    'allocation shall reach displacement and longitudinal-'
                    'moment equilibrium within the generated hull draft grid'
                ),
                threshold_value=failed_row['Threshold Value'],
                constraint_value=failed_row['Constraint Value'],
                units='valid candidates',
                comparison='minimum',
                result_class='Loading Condition Feasibility',
                constraint_rows=pd.concat(
                    [
                        self.HD.HullConstraints[
                            V1P1_CONSTRAINT_COLUMNS
                        ],
                        self.SystemFeasibilityConstraintsV1p1,
                    ],
                    ignore_index=True,
                ),
                diagnostics=diagnostics,
            ) from error

        # The capacity-extreme segment is fast and normally sufficient because
        # trim is controlled primarily by cargo longitudinal moment.  If its
        # best physical solution remains outside the approved trim band, allow
        # all hold fractions to vary.  This captures solutions that cannot be
        # represented by one interpolation between two extreme allocations.
        segment_state = search['state']
        multivariable_used = bool(
            abs(segment_state['trim_m']) > segment_state['trim_limit_m']
        )
        multivariable_success = False
        multivariable_status = 0
        multivariable_message = 'Not required; segment solution passed trim.'
        multivariable_iterations = 0
        multivariable_evaluation_count = 0
        multivariable_invalid_count = 0

        if multivariable_used:
            valid_candidates = [
                {
                    'allocation': np.asarray(
                        search['allocation'],
                        dtype=float,
                    ).copy(),
                    'state': dict(search['state']),
                }
            ]

            def close_to_capacity_simplex(candidate):
                """Close SLSQP roundoff while preserving every capacity."""

                candidate = np.asarray(candidate, dtype=float).copy()
                candidate = np.clip(candidate, 0.0, hold_capacity_fractions)
                residual = 1.0-float(np.sum(candidate))

                # SLSQP owns the equality constraint.  Only close its small
                # numerical residual here; a materially infeasible proposal
                # receives a penalty without entering the physical evaluator.
                if abs(residual) > 1.0e-5:
                    return None

                if residual > 0.0:
                    spare = hold_capacity_fractions-candidate

                    for index in np.argsort(spare)[::-1]:
                        addition = min(float(spare[index]), residual)
                        candidate[index] += addition
                        residual -= addition

                        if residual <= 1.0e-12:
                            break
                elif residual < 0.0:
                    excess = -residual

                    for index in np.argsort(candidate)[::-1]:
                        removal = min(float(candidate[index]), excess)
                        candidate[index] -= removal
                        excess -= removal

                        if excess <= 1.0e-12:
                            break

                    residual = -excess

                if abs(1.0-float(np.sum(candidate))) > 1.0e-9:
                    return None

                if np.any(candidate > hold_capacity_fractions + 1.0e-9):
                    return None

                return candidate

            def fill_uniformity_penalty(candidate):
                """Prefer even relative hold filling after minimizing trim."""

                active = hold_capacity_fractions > 0.0
                relative_fill = (
                    candidate[active]/hold_capacity_fractions[active]
                )
                return float(np.var(relative_fill))

            def physical_trim_objective(candidate):
                """Evaluate exact normalized trim for one SLSQP proposal."""

                nonlocal multivariable_evaluation_count
                nonlocal multivariable_invalid_count

                candidate = close_to_capacity_simplex(candidate)

                if candidate is None:
                    return 1.0e6

                multivariable_evaluation_count += 1

                try:
                    state = evaluate_distribution(candidate)
                except LongitudinalEquilibriumError:
                    multivariable_invalid_count += 1
                    return 1.0e6

                valid_candidates.append(
                    {
                        'allocation': candidate.copy(),
                        'state': dict(state),
                    }
                )
                normalized_trim = (
                    float(state['trim_m'])/float(state['trim_limit_m'])
                )

                # Trim dominates by six orders of magnitude.  The secondary
                # term only chooses a more uniform fill among solutions with
                # effectively identical physical trim.
                return (
                    normalized_trim**2
                    + 1.0e-6*fill_uniformity_penalty(candidate)
                )

            capacity_weighted = (
                hold_capacity_fractions
                / float(np.sum(hold_capacity_fractions))
            )
            multivariable_starts = [
                np.asarray(search['allocation'], dtype=float),
                np.asarray(direct_allocation['allocation'], dtype=float),
                capacity_weighted,
                0.5*(forwardmost_allocation + aftmost_allocation),
            ]
            unique_starts = []

            for start in multivariable_starts:
                start = close_to_capacity_simplex(start)

                if start is None:
                    continue

                if any(
                    np.allclose(start, existing, rtol=0.0, atol=1.0e-12)
                    for existing in unique_starts
                ):
                    continue

                unique_starts.append(start)

            optimizer_results = []
            equality_constraint = {
                'type': 'eq',
                'fun': lambda candidate: float(np.sum(candidate)-1.0),
            }
            bounds = [
                (0.0, float(capacity))
                for capacity in hold_capacity_fractions
            ]

            for start in unique_starts:
                result = minimize(
                    physical_trim_objective,
                    start,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[equality_constraint],
                    options={
                        'maxiter': 80,
                        'ftol': 1.0e-10,
                        'disp': False,
                    },
                )
                optimizer_results.append(result)
                multivariable_iterations += int(result.nit)

            # Select from every physically evaluated candidate instead of
            # trusting an optimizer status flag as proof of ship feasibility.
            best_candidate = min(
                valid_candidates,
                key=lambda candidate: (
                    abs(
                        candidate['state']['trim_m']
                        / candidate['state']['trim_limit_m']
                    ),
                    fill_uniformity_penalty(candidate['allocation']),
                ),
            )
            best_state = evaluate_distribution(best_candidate['allocation'])
            multivariable_success = bool(
                abs(best_state['trim_m']) <= best_state['trim_limit_m']
            )
            successful_results = [
                result for result in optimizer_results if bool(result.success)
            ]

            if successful_results:
                reporting_result = min(
                    successful_results,
                    key=lambda result: float(result.fun),
                )
            else:
                reporting_result = min(
                    optimizer_results,
                    key=lambda result: float(result.fun),
                )

            multivariable_status = int(reporting_result.status)
            multivariable_message = str(reporting_result.message)
            search = {
                'allocation': best_candidate['allocation'],
                'state': best_state,
                # A full all-hold allocation generally does not lie on the
                # original capacity-extreme interpolation segment.
                'alpha': np.nan,
                'evaluation_count': (
                    int(search['evaluation_count'])
                    + int(multivariable_evaluation_count)
                ),
                'invalid_evaluation_count': (
                    int(search['invalid_evaluation_count'])
                    + int(multivariable_invalid_count)
                ),
                'valid_evaluation_count': (
                    int(search['valid_evaluation_count'])
                    + int(multivariable_evaluation_count)
                    - int(multivariable_invalid_count)
                ),
                'sign_change_bracket_count': int(
                    search['sign_change_bracket_count']
                ),
            }

        cargo_fractions = search['allocation']
        equilibrium_constraint = evaluate_cargo_equilibrium_constraint(
            search['valid_evaluation_count']
        )
        self.SystemFeasibilityConstraintsV1p1 = pd.concat(
            [
                self.SystemFeasibilityConstraintsV1p1,
                equilibrium_constraint,
            ],
            ignore_index=True,
        )
        final_state = search['state']
        trim_margin_m = (
            final_state['trim_limit_m'] - abs(final_state['trim_m'])
        )
        capacity_margin = np.min(
            hold_capacity_fractions - cargo_fractions
        )

        # Publish the allocation result separately from regulatory constraints.
        # A design outside the trim band is retained and receives a failed trim
        # constraint later; it is not mislabeled as a numerical solver crash.
        self.CargoOptimizationSummary = pd.DataFrame(
            [
                {
                    'Solver': 'Hybrid physical-trim cargo allocation',
                    'Solver_Success': True,
                    'Solver_Status': int(multivariable_status),
                    'Solver_Message': (
                        'Capacity-extreme physical trim solve passed without '
                        'requiring the all-hold optimizer.'
                        if not multivariable_used
                        else
                        multivariable_message
                    ),
                    'Solver_Phase': (
                        'capacity_extreme_physical_root'
                        if not multivariable_used
                        else 'bounded_all_hold_slsqp'
                    ),
                    'Multivariable_Optimizer_Used': multivariable_used,
                    'Multivariable_Optimizer_Success': (
                        multivariable_success
                    ),
                    'Multivariable_Optimizer_Iterations': int(
                        multivariable_iterations
                    ),
                    'Multivariable_Physical_Evaluation_Count': int(
                        multivariable_evaluation_count
                    ),
                    'Iterations': int(search['evaluation_count']),
                    'Valid_Equilibrium_Evaluation_Count': int(
                        search['valid_evaluation_count']
                    ),
                    'Invalid_Equilibrium_Evaluation_Count': int(
                        search['invalid_evaluation_count']
                    ),
                    'Interpolation_Alpha': float(search['alpha']),
                    'Trim_Sign_Change_Bracket_Count': int(
                        search['sign_change_bracket_count']
                    ),
                    'Required_Cargo_LCG_m': float(
                        direct_allocation['required_cargo_lcg_m']
                    ),
                    'Forwardmost_Cargo_LCG_m': float(
                        direct_allocation['forwardmost_cargo_lcg_m']
                    ),
                    'Aftmost_Cargo_LCG_m': float(
                        direct_allocation['aftmost_cargo_lcg_m']
                    ),
                    'Achieved_Geometric_Cargo_LCG_m': float(
                        np.dot(cargo_fractions, hold_centers_m)
                    ),
                    'Cargo_LCG_Deficit_m': float(
                        direct_allocation['required_cargo_lcg_m']
                        - np.dot(cargo_fractions, hold_centers_m)
                    ),
                    'Cargo_LCG_Target_Reachable': bool(
                        direct_allocation['target_reachable']
                    ),
                    'Cargo_Allocation_Alpha_Unbounded': float(
                        direct_allocation['alpha_unbounded']
                    ),
                    'Equilibrium_Draft_m': final_state['draft_m'],
                    'Draft_Forward_m': final_state['draft_forward_m'],
                    'Draft_Aft_m': final_state['draft_aft_m'],
                    'Signed_Trim_Aft_Minus_Forward_m': final_state['trim_m'],
                    'Absolute_Trim_m': abs(final_state['trim_m']),
                    'Trim_Limit_One_Percent_Draft_m': final_state[
                        'trim_limit_m'
                    ],
                    'Trim_Margin_m': trim_margin_m,
                    # Permit only floating-point subtraction noise at the
                    # exact one-percent boundary.  This is not an added
                    # physical trim allowance or an engineering tolerance.
                    'Trim_Constraint_Passed': trim_margin_m >= (
                        -64.0*np.finfo(float).eps
                        * max(
                            1.0,
                            abs(final_state['trim_m']),
                            abs(final_state['trim_limit_m']),
                        )
                    ),
                    'Minimum_Hold_Capacity_Margin_Fraction': capacity_margin,
                    'Required_Cargo_Mass_kg': float(
                        self.Weights.cargo_weight
                    ),
                    'Available_Cargo_Hold_Capacity_kg': (
                        available_cargo_mass_kg
                    ),
                    'Total_Cargo_Hold_Capacity_Margin_kg': (
                        available_cargo_mass_kg
                        - self.Weights.cargo_weight
                    ),
                    'Hold_Capacity_Constraints_Passed': (
                        capacity_margin >= (
                            -64.0*np.finfo(float).eps
                            * max(1.0, float(np.max(hold_capacity_fractions)))
                        )
                    ),
                    'Assumption_IDs': 'HDF-ASM-019; HDF-ASM-033',
                }
            ]
        )

        return cargo_fractions

            
    

        
