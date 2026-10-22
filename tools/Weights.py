#This script calculates the weight distribution properties of a ship based on its hull parameters and structural elements. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tools.HullParameterization import Hull_Parameterization as HP
from tools.EvaluationErrors import PhysicalConstraintFailure
from tools.EarlyDesignMasses import build_lightship_margin_component
from tools.System_Feasibility_Constraints_v1p1 import (
    evaluate_lightship_displacement_constraint,
    evaluate_positive_cargo_payload_constraint,
)


class LongitudinalEquilibriumError(ValueError):
    """Identify an allocation-specific force/moment equilibrium failure.

    Cargo allocation deliberately explores the longitudinal-moment extremes
    of the feasible hold-capacity polytope.  An extreme load can place LCG
    outside the buoyancy-centroid range represented by the hull draft grid
    even when interior allocations remain fully solvable.  The dedicated type
    lets the cargo search skip only this known physical candidate failure
    without swallowing unrelated numerical or implementation exceptions.
    """

    def __init__(self, message, **diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _scale_distribution_to_mass(distribution, x_coordinates, target_mass_kg):
    """Scale a kg/m distribution so trapezoidal integration returns mass.

    Component scalar masses drive displacement, CG, and performance.  Their
    distributed representations drive still-water shear and bending.  This
    helper keeps those two representations mathematically identical without
    changing the component's calculated longitudinal shape.
    """

    distribution = np.asarray(distribution, dtype=float).copy()
    target_mass_kg = float(target_mass_kg)

    if target_mass_kg == 0.0:
        return np.zeros_like(distribution)

    integrated_mass_kg = np.trapezoid(
        distribution,
        x=np.asarray(x_coordinates, dtype=float),
    )

    if integrated_mass_kg == 0.0:
        raise ValueError(
            'A nonzero scalar weight has no longitudinal distribution.'
        )

    return distribution * target_mass_kg/integrated_mass_kg


def interp(A,Z,z):
        # This function interpolates data to approximate A(z)  given values of A(Z) 
        # Ths assumes that Z is sorted in ascending order and z is within the range of Z

        # Drafts at the baseline or top modeled waterline are valid and do
        # not require an extrapolated interval.
        if z <= Z[0]:
            return A[0]

        if z >= Z[-1]:
            return A[-1]

        idx = np.searchsorted(Z, z, side='right') - 1
    
        frac = (z - Z[idx])/(Z[idx+1] - Z[idx])
    
        return A[idx] + frac*(A[idx+1] - A[idx])


def solve_longitudinal_buoyancy_equilibrium(
    x_coordinates,
    draft_grid,
    sectional_area_grid,
    target_displacement_volume_m3,
    target_lcg_m,
    initial_draft_m,
):
    """Solve sinkage and trim from displacement and longitudinal moment.

    The immersed area at every station is interpolated on the existing
    upright sectional-area grid.  A straight waterplane between the forward
    and aft grid ordinates defines the local draft.  The two unknown endpoint
    drafts are then selected so that both displaced volume and its
    longitudinal first moment equal the selected weight condition.

    This is the physically balanced small-trim model required before the
    still-water load can be integrated into shear and bending moment.  It does
    not apply an artificial correction to either derived curve.
    """

    x_coordinates = np.asarray(x_coordinates, dtype=float)
    draft_grid = np.asarray(draft_grid, dtype=float)
    sectional_area_grid = np.asarray(sectional_area_grid, dtype=float)
    target_displacement_volume_m3 = float(
        target_displacement_volume_m3
    )
    target_lcg_m = float(target_lcg_m)
    initial_draft_m = float(initial_draft_m)

    forward_x_m = float(x_coordinates[0])
    aft_x_m = float(x_coordinates[-1])
    modeled_length_m = aft_x_m - forward_x_m
    target_first_moment_m4 = (
        target_displacement_volume_m3
        * (target_lcg_m - forward_x_m)
    )

    def properties(endpoint_drafts_m):
        """Return local drafts, areas, volume, and first moment."""

        forward_draft_m, aft_draft_m = endpoint_drafts_m
        local_drafts_m = (
            forward_draft_m
            + (aft_draft_m - forward_draft_m)
            * (x_coordinates - forward_x_m)
            / modeled_length_m
        )
        local_areas_m2 = np.asarray(
            [
                np.interp(
                    local_draft_m,
                    draft_grid,
                    sectional_area_grid[station_index, :],
                )
                for station_index, local_draft_m in enumerate(
                    local_drafts_m
                )
            ],
            dtype=float,
        )
        volume_m3 = float(
            np.trapezoid(local_areas_m2, x=x_coordinates)
        )
        first_moment_m4 = float(
            np.trapezoid(
                local_areas_m2*(x_coordinates - forward_x_m),
                x=x_coordinates,
            )
        )

        return (
            local_drafts_m,
            local_areas_m2,
            volume_m3,
            first_moment_m4,
        )

    volume_scale_m3 = max(
        abs(target_displacement_volume_m3),
        np.finfo(float).eps,
    )
    moment_scale_m4 = max(
        abs(target_displacement_volume_m3*modeled_length_m),
        np.finfo(float).eps,
    )

    def normalized_residual(endpoint_drafts_m):
        """Scale unlike equilibrium equations for the numerical solve."""

        _, _, volume_m3, first_moment_m4 = properties(
            endpoint_drafts_m
        )

        return np.asarray(
            [
                (
                    volume_m3 - target_displacement_volume_m3
                )/volume_scale_m3,
                (
                    first_moment_m4 - target_first_moment_m4
                )/moment_scale_m4,
            ],
            dtype=float,
        )

    bounded_initial_draft_m = float(
        np.clip(initial_draft_m, draft_grid[0], draft_grid[-1])
    )
    solution = least_squares(
        normalized_residual,
        x0=np.asarray(
            [bounded_initial_draft_m, bounded_initial_draft_m],
            dtype=float,
        ),
        bounds=(
            np.asarray([draft_grid[0], draft_grid[0]], dtype=float),
            np.asarray([draft_grid[-1], draft_grid[-1]], dtype=float),
        ),
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=200,
    )
    (
        local_drafts_m,
        local_areas_m2,
        volume_m3,
        first_moment_m4,
    ) = properties(solution.x)
    final_normalized_residual = normalized_residual(solution.x)

    # This tolerance controls only convergence of two normalized algebraic
    # equilibrium equations.  Engineering acceptance tolerances for the saved
    # load-closure metrics remain a separate project decision.
    equilibrium_solved = bool(
        solution.success
        and np.max(np.abs(final_normalized_residual)) <= 1.0e-9
    )

    return {
        'Forward_Draft_m': float(solution.x[0]),
        'Aft_Draft_m': float(solution.x[1]),
        'Local_Drafts_m': local_drafts_m,
        'Local_Sectional_Areas_m2': local_areas_m2,
        'Displacement_Volume_m3': volume_m3,
        'Displacement_First_Moment_m4': first_moment_m4,
        'Target_Displacement_First_Moment_m4': target_first_moment_m4,
        'Normalized_Volume_Residual': float(
            final_normalized_residual[0]
        ),
        'Normalized_First_Moment_Residual': float(
            final_normalized_residual[1]
        ),
        'Equilibrium_Solved': equilibrium_solved,
        'Solver_Status': int(solution.status),
        'Solver_Message': str(solution.message),
        'Function_Evaluations': int(solution.nfev),
    }

class Weights:
    def __init__(
        self,
        hull: HP,
        structural_elements: pd.DataFrame,
        V_disp: float,
        desgin_draft: float,
        Engine_Power_kW: float,
        Days_at_Sea: float,
        fuel_weight_kg=None,
    ):
        '''
        Outfit, machinery, fuel, and other weights not included in the hull structure and deadweight calculations are 
        estimated by Hart and Vlahopoulos 2008 from their integrated multidisciplinary design approach for ship design.
        https://link.springer.com/article/10.1007/s00158-009-0414-0

        
        '''
        self.hull = hull
        self.V_disp = V_disp #Displacement volume in m^3
        self.design_draft = desgin_draft #Design draft in meters
        # HDF-103: HullParameterization stores ``Bd`` as the half-breadth.
        # The block-coefficient denominator requires the full molded breadth.
        # HDF-ASM-006 permits LOA as the conservative working length where a
        # distinct regulatory length is not yet represented.
        full_beam = 2.0 * self.hull.Bd
        self.Cb = self.V_disp / (
            self.hull.LOA * full_beam * self.design_draft
        )
        self.structural_elements = structural_elements
        self.Db_height = self.structural_elements[self.structural_elements['Object_ID']=='Side_Shell']['z_loc'].values[0]
        self.rho = 1025 #Density of seawater in kg/m^3
        self.rho_steel = 7850 #Density of steel in kg/m^3
        self.g = 9.81 #Gravitational acceleration in m/s^2
        self.Engine_Power_kW = Engine_Power_kW #Engine power in kW
        self.Days_at_Sea = Days_at_Sea #Days at sea for fuel weight
        
        # The integrated system path supplies the fuel mass calculated by
        # HullDesign so an explicit powering scenario cannot be discarded at
        # the weights boundary.  Keep the old equation only as a backwards-
        # compatible fallback for focused legacy callers which construct
        # Weights directly.
        if fuel_weight_kg is None:
            self.fuel_weight = (
                (0.19*24*Engine_Power_kW/1000.0 + 0.2)*1000.0
            )*(Days_at_Sea + 5)
            self.fuel_weight_basis = "legacy_weights_fallback"
        else:
            self.fuel_weight = float(fuel_weight_kg)
            self.fuel_weight_basis = "hull_design_powering_scenario"
 
        

        self.Weights = pd.DataFrame(columns=['Name', 'Total_Weight_kg', 'VCG', 'LCG', 'Dist_Weight']) #Dataframe to store weight and VCG of each weight class

        # Populated by the load, shear, and bending calculations.  Keeping the
        # residuals explicit makes numerical closure inspectable without
        # modifying the underlying loads to force endpoint values to zero.
        self.LoadClosureDiagnostics = {}


        
    def upload_Bulkhead_info(self, bulkhead_locs:list):
        self.bulkhead_locs = bulkhead_locs #list of X indices where bulkheads are located
        
   

    def upload_PC_SA_values(self, X, Z, Y, sectional_areas, sectional_areas_Z_centroids): 
        self.X = X #X coordinates along the length of the ship in meters
        self.Z = Z #Z coordinates along the depth of the ship in meters
        self.Y = Y #Y coordinates along the width of the ship in meters (in a len X by len Z grid)
        self.sectional_Areas = sectional_areas #Sectional areas below draftmark along the length of the ship in m^2 format is lenX by lenZ
        self.sectional_Areas_Z_centroids = sectional_areas_Z_centroids #Sectional areas below draftmark along the length of the ship in m^2 format is lenX by lenZ


    def upload_CargoBallastFuel_Weights_values(self, CargoHolds:pd.DataFrame, BallastTanks:pd.DataFrame, FuelTanks:pd.DataFrame, rho_cargo:float):
        
        #First check for instances of Cargo_Hold, Ballast_Tank, and Fuel_Tank  dataframes and remove them from self.Weights
        self.Weights = self.Weights[self.Weights['Name'] != 'Cargo_Hold']
        self.Weights = self.Weights[self.Weights['Name'] != 'Ballast_Tank']
        self.Weights = self.Weights[self.Weights['Name'] != 'Fuel_Tank']
        
        #Add these to the Weights dataframe
        for i in range(len(CargoHolds)):
            item = pd.DataFrame({'Name': ['Cargo_Hold'], 'Total_Weight_kg': [CargoHolds.iloc[i]['Total_Weight_kg']], 'VCG': [CargoHolds.iloc[i]['VCG']], 'LCG': [CargoHolds.iloc[i]['LCG']], 'Dist_Weight': [CargoHolds.iloc[i]['Dist_Weight']]})   
            self.Weights = pd.concat([self.Weights, item], ignore_index=True)
        for i in range(len(BallastTanks)):
            item = pd.DataFrame({'Name': ['Ballast_Tank'], 'Total_Weight_kg': [BallastTanks.iloc[i]['Total_Weight_kg']], 'VCG': [BallastTanks.iloc[i]['VCG']], 'LCG': [BallastTanks.iloc[i]['LCG']], 'Dist_Weight': [BallastTanks.iloc[i]['Dist_Weight']]})   
            self.Weights = pd.concat([self.Weights, item], ignore_index=True)

        for i in range(len(FuelTanks)):
            item = pd.DataFrame({'Name': ['Fuel_Tank'], 'Total_Weight_kg': [FuelTanks.iloc[i]['Total_Weight_kg']], 'VCG': [FuelTanks.iloc[i]['VCG']], 'LCG': [FuelTanks.iloc[i]['LCG']], 'Dist_Weight': [FuelTanks.iloc[i]['Dist_Weight']]})   
            self.Weights = pd.concat([self.Weights, item], ignore_index=True)
        
        self.rho_cargo = rho_cargo
        self.cargo_weight = CargoHolds['Total_Weight_kg'].sum()


    def calc_unit_Bulkhead_Weight(self): 
        '''
        This function calculates the weight of the bulkheads in the ship based on the structural elements data.

        Then we want to normalize the bulkhead weights by sectional area of the structural design to get a weight per unit SA
        '''
        #First, collect all elements on one bulkhead: 
        #get list of all bulkheads
        blkhds = self.structural_elements[self.structural_elements['Class'] == 'Transverse_Bulkhead']
        #Collect all elements on the first bulkhead
        bulkhead_elements = self.structural_elements[self.structural_elements['Parent_Struct'] == blkhds.iloc[0]['Object_ID']]

        #append bulkhead to elements list
        bulkhead_elements = pd.concat([bulkhead_elements, blkhds.iloc[0:1]], ignore_index=True)

        bulkhead_mass = bulkhead_elements['volume_pannel'].sum() * self.rho_steel #in kg

        self.Bulkhead_unitWeight = bulkhead_mass / (blkhds.iloc[0]['area_pannel']) #in kg/m^2 (weight per unit area of bulkhead)


    def calc_Bulkhead_Weights(self):
        '''
        Use unit weight and sectional areas to calculate total bulkhead weights
        bulkhead_locs is a sorted list of X indices where bulkheads are located
        '''
        bulkhead_weights = np.zeros((len(self.bulkhead_locs),)) #in kg
    
        bulkhead_VCGs = np.zeros((len(self.bulkhead_locs),)) #in meters
        pd_items = pd.DataFrame(columns=['Name', 'Total_Weight_kg', 'VCG', 'LCG', 'Dist_Weight'])

        for i in range(len(self.bulkhead_locs)):
            bulkhead_weights[i] = self.Bulkhead_unitWeight * self.sectional_Areas[self.bulkhead_locs[i], -1] #in kg (sectional area at the full depth of hull)
            bulkhead_VCGs[i] = self.sectional_Areas_Z_centroids[self.bulkhead_locs[i], -1] #in meters (VCG at the is the centroid of the sectional area of the full depth of hull)
            dist_weight = np.zeros((len(self.X),))
            # A nonzero value at the physical bulkhead station establishes
            # the point-load shape.  The normalization below supplies the
            # correct grid-dependent magnitude for trapezoidal integration.
            dist_weight[self.bulkhead_locs[i]] = 1.0

            # Point-like bulkhead weights are represented on the longitudinal
            # grid.  Normalize them with the same integration rule as every
            # distributed load so endpoint/grid placement cannot change mass.
            dist_weight = _scale_distribution_to_mass(
                dist_weight,
                self.X,
                bulkhead_weights[i],
            )
            item = pd.DataFrame({'Name': ['Transverse_Bulkhead ' + str(i)], 'Total_Weight_kg': [bulkhead_weights[i]], 'VCG': [bulkhead_VCGs[i]], 'LCG': [self.X[self.bulkhead_locs[i]]], 'Dist_Weight': [dist_weight]})
            pd_items.loc[i] = item.loc[0]
            
            
          
        self.Weights = pd.concat([self.Weights, pd_items], ignore_index=True)
        self.bulkhead_weights = bulkhead_weights

    def calc_unit_Long_Struct_Weight(self):
        '''
        This function calculates the weight of the longitudinal structures in the ship based on the structural elements data.

        Then we want to normalize the weight of all structure that is not a bulkhead by
        the length of the structure design to get a weight per unit length and sectional area

        '''
        # Identify all the bulkheads and their sub structures
        bulkheads = self.structural_elements[self.structural_elements['Class'] == 'Transverse_Bulkhead']

        L = self.structural_elements['L1'].max()

        bulkhead_ids = bulkheads['Object_ID']

        # Exclude every transverse bulkhead and all of its attached stiffeners
        # and flanges from the longitudinal structural-weight calculation.
        is_bulkhead_structure = (
            (self.structural_elements['Class'] == 'Transverse_Bulkhead')
            | self.structural_elements['Parent_Struct'].isin(bulkhead_ids)
        )

        # All remaining elements contribute to longitudinal structure weight.
        long_weight_structures = self.structural_elements[~is_bulkhead_structure]
        long_weight = long_weight_structures['volume_pannel'].sum() * self.rho_steel #in kg 
 
        self.Longitudinal_Structure_unitWeight = long_weight / (L * (bulkheads.iloc[0]['area_pannel'])) #in kg/m^3 (weight per unit length and sectional area of bullkhead)
        moment = (long_weight_structures['z_centroid_pannel'] * long_weight_structures['volume_pannel']).sum() #in kg*m
        self.Longitudinal_Structre_VCG = moment/(long_weight_structures['volume_pannel'].sum()) #in meters (approximate VCG of longitudinal structure as the mean of all elements)

    def calc_distributed_Long_Struct_Weight(self): 
        '''
        Use unit weight and sectional areas to calculate distributed longitudinal structural weight distribution
        '''
        long_struct_weight_dist = self.Longitudinal_Structure_unitWeight * self.sectional_Areas[:, -1] #in kg/m (sectional area at the full depth of hull)
        # HDF-ASM-028: use the composite trapezoidal rule for distributed
        # weights and their longitudinal first moments.
        Total_Longitudinal_Struct_Weight = np.trapezoid(
            long_struct_weight_dist,
            x=self.X,
        ) #in kg
        long_struct_LCG = np.trapezoid(
            long_struct_weight_dist * self.X,
            x=self.X,
        ) / Total_Longitudinal_Struct_Weight #in meters


        pd_item = pd.DataFrame({'Name': ['Structure'], 'Total_Weight_kg': [Total_Longitudinal_Struct_Weight], 'VCG': [self.Longitudinal_Structre_VCG], 'LCG': long_struct_LCG, 'Dist_Weight': [long_struct_weight_dist]})
        self.Weights = pd.concat([self.Weights, pd_item], ignore_index=True)
        
        #print('Longitudinal Structure Weight (kg): ', Total_Longitudinal_Struct_Weight)
        #print('integral of long struct weight (kg): ', np.trapezoid(long_struct_weight_dist, self.X))


        self.Total_Longitudinal_Struct_Weight = Total_Longitudinal_Struct_Weight

        
    
    
    def calc_NonStruct_Weights(
        self,
        additional_lightship_components=None,
        lightship_margin_inputs=None,
    ):
        '''
        This function calculates the outfit and machinery weights based on empirical formulas from Hart and Vlahopoulos 2008.
        Run this function after calculating steel sturctural weights

        engine power in kW is approximated by drag estimates from resistance calculations
        '''
        # Rebuild these rows whenever the nonstructural mass calculation is
        # rerun.  They are ordinary project feasibility constraints, not ABS
        # or statutory rules, and use calculated design values throughout.
        self.SystemFeasibilityConstraintsV1p1 = pd.DataFrame()

        # Calculate approximate unit distribution based on sectional areas
        
        idx_fwd_fuel_bulkhead = self.bulkhead_locs[-4]#Location of forward fuel tank bulkhead in meters
        SA_unit_dist = np.zeros((len(self.X),))
        SA_unit_dist[idx_fwd_fuel_bulkhead:] = self.sectional_Areas[idx_fwd_fuel_bulkhead:, -1] #relative area distribution from the fuel tank to the stern
        
        # Bias outfit weight to be more in the aft half of the ship from the fuel tank to the stern
        self.outfit_weight = (1.0*(self.hull.LOA**0.8) * ((2.0*self.hull.Bd)**0.6) *(self.hull.Dd**0.3) * self.Cb**0.5) * 1000.0 #in kg
        outfit_dist_weight = _scale_distribution_to_mass(
            SA_unit_dist,
            self.X,
            self.outfit_weight,
        ) #in kg/m distributed along the length of the ship
        outfit_VCG = np.trapezoid(
            outfit_dist_weight*self.sectional_Areas_Z_centroids[:, -1],
            x=self.X,
        )/self.outfit_weight #in meters
        outfit_LCG = np.trapezoid(
            outfit_dist_weight*self.X,
            x=self.X,
        )/self.outfit_weight #in meters


        item = pd.DataFrame({'Name': ['Outfit'], 'Total_Weight_kg': [self.outfit_weight], 'VCG': [outfit_VCG], 'LCG': outfit_LCG, 'Dist_Weight': [outfit_dist_weight]})
        if self.Weights.empty:
            # Preserve pandas' inferred numeric/object dtypes when outfit is
            # the first component in a focused calculation.  Concatenating
            # against an all-object empty template emits a pandas deprecation
            # warning and could change dtypes in a future release.
            self.Weights = item.copy()
        else:
            self.Weights = pd.concat(
                [self.Weights, item],
                ignore_index=True,
            )

        # Machinery weight is estimated from engine power
        machinery_weight = 0.17*(self.Engine_Power_kW**0.9) * 1000.0 #in kg distributed in engine room only
        machinery_dist_weight = np.zeros((len(self.X),))
        engine_room_start_idx = self.bulkhead_locs[-3]+1 #+1 to start just after the bulkhead
        engine_room_end_idx = self.bulkhead_locs[-2]-1 #-1 to end just before the bulkhead
        machinery_dist_weight[engine_room_start_idx:engine_room_end_idx] = 1.0
        machinery_dist_weight = _scale_distribution_to_mass(
            machinery_dist_weight,
            self.X,
            machinery_weight,
        )
        #Machinery VCG is approximately 1/3 of the engine room height above the double bottom 
        machinery_VCG = self.Db_height + (self.hull.Dd - self.Db_height)/3.0 #in meters
        machinery_LCG = np.trapezoid(
            machinery_dist_weight*self.X,
            x=self.X,
        )/machinery_weight #in meters

        item = pd.DataFrame({'Name': ['Machinery'], 'Total_Weight_kg': [machinery_weight], 'VCG': [machinery_VCG], 'LCG': machinery_LCG, 'Dist_Weight': [machinery_dist_weight]})
        self.Weights = pd.concat([self.Weights, item], ignore_index=True)
        
        self.structure_weight = self.Total_Longitudinal_Struct_Weight + self.bulkhead_weights.sum()

        # HDF-ASM-021 through HDF-ASM-025: hatch covers are explicit
        # preliminary-design lightship components.  Their scalar masses,
        # centers, and distributed representations are supplied by
        # ``tools.HatchCovers`` after the cargo-hold boundaries are known.
        # Include them before solving the residual deadweight so adding CAD
        # detail cannot silently make total ship mass exceed the displacement
        # selected by the operating-condition draft.
        additional_lightship_weight_kg = 0.0
        self.LightshipMarginComponent = pd.DataFrame()
        self.lightship_margin_weight = 0.0

        if additional_lightship_components is not None:
            additional_lightship_components = (
                additional_lightship_components.copy().reset_index(drop=True)
            )

            if not additional_lightship_components.empty:
                additional_lightship_weight_kg = float(
                    additional_lightship_components['Total_Weight_kg'].sum()
                )

                # The working weight table intentionally keeps only the common
                # load fields.  Hatch-specific dimensions remain available in
                # ShipSystemDesign.HatchCoverComponents and the detail export.
                component_rows = additional_lightship_components[
                    ['Name', 'Total_Weight_kg', 'VCG', 'LCG', 'Dist_Weight']
                ].copy()
                self.Weights = pd.concat(
                    [self.Weights, component_rows],
                    ignore_index=True,
                )

        # A growth allowance is calculated only when the caller supplies its
        # fraction and all mass-center coordinates.  Its base is the complete
        # modeled lightship assembled immediately above: structure, empirical
        # outfit/machinery, hatch covers, and any explicitly supplied
        # deckhouse.  The helper creates a point-equivalent grid load whose
        # trapezoidal mass and first moment match the assigned scalar values.
        modeled_lightship_before_margin_kg = (
            self.structure_weight
            + self.outfit_weight
            + machinery_weight
            + additional_lightship_weight_kg
        )

        if lightship_margin_inputs is not None:
            self.LightshipMarginComponent = (
                build_lightship_margin_component(
                    self.X,
                    modeled_lightship_mass_kg=(
                        modeled_lightship_before_margin_kg
                    ),
                    **dict(lightship_margin_inputs),
                )
            )
            self.lightship_margin_weight = float(
                self.LightshipMarginComponent['Total_Weight_kg'].sum()
            )
            margin_rows = self.LightshipMarginComponent[
                ['Name', 'Total_Weight_kg', 'VCG', 'LCG', 'Dist_Weight']
            ].copy()
            self.Weights = pd.concat(
                [self.Weights, margin_rows],
                ignore_index=True,
            )

        self.modeled_lightship_weight = (
            modeled_lightship_before_margin_kg
            + self.lightship_margin_weight
        )

        self.deadweight = (
            self.rho*self.V_disp
            - self.structure_weight
            - self.outfit_weight
            - machinery_weight
            - additional_lightship_weight_kg
            - self.lightship_margin_weight
        ) #in kg

        lightship_constraint = evaluate_lightship_displacement_constraint(
            self.rho*self.V_disp,
            self.modeled_lightship_weight,
        )
        self.SystemFeasibilityConstraintsV1p1 = pd.concat(
            [
                self.SystemFeasibilityConstraintsV1p1,
                lightship_constraint,
            ],
            ignore_index=True,
        )

        if self.deadweight <= 0.0:
            failed_row = lightship_constraint.iloc[0]
            raise PhysicalConstraintFailure(
                'Calculated lightship, including all active preliminary '
                'components, equals '
                'or exceeds displacement at the selected draft.',
                name='Modeled lightship fits selected displacement',
                requirement=(
                    'Modeled lightship mass shall be less than displacement '
                    'mass at the selected operating draft'
                ),
                threshold_value=failed_row['Threshold Value'],
                constraint_value=failed_row['Constraint Value'],
                units='kg',
                comparison='maximum',
                result_class='System Mass Feasibility',
                constraint_rows=(
                    self.SystemFeasibilityConstraintsV1p1.copy()
                ),
                diagnostics={
                    'Displacement_Mass_kg': self.rho*self.V_disp,
                    'Modeled_Lightship_Weight_kg': (
                        self.modeled_lightship_weight
                    ),
                    'Lightship_Displacement_Margin_kg': self.deadweight,
                },
            )
 
        misc_dwt = (2.0*(self.deadweight/1000.0)**0.5) *1000.0 #in kg (original formula in metric tons) Constraint
        self.cargo_weight = self.deadweight - self.fuel_weight - misc_dwt #in kg Constraint

        cargo_payload_constraint = (
            evaluate_positive_cargo_payload_constraint(self.cargo_weight)
        )
        self.SystemFeasibilityConstraintsV1p1 = pd.concat(
            [
                self.SystemFeasibilityConstraintsV1p1,
                cargo_payload_constraint,
            ],
            ignore_index=True,
        )

        if self.cargo_weight <= 0.0:
            failed_row = cargo_payload_constraint.iloc[0]
            raise PhysicalConstraintFailure(
                'No positive cargo payload remains after lightship, fuel, and '
                'miscellaneous deadweight are assigned at the selected draft.',
                name='Positive available cargo payload',
                requirement=(
                    'Available cargo mass after lightship, voyage fuel, and '
                    'miscellaneous deadweight shall be positive'
                ),
                threshold_value=failed_row['Threshold Value'],
                constraint_value=failed_row['Constraint Value'],
                units='kg',
                comparison='minimum',
                result_class='System Mass Feasibility',
                constraint_rows=(
                    self.SystemFeasibilityConstraintsV1p1.copy()
                ),
                diagnostics={
                    'Displacement_Mass_kg': self.rho*self.V_disp,
                    'Modeled_Lightship_Weight_kg': (
                        self.modeled_lightship_weight
                    ),
                    'Lightship_Displacement_Margin_kg': self.deadweight,
                    'Required_Fuel_Weight_kg': self.fuel_weight,
                    'Miscellaneous_Deadweight_kg': misc_dwt,
                    'Cargo_Weight_kg': self.cargo_weight,
                    'Available_Cargo_Displacement_Fraction': (
                        self.cargo_weight/(self.rho*self.V_disp)
                    ),
                },
            )
        misc_dwt_dist_weight = np.zeros((len(self.X),))
        misc_dwt_start_idx = self.bulkhead_locs[-3]+1 # Start at Fuel tank bulkhead
        misc_dwt_end_idx = len(self.X)-1
        #print('Misc Deadweight (kg): ', misc_dwt)
        #print('Cargo Weight (kg): ', self.cargo_weight)

        misc_dwt_dist_weight[misc_dwt_start_idx:misc_dwt_end_idx] = 1.0
        misc_dwt_dist_weight = _scale_distribution_to_mass(
            misc_dwt_dist_weight,
            self.X,
            misc_dwt,
        )
        misc_dwt_VCG = self.Db_height + (self.hull.Dd - self.Db_height)/2.0 #in meters
        # HDF-ASM-028: integrate the distributed deadweight first moment with
        # the same composite trapezoidal convention as the hull modules.
        misc_dwt_LCG = np.trapezoid(
            misc_dwt_dist_weight * self.X,
            x=self.X,
        ) / misc_dwt

        item = pd.DataFrame({'Name': ['Misc_DWT'], 'Total_Weight_kg': [misc_dwt], 'VCG': [misc_dwt_VCG], 'LCG': misc_dwt_LCG, 'Dist_Weight': [misc_dwt_dist_weight]})
        self.Weights = pd.concat([self.Weights, item], ignore_index=True)

        #print('--- Weight Summary --- line 224 weights')
        total_weight = self.Weights['Total_Weight_kg'].to_numpy().sum() + self.fuel_weight + self.cargo_weight

        #print('total weight (kg): ', total_weight)
        #print('total weight (tons): ', total_weight/1000.0)
    
    def calc_CG(self):
        '''
        This function calculates the center of gravity of the ship based on the weights and VCGs of each weight class.
        1) Calculate with full cargo, empty ballast, and fuel weights
        2) Calculate with empty cargo, full ballast, and empty fuel weights

        '''
  

        weights_1 = self.Weights[self.Weights['Name'] != 'Ballast_Tank']
        total_weight_1 = weights_1['Total_Weight_kg'].to_numpy().sum() #
        VCG_1 = (weights_1['Total_Weight_kg'].to_numpy() * weights_1['VCG'].to_numpy()).sum() / total_weight_1
        LCG_1 = (weights_1['Total_Weight_kg'].to_numpy() * weights_1['LCG'].to_numpy()).sum() / total_weight_1

        weights_2 = self.Weights[self.Weights['Name'] != 'Cargo_Hold']
        total_weight_2 = weights_2['Total_Weight_kg'].to_numpy().sum() #
        VCG_2 = (weights_2['Total_Weight_kg'].to_numpy() * weights_2['VCG'].to_numpy()).sum() / total_weight_2
        LCG_2 = (weights_2['Total_Weight_kg'].to_numpy() * weights_2['LCG'].to_numpy()).sum() / total_weight_2
        

        Ship_CG = pd.DataFrame({'Condition': ['Full Cargo', 'Full Ballast'], 
                                'Total_Weight_kg': [total_weight_1, total_weight_2], 
                                'Displacement_t': [total_weight_1/1000.0, total_weight_2/1000.0],
                                'VCG': [VCG_1, VCG_2], 
                                'LCG': [LCG_1, LCG_2]})
        self.Ship_CG = Ship_CG

        return Ship_CG
    
    def calc_Distributed_Load(self, load_condition:str):
        '''
        This function calculates the total distributed weight along the length of the ship.
        '''
        if load_condition == 'Full Cargo':
            weights = self.Weights[self.Weights['Name'] != 'Ballast_Tank']
            #print('Weight Condition: Full Cargo tons ', weights['Total_Weight_kg'].to_numpy().sum()/1000.0)
        elif load_condition == 'Full Ballast':
            weights = self.Weights[self.Weights['Name'] != 'Cargo_Hold']
        
        #print('Total Weight (kg): ', weights['Total_Weight_kg'].to_numpy().sum())
        #print('Number of Weight Components: ', len(weights))

        #print('--- Weight Breakdown ---')
        #print(weights)
        total_mass_distribution_kg_per_m = np.zeros(len(self.X))

        for row_index, weight in weights.iterrows():

            # Normalize again at the aggregation boundary.  Cargo and tank
            # distributions are produced by another module, and this check
            # guarantees that every component entering longitudinal strength
            # carries exactly the mass recorded in the scalar weight table.
            normalized_distribution = _scale_distribution_to_mass(
                weight['Dist_Weight'],
                self.X,
                weight['Total_Weight_kg'],
            )
            self.Weights.at[row_index, 'Dist_Weight'] = (
                normalized_distribution
            )
            total_mass_distribution_kg_per_m += normalized_distribution

        self.Distributed_Load = (
            -total_mass_distribution_kg_per_m*self.g
        ) #in N/m (negative for downward force)

        selected_scalar_mass_kg = weights['Total_Weight_kg'].sum()
        integrated_distributed_mass_kg = np.trapezoid(
            total_mass_distribution_kg_per_m,
            x=self.X,
        )

        # Mass closure alone cannot reveal a longitudinal placement error.
        # Compare the first moment of every distributed component with the
        # scalar mass/LCG table while leaving the physical load shape intact.
        # A nonzero value is diagnostic; no artificial load shift is applied.
        selected_scalar_first_moment_kg_m = (
            weights['Total_Weight_kg'].to_numpy(dtype=float)
            * (
                weights['LCG'].to_numpy(dtype=float)
                - self.X[0]
            )
        ).sum()
        integrated_distributed_first_moment_kg_m = np.trapezoid(
            total_mass_distribution_kg_per_m*(self.X - self.X[0]),
            x=self.X,
        )

        # Solve sinkage *and trim* on the same sectional-area grid that
        # supplies distributed buoyancy.  Matching volume alone closes force
        # but leaves an unbalanced first moment whenever LCG and upright LCB
        # differ.  The two-endpoint waterplane solve below closes both physical
        # equilibrium equations before shear and bending are integrated.
        displacement_volume_curve_m3 = np.trapezoid(
            self.sectional_Areas,
            x=self.X,
            axis=0,
        )
        target_displacement_volume_m3 = selected_scalar_mass_kg/self.rho
        target_lcg_m = float(
            selected_scalar_first_moment_kg_m/selected_scalar_mass_kg
            + self.X[0]
        )

        if (
            target_displacement_volume_m3
            < displacement_volume_curve_m3[0]
            or target_displacement_volume_m3
            > displacement_volume_curve_m3[-1]
        ):
            raise LongitudinalEquilibriumError(
                'Selected loading-condition mass lies outside the generated '
                'hull displacement-volume range.',
                Target_Displacement_Volume_m3=(
                    target_displacement_volume_m3
                ),
                Target_LCG_m=target_lcg_m,
                Minimum_Generated_Displacement_Volume_m3=float(
                    displacement_volume_curve_m3[0]
                ),
                Maximum_Generated_Displacement_Volume_m3=float(
                    displacement_volume_curve_m3[-1]
                ),
                Normalized_Displacement_Range_Margin=float(
                    min(
                        (
                            target_displacement_volume_m3
                            - displacement_volume_curve_m3[0]
                        ),
                        (
                            displacement_volume_curve_m3[-1]
                            - target_displacement_volume_m3
                        ),
                    )
                    / max(
                        abs(target_displacement_volume_m3),
                        np.finfo(float).eps,
                    )
                ),
            )

        equilibrium = solve_longitudinal_buoyancy_equilibrium(
            self.X,
            self.Z,
            self.sectional_Areas,
            target_displacement_volume_m3,
            target_lcg_m,
            self.design_draft,
        )

        if not equilibrium['Equilibrium_Solved']:
            raise LongitudinalEquilibriumError(
                'The selected loading condition cannot reach simultaneous '
                'displacement and longitudinal-moment equilibrium within the '
                'generated hull draft grid. '
                f"Solver result: {equilibrium['Solver_Message']}",
                Target_Displacement_Volume_m3=(
                    target_displacement_volume_m3
                ),
                Target_LCG_m=target_lcg_m,
                Minimum_Generated_Draft_m=float(self.Z[0]),
                Maximum_Generated_Draft_m=float(self.Z[-1]),
                Normalized_Displacement_Volume_Residual=float(
                    equilibrium['Normalized_Volume_Residual']
                ),
                Normalized_Displacement_First_Moment_Residual=float(
                    equilibrium['Normalized_First_Moment_Residual']
                ),
                Solver_Status=int(equilibrium['Solver_Status']),
                Solver_Function_Evaluations=int(
                    equilibrium['Function_Evaluations']
                ),
            )

        forward_equilibrium_draft_m = equilibrium['Forward_Draft_m']
        aft_equilibrium_draft_m = equilibrium['Aft_Draft_m']
        load_equilibrium_draft_m = 0.5*(
            forward_equilibrium_draft_m + aft_equilibrium_draft_m
        )

        # Convert the physically equilibrated local immersed section areas to
        # an upward force per unit length.
        self.buoyancy_dist = (
            self.rho
            * self.g
            * equilibrium['Local_Sectional_Areas_m2']
        )

        integrated_buoyancy_force_n = np.trapezoid(
            self.buoyancy_dist,
            x=self.X,
        )
        integrated_weight_force_n = np.trapezoid(
            self.Distributed_Load,
            x=self.X,
        )
        net_vertical_force_n = (
            integrated_weight_force_n + integrated_buoyancy_force_n
        )
        integrated_weight_first_moment_nm = np.trapezoid(
            self.Distributed_Load*(self.X - self.X[0]),
            x=self.X,
        )
        integrated_buoyancy_first_moment_nm = np.trapezoid(
            self.buoyancy_dist*(self.X - self.X[0]),
            x=self.X,
        )

        self.Net_Load = self.Distributed_Load + self.buoyancy_dist
        self.LoadClosureDiagnostics = {
            'Load_Condition': load_condition,
            'Nominal_Design_Draft_m': float(self.design_draft),
            'Load_Equilibrium_Draft_m': load_equilibrium_draft_m,
            'Load_Equilibrium_Forward_Draft_m': float(
                forward_equilibrium_draft_m
            ),
            'Load_Equilibrium_Aft_Draft_m': float(
                aft_equilibrium_draft_m
            ),
            'Load_Equilibrium_Trim_m': float(
                aft_equilibrium_draft_m - forward_equilibrium_draft_m
            ),
            'Load_Equilibrium_Sinkage_m': float(
                load_equilibrium_draft_m - self.design_draft
            ),
            'Longitudinal_Equilibrium_Solved': bool(
                equilibrium['Equilibrium_Solved']
            ),
            'Longitudinal_Equilibrium_Solver_Status': int(
                equilibrium['Solver_Status']
            ),
            'Longitudinal_Equilibrium_Function_Evaluations': int(
                equilibrium['Function_Evaluations']
            ),
            'Normalized_Displacement_Volume_Residual': float(
                equilibrium['Normalized_Volume_Residual']
            ),
            'Normalized_Displacement_First_Moment_Residual': float(
                equilibrium['Normalized_First_Moment_Residual']
            ),
            'Target_Displacement_Volume_m3': float(
                target_displacement_volume_m3
            ),
            'Selected_Scalar_Mass_kg': float(selected_scalar_mass_kg),
            'Integrated_Distributed_Mass_kg': float(
                integrated_distributed_mass_kg
            ),
            'Mass_Distribution_Residual_kg': float(
                integrated_distributed_mass_kg - selected_scalar_mass_kg
            ),
            'Selected_Scalar_First_Moment_kg_m': float(
                selected_scalar_first_moment_kg_m
            ),
            'Integrated_Distributed_First_Moment_kg_m': float(
                integrated_distributed_first_moment_kg_m
            ),
            'Distribution_First_Moment_Residual_kg_m': float(
                integrated_distributed_first_moment_kg_m
                - selected_scalar_first_moment_kg_m
            ),
            'Integrated_Weight_Force_N': float(integrated_weight_force_n),
            'Integrated_Buoyancy_Force_N': float(
                integrated_buoyancy_force_n
            ),
            'Net_Vertical_Force_Residual_N': float(net_vertical_force_n),
            'Net_Vertical_Force_Residual_Fraction': float(
                net_vertical_force_n/
                max(abs(integrated_weight_force_n), np.finfo(float).eps)
            ),
            'Integrated_Weight_First_Moment_Nm': float(
                integrated_weight_first_moment_nm
            ),
            'Integrated_Buoyancy_First_Moment_Nm': float(
                integrated_buoyancy_first_moment_nm
            ),
            'Net_Vertical_First_Moment_Residual_Nm': float(
                integrated_weight_first_moment_nm
                + integrated_buoyancy_first_moment_nm
            ),
        }
        
    def calc_ShearForce(self):
        '''
        Integrate the physical net load into longitudinal shear force.

        The former implementation subtracted a linear ramp solely to force
        the final shear value to zero.  That operation hid weight/buoyancy
        imbalance and broke the differential relationship between load and
        shear.  The cumulative trapezoidal integral below preserves the
        calculated load and publishes the endpoint residual for review.
        '''
        self.Shear_Force = np.zeros((len(self.X),))

        net_load = self.Distributed_Load + self.buoyancy_dist
        interval_load_n = (
            0.5*(net_load[:-1] + net_load[1:])*np.diff(self.X)
        )
        self.Shear_Force[1:] = np.cumsum(interval_load_n)

        self.LoadClosureDiagnostics.update(
            {
                'Forward_Shear_Force_N': float(self.Shear_Force[0]),
                'Aft_Shear_Force_Residual_N': float(
                    self.Shear_Force[-1]
                ),
            }
        )

        return np.min(self.Shear_Force[:]), np.max(self.Shear_Force[:])


    def calc_BendingMoment(self):
        '''
        Integrate shear into bending moment without forcing the aft endpoint.

        A free-floating, equilibrated load case should naturally return to
        zero force and moment.  Any aft-end residual is evidence about the
        current load/CG equilibrium and remains visible in the diagnostic.
        '''
        self.Bending_Moment = np.zeros((len(self.X),))

        interval_moment_nm = (
            0.5*(self.Shear_Force[:-1] + self.Shear_Force[1:])
            * np.diff(self.X)
        )
        self.Bending_Moment[1:] = np.cumsum(interval_moment_nm)

        net_load = self.Distributed_Load + self.buoyancy_dist
        net_load_first_moment_nm = np.trapezoid(
            net_load*(self.X - self.X[0]),
            x=self.X,
        )
        self.LoadClosureDiagnostics.update(
            {
                'Forward_Bending_Moment_Nm': float(
                    self.Bending_Moment[0]
                ),
                'Aft_Bending_Moment_Residual_Nm': float(
                    self.Bending_Moment[-1]
                ),
                'Net_Load_First_Moment_About_Forward_Nm': float(
                    net_load_first_moment_nm
                ),
            }
        )
            
        # Return the max sagging and hogging bending moments
        return np.min(self.Bending_Moment), np.max(self.Bending_Moment) #in Nm (min is sagging, max is hogging)
    
    def plot_Loads(self):
        '''
        This function plots the distributed load, shear force, and bending moment along the length of the ship.
        '''
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        bow, stern = self.hull.gen_Bow_and_Stern_Profiles(self.Z)
        min_bow = np.min(bow)
        bow = bow - min_bow  # Shift bow to start at 0
        stern = stern - min_bow # Shift stern by same amount as bow
        
        '''
        ax_twin = axs[0].twinx()
        
        ax_twin.plot(self.Z, bow, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot(self.Z, stern, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[0], stern[0]], [self.Z[0], self.Z[0]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[-1], stern[-1]], [self.Z[-1], self.Z[-1]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.set_ylabel('Bow and Stern Profile (m)')
        ax_twin.set_ylim([self.Z[0], 4*self.Z[-1]])
        '''


        #axs[0].fill_betweenx(self.Z, 0, self.hull.bow_profile(self.Z), color=[0.8,0.8,0.8], alpha=0.5)
        axs[0].plot(self.X, self.Distributed_Load/1000.0, linewidth=1, color=[0,0,1])
        axs[0].plot(self.X, self.buoyancy_dist/1000.0, linewidth=1, color=[0,1,0])
        axs[0].plot(self.X, (self.Distributed_Load + self.buoyancy_dist)/1000.0, linewidth=1, color=[1,0,0])
        axs[0].set_title('Distributed Load (kN/m)')
        axs[0].set_xlabel('Length (m)')
        axs[0].set_ylabel('Load (kN/m)')
        axs[0].grid()

        '''
        ax_twin = axs[1].twinx()
        ax_twin.plot(self.X, self.Distributed_Load/1000.0, linewidth=1, color=[0,0,1])
        ax_twin.plot(self.Z, bow, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot(self.Z, stern, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[0], stern[0]], [self.Z[0], self.Z[0]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[-1], stern[-1]], [self.Z[-1], self.Z[-1]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.set_ylabel('Bow and Stern Profile (m)')
        ax_twin.set_ylim([self.Z[0], 4*self.Z[-1]])
        '''

        axs[1].plot(self.X, self.Shear_Force[:]/1000.0, linewidth=1, color=[1,0,0])
        #axs[1].plot(self.X, self.Shear_Force[:,1]/1000.0, linewidth=1, color=[0,0,1])
        axs[1].set_title('Shear Force (kN)')
        axs[1].set_xlabel('Length (m)')
        axs[1].set_ylabel('Shear Force (kN)')
        axs[1].grid()
        
        '''
        ax_twin = axs[2].twinx()
        ax_twin.plot(self.X, self.Distributed_Load/1000.0, linewidth=1, color=[0,0,1])
        ax_twin.plot(self.Z, bow, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot(self.Z, stern, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[0], stern[0]], [self.Z[0], self.Z[0]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[-1], stern[-1]], [self.Z[-1], self.Z[-1]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.set_ylabel('Bow and Stern Profile (m)')
        ax_twin.set_ylim([self.Z[0], 4*self.Z[-1]])
        '''
        axs[2].plot(self.X, self.Bending_Moment/1e6, linewidth=1, color=[0,0.5,0])
        axs[2].set_title('Bending Moment (MN-m)')
        axs[2].set_xlabel('Length (m)')
        axs[2].set_ylabel('Bending Moment (MN-m)')
        axs[2].grid()

        plt.tight_layout()
        #plt.show()
        return fig, axs



        
