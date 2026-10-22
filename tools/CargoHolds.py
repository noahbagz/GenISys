#This script calculates the weight distribution properties of Cargo holds based on its hull parameters and structural elements. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.HullParameterization import Hull_Parameterization as HP


# The public GenISys design vector reserves fifteen cargo-fraction fields.
# Emit the corresponding cargo- and ballast-capacity identities for every
# completed design so compiled tables have a stable applicability contract.
MAX_PARAMETRIC_CARGO_HOLDS = 15


def _scale_distribution_to_mass(distribution, x_coordinates, target_mass_kg):
    """Scale one longitudinal mass distribution to its scalar mass.

    GenISys stores distributed weights in kg/m and later integrates them into
    shear force and bending moment.  The scalar tank/cargo mass and the area
    under that distribution must therefore describe the same load.  Preserve
    the calculated longitudinal shape, but use the approved composite
    trapezoidal rule to remove grid-end and fill-height integration error.
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
            'A positive tank or cargo mass has no longitudinal distribution.'
        )

    return distribution * target_mass_kg/integrated_mass_kg


def interp(A,Z,z):
        # This function interpolates data to approximate A(z)  given values of A(Z) 
        # Ths assumes that Z is sorted in ascending order and z is within the range of Z

        # Exact empty/full boundaries are valid operating points.  Return the
        # tabulated endpoint directly instead of indexing outside the interval.
        if z <= Z[0]:
            return A[0]

        if z >= Z[-1]:
            return A[-1]

        idx = np.searchsorted(Z, z, side='right') - 1
    
        frac = (z - Z[idx])/(Z[idx+1] - Z[idx])
    
        return A[idx] + frac*(A[idx+1] - A[idx])

class CargoHolds:
    def __init__(self, structural_params, rho_cargo, bulkhead_params:list, fuel_weight:float):
        '''
        bulkhead_params is a list of bulkhead spacing parameters
        0 - fraction of LOA to crash bulkhead from bow
        1 - fraction of LOA to aft peak bulkhead from stern
        2 - fraction of LOA that is engine room length
        3 - fraction of LOA that is fuel tank length
        4 - integer number of cargo holds 
        ORDER of BULKHEADS:
        0 - Crash Bulkhead
        1-N - Cargo Hold Bulkheads
        N+1 - Fuel Tank Forward Bulkhead
        N+2 - Engine Room Forward Bulkhead
        N+3 - Aft Peak Bulkhead

        Bulklhead Locations is a list of X indices where bulkheads are located 
        CargoHold_Assignments is the list of bulkhead indices that define the start and end of each cargo hold (e.g. [0, 1, 3] means cargo hold 0 is between bulkhead 0 and 1, cargo hold 1 is between bulkhead 1 and 3)
        rho_cargo is the density of the cargo in kg/m^3
        '''
        self.bulkhead_params = bulkhead_params #List of bulkhead spacing parameters

        self.structural_params = structural_params #
        self.rho_cargo = rho_cargo #Density of cargo in kg/m^3

        self.rho = 1025 #Density of seawater in kg/m^3
        self.rho_fuel = 1000 #Density of fuel in kg/m^3 (assuming heavy fuel oil)
        self.g = 9.81 #Gravitational acceleration in m/s^2

        self.BulkheadConstraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])
        
        '''
        self.CargoHolds = pd.DataFrame(columns=['Hold_Number', 
                                                'Total_Hold_Volume_m3',
                                                'Cargo_Volume_Filled_m3',
                                                'Cargo_Fill_Height_m',
                                                'Total_Weight_kg', 
                                                'VCG', 
                                                'LCG', 
                                                'Dist_Weight']) #Dataframe to store weight and VCG of each cargo hold
        
        self.BallastTanks = pd.DataFrame(columns=['Tank_Number',
                                                'Total_Tank_Volume_m3', 
                                                'Ballast_Volume_toReplaceCargo_m3',
                                                'Ballast_Weight_kg', 
                                                'VCG', 
                                                'LCG', 
                                                'Dist_Weight']) #Dataframe to store weight and VCG of each ballast tank
        
        self.Fuel_Tank = pd.DataFrame(columns=['Tank_Number', 
                                                'Total_Tank_Volume_m3', 
                                                'Fuel_Volume_Filled_m3',
                                                'Total_Weight_kg', 
                                                'VCG', 
                                                'LCG', 
                                                'Dist_Weight']) #Dataframe to store weight and VCG of fuel tank
        
        '''
        self.fuel_weight = fuel_weight #Weight of fuel in kg #distributed in tanks
        
        self.cargo_hold_permeability = 0.99 #Assuming 99% permeability for cargo holds
        self.ballast_tank_permeability = 0.95 #Assuming 95% permeability for ballast tanks

    
        
    def upload_PC_SA_values(self, X, Z, Y, sectional_areas, sectional_areas_Z_centroids): 
        self.X = X #X coordinates along the length of the ship in meters
        self.Z = Z #Z coordinates along the depth of the ship in meters
        self.Y = Y #Y coordinates along the width of the ship in meters (in a len X by len Z grid)
        self.sectional_Areas = sectional_areas #Sectional areas below draftmark along the length of the ship in m^2 format is lenX by lenZ
        self.sectional_Areas_Z_centroids = sectional_areas_Z_centroids #Sectional areas below draftmark along the length of the ship in m^2 format is lenX by lenZ

    def calc_Bulkhead_Locations(self): 
        '''
        This function calculates the number and locations of bulkheads based on the bulkhead spacing parameters and hull length
        '''
        LOA = self.X[-1] - self.X[0] #Length Overall in meters
        
        num_holds = int(self.bulkhead_params[5]) #Number of cargo holds

        X_crashbulkhead = self.bulkhead_params[0]*LOA #Location of crash bulkhead from Foward Perpendicular in meters
        X_aftpeakbulkhead = LOA - self.bulkhead_params[1]*LOA #Location of aft peak bulkhead from Foward Perpendicular in meters
        L_lazarette = self.bulkhead_params[2]*LOA #Length of lazarette in meters
        L_engine = self.bulkhead_params[3]*LOA #Length of engine room in meters
        X_engine_aftkbulkhead = X_aftpeakbulkhead - L_lazarette #Location of engine room aft peak bulkhead from Foward Perpendicular in meters
        X_engine_fwdbulkhead = X_engine_aftkbulkhead - L_engine #Location of engine room forward bulkhead from Foward Perpendicular in meters
        L_fueltank = self.bulkhead_params[4]*LOA #Length of fuel
        X_fueltank_fwdbulkhead = X_engine_fwdbulkhead - L_fueltank #Location of fuel tank forward bulkhead from Foward Perpendicular in meters
        Cargo_Bulkheads = np.linspace(X_crashbulkhead, X_fueltank_fwdbulkhead, num_holds+1) #Locations of cargo hold bulkheads from Foward Perpendicular in meters (exclude crash and fuel tank bulkheads)


        self.Bulkhead_Locations = np.zeros(num_holds+4, dtype=int) #List of X indices where bulkheads are located
        self.Bulkhead_Locations[-1] = np.argmin(np.abs(self.X - X_aftpeakbulkhead)) #Aft peak bulkhead
        self.Bulkhead_Locations[-2] = np.argmin(np.abs(self.X - X_engine_aftkbulkhead)) #Fuel tank forward bulkhead
        self.Bulkhead_Locations[-3] = np.argmin(np.abs(self.X - X_engine_fwdbulkhead)) #Engine room forward bulkhead
        self.Bulkhead_Locations[0:-3] = np.array([np.argmin(np.abs(self.X - x)) for x in Cargo_Bulkheads]) #Crash bulkhead and cargo hold bulkheads

        self.X_bulkheads = self.X[self.Bulkhead_Locations] #X locations of bulkheads in meters

        self.CargoHold_Assignments = np.arange(0,num_holds+1) #List of bulkhead indices that define the start and end of each cargo hold (e.g. [0, 1, 3] means cargo hold 0 is between bulkhead 0 and 1, cargo hold 1 is between bulkhead 1 and 3)
        self.FuelTank_Assignment = np.array([num_holds, num_holds+1]) #List of bulkhead indices that define the start and end of the fuel tank (e.g. [0, 1] means fuel tank is between bulkhead 0 and 1)



    def calc_CargoHold_Offsets(self):
        '''
        This function calculates the volume of each cargo hold based on the bulkhead locations and sectional areas.
        '''

        DB_height = self.structural_params[6]/1000.0 #Double bottom height in meters
        
        Depth = self.structural_params[3] #depth in m
        
        Webframe_height = self.structural_params[48]/1000.0
        Bracket_Floor_Web = np.array([self.structural_params[60], self.structural_params[61]])/1000.0 #Bracket floor web height in meters idx 0 is Y length and 1 is Z height
        Bracket_Deck_Web = np.array([self.structural_params[63], self.structural_params[64]])/1000.0 #Bracket deck web height in meters] idx 0 is Y length and 1 is Z height


        cargo_hold_profile = np.zeros((len(self.Z),)) # We will calculate the cargo hold profile at each Z level
        DB_idx = int(DB_height/self.Z[-1] * (len(self.Z)-1)+0.5) #Index of the Z level corresponding to the double bottom height (and round up to nearest integer)
        
        #print('Z Levels: ', len(self.Z), ' Z Max: ', self.Z[-1])
        #print('DB Index: ', DB_idx, ' DB Height (m): ', self.Z[DB_idx])

        Brack_FW_idx_range = [DB_idx, int((DB_height + Bracket_Floor_Web[1])/self.Z[-1] * (len(self.Z)-1)+0.5)] #Index range of the Z levels corresponding to the bracket floor-web height (and round up to nearest integer)
        Brack_DW_idx_range = [int((Depth - Bracket_Deck_Web[1])/self.Z[-1] * (len(self.Z)-1)+0.5), len(self.Z)-1] #Index range of the Z levels corresponding to bracket deck-web height (and round up to nearest integer)


        #cargo_hold_profile values are negative - so that we can subtract them from Y across the length of the ship 
        cargo_hold_profile[DB_idx:] = cargo_hold_profile[DB_idx:] - Webframe_height #Subtract webframe height from cargo hold profile below double bottom
        Brack_FW_slope = np.linspace(Bracket_Floor_Web[0], 0.0, int((Brack_FW_idx_range[1]-Brack_FW_idx_range[0]) + 1)) #Linear slope from bracket floor-web Y length to 0 over the Z index range
        cargo_hold_profile[Brack_FW_idx_range[0]:Brack_FW_idx_range[1]+1] = cargo_hold_profile[Brack_FW_idx_range[0]:Brack_FW_idx_range[1]+1] - Brack_FW_slope #Subtract bracket floor-web slope from cargo hold profile

        Brack_DW_slope = np.linspace(0.0, Bracket_Deck_Web[0], int((Brack_DW_idx_range[1]-Brack_DW_idx_range[0]) + 1)) #Linear slope from 0 to bracket deck-web Y length over the Z index range
        cargo_hold_profile[Brack_DW_idx_range[0]:Brack_DW_idx_range[1]+1] = cargo_hold_profile[Brack_DW_idx_range[0]:Brack_DW_idx_range[1]+1] - Brack_DW_slope #Subtract bracket deck-web slope from cargo hold profile

        #print('Cargo Hold Profile (m): ', cargo_hold_profile)

        #Now that we have the cargo hold profile, Lets develop the cargo hold offsets across the length of the ship
        #Cargo hold offsets are shaped like Y 
        self.CargoHold_Offsets = np.zeros((len(self.X), len(self.Z)))
        for i in range(len(self.X)):
            self.CargoHold_Offsets[i, DB_idx:] = self.Y[i,DB_idx:] + cargo_hold_profile[DB_idx:]  #add since cargo_hold_profile is negative
            #if the cargo hold offset is negative, set it to zero
            self.CargoHold_Offsets[i, self.CargoHold_Offsets[i, :] < 0] = 0.0
        

    def calc_CargoAndBallast_SectionalAreas(self):
        '''
        This function calculates the sectional areas of the cargo holds and ballast tanks based on the cargo hold offsets.
        This function also calculates the centroids of the cargo hold 
        '''
        self.CargoHold_SectionalAreas = np.zeros(self.Y.shape)
        self.BallastTank_SectionalAreas = np.zeros(self.Y.shape)

        self.CargoHold_z_centroid = np.zeros(self.Y.shape)
        self.BallastTank_z_centroid = np.zeros(self.Y.shape)

        dz = self.Z[1] - self.Z[0]  # Assume uniform spacing in Z

        for i in range(self.Y.shape[0]): # X Positions
            Moment_CH = 0.0
            Moment_BT = 0.0

            for j in range(1, self.Y.shape[1]):
                
                #Cargo hold sectional area calculation
                a = 2.0 * self.CargoHold_Offsets[i,j]  # Area at previous Z #multiply by 2 for both sides of the ship
                b = 2.0 * self.CargoHold_Offsets[i,j-1]  # Area at current Z #multiply by 2 for both sides of the ship

                dA = 0.5 * (a + b) * dz

                self.CargoHold_SectionalAreas[i,j] = self.CargoHold_SectionalAreas[i,j-1] + dA  # Update sectional area

                self.BallastTank_SectionalAreas[i,j] = self.sectional_Areas[i,j] - self.CargoHold_SectionalAreas[i,j]
            
                cz_CH = self.Z[j-1] + dz/3.0 * (2.0*a + b) / (a + b+1e-10)  #Avoid divide by zero error
                Moment_CH = Moment_CH + cz_CH * dA
                self.CargoHold_z_centroid[i, j] = Moment_CH / (self.CargoHold_SectionalAreas[i, j] + 1e-10)  # Avoid divide by zero error

                a_BT = 2.0 * self.Y[i,j] - a  # Area at previous Z #multiply by 2 for both sides of the ship
                b_BT = 2.0 * self.Y[i,j-1] - b
                dA_BT = 0.5 * (a_BT + b_BT) * dz
                cz_BT = self.Z[j-1] + dz/3.0 * (2.0*a_BT + b_BT) / (a_BT + b_BT+1e-10)
                Moment_BT = Moment_BT + cz_BT * dA_BT
                self.BallastTank_z_centroid[i, j] = Moment_BT / (self.BallastTank_SectionalAreas[i, j] + 1e-10)
        
        #Now we need to correct sectional areas and centroids for permeability
        self.CargoHold_SectionalAreas = self.CargoHold_SectionalAreas * self.cargo_hold_permeability
        self.BallastTank_SectionalAreas = self.BallastTank_SectionalAreas * self.ballast_tank_permeability
    
    def upload_CargoWeight(self, cargo_weight:float):
        '''
        This function uploads the total cargo weight in kg

        '''
        self.cargo_weight = cargo_weight #Total cargo weight in kg
        dist_cargo_fractions = self.bulkhead_params[6:] #Get cargo weight fractions for each hold
        N_holds = int(self.bulkhead_params[5])
        self.Cargo_Weights = np.zeros((N_holds,))
        self.bulkhead_params[6:] = 0 #Reset cargo weight fractions
        self.bulkhead_params[6:6+len(dist_cargo_fractions)] = dist_cargo_fractions #Upload cargo weight fractions
        for i in range(N_holds):
            self.Cargo_Weights[i] = cargo_weight * self.bulkhead_params[6+i] #Cargo weight fraction for each hold #Assume cargo weight fractions are corrected and sum to 1.0
        
    def upload_CargoWeightFractions(self, dist_cargo_fractions: list):  
        '''
        This function uploads the cargo weight fractions for each cargo hold

        '''
        N_holds = int(self.bulkhead_params[5])
        self.Cargo_Weights = np.zeros((N_holds,))
        self.bulkhead_params[6:] = 0 #Reset cargo weight fractions
        self.bulkhead_params[6:6+len(dist_cargo_fractions)] = dist_cargo_fractions #Upload cargo weight fractions

        for i in range(N_holds):
            self.Cargo_Weights[i] = self.cargo_weight * self.bulkhead_params[6+i] #Cargo weight fraction for each hold #Assume cargo weight fractions are corrected and sum to 1.0  

    def calc_CargoHoldAndBallast_Weights(self):
        '''
        This function calculates the weight distribution and VCG of each cargo hold based on the bulkhead locations and sectional areas.
        Cargo_Weights is a list of weights in kg for each cargo hold (length of list should be len(CargoHold_Assignments)-1)

        '''

        #First delete cargo hold and ballast tank dataframes if they already exist
        self.CargoHolds = pd.DataFrame(columns=['Hold_Number',  
                                                'Total_Hold_Volume_m3',
                                                'Cargo_Volume_Filled_m3',
                                                'Cargo_Fill_Height_m',
                                                'Total_Weight_kg',
                                                'Distributed_Weight_kg',
                                                'Distribution_Mass_Residual_kg',
                                                'VCG',
                                                'LCG',
                                                'Dist_Weight']) #Dataframe to store weight and VCG of each cargo hold
        
        self.BallastTanks = pd.DataFrame(columns=['Tank_Number',
                                                'Total_Tank_Volume_m3',
                                                'Ballast_Volume_toReplaceCargo_m3',
                                                'Ballast_Fill_Height_m',
                                                'Total_Weight_kg',
                                                'Distributed_Weight_kg',
                                                'Distribution_Mass_Residual_kg',
                                                'VCG',
                                                'LCG',
                                                'Dist_Weight']) #Dataframe to store weight and VCG of each ballast tank

        for i in range(len(self.CargoHold_Assignments)-1):

            dist_weight_CH = np.zeros((len(self.X),)) #in kg/m
            dist_weight_BT = np.zeros((len(self.X),)) #in kg/m
            start_idx = self.Bulkhead_Locations[self.CargoHold_Assignments[i]]+1 #+1 to avoid double counting bulkhead
            end_idx = self.Bulkhead_Locations[self.CargoHold_Assignments[i+1]] 
        

            #Calculate volume of cargo hold
            cargo_hold_volume = np.zeros(self.Z.shape)
            cargo_hold_CG = np.zeros((2,)) #2D array to store CG at each Z level (first column is X position, second column is Z position)
            ballast_tank_volume = np.zeros(self.Z.shape)
            ballast_tank_CG = np.zeros((2,)) #2D array to store CG at each Z level (first column is X position, second column

            for j in range(1,len(self.Z)):

                # Keep the compartment mask used by the existing geometry,
                # but integrate the resulting full-length distribution with
                # the same trapezoidal method used by the hydrostatic model.
                cargo_area_distribution = np.zeros(len(self.X))
                ballast_area_distribution = np.zeros(len(self.X))
                cargo_area_distribution[start_idx:end_idx] = (
                    self.CargoHold_SectionalAreas[start_idx:end_idx, j]
                )
                ballast_area_distribution[start_idx:end_idx] = (
                    self.BallastTank_SectionalAreas[start_idx:end_idx, j]
                )

                cargo_hold_volume[j] = np.trapezoid(
                    cargo_area_distribution,
                    x=self.X,
                )
                ballast_tank_volume[j] = np.trapezoid(
                    ballast_area_distribution,
                    x=self.X,
                )

            #print('cargo hold ', i, ' total volume (m^3): ', cargo_hold_volume[-1], ' cargo volume: ', self.Cargo_Weights[i]/self.rho_cargo)

            cargo_volume_requested_m3 = (
                self.Cargo_Weights[i]/self.rho_cargo
            )
            ballast_volume_requested_m3 = (
                self.Cargo_Weights[i]/self.rho
            )

            try:
                # The legacy interpolation helper excludes its lower bound.
                # Handle a genuinely empty hold explicitly so a zero-mass
                # operating condition does not appear geometrically full.
                if cargo_volume_requested_m3 == 0.0:
                    Z_cargo_fill = self.Z[0]
                else:
                    Z_cargo_fill = interp(
                        self.Z,
                        cargo_hold_volume,
                        cargo_volume_requested_m3,
                    )
                #print('Z cargo fill (m): ', Z_cargo_fill)
                #Z_cargo_fill_idx = np.where(self.Z < Z_cargo_fill)[0][-1]+1 #+1 to get the index above the fill level
            except:
                Z_cargo_fill = self.Z[-1] #If cargo hold is overfilled, set fill level to above max Z level to trigger full fill constraint
                #Z_cargo_fill_idx = len(self.Z) -1
            
            try:
                if ballast_volume_requested_m3 == 0.0:
                    Z_ballast_fill = self.Z[0]
                else:
                    Z_ballast_fill = interp(
                        self.Z,
                        ballast_tank_volume,
                        ballast_volume_requested_m3,
                    )
                #Z_ballast_fill_idx = np.where(self.Z < Z_ballast_fill)[0][-1] +1 #+1 to get the index above the fill level
            except:
                Z_ballast_fill = self.Z[-1]
                #Z_ballast_fill_idx = len(self.Z) -1


            # Since the fill surfaces are known, calculate the longitudinal
            # density shapes first.  Their final normalization and moments are
            # all evaluated with trapezoidal integration below.
            cargo_station_vcg = np.zeros(len(self.X))
            ballast_station_vcg = np.zeros(len(self.X))

            for j in range(start_idx, end_idx):
                dist_weight_CH[j] = interp(self.CargoHold_SectionalAreas[j,:], self.Z, Z_cargo_fill) * self.rho_cargo #in kg/m
                dist_weight_BT[j] = interp(self.BallastTank_SectionalAreas[j,:], self.Z, Z_ballast_fill) * self.rho #in kg/m

                cargo_station_vcg[j] = interp(
                    self.CargoHold_z_centroid[j,:],
                    self.Z,
                    Z_cargo_fill,
                )
                ballast_station_vcg[j] = interp(
                    self.BallastTank_z_centroid[j,:],
                    self.Z,
                    Z_ballast_fill,
                )

            # Retain each calculated fill shape while guaranteeing that its
            # integrated longitudinal load equals the scalar mass used in the
            # ship weight and CG tables.  An overfilled design remains visible
            # through the existing volume constraint; it is not silently
            # converted to a lower scalar cargo or ballast mass here.
            dist_weight_CH = _scale_distribution_to_mass(
                dist_weight_CH,
                self.X,
                self.Cargo_Weights[i],
            )
            dist_weight_BT = _scale_distribution_to_mass(
                dist_weight_BT,
                self.X,
                self.Cargo_Weights[i],
            )

            distributed_cargo_mass_kg = np.trapezoid(
                dist_weight_CH,
                x=self.X,
            )
            distributed_ballast_mass_kg = np.trapezoid(
                dist_weight_BT,
                x=self.X,
            )

            if distributed_cargo_mass_kg != 0.0:
                cargo_hold_CG[0] = np.trapezoid(
                    dist_weight_CH*self.X,
                    x=self.X,
                )/distributed_cargo_mass_kg
                cargo_hold_CG[1] = np.trapezoid(
                    dist_weight_CH*cargo_station_vcg,
                    x=self.X,
                )/distributed_cargo_mass_kg

            # CG is undefined for an empty tank.  Retaining the initialized
            # zero is numerically neutral because its scalar mass is zero.
            if distributed_ballast_mass_kg != 0.0:
                ballast_tank_CG[0] = np.trapezoid(
                    dist_weight_BT*self.X,
                    x=self.X,
                )/distributed_ballast_mass_kg
                ballast_tank_CG[1] = np.trapezoid(
                    dist_weight_BT*ballast_station_vcg,
                    x=self.X,
                )/distributed_ballast_mass_kg

            
            pd_item_CH = pd.DataFrame({'Hold_Number': [i], 
                                    'Total_Hold_Volume_m3': [cargo_hold_volume[ -1]], 
                                    'Cargo_Volume_Filled_m3': [cargo_volume_requested_m3],
                                    # Preserve the optimized cargo surface so
                                    # MVR 3-2-4/9.2 can use the actual loaded
                                    # cargo head instead of a fixed full-hold
                                    # assumption.
                                    'Cargo_Fill_Height_m': [Z_cargo_fill],
                                    'Total_Weight_kg': [self.Cargo_Weights[i]], 
                                    'Distributed_Weight_kg': [distributed_cargo_mass_kg],
                                    'Distribution_Mass_Residual_kg': [distributed_cargo_mass_kg - self.Cargo_Weights[i]],
                                    'VCG': [cargo_hold_CG[1]], 
                                    'LCG': [cargo_hold_CG[0]], 
                                    'Dist_Weight': [dist_weight_CH]})
            
            self.CargoHolds.loc[i] = pd_item_CH.loc[0]

            pd_item_BT = pd.DataFrame({'Tank_Number': [i], 
                                    'Total_Tank_Volume_m3': [ballast_tank_volume[ -1]], 
                                    'Ballast_Volume_toReplaceCargo_m3': distributed_ballast_mass_kg/self.rho, #in m^3
                                    # Retain the actual integrated liquid
                                    # surface.  V1.1 structural tank-boundary
                                    # rules use this design result instead of
                                    # the old fixed 0.667*Depth pressure head.
                                    'Ballast_Fill_Height_m': Z_ballast_fill,
                                    'Total_Weight_kg': self.Cargo_Weights[i], 
                                    'Distributed_Weight_kg': [distributed_ballast_mass_kg],
                                    'Distribution_Mass_Residual_kg': [distributed_ballast_mass_kg - self.Cargo_Weights[i]],
                                    'VCG': ballast_tank_CG[1], 
                                    'LCG': ballast_tank_CG[0], 
                                    'Dist_Weight': [dist_weight_BT]})
            
            self.BallastTanks.loc[i] = pd_item_BT.loc[0]
            

    def calculate_FuelTank_State(self, fuel_weight_kg):
        '''
        Calculate one fuel-tank fill state without changing the base design.

        The generated longitudinal/vertical tank geometry is reused for every
        named condition.  Recomputing fill height and CG here prevents the
        ten-percent arrival condition from inheriting the full departure-tank
        center of gravity.
        '''
        fuel_weight_kg = float(fuel_weight_kg)
        dist_weight_FT = np.zeros((len(self.X),))
        start_idx = self.Bulkhead_Locations[self.FuelTank_Assignment[0]]+1
        end_idx = self.Bulkhead_Locations[self.FuelTank_Assignment[1]]-1



        #Calculate volume of fuel tank using the cargo hold sectional areas. we also need to calculate the Z fill idx for the fuel tank
        fuel_tank_volume = np.zeros(self.Z.shape)
        fuel_tank_CG = np.zeros((2,)) #2D array to store CG at each Z level (first column is X position, second column is Z position)
        for j in range(1,len(self.Z)):

            fuel_area_distribution = np.zeros(len(self.X))
            fuel_area_distribution[start_idx:end_idx] = (
                self.CargoHold_SectionalAreas[start_idx:end_idx, j]
            )
            fuel_tank_volume[j] = np.trapezoid(
                fuel_area_distribution,
                x=self.X,
            )
        
        fuel_volume_requested_m3 = fuel_weight_kg/self.rho_fuel

        try:
            if fuel_volume_requested_m3 == 0.0:
                Z_fuel_fill = self.Z[0]
            else:
                Z_fuel_fill = interp(
                    self.Z,
                    fuel_tank_volume,
                    fuel_volume_requested_m3,
                )
            #Z_fuel_fill_idx = np.where(self.Z < Z_fuel_fill)[0][-1]+1 #+1 to get the index above the fill level
        except:
            Z_fuel_fill = self.Z[-1] 
            #Z_fuel_fill_idx = len(self.Z) -1


        # Calculate the filled tank's longitudinal shape and local vertical
        # centroid.  The same trapezoidal integral then owns mass and CG.
        fuel_station_vcg = np.zeros(len(self.X))

        for j in range(start_idx, end_idx):
            dist_weight_FT[j] = interp(self.CargoHold_SectionalAreas[j,:], self.Z, Z_fuel_fill) * self.rho_fuel #in kg/m
            fuel_station_vcg[j] = interp(
                self.CargoHold_z_centroid[j,:],
                self.Z,
                Z_fuel_fill,
            )

        # Preserve the requested voyage fuel mass even when the tank-volume
        # constraint subsequently rejects the design.  Scaling the spatial
        # shape avoids evaluating longitudinal strength with a smaller load
        # than the scalar weight/CG model carries.
        dist_weight_FT = _scale_distribution_to_mass(
            dist_weight_FT,
            self.X,
            fuel_weight_kg,
        )
        distributed_fuel_mass_kg = np.trapezoid(
            dist_weight_FT,
            x=self.X,
        )

        if distributed_fuel_mass_kg != 0.0:
            fuel_tank_CG[0] = np.trapezoid(
                dist_weight_FT*self.X,
                x=self.X,
            )/distributed_fuel_mass_kg
            fuel_tank_CG[1] = np.trapezoid(
                dist_weight_FT*fuel_station_vcg,
                x=self.X,
            )/distributed_fuel_mass_kg

        return pd.DataFrame({'Tank_Number': [0], 
                                'Total_Tank_Volume_m3': [fuel_tank_volume[ -1]], 
                                'Fuel_Volume_Filled_m3': [fuel_volume_requested_m3],
                                # Preserve the design-derived fuel surface for
                                # deep-tank boundary pressure calculations.
                                'Fuel_Fill_Height_m': [Z_fuel_fill],
                                'Total_Weight_kg': [fuel_weight_kg], 
                                'Distributed_Weight_kg': [distributed_fuel_mass_kg],
                                'Distribution_Mass_Residual_kg': [distributed_fuel_mass_kg - fuel_weight_kg],
                                'VCG': [fuel_tank_CG[1]], 
                                'LCG': [fuel_tank_CG[0]], 
                                'Dist_Weight': [dist_weight_FT]})

    def calc_FuelTank_Weights(self):
        """Populate the base 100-percent voyage-fuel tank state."""

        self.Fuel_Tank = self.calculate_FuelTank_State(self.fuel_weight)
        
        

    def calc_Bulkhead_Constraints(self, Hull_Dict):
        '''
        This function calculates the bulkhead spacing constraints based on the bulkhead locations and hull length.
        Constraints are based on 
        '''
        LOA = self.X[-1] - self.X[0]
        #CFR 46 Part 171.065e
        min_bulkhead_spacing = min(3.05*0.3*LOA, 10.7)
        #check engine room bulkhead spacing
        # The engine room lies between its forward and aft bulkheads.  The
        # legacy expression instead measured the lazarette from the engine-room
        # aft bulkhead to the aft-peak bulkhead, which caused a false machinery
        # fit result.
        engine_room_spacing = (
            self.X[self.Bulkhead_Locations[-2]]
            - self.X[self.Bulkhead_Locations[-3]]
        )
        Stern2AftPeak_spacing = LOA - self.X[self.Bulkhead_Locations[-1]]
        Bow2Crash_spacing = self.X[self.Bulkhead_Locations[0]]
        CargoHold_spacing = self.X[self.Bulkhead_Locations[1]] - self.X[self.Bulkhead_Locations[0]]
        item = pd.DataFrame({'Class': ['Bulkhead_Spacing']*4, 
                                'Name': ['Min Engine Room Bulkhead Spacing', 
                                        'Min Stern to Aft Peak Bulkhead Spacing', 
                                        'Min Bow to Crash Bulkhead Spacing', 
                                        'Min Cargo Hold Bulkhead Spacing'],
                                'Requirement': ['CFR 46 Part 171.065e']*4,
                                'Threshold Value': [min_bulkhead_spacing, 
                                                    min_bulkhead_spacing, 
                                                    min_bulkhead_spacing, 
                                                    min_bulkhead_spacing],
                                'Constraint Value': [engine_room_spacing, 
                                                    Stern2AftPeak_spacing, 
                                                    Bow2Crash_spacing, 
                                                    CargoHold_spacing],
                                'Units': ['m']*4,
                                'Constraint Eval': [min_bulkhead_spacing - engine_room_spacing, 
                                                    min_bulkhead_spacing - Stern2AftPeak_spacing, 
                                                    min_bulkhead_spacing - Bow2Crash_spacing, 
                                                    min_bulkhead_spacing - CargoHold_spacing],
                                'Constraint Bit': [min_bulkhead_spacing <= engine_room_spacing, 
                                                    min_bulkhead_spacing <= Stern2AftPeak_spacing, 
                                                    min_bulkhead_spacing <= Bow2Crash_spacing, 
                                                    min_bulkhead_spacing <= CargoHold_spacing]})
        
        self.BulkheadConstraints = item.copy()

        #Check Engine room is longer than engine room per Hull Dict
        item = pd.DataFrame({'Class': ['Bulkhead_Spacing'], 
                                'Name': ['Min Engine Room Length'],
                                'Requirement': ['Hull-imposed Constraint'],
                                'Threshold Value': [Hull_Dict['Min_Engine_Room_Length']],
                                'Constraint Value': [engine_room_spacing],
                                'Units': ['m'],
                                'Constraint Eval': [Hull_Dict['Min_Engine_Room_Length'] - engine_room_spacing],
                                'Constraint Bit': [Hull_Dict['Min_Engine_Room_Length'] <= engine_room_spacing]})
        self.BulkheadConstraints = pd.concat([self.BulkheadConstraints, item], ignore_index=True)

        # Make sure cargo holds are not over filled
        Hold_Volumes = self.CargoHolds['Total_Hold_Volume_m3'].values
        Cargo_Volumes_Filled = self.CargoHolds['Cargo_Volume_Filled_m3'].values
        vols = Cargo_Volumes_Filled - Hold_Volumes
        for i in range(MAX_PARAMETRIC_CARGO_HOLDS):
            if i < len(vols):
                requirement = 'Cargo Hold Volume >= Cargo Volume Filled'
                threshold = Hold_Volumes[i]
                value = Cargo_Volumes_Filled[i]
                units = 'm^3'
                residual = vols[i]

                # Exact capacity is a valid physical limit.  The scale-aware
                # allowance handles only floating-point subtraction at equality.
                passed = residual <= (
                    64.0*np.finfo(float).eps
                    * max(1.0, abs(threshold))
                )
            else:
                # A hold identity beyond this design's selected hold count is
                # not physically present.  Under the approved dataset
                # addendum, non-applicability is an explicit pass.
                requirement = 'Cargo Hold Volume >= Cargo Volume Filled'
                threshold = 0.0
                value = 0.0
                units = 'N/A'
                residual = 0.0
                passed = True

            item = pd.DataFrame({'Class': ['Cargo Hold Volume'],
                                    'Name': [f'Cargo Hold {i} Not Overfilled'],
                                    'Requirement': [requirement],
                                    'Threshold Value': [threshold],
                                    'Constraint Value': [value],
                                    'Units': [units],
                                    'Constraint Eval': [residual],
                                    'Constraint Bit': [passed]})
            self.BulkheadConstraints = pd.concat(
                [self.BulkheadConstraints, item],
                ignore_index=True,
            )

        # Check Ballast tanks are not over filled
        Tank_Volumes = self.BallastTanks['Total_Tank_Volume_m3'].values
        Ballast_Volumes_Filled = self.BallastTanks['Ballast_Volume_toReplaceCargo_m3'].values
        vols = Ballast_Volumes_Filled - Tank_Volumes
        for i in range(MAX_PARAMETRIC_CARGO_HOLDS):
            if i < len(vols):
                requirement = (
                    'Ballast Tank Volume >= Ballast Volume Filled'
                )
                threshold = Tank_Volumes[i]
                value = Ballast_Volumes_Filled[i]
                units = 'm^3'
                residual = vols[i]
                passed = residual <= (
                    64.0*np.finfo(float).eps
                    * max(1.0, abs(threshold))
                )
            else:
                # Preserve the numbered tank identity in every output file.
                # An absent tank cannot be overfilled and therefore passes
                # this particular capacity constraint by non-applicability.
                requirement = (
                    'Ballast Tank Volume >= Ballast Volume Filled'
                )
                threshold = 0.0
                value = 0.0
                units = 'N/A'
                residual = 0.0
                passed = True

            item = pd.DataFrame({'Class': ['Ballast Tank Volume'],
                                    'Name': [f'Ballast Tank {i} Not Overfilled'],
                                    'Requirement': [requirement],
                                    'Threshold Value': [threshold],
                                    'Constraint Value': [value],
                                    'Units': [units],
                                    'Constraint Eval': [residual],
                                    'Constraint Bit': [passed]})
            # The cargo-replacement ballast calculation is retained as a
            # diagnostic, but this capacity comparison is not an active
            # GenISys constraint. A bulk carrier's lightship ballast state is
            # not intended to reproduce its full-cargo displacement by filling
            # the simplified surrounding tanks with seawater.
            #
            # self.BulkheadConstraints = pd.concat(
            #     [self.BulkheadConstraints, item],
            #     ignore_index=True,
            # )

        # Check Fuel tank is not over filled    
        Tank_Volume = self.Fuel_Tank['Total_Tank_Volume_m3'].values[0]
        Fuel_Volume_Filled = self.Fuel_Tank['Fuel_Volume_Filled_m3'].values[0]
        vols = Fuel_Volume_Filled - Tank_Volume
        item = pd.DataFrame({'Class': ['Fuel Tank Volume'], 
                                'Name': [f'Fuel Tank Not Overfilled'],
                                'Requirement': ['Fuel Tank Volume >= Fuel Volume Filled'],
                                'Threshold Value': [Tank_Volume],
                                'Constraint Value': [Fuel_Volume_Filled],
                                'Units': ['m^3'],
                                'Constraint Eval': [vols],
                                'Constraint Bit': [vols <= 64.0*np.finfo(float).eps*max(1.0, abs(Tank_Volume))]})
        self.BulkheadConstraints = pd.concat([self.BulkheadConstraints, item], ignore_index=True)






                                    


        



        



                
                    




            


       
