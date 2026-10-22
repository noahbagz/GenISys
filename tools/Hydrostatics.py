#This script calculates the hydrostatic properties of a ship based on its hull parameters, including damage stability and floodable length. These Funcitons are all discrete for the given the WL and NumPointsPerWL. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def interp(A, Z, z):
    """Linearly interpolate ``A`` on the monotonic coordinate ``Z``.

    The previous helper failed at the first ordinate and silently indexed past
    the array at the last ordinate.  Flooding searches regularly encounter
    both limits, so boundary handling is part of the hydrostatic foundation
    rather than defensive input validation.
    """

    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if z < Z[0] or z > Z[-1]:
        raise ValueError(
            f"Interpolation target {z} is outside [{Z[0]}, {Z[-1]}]."
        )

    return float(np.interp(z, Z, A))


def evaluate_trim_constraint(draft_forward, draft_aft, equilibrium_draft):
    """Evaluate the approved plus/minus one-percent-of-draft trim band.

    HDF-ASM-033 defines physical trim as aft draft minus forward draft.  The
    returned margin is positive inside the permitted band and negative when
    the loading condition exceeds the project limit.
    """

    signed_trim = float(draft_aft - draft_forward)
    absolute_trim = abs(signed_trim)
    trim_limit = 0.01*float(equilibrium_draft)
    trim_margin = trim_limit - absolute_trim

    return signed_trim, absolute_trim, trim_limit, trim_margin


def floating_point_length_tolerance(*values, operation_count=1):
    """Return a dimensional roundoff allowance for length comparisons.

    This is deliberately a *numerical* tolerance, not an engineering or
    regulatory margin.  ``numpy.spacing(scale)`` is one representable floating
    point increment at the largest length being compared.  Multiplying by the
    number of accumulated grid operations provides a scale-aware allowance for
    the trapezoidal sums that precede the flooding comparison.

    HDF-083/HDF-ASM-028/HDF-ASM-029: the helper replaces the legacy arbitrary
    5 mm offset without relaxing the physical critical-draft boundary by a
    fixed real-world distance.
    """

    finite_values = [
        abs(float(value))
        for value in values
        if np.isfinite(value)
    ]
    scale = max([1.0] + finite_values)

    return float(np.spacing(scale) * max(1, int(operation_count)))

class Hydrostatics:
    def __init__(self, hull, hull_dict, CG=None,  rho = 1025):
        '''
        Stability requirements in this section are cited at the function definition. In general, rules follow:
        CFR Title 46 Chapter I Subchapter S Part 170 for stability requirements
        CFR Title 46 Chapter I Subchapter S Part 171 for subdivision and damage stability requirements
        
        '''


        self.rho = rho  # Density of water in kg/m^3
        self.g = 9.81  # Gravitational acceleration in m/s^2
        self.hull = hull  # hull already has the hull parameters defined and volume properties calculated
        self.Vol_Disp = hull_dict['Vol_Disp']  # Displacement volume in m^3
        self.LCB_intact = hull_dict['LCB']  # Longitudinal center of buoyancy at design draft in meters from midship, positive forward
        self.VCB_intact = hull_dict['VCB']  # Vertical center of buoyancy at design draft in meters from baseline  
        self.depth = self.hull.Dd
        self.draft = hull_dict['d']  # Design draft in meters
        self.generate_PC_for_hydrostatics()
        self.calc_Sectional_Areas()

        self.CG = CG # Center of Gravity of the weight of the ship [X, Y, Z] in meters

        self.StabilityConstraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])



    def update_CG(self, CG):
        #CG is a list or array of [X, Y, Z] coordinates of the center of gravity of the ship in meters
        self.CG = CG

    def calc_intact_GM_requirement(
        self,
        deckhouse_dims=[20.0, 20.0],
        deck_cargo_dims=[0, 0],
        displacement_volume=None,
    ):
        '''
        CFR Title 46 Chapter I Subchapter S Part 170 Subpart E 170.170 USCG weather GM requirement
        This function calculates the minimum intact GM requirement for the ship

        deckhouse_dims: [length, height] of the deckhouse in meters. This is used to calculate the area of the ship above the waterline this sits on the main deck
        '''

        P = 0.055 + (self.hull.LOA/1309)**2.0

        f = self.depth - self.draft # Freeboard in meters

        A = f * self.hull.LOA + deckhouse_dims[0] * deckhouse_dims[1] + deck_cargo_dims[0]*deck_cargo_dims[1]  # Area of the deckhouse and the area of the ship above the waterline

        centroid_A = np.sum([f*self.hull.LOA*(f/2.0 + self.draft), 
                            deckhouse_dims[0]*deckhouse_dims[1]*(self.hull.Dd + deckhouse_dims[1]/2.0),
                            deck_cargo_dims[0]*deck_cargo_dims[1]*(self.hull.Dd + deck_cargo_dims[1]/2.0)]) / A
        
        H = centroid_A - self.draft/2.0  # Height of the centroid of the area above the waterline from the waterline

        # Loading-condition evaluations must use the mass supported in that
        # condition rather than silently reverting to the design-draft volume.
        # Existing callers retain the design condition through the default.
        if displacement_volume is None:
            displacement_volume = self.Vol_Disp

        W = displacement_volume * self.rho/1000.0  # Displacement in metric tons

        angle_trim_half_f  = np.arctan2(f/2.0, self.hull.Bd) # Angle of heel at half the freeboard submersion in radians (self.hull.Bd is half breadth)

        T = min([14.0*np.pi/180.0, angle_trim_half_f])  # Trim angle in radians (max 14 degrees or angle at half freeboard submersion)

        GM_req = P*A*H/(W* np.tan(T))

        return GM_req
    
    def check_damaged_GM_requirement(self, damaged_GMt, label):
        '''
        CFR Title 46 Chapter I Subchapter E Part 42 Subpart 20
        This function calculates the minimum damaged GM requirement for the ship - after flooding, 

        GM must be postive and the righting arm at 20 degrees of heel must be at least 0.1m 
        area under the righting curve must be 0.0175 m-rad

        function returns an array of booleans for each requirement - an array of all zeros means the ship passes damage stability for a given damaged GMt
        '''

        A = damaged_GMt > 0.0  # Check if GM is positive
        B = damaged_GMt*np.sin(20.0*np.pi/180.0) >= 0.1  # Check if righting arm at 20 degrees is at least 0.1m
        C = damaged_GMt*np.sin(20.0*np.pi/180.0)*0.5*20.0*np.pi/180.0 >= 0.0175  # Check if area under righting curve is at least 0.0175 m-rad (approximation as a triangle)

        item = pd.DataFrame({'Class': ['Damage Stability']*3,
                             'Name': [label + ' Positive GM', label +' Righting Arm at 20 deg', label + ' Area under Righting Curve'],
                            'Requirement': ['CFR Title 46 Chapter I Subchapter E Part 42 Subpart 20']*3,
                            'Threshold Value': [0.0, 0.1, 0.0175],
                            'Constraint Value': [damaged_GMt, damaged_GMt*np.sin(20.0*np.pi/180.0), damaged_GMt*np.sin(20.0*np.pi/180.0)*0.5*20.0*np.pi/180.0],
                            'Units': ['m', 'm', 'm-rad'],
                            'Constraint Eval': [-damaged_GMt, 0.1 - damaged_GMt*np.sin(20.0*np.pi/180.0), 0.0175 - damaged_GMt*np.sin(20.0*np.pi/180.0)*0.5*20.0*np.pi/180.0],
                            'Constraint Bit': [A, B, C]})
        self.StabilityConstraints = pd.concat([self.StabilityConstraints, item], ignore_index=True)
        


    def generate_PC_for_hydrostatics(self, num_wl=101, points_per_wl=1001):


        X, Z, Y, WL = self.hull.gen_PC_for_Cw(self.depth, NUM_WL=num_wl, PointsPerWL=points_per_wl)

        # Find last position of Y[:, -1] that is zero at beginning and end of the array (limits of ship deck)
        self.bow_deck_idx = np.where(Y[:, -1]> 0)[0][0] -1
        self.stern_deck_idx = np.where(Y[:, -1] > 0)[0][-1]+1
        # HDF-ASM-031: place the forward perpendicular where the modeled bow
        # intersects the design waterline.  Retain the absolute longitudinal
        # value before shifting the working hydrostatic grid.
        self.X_FP = float(self.hull.bow_profile(self.draft))

        # HDF-ASM-030: place the aft perpendicular at the modeled
        # stern/transom plane.  Keep LBP separate from the structural LOA
        # proxy so the two definitions cannot be silently interchanged.
        self.X_AP = float(self.hull.LOA)
        self.LBP = self.X_AP - self.X_FP

        #Round the values to 3 decimal places
        self.X = X-self.X_FP # Shift X so that FP is at 0
        self.Z = Z+self.depth # Shift Z so that baseline is at 0
        self.Y = Y
        #Closest Z value to the draft
        self.draft_index = np.argmin(np.abs(Z - self.draft))


    
    def calc_Sectional_Areas(self):  #Calculate the sectional area of a ship below each Z at each X position
        """Calculate cumulative sectional area and its vertical centroid.

        HDF-ASM-028: all integrations in this hydrostatic foundation use the
        composite trapezoidal rule on the generated hull grid.  The explicit
        loop is retained so that the cumulative first moment is available at
        every waterline, not only at the design draft.
        """

        self.sectional_Areas = np.zeros(self.Y.shape)
        self.SA_z_centroid = np.zeros(self.Y.shape)  

        for i in range(self.Y.shape[0]): # X Positions
            Moment = 0.0
            for j in range(1, self.Y.shape[1]):

                # Use the actual vertical spacing so the same method remains
                # correct if the waterline grid later becomes nonuniform.
                dz = self.Z[j] - self.Z[j-1]
                
                a = 2.0 * self.Y[i,j]  # Area at previous Z #multiply by 2 for both sides of the ship
                b = 2.0 * self.Y[i,j-1]  # Area at current Z #multiply by 2 for both sides of the ship

                dA = 0.5 * (a + b) * dz

                self.sectional_Areas[i,j] = self.sectional_Areas[i,j-1] + dA  # Update sectional area
            
                cz = self.Z[j-1] + dz/3.0 * (2.0*a + b) / (a + b+1e-10)  #Avoid divide by zero error

                Moment = Moment + cz * dA

                self.SA_z_centroid[i, j] = Moment / (self.sectional_Areas[i, j] + 1e-10)  # Avoid divide by zero error
    
    def _integrate_section_properties(self, z_idx, cell_is_included):
        """Integrate volume and first moments at one waterline.

        HDF-ASM-028: sectional areas and both first moments are integrated
        longitudinally with the composite trapezoidal rule.  Damage is
        represented by selecting whole longitudinal grid cells; a later
        compartment model can replace the selector without changing the
        numerical integration convention.
        """

        volume = 0.0
        longitudinal_moment = 0.0
        vertical_moment = 0.0

        for j in range(1, len(self.X)):

            if not cell_is_included(j):
                continue

            x_0 = self.X[j-1]
            x_1 = self.X[j]
            dx = x_1 - x_0

            area_0 = self.sectional_Areas[j-1, z_idx]
            area_1 = self.sectional_Areas[j, z_idx]

            # Composite trapezoidal volume for the longitudinal cell.
            cell_volume = 0.5 * (area_0 + area_1) * dx
            volume += cell_volume

            # Exact first moment of the linear area interpolation used by the
            # trapezoid.  This is equivalent to applying the trapezoid's
            # centroid location to ``cell_volume``.
            longitudinal_moment += dx / 6.0 * (
                area_0 * (2.0*x_0 + x_1)
                + area_1 * (x_0 + 2.0*x_1)
            )

            # Apply the trapezoidal rule to the sectional vertical first
            # moment A(z)*VCB(z) at the two bounding stations.
            vertical_moment += 0.5 * dx * (
                area_0 * self.SA_z_centroid[j-1, z_idx]
                + area_1 * self.SA_z_centroid[j, z_idx]
            )

        if volume <= 0.0:
            return 0.0, 0.0, 0.0

        return (
            volume,
            longitudinal_moment / volume,
            vertical_moment / volume,
        )

    def calc_Volumes(self, lost_vol_idx_range=None):
        """Return intact remaining buoyant volume through every waterline."""

        if lost_vol_idx_range is None:
            lost_vol_idx_range = [-1, -1]

        lost_start_idx, lost_end_idx = lost_vol_idx_range

        # Calculate the remaining buoyant volume of the ship.  Each ordinate
        # is integrated from the already-cumulative sectional area at that
        # waterline.  This avoids carrying first moments between waterlines.
        Volumes = np.zeros(self.Z.shape)
        LCB = np.zeros(self.Z.shape)
        VCB = np.zeros(self.Z.shape)

        for i in range(1, len(Volumes)):

            def cell_is_intact(j):
                # Cell j spans X[j-1] to X[j].  A damage range [start, end]
                # therefore removes cells start < j <= end.
                return not (j > lost_start_idx and j <= lost_end_idx)

            Volumes[i], LCB[i], VCB[i] = (
                self._integrate_section_properties(i, cell_is_intact)
            )
        
        return Volumes, LCB, VCB
    
    
    def calc_Damaged_Volumes(self, lost_vol_idx_range=None, permeability = 0.95):
        """Return residual buoyancy inside the coarsely flooded slice.

        A permeability of 1.0 removes all buoyancy in the selected slice; a
        permeability of 0.0 restores it completely.  HDF-083/HDF-ASM-029:
        this is a whole-slice early-design diagnostic and is not a regulatory
        compartment-flooding model.
        """

        Damaged_Volumes = np.zeros((len(self.Z)))
        Damaged_VCB = np.zeros((len(self.Z)))
        Damaged_LCB = np.zeros((len(self.Z)))

        #print('permeability:', permeability)

        if lost_vol_idx_range is None:
            return Damaged_Volumes, Damaged_LCB, Damaged_VCB

        lost_start_idx, lost_end_idx = lost_vol_idx_range
        residual_fraction = 1.0 - permeability

        for i in range(1, len(self.Z)):

            def cell_is_damaged(j):
                return j > lost_start_idx and j <= lost_end_idx

            full_slice_volume, slice_lcb, slice_vcb = (
                self._integrate_section_properties(i, cell_is_damaged)
            )

            Damaged_Volumes[i] = residual_fraction * full_slice_volume
            Damaged_LCB[i] = slice_lcb
            Damaged_VCB[i] = slice_vcb

        return Damaged_Volumes, Damaged_LCB, Damaged_VCB


 
    def calc_WaterPlane(
        self,
        draft,
        lost_vol_idx_range=None,
        permeability=1.0,
    ):
        """Calculate full-waterplane properties at ``draft``.

        HDF-082: ``Y`` stores half-breadths, so every waterplane property must
        be based on the full breadth ``2*Y``.  The old implementation used a
        half-waterplane area and bracketed the waterline above the requested
        draft, which produced a negative interpolation fraction.

        HDF-083/HDF-ASM-029: the same permeability used to retain residual
        buoyant volume in a flooded whole-slice is applied linearly to that
        slice's waterplane area and inertia contributions.  This keeps the
        coarse lost-buoyancy model internally consistent; it does not turn the
        diagnostic into a regulatory compartment-damage calculation.
        """

        if lost_vol_idx_range is None:
            lost_vol_idx_range = [-1, -1]

        lost_start_idx, lost_end_idx = lost_vol_idx_range

        if draft < self.Z[0] or draft > self.Z[-1]:
            raise ValueError(
                f"Draft {draft} is outside hull grid "
                f"[{self.Z[0]}, {self.Z[-1]}]."
            )

        # Bracket the requested draft from below and above.  At an exact grid
        # boundary both indices are the same, which also handles the top and
        # bottom waterlines without indexing beyond the array.
        upper_idx = int(np.searchsorted(self.Z, draft, side='left'))

        # Treat only an exactly represented ordinate as an exact grid hit.
        # ``numpy.isclose`` has a default relative tolerance of 1e-5; at a
        # 20 m draft that can collapse a genuine 0.2 mm interval onto the
        # upper waterline and bypass the required interpolation.  Ordinary
        # linear interpolation is already stable as the interval tends to
        # zero, so no engineering-scale proximity tolerance belongs here.
        if upper_idx == 0 or draft == self.Z[upper_idx]:
            lower_idx = upper_idx
        else:
            lower_idx = upper_idx - 1

        waterline_indices = [lower_idx]

        if upper_idx != lower_idx:
            waterline_indices.append(upper_idx)

        properties = []

        for z_idx in waterline_indices:
            waterplane_area = 0.0
            first_moment_x = 0.0
            second_moment_x = 0.0
            second_moment_y_origin = 0.0

            for j in range(1, len(self.X)):

                # A permeability of one removes the damaged cell completely;
                # zero restores its complete contribution.  Scale each
                # integral contribution, rather than scaling breadth before
                # cubing it, because permeability is a volume/area fraction in
                # this whole-cell lost-buoyancy approximation.
                cell_is_damaged = (
                    j > lost_start_idx and j <= lost_end_idx
                )
                buoyancy_fraction = (
                    1.0 - permeability if cell_is_damaged else 1.0
                )

                x_0 = self.X[j-1]
                x_1 = self.X[j]
                dx = x_1 - x_0

                # Hull offsets are half-breadths.  Convert once here and use
                # the full widths consistently in area and inertia terms.
                width_0 = 2.0 * self.Y[j-1, z_idx]
                width_1 = 2.0 * self.Y[j, z_idx]

                waterplane_area += (
                    buoyancy_fraction
                    * 0.5
                    * (width_0 + width_1)
                    * dx
                )

                first_moment_x += buoyancy_fraction * dx / 6.0 * (
                    width_0 * (2.0*x_0 + x_1)
                    + width_1 * (x_0 + 2.0*x_1)
                )

                # Transverse waterplane inertia: integrate width^3/12 for a
                # breadth that varies linearly through the cell.
                second_moment_x += buoyancy_fraction * dx / 48.0 * (
                    width_0**3.0
                    + width_0**2.0 * width_1
                    + width_0 * width_1**2.0
                    + width_1**3.0
                )

                # Longitudinal second moment about the global X origin for a
                # linearly varying breadth.  It is shifted to the LCF after
                # all cells have been accumulated.
                second_moment_y_origin += buoyancy_fraction * dx / 12.0 * (
                    width_0 * (
                        3.0*x_0**2.0 + 2.0*x_0*x_1 + x_1**2.0
                    )
                    + width_1 * (
                        x_0**2.0 + 2.0*x_0*x_1 + 3.0*x_1**2.0
                    )
                )

            if waterplane_area <= 0.0:
                lcf = 0.0
                second_moment_y = 0.0
            else:
                lcf = first_moment_x / waterplane_area
                second_moment_y = (
                    second_moment_y_origin
                    - waterplane_area * lcf**2.0
                )

            properties.append(
                np.array(
                    [
                        waterplane_area,
                        lcf,
                        second_moment_x,
                        second_moment_y,
                    ],
                    dtype=float,
                )
            )

        if len(properties) == 1:
            return tuple(properties[0])

        fraction = (
            (draft - self.Z[lower_idx])
            / (self.Z[upper_idx] - self.Z[lower_idx])
        )
        interpolated = properties[0] + fraction * (
            properties[1] - properties[0]
        )

        return tuple(interpolated)


    def calc_Damaged_Displacement_Condition(
        self,
        lost_vol_idx_range=None,
        permeability=0.95,
        displacement_volume=None,
    ):
        """Solve the upright sinkage condition for one coarse damage slice."""

         # Calculate the volume of the ship
        intact_Volumes, intact_LCB, intact_VCB = self.calc_Volumes(lost_vol_idx_range)

        Damaged_Volumes, Damaged_LCB, Damaged_VCB = self.calc_Damaged_Volumes(lost_vol_idx_range, permeability)

        #Calculate Flooded Bouyancy of the ship across all drafts
        Volumes = intact_Volumes + Damaged_Volumes

        #print('Intact Volumes:', intact_Volumes)
        #print('Damaged Volumes:', Damaged_Volumes)
        #print('Total Volumes:', Volumes)
        
        LCB = np.zeros(Volumes.shape)
        VCB = np.zeros(Volumes.shape)

        for i in range(1, len(Volumes)):

            # Avoid inserting a dimensional epsilon into the centroid
            # denominator.  A zero-volume ordinate has no defined buoyancy
            # center and remains at the initialized zero value.
            if Volumes[i] <= 0.0:
                continue

            LCB[i] = (
                intact_LCB[i]*intact_Volumes[i]
                + Damaged_LCB[i]*Damaged_Volumes[i]
            ) / Volumes[i]
            VCB[i] = (
                intact_VCB[i]*intact_Volumes[i]
                + Damaged_VCB[i]*Damaged_Volumes[i]
            ) / Volumes[i]
        
        #print('LCB')   
        #print(LCB)
        #print(self.hull.VolumeCentroids[:,0])

        #print('VCB')
        #print(VCB)
        #print(self.hull.VolumeCentroids[:,1])
        
        if displacement_volume is None:
            displacement_volume = self.Vol_Disp

        # Interpolate to find draft based on the selected loading-condition
        # displacement volume.  Existing callers retain the design condition.
        draft_cond = interp(self.Z, Volumes, displacement_volume)
        LCB_cond = interp(LCB, self.Z, draft_cond)  # Interpolate to find LCB at the draft
        VCB_cond = interp(VCB, self.Z, draft_cond)  # Interpolate to find VCB at the draft

        return Volumes, draft_cond, LCB_cond, VCB_cond
    

        
    
    def calc_GM_TrimMoments(
        self,
        Volumes_cond,
        draft_cond,
        LCB_Cond,
        VCB_Cond,
        lost_vol_idx_range=None,
        permeability=1.0,
        displacement_volume=None,
    ):
        """Calculate upright GM and linear trim for one condition.

        ``displacement_volume`` is the actual buoyant volume supported by the
        evaluated loading condition.  When a complete volume curve is passed,
        it is recovered at ``draft_cond`` using the same interpolation used by
        the damaged-equilibrium solver.  The legacy design volume is retained
        only for lightweight test/mocked callers that do not carry a volume
        grid.

        The calculation assumes YCG = 0 and uses the project-approved linear
        upright trim approximation; it is not a finite-angle stability model.
        """
    
        # HDF-083/HDF-ASM-029: waterplane loss uses the same permeability as
        # the flooded-volume calculation for this coarse whole-slice model.
        WP, LCF, Ixx, Iyy = self.calc_WaterPlane(
            draft_cond,
            lost_vol_idx_range,
            permeability=permeability,
        )

        if displacement_volume is None:

            # Real Hydrostatics objects carry the full Z grid and volume curve.
            # This path makes intact/damaged loading-condition calculations use
            # their own displacement instead of the design-draft displacement.
            volume_curve = np.asarray(Volumes_cond, dtype=float)

            if (
                hasattr(self, 'Z')
                and volume_curve.ndim == 1
                and len(volume_curve) == len(self.Z)
            ):
                displacement_volume = interp(
                    volume_curve,
                    self.Z,
                    draft_cond,
                )
            else:
                # Preserve compatibility for isolated numerical tests and
                # legacy callers that pass only a placeholder curve.
                displacement_volume = self.Vol_Disp

        #print('WP:', WP, 'LCF:', LCF, 'Ixx:', Ixx, 'Iyy:', Iyy)

        #print(VCB_Cond, LCB_Cond, draft_cond, self.CG[2])
        #print('Ixx:', Ixx, 'Iyy:', Iyy, 'Vol_Disp:', self.Vol_Disp, 'CG[2]:', self.CG[2])
        #print('Hull_I_WP:', self.hull.I_WP[66])

        GM_T = VCB_Cond + (Ixx/displacement_volume) - self.CG[2]
        GM_L = VCB_Cond + (Iyy/displacement_volume) - self.CG[2]

        #print('GM_T:', GM_T, 'GM_L:', GM_L)

        # HDF-ASM-030/HDF-ASM-031: use the distance between the approved
        # perpendiculars for moment-to-change-trim and for distributing the
        # one-centimetre change between the two perpendicular drafts.
        MTC = (
            self.rho
            * self.g
            * displacement_volume
            * GM_L
            * 0.01
            / self.LBP
        )

        trim_moment = (
            (self.CG[0] - LCB_Cond)
            * self.rho
            * self.g
            * displacement_volume
        )  # Trim moment in N m

        #Linear approximation of trim  by moment (+trim means bow up)
        trim = trim_moment / MTC  # Trim in centimeters

        # ``self.X`` is shifted so the forward perpendicular is zero and the
        # aft perpendicular is LBP.  Positive trim retains the legacy sign
        # convention of bow up / greater aft draft.
        draft_fwd = draft_cond - 0.01*trim*LCF/self.LBP
        draft_aft = draft_cond + 0.01*trim*(self.LBP - LCF)/self.LBP

        return GM_T, GM_L, MTC, trim, draft_fwd, draft_aft
    

    
    def _flooding_candidate_passes(
        self,
        x_idx,
        half_steps,
        permeability,
        critical_draft,
    ):
        """Evaluate one symmetric whole-slice damage candidate.

        Keeping this predicate outside the binary-search loop lets the
        integrated diagnostic independently sample the same pass/fail boundary
        and record whether its monotonic-search assumption is observed.
        """

        lost_volume_indices = [
            x_idx - half_steps,
            x_idx + half_steps,
        ]

        try:
            Volumes, draft_cond, LCB_cond, VCB_cond = (
                self.calc_Damaged_Displacement_Condition(
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=permeability,
                )
            )

            GM_T, GM_L, MTC, trim, draft_fwd, draft_aft = (
                self.calc_GM_TrimMoments(
                    Volumes,
                    draft_cond,
                    LCB_cond,
                    VCB_cond,
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=permeability,
                )
            )

        except (
            IndexError,
            ValueError,
            FloatingPointError,
            ZeroDivisionError,
            UnboundLocalError,
        ):
            # A failed equilibrium solution cannot demonstrate survival of
            # this candidate damage length.
            return False

        # A non-finite draft also cannot demonstrate a passing damaged
        # equilibrium condition.
        if not np.isfinite(draft_fwd) or not np.isfinite(draft_aft):
            return False

        # HDF-083/HDF-ASM-029: accept only discrepancies attributable to the
        # floating-point work in the trapezoidal sums.  This replaces the old
        # fixed 5 mm allowance and therefore is not a physical safety margin.
        length_tolerance = floating_point_length_tolerance(
            draft_fwd,
            draft_aft,
            critical_draft,
            operation_count=max(len(self.X), len(getattr(self, 'Z', []))),
        )

        return (
            draft_fwd <= critical_draft + length_tolerance
            and draft_aft <= critical_draft + length_tolerance
        )

    def _check_flooding_boundary_monotonicity(
        self,
        permeability,
        critical_draft,
        maximum_station_checks=5,
        maximum_candidates_per_station=9,
    ):
        """Spot-check the binary search's monotonic pass/fail assumption.

        HDF-ASM-029 permits binary search only after the selected numerical
        model has been checked for a pass-then-fail boundary.  Exhaustively
        evaluating every damage length would defeat the purpose of the search,
        so this diagnostic samples up to five longitudinal stations and nine
        ordered positive damage lengths at each station.  A pass observed
        after a failure invalidates the assumption for this smoke result.
        """

        if len(self.FL_X_idx) == 0:
            return False, 0, 0

        station_positions = np.unique(
            np.linspace(
                0,
                len(self.FL_X_idx) - 1,
                min(maximum_station_checks, len(self.FL_X_idx)),
                dtype=int,
            )
        )

        monotonic_boundary_observed = True
        candidate_evaluations = 0

        for station_position in station_positions:
            x_idx = self.FL_X_idx[station_position]
            max_half_steps = min(x_idx, len(self.X) - 1 - x_idx)

            if max_half_steps <= 0:
                continue

            half_step_samples = np.unique(
                np.linspace(
                    1,
                    max_half_steps,
                    min(maximum_candidates_per_station, max_half_steps),
                    dtype=int,
                )
            )

            failure_observed = False

            for half_steps in half_step_samples:
                passes = self._flooding_candidate_passes(
                    x_idx,
                    int(half_steps),
                    permeability,
                    critical_draft,
                )
                candidate_evaluations += 1

                if not passes:
                    failure_observed = True
                elif failure_observed:
                    # A later pass after an earlier failure is incompatible
                    # with the binary search's single-boundary assumption.
                    monotonic_boundary_observed = False

        return (
            monotonic_boundary_observed,
            len(station_positions),
            candidate_evaluations,
        )

    def calc_Floodable_Length_Curves(self, permeabilities = [0.7, 0.85, 0.9, 0.95], critical_draft=None, step=1):

        '''
        This function calculates the floodable lengths along self.X 

        The maximum floodable length is the length along the ship where the ship can flood without immersing the critical draft along any point of the ship

        At each longitudinal evaluation position, the calculation uses binary
        search to find the largest passing symmetric damage length.  This
        assumes that once a damage length fails, all longer damage lengths at
        that position also fail.
        
        '''
        if critical_draft is None:
            # HDF-ASM-014/HDF-ASM-029: openings and true downflooding points
            # are outside the GenISys scope.  The topmost continuous modeled
            # deck is therefore used as a simple deck-edge immersion limit.
            # This produces a useful early-design diagnostic, but it is not a
            # regulatory downflooding or damage-stability criterion.
            critical_draft = self.hull.Dd
        
        self.Floodable_Lengths = np.zeros((len(self.X), len(permeabilities)))

        # The same longitudinal positions are evaluated for every
        # permeability.  Store this list once so downstream indexing does not
        # contain duplicate positions when multiple permeabilities are used.
        self.FL_X_idx = list(range(1, len(self.X)-1, step))

        for i in range(len(permeabilities)):
            print('Calculating Floodable Lengths for permeability:', permeabilities[i])

            for j in self.FL_X_idx:

                # The damage length is symmetric about station j.  Search in
                # integer half-steps so the result keeps the existing
                # resolution of two longitudinal grid intervals.
                max_half_steps = min(j, len(self.X)-1-j)

                # If the largest geometrically possible damage length passes,
                # no additional candidate evaluations are needed.
                if self._flooding_candidate_passes(
                    j,
                    max_half_steps,
                    permeabilities[i],
                    critical_draft,
                ):
                    passing_half_steps = max_half_steps

                else:
                    # Zero is a sentinel for no positive passing damage
                    # length.  The failed geometric upper bound brackets the
                    # other end of the binary search.
                    low = 0
                    high = max_half_steps

                    while high - low > 1:

                        mid = (low + high) // 2

                        if self._flooding_candidate_passes(
                            j,
                            mid,
                            permeabilities[i],
                            critical_draft,
                        ):
                            low = mid
                        else:
                            high = mid

                    passing_half_steps = low

                self.Floodable_Lengths[j, i] = (
                    self.X[j + passing_half_steps]
                    - self.X[j - passing_half_steps]
                )
        
        return self.Floodable_Lengths
    
    def calc_Subdivision_Floodable_Lengths(self, bulkhead_indices = np.array([101, 301, 501, 701, 901])):
        '''
        This function calcultes the floodable lengths supported by the subdivision of the ship into compartments by bulkheads.
        The bulkhead indices are the X indices of the bulkheads in the hull parameterization grid.
        
        The function returns the maximum floodable length supported by the subdivision of the ship using single and 2- compartment flooding
        '''

        self.single_compartment_FL = np.zeros((len(self.X)),)
        self.two_compartment_FL = np.zeros((len(self.X), 2)) # Two compartment flooding starting at the odd and even bulkheads

        dX = self.X[1] - self.X[0]  #Assume uniform spacing in X direction

        #Add 0 and len(X)-1 to the bulkhead indices to include the ends of the ship
        bulkhead_indices = np.concatenate(([0], bulkhead_indices, [len(self.X)-1]))

        for i in range(0, len(bulkhead_indices)-1):
            L = self.X[bulkhead_indices[i+1]] - self.X[bulkhead_indices[i]]
            mid_idx = (bulkhead_indices[i] + bulkhead_indices[i+1]) // 2
            
            #Single compartment flooding is a triangular distribution from 0 at the bulkheads to L at the mid point, where L is the lengthe between the subdivisions
            idx = np.arange(bulkhead_indices[i]+1, mid_idx+1)
            self.single_compartment_FL[idx] = L*(idx - bulkhead_indices[i])/(mid_idx - bulkhead_indices[i])

            idx = np.arange(mid_idx+1, bulkhead_indices[i+1]-1)
            self.single_compartment_FL[idx] = L* (1 - (idx - mid_idx)/(bulkhead_indices[i+1] - mid_idx))

        for i in range(0, len(bulkhead_indices)-2):
            L = self.X[bulkhead_indices[i+2]] - self.X[bulkhead_indices[i]]
            mid_idx = (bulkhead_indices[i] + bulkhead_indices[i+2]) // 2

            #Two compartment flooding is a triangular distribution from 0 at the bulkheads to L at the mid point, where L is the lengthe between the subdivisions
            odds_idx = i%2  # Start at even or odd bulkhead
            idx = np.arange(bulkhead_indices[i]+1, mid_idx+1)
            self.two_compartment_FL[idx, odds_idx] = L*(idx - bulkhead_indices[i])/(mid_idx - bulkhead_indices[i])

            idx = np.arange(mid_idx+1, bulkhead_indices[i+2]-1)
            self.two_compartment_FL[idx, odds_idx] = L* (1 - (idx - mid_idx)/(bulkhead_indices[i+2] - mid_idx))

        return self.single_compartment_FL, self.two_compartment_FL
    
    def calc_Damaged_Stability(self, bulkhead_indices = np.array([101, 301, 501, 701, 901])):
        '''
        Loop through all single and two compartment flooding scenarios and check if the ship meets the damaged stability requirements
        '''
        bulkhead_indices = np.concatenate(([0], bulkhead_indices, [len(self.X)-1]))

        self.single_compartment_GM = np.zeros((len(bulkhead_indices)-1,))
        self.two_compartment_GM = np.zeros((len(bulkhead_indices)-2,2))

        for i in range(0, len(bulkhead_indices)-1):
            lost_volume_indices = [
                bulkhead_indices[i],
                bulkhead_indices[i+1],
            ]
            Volumes, draft_cond, LCB_cond, VCB_cond = (
                self.calc_Damaged_Displacement_Condition(
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=0.95,
                )
            )
            GM_T, GM_L, MTC, trim, draft_fwd, draft_aft = (
                self.calc_GM_TrimMoments(
                    Volumes,
                    draft_cond,
                    LCB_cond,
                    VCB_cond,
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=0.95,
                )
            )
            self.single_compartment_GM[i] = GM_T
            self.check_damaged_GM_requirement(GM_T, label=f'Single Compartment Flooding ' + str(i))

        for i in range(0, len(bulkhead_indices)-2):

            lost_volume_indices = [
                bulkhead_indices[i],
                bulkhead_indices[i+2],
            ]
            Volumes, draft_cond, LCB_cond, VCB_cond = (
                self.calc_Damaged_Displacement_Condition(
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=0.95,
                )
            )
            GM_T, GM_L, MTC, trim, draft_fwd, draft_aft = (
                self.calc_GM_TrimMoments(
                    Volumes,
                    draft_cond,
                    LCB_cond,
                    VCB_cond,
                    lost_vol_idx_range=lost_volume_indices,
                    permeability=0.95,
                )
            )
            self.two_compartment_GM[i] = GM_T
            self.check_damaged_GM_requirement(GM_T, label=f'Two Compartment Flooding ' + str(i))
        


    
    def calc_Floodable_Length(self, permeabilities = [0.95], critical_draft=None, bulkhead_indices = np.array([101, 301, 501, 701, 901]), plot=False):
        '''
        This function calculates the floodable lengths along self.X 

        The maximum floodable length is the length along the ship where the ship can flood without immersing the critical draft along any point of the ship
        
        HDF-083/HDF-ASM-029: this legacy whole-slice comparison is retained
        as a non-regulatory early-design diagnostic.  The binary search is a
        validated numerical search, but the underlying damage extent,
        downflooding limit, and survival criteria are not a complete CFR,
        SOLAS, or IMO damage-stability implementation.
    
        '''
        print( 'Calculating Floodable Lengths...')
        self.Floodable_Lengths = self.calc_Floodable_Length_Curves(permeabilities, critical_draft, step=15) 

        self.single_compartment_FL, self.two_compartment_FL = self.calc_Subdivision_Floodable_Lengths(bulkhead_indices)

        if plot:
            self.plot_Floodable_Lengths(self.Floodable_Lengths, permeabilities, self.single_compartment_FL, self.two_compartment_FL)

        criteria = np.zeros((len(permeabilities), 2, len(self.X[self.FL_X_idx])))  # Criteria for each permeability, single and two compartment flooding, and X positions
        for i in range(len(permeabilities)):
            criteria[i, 0, :] = self.Floodable_Lengths[self.FL_X_idx, i] - self.single_compartment_FL[self.FL_X_idx]
            criteria[i, 1, :] = self.Floodable_Lengths[self.FL_X_idx, i] - np.max(self.two_compartment_FL[self.FL_X_idx], axis=1)

        # HDF-083/HDF-ASM-029: use only a floating-point accumulation
        # allowance.  The earlier fixed 5 mm offset was an undocumented
        # engineering margin embedded in a numerical comparison.
        length_tolerance = floating_point_length_tolerance(
            np.max(np.abs(self.X)),
            operation_count=len(self.X),
        )
        
        #FL Eval If any value in criteria is negative, the floodable length is less than the subdivision floodable length
        FL_Eval = np.ones((len(permeabilities), 2), dtype=bool)
        for i in range(len(permeabilities)):
            FL_Eval[i, 0] = np.all(criteria[i, 0, :] >= -length_tolerance)
            FL_Eval[i, 1] = np.all(criteria[i, 1, :] >= -length_tolerance)
        
        return permeabilities, FL_Eval

    def calc_Flooding_Diagnostic(
        self,
        bulkhead_indices,
        permeabilities=(0.95,),
        critical_draft=None,
        step=15,
    ):
        """Run and tabulate the integrated binary-search flooding diagnostic.

        This wrapper deliberately keeps diagnostic results separate from
        ``StabilityConstraints``.  The binary search is integrated into the
        main system-design path, while the audit continues to identify
        HDF-083 as non-regulatory until explicit compartments, openings,
        downflooding points, and the selected damage-survival criteria exist.
        """

        if critical_draft is None:
            # HDF-ASM-014: the uppermost continuous modeled deck is the
            # freeboard deck.  It is only a diagnostic immersion boundary.
            critical_draft = self.hull.Dd

        permeabilities = list(permeabilities)
        bulkhead_indices = np.asarray(bulkhead_indices, dtype=int)

        # HDF-083: the production calculation calls the binary-search curve
        # method.  Focused tests compare that search with an exhaustive
        # monotonic reference so this integration does not revert to the old
        # linear candidate scan.
        floodable_lengths = self.calc_Floodable_Length_Curves(
            permeabilities=permeabilities,
            critical_draft=critical_draft,
            step=step,
        )
        single_demand, two_demand = self.calc_Subdivision_Floodable_Lengths(
            bulkhead_indices=bulkhead_indices,
        )

        evaluated_indices = np.asarray(self.FL_X_idx, dtype=int)
        curve_rows = []
        summary_rows = []

        for permeability_idx, permeability in enumerate(permeabilities):
            (
                monotonic_boundary_observed,
                monotonic_station_count,
                monotonic_candidate_count,
            ) = self._check_flooding_boundary_monotonicity(
                permeability=permeability,
                critical_draft=critical_draft,
            )

            available = floodable_lengths[
                evaluated_indices,
                permeability_idx,
            ]
            required_single = single_demand[evaluated_indices]
            required_two = np.max(
                two_demand[evaluated_indices],
                axis=1,
            )

            single_margin = available - required_single
            two_margin = available - required_two

            # This is a numerical roundoff allowance only.  It scales with
            # the longitudinal grid work and is intentionally many orders of
            # magnitude smaller than a ship-design tolerance.
            length_tolerance = floating_point_length_tolerance(
                np.max(np.abs(self.X)),
                critical_draft,
                operation_count=len(self.X),
            )

            for row_idx, x_idx in enumerate(evaluated_indices):
                curve_rows.append(
                    {
                        'X Position': self.X[x_idx],
                        'X Index': x_idx,
                        'Permeability': permeability,
                        'Floodable Length': available[row_idx],
                        'Single-Compartment Demand': required_single[row_idx],
                        'Two-Compartment Demand': required_two[row_idx],
                        'Single-Compartment Margin': single_margin[row_idx],
                        'Two-Compartment Margin': two_margin[row_idx],
                        'Units': 'm',
                    }
                )

            summary_rows.append(
                {
                    'Class': 'Flooding Diagnostic',
                    'Name': (
                        'Whole-slice floodable-length diagnostic at '
                        f'{permeability:.2f} permeability'
                    ),
                    'Requirement': 'HDF-083 / HDF-ASM-029',
                    'Critical Draft': critical_draft,
                    'Permeability': permeability,
                    'Evaluation Step': step,
                    'Sample Count': len(evaluated_indices),
                    'Monotonicity Check Stations': monotonic_station_count,
                    'Monotonicity Candidate Evaluations': (
                        monotonic_candidate_count
                    ),
                    'Monotonic Boundary Observed': bool(
                        monotonic_boundary_observed
                    ),
                    'Minimum Single-Compartment Margin': np.min(
                        single_margin
                    ),
                    'Minimum Two-Compartment Margin': np.min(two_margin),
                    'Numerical Feasibility Tolerance': length_tolerance,
                    'Diagnostic Single Pass': bool(
                        np.all(single_margin >= -length_tolerance)
                    ),
                    'Diagnostic Two Pass': bool(
                        np.all(two_margin >= -length_tolerance)
                    ),
                    'Units': 'm',
                    'Implementation Status': (
                        'diagnostic_non_regulatory'
                        if monotonic_boundary_observed
                        else 'diagnostic_monotonicity_failed'
                    ),
                    'Result Basis': (
                        'Binary-search whole-slice lost-buoyancy model; '
                        'not regulatory damage-stability compliance.'
                    ),
                }
            )

        self.FloodingDiagnosticSummary = pd.DataFrame(summary_rows)
        self.FloodableLengthCurve = pd.DataFrame(curve_rows)

        return self.FloodingDiagnosticSummary, self.FloodableLengthCurve
    
    def calc_Stability_Constraints(self, bulkhead_indices = np.array([101, 301, 501, 701, 901])):
        '''
        This function calculates the stability constraints for the ship and stores them in the StabilityConstraints dataframe
        '''
    
        #Intact GM requirement
        GM_req = self.calc_intact_GM_requirement()
        Volumes, draft_cond, LCB_cond, VCB_cond = self.calc_Damaged_Displacement_Condition()

        #print('VCB:', VCB_cond, 'LCB:', LCB_cond, 'Draft:', draft_cond)

        GM_T, GM_L, MTC, trim, draft_fwd, draft_aft = self.calc_GM_TrimMoments(Volumes, draft_cond, LCB_cond, VCB_cond)

        #print(f'Intact GM: {GM_T} m, Required GM: {GM_req:.2f} m')

        # HDF-051: 46 CFR 170.170 is not the selected criterion for this new,
        # unrestricted-ocean bulk carrier.  Preserve the calculation for
        # regression visibility, but label it as a legacy diagnostic rather
        # than representing it as an implemented regulatory constraint.
        intact_GM_constraint = {
            'Class': 'Legacy Stability Diagnostic',
            'Name': 'Legacy weather-GM diagnostic (not selected)',
            'Requirement': 'HDF-051; 46 CFR 170.170 route not selected',
            'Threshold Value': GM_req,
            'Constraint Value': GM_T,
            'Units': 'm',
            'Constraint Eval': GM_req - GM_T,
            'Constraint Bit': GM_T >= GM_req
        }

        self.StabilityConstraints =  pd.DataFrame([intact_GM_constraint]).copy()

        # LCB within 2.5% of LCG
        LCG = self.CG[0]

        print(f'Intact LCB: {LCB_cond} m, LCG: {LCG} m')
        print(f'Intact VCB: {VCB_cond} m, VCG: {self.CG[2]} m')
        
        print(f'Intact Draft: {draft_cond} m, Draft Fwd: {draft_fwd} m, Draft Aft: {draft_aft} m')


        LCB_diff = np.abs(LCB_cond - LCG)
        LCB_threshold = 0.025 * self.hull.LOA
        LCB_constraint = {
            'Class': 'Intact Stability',
            'Name': 'LCB within 2.5 percent of LCG',
            'Requirement': 'Self-imposed',
            'Threshold Value': LCB_threshold,
            'Constraint Value': LCB_diff,
            'Units': 'm',
            'Constraint Eval': LCB_diff - LCB_threshold,
            'Constraint Bit': LCB_diff <= LCB_threshold
        }
        self.StabilityConstraints = pd.concat([self.StabilityConstraints, pd.DataFrame([LCB_constraint])], ignore_index=True)
        
        # HDF-ASM-033: constrain physical aft-versus-forward draft difference
        # to plus/minus one percent of the equilibrium draft.  This is a
        # GenISys design constraint, not a statutory stability criterion.
        _, trim_diff, trim_threshold, _ = evaluate_trim_constraint(
            draft_fwd,
            draft_aft,
            draft_cond,
        )
        trim_constraint = {
            'Class': 'Intact Stability',
            'Name': 'Trim within 1 percent of Draft',
            'Requirement': 'Self-imposed',
            'Threshold Value': trim_threshold,
            'Constraint Value': trim_diff,
            'Units': 'm',
            'Constraint Eval': trim_diff - trim_threshold,
            'Constraint Bit': trim_diff <= trim_threshold
        }
        self.StabilityConstraints = pd.concat([self.StabilityConstraints, pd.DataFrame([trim_constraint])], ignore_index=True)
        
        print(
            'Regulatory floodable-length constraints are not included in '
            'the legacy stability table; use the separate HDF-083 diagnostic.'
        )
        '''
        #Floodable Length Constraints
        permeabilities, FL_Eval = self.calc_Floodable_Length(bulkhead_indices = bulkhead_indices, plot=False)
        idx_095 = permeabilities.index(0.95)
        # We want to pass one and two compartment flooding at 0.95 permeability
        # FL_Eval[permeabilities.index(0.95), 0] for single compartment flooding
        # FL_Eval[permeabilities.index(0.95), 1] for two compartment flooding

        item = pd.DataFrame({
            'Class': ['Damage Stability', 'Damage Stability'],
            'Name': ['Single Compartment Flooding at 0.95 Permeability', 'Two Compartment Flooding at 0.95 Permeability'],
            'Requirement': ['CFR Title 46 Chapter I Subchapter S Part 171 Subpart C Section 171.065-068', 'CFR Title 46 Chapter I Subchapter S Part 171 Subpart C Section 171.065-068'],
            'Threshold Value': [True, True],
            'Constraint Value': [FL_Eval[idx_095, 0], FL_Eval[idx_095, 1]],
            'Units': ['Boolean', 'Boolean'],
            'Constraint Eval': [1 - FL_Eval[idx_095, 0], 1 - FL_Eval[idx_095, 0]],
            'Constraint Bit':  [FL_Eval[idx_095, 0],FL_Eval[idx_095, 1] ]

        })
        
        self.StabilityConstraints = pd.concat([self.StabilityConstraints, item], ignore_index=True)
        
        #Damaged GM requirements
        # We want to pass all damaged GM requirements for 0.95 permeability
        self.calc_Damaged_Stability(bulkhead_indices)
        '''
    
    
    def plot_Floodable_Lengths(self, permeabilities, bulkhead_indices, draft):
        '''
        Later: add functionality to draw the hull profile and bulkheads on the plot, save the plot, and customize the plot appearance
        This function plots the floodable length curves along with the single and two compartment floodable lengths
        '''
        
        #plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for i in range(self.Floodable_Lengths.shape[1]):
            ax.plot(self.X[self.FL_X_idx], self.Floodable_Lengths[self.FL_X_idx, i], label=f'Permeability {permeabilities[i]}')
        
        ax.plot(self.X, self.two_compartment_FL[:,0], 'r--', label='Two Compartment FL')
        ax.plot(self.X, self.two_compartment_FL[:,1], 'r--')
        ax.plot(self.X, self.single_compartment_FL, 'k--', label='Single Compartment FL')

        plt.legend()
        #draft_idx = np.argmin(np.abs(self.Z - draft))
        
        bow, stern = self.hull.gen_Bow_and_Stern_Profiles(self.Z)
        min_bow = np.min(bow)
        bow = bow - min_bow  # Shift bow to start at 0
        stern = stern - min_bow # Shift stern by same amount as bow

        ax_twin = ax.twinx()

        #add vertical lines at bulkhead positions
        for bulkhead in bulkhead_indices:
            ax_twin.plot([self.X[bulkhead], self.X[bulkhead]], [self.Z[0], self.Z[-1]], linewidth=0.5, color=[0,0,0], linestyle='--')

        ax_twin.plot(bow, self.Z, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot(stern, self.Z, linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[0], stern[0]], [self.Z[0], self.Z[0]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([bow[-1], stern[-1]], [self.Z[-1], self.Z[-1]], linewidth=0.5, color=[0,0,0], linestyle='-')
        ax_twin.plot([min(bow), max(stern)], [draft, draft], linewidth=0.5, color=[1,0,0], linestyle='--', label='Critical Draft')
        ax_twin.set_ylabel('Bow and Stern Profile (m)')
        ax_twin.set_ylim([self.Z[0], 4*self.Z[-1]])

        ax.set_xlabel('Position along ship (m)')
        ax.set_ylabel('Floodable Length (m)')
        plt.title('Floodable Length Curves')
        
        plt.grid()
        plt.show()
        return fig, ax



        

            
                



            
            




   


        

        

        
        




      
