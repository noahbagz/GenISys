"""
=========================================================
Written by Noah J. Bagazinski

This code calculates minimum plate thicknesses and spacing requirements for ship structures according to the ABS Part 3 Guidelines for Hull Construction

The ABS rules on Hull Construction and Equipent are found at
https://ww2.eagle.org/content/dam/eagle/rules-and-guides/current/other/1-rules-for-building-and-classing-marine-vessels-2024/1-mvr-part-3-jan24.pdf


The code is separated into the following sections:



ALL CALCULATIONS ARE DONE ACCORDING TO ABS's Metric Units, sign conventions, and constraints.

================================================================================================
MAJOR ASSUMPTIONS FOR USING THIS CODE TO DESIGN SHIP STRUCTURES:
================================================================================================
0) This is for Longitudinally Framed Steel Ships
1) The Freeboard Deck = Bulkhead Deck = Main Deck
2) No Suberstructure or Deckhouse Stengths are calculated in this code
3) The Lowest Deck is the Floor of the Double Bottom
4) The highest deck is the Freeboard Deck
5) There are no tween decks
6) The double bottom is a solid floor with no openings
7) Shell plating thickness can only have two thicknesses across the ship: bottom and side thickness (ABS Part 3 Ch 2 Sec 2 / 3.1)
8) There are 4 types of transverse members: Side Frames, Bottom Frames, Deck Beams, and Water Tight Bulkheads
9) There are 2 types of longitudinal members: Longitudinal Girders Stiffeners
10) There is no calculation for plate thickness where structural members could also be tanks.
11) There is no consideration for appendeges such as bilge keels or skegs
12) None of the optional corrections to thickness are considered in this code

Note: Afer running this code, all thicknesses should be rounded to the nearest 0.5 mm per ABS Part 3 Ch 1 Sec 2 / 5.3

"""

import numpy as np
import pandas as pd

class ABS_Part3_Constraints:
    """
    This class calculates minimum plate thicknesses and spacing requirements for ship structures according to the ABS Part 3 Guidelines for Hull Construction
    """
    def __init__(self, Hull_Dict):
        """
            Hull_Dict = {
        "L": Scantling Length of the Ship, L <= 305m for offshore support vessels, L <= 500m for other steel vessels (ABS Part 3 Ch 1 Sec 1 / 3.1)
        "B": Greatest Molded Breadth of Hull in meters
        "D": Molded Depth of Hull in meters = Scantling Depth Defined in ABS Part 3 Ch 1 Sec 1 / 7.3 
        "d": Molded draft of the ship in meters (baseline to summer load line)
        "Cb": Block Coefficient of the Ship (ABS Part 3 Ch 1 Sec 1 / 13.3) Cb >= 0.6
        "C_proportion": Constant for Proportionality (ABS Part 3 Ch 1 Sec 2 / 7) Cp = 3.0 for offshore support vessels, Cp = 2.0 for other steel vessels under 90m in length, and Cp = 2.5 for other steel vessels over 90m in length
    
    
        """
        self.Hull_Dict = Hull_Dict

        # CSR-H Pt 1, Ch 3, Sec 1 [2.1.1] and project assumption ASM-005.
        # Keep the active material definition beside the rule calculator so
        # every downstream structural check uses the same values.
        self.E_steel = 206000.0  # N/mm^2
        self.nu_steel = 0.30
        self.sigma_yield_steel = 235.0  # N/mm^2
        self.Q_steel = 1.0

    @staticmethod
    def round_rule_thickness(thickness_mm):
        """Round a calculated plate thickness to the nearest 0.5 mm.

        ABS MVR Part 3, Ch 1, Sec 2/5.3 explicitly places exact quarter-
        millimetre ties in the upper half-millimetre interval.  Python's
        built-in ``round`` uses bankers' rounding, so it cannot be used for
        this Rule operation.
        """

        return np.floor(2.0*float(thickness_mm) + 0.5)/2.0

    def calc_wave_coefficient(self):
        """Return C1 from ABS MVR 3-2-1/3.5.1 for 61 m <= L <= 500 m."""

        L = self.Hull_Dict["L"]

        if L <= 90.0:
            return 0.044*L + 3.75
        elif L < 300.0:
            return 10.75 - ((300.0-L)/100.0)**1.5
        elif L < 350.0:
            return 10.75

        return 10.75 - ((L - 350.0)/150.0)**1.5
        
    def Hull_Dim_Constraints(self):
        
        #Basic Checks:
    
        if self.Hull_Dict["L"] < 90:
            self.Hull_Dict["C_proportion"] = 2.0
        elif self.Hull_Dict["L"] >= 90:
            self.Hull_Dict["C_proportion"] = 2.5

        max_L = 500.0 # meters
        max_B = self.Hull_Dict["L"]*0.2 # meters
        max_B_from_D = self.Hull_Dict["D"]*self.Hull_Dict["C_proportion"] # meters
        return max_L, max_B, max_B_from_D

    def calc_general_constraints(self):
        '''
        This function returns general constraints on the structure design
        '''
        stiffener_aspect_ratio_withFlange = 15.0 # ABS Part 3 Ch 1 Sec 2 / 13.5.2
        stiffener_aspect_ratio_noFlange = 50.0 # ABS Part 3 Ch 1 Sec 2 / 13.5.2
        bending_stress_in_port = 13.13 # kN/cm^2 - ABS Part 3 Ch 2 Sec 1 / 7.3.2
        shear_stress_in_port = 10.0 # kN/cm^2 - ABS Part 3 Ch 2 Sec 1 / 7.3.2

        return stiffener_aspect_ratio_withFlange, stiffener_aspect_ratio_noFlange, bending_stress_in_port, shear_stress_in_port



    def calc_Min_Longitudinal_Section_Modulus_61m(self):
        """
        This function calculates the minimum section modulus requirements for ship structures according to the ABS Part 3 Ch 2 Sec 1 3.1 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or less
        inputs:
        None

        outputs:
        SM = Minimum Section Modulus in m^3
        """
        #ABS Part 3 Ch 2 Sec 1 / 3.1
        L = self.Hull_Dict["L"]
        B = self.Hull_Dict["B"]
        Cb = self.Hull_Dict["Cb"]

        #Check Cb > 0.6
        if Cb < 0.6:
            raise ValueError("Block Coefficient Cb must be greater than or equal to 0.6 for ABS Part 3 Ch 2 Sec 1 / 3.1")

        if L >= 12.0 and L < 18.0:
            C1 = 30.67 - 0.98*L
        elif L >= 18.0 and L < 24.0:
            C1 = 22.4 - 0.52*L
        elif L >= 24.0 and L < 35.0:
            C1 = 15.2 - 0.22*L
        elif L >= 35.0 and L < 45:
            C1 = 11.35 - 0.11*L
        else: # L between 45 and 61 meters
            C1 = 6.40

        C2 = 0.01

        SM = C1*C2*(L**2.0)*B*(Cb+0.7) # cm^2-m
        return SM

    def calc_Absolute_Minimum_Hull_Girder_SM(self):
        """Calculate ABS MVR 3-2-1/3.7.1(b) for vessels 61 m and longer."""

        L = self.Hull_Dict["L"]
        B = self.Hull_Dict["B"]
        # MVR 3-2-1/3.5.1 does not permit Cb below 0.60 in this formula.
        Cb = max(float(self.Hull_Dict["Cb"]), 0.6)
        C1 = self.calc_wave_coefficient()
        C2 = 0.01

        return C1*C2*L**2.0*B*(Cb + 0.7)  # cm^2-m

    def calc_WaveBending_Moments(self):
        """
        This function calculates the wave bending moments for ship structures according to the ABS Part 3 Ch 2 Sec 1 / 3.5.1 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or longer
        inputs:
        None

        outputs:
        M_ws = Wave bending sagging moment in kN*m
        M_wh = Wave bending hogging moment in kN*m
        """

        k1 = 110
        k2 = 190
        L = self.Hull_Dict["L"]
        B = self.Hull_Dict["B"]
        # MVR 3-2-1/3.5.1 does not permit Cb below 0.60 in this formula.
        Cb = max(float(self.Hull_Dict["Cb"]), 0.6)

        C1 = self.calc_wave_coefficient()

        M_ws = -k1*C1*(L**2.0)*B*(Cb+0.7)*.001 # Wave bending sagging moment in kN*m

        M_wh = k2*C1*(L**2.0)*B*(Cb)*.001 # Wave bending hogging moment in kN*m

        return M_ws, M_wh

    @staticmethod
    def calc_wave_bending_distribution_factor(x_over_L):
        """Return the MVR 3-2-1/3.5.2 Figure 2 factor at ``x/L``.

        Figure 2 is zero at both perpendiculars, rises linearly to unity at
        ``x/L = 0.40``, remains unity through ``x/L = 0.65``, and then falls
        linearly to zero at the forward perpendicular.  Keeping this
        distribution separate from the amidship moments is important: the
        Rule still-water and wave moments must be added at the *same station*.
        """

        x = np.asarray(x_over_L, dtype=float)

        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError(
                'ABS wave-bending x/L coordinates must lie in [0, 1]'
            )

        return np.interp(
            x,
            np.array([0.0, 0.40, 0.65, 1.0]),
            np.array([0.0, 1.0, 1.0, 0.0]),
        )
    
    def calc_wave_shear_distribution_factors(self, x_over_L):
        """Return the Figure 3/4 wave-shear factors at ``x/L``.

        ABS MVR 3-2-1/3.5.3 defines the positive and negative wave-shear
        envelopes with the piecewise-linear factors ``F1`` and ``F2`` shown
        in Figures 3 and 4.  The legacy evaluator retained only the maximum
        values (``F1 = 1`` and ``F2 = 0.92``), which is not enough to combine
        wave and still-water shear at the same longitudinal station.

        This helper digitizes the *published break points*, not an
        approximation to the plotted lines.  It accepts either a scalar or
        an array and returns arrays with the same broadcast shape.
        """

        x = np.asarray(x_over_L, dtype=float)

        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError('ABS wave-shear x/L coordinates must lie in [0, 1]')

        Cb = max(float(self.Hull_Dict['Cb']), 0.6)

        # Figure 3 ties the after-body plateau to the negative-wave load
        # coefficient so the positive and negative envelope branches meet.
        after_positive = 0.92*190.0*Cb/(110.0*(Cb + 0.7))

        # Figure 4 uses the reciprocal coefficient relationship for its
        # forward-body plateau.  Both ratios come directly from the figure.
        forward_negative = 190.0*Cb/(110.0*(Cb + 0.7))

        def piecewise_factor(after_plateau, forward_plateau):
            """Evaluate the common seven-segment Figure 3/4 polyline."""

            return np.piecewise(
                x,
                [
                    x <= 0.20,
                    (x > 0.20) & (x <= 0.30),
                    (x > 0.30) & (x <= 0.40),
                    (x > 0.40) & (x <= 0.60),
                    (x > 0.60) & (x <= 0.70),
                    (x > 0.70) & (x <= 0.85),
                    x > 0.85,
                ],
                [
                    lambda q: after_plateau*q/0.20,
                    after_plateau,
                    lambda q: after_plateau
                    + (0.70-after_plateau)*(q-0.30)/0.10,
                    0.70,
                    lambda q: 0.70
                    + (forward_plateau-0.70)*(q-0.60)/0.10,
                    forward_plateau,
                    lambda q: forward_plateau*(1.0-q)/0.15,
                ],
            )

        F1 = piecewise_factor(after_positive, 1.0)
        F2 = piecewise_factor(0.92, forward_negative)

        return F1, F2

    def calc_Wave_Shear_Force_Envelopes(self, x_over_L):
        """Return signed MVR 3-2-1/3.5.3 shear envelopes in kN."""

        k = 30.0
        L = float(self.Hull_Dict['L'])
        B = float(self.Hull_Dict['B'])
        Cb = max(float(self.Hull_Dict['Cb']), 0.6)
        C1 = self.calc_wave_coefficient()
        F1, F2 = self.calc_wave_shear_distribution_factors(x_over_L)

        common = k*C1*L*B*(Cb + 0.7)*1.0e-2

        return common*F1, -common*F2

    def calc_Wave_Shear_Forces(self):
        """
        This function calculates the wave shear forces for ship structures according to the ABS Part 3 Ch 2 Sec 1 / 3.5.3 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or longer
        inputs:
        None

        outputs:
        """
        # Preserve the historical scalar API, but obtain its extrema from the
        # complete rule distributions instead of hard-coding two factors.
        break_points = np.array([0.0, 0.20, 0.30, 0.40,
                                 0.60, 0.70, 0.85, 1.0])
        positive, negative = self.calc_Wave_Shear_Force_Envelopes(
            break_points
        )

        return float(np.max(positive)), float(np.min(negative))

    def calc_freeboard_deck_head(self):
        """Return the MVR 3-2-7/3.1 Table 2 column-``a`` deck head.

        GenISys has one upper weather/freeboard deck and no tween decks.  It
        therefore maps to Table 1's ``freeboard deck having no continuous
        decks below`` category.  With no deck cargo (ASM-013), the governing
        head is Table 2 column ``a`` rather than the separate 3.66 m deck-
        cargo value.  Intermediate lengths are linearly interpolated exactly
        as directed below Table 2; the value is capped at 2.90 m for
        ``L >= 110 m`` by the published table.
        """

        lengths_m = np.array([
            30.0, 40.0, 50.0, 60.0, 70.0,
            80.0, 90.0, 100.0, 110.0, 120.0, 122.0,
        ])
        column_a_heads_m = np.array([
            1.36, 1.56, 1.76, 1.96, 2.16,
            2.36, 2.56, 2.76, 2.90, 2.90, 2.90,
        ])

        L = float(self.Hull_Dict['L'])

        return float(np.interp(
            L,
            lengths_m,
            column_a_heads_m,
            left=column_a_heads_m[0],
            right=column_a_heads_m[-1],
        ))
    
    def calc_Min_Longitudinal_Section_Modulus_Constraint(self,Mnsw, Mpsw):
        """
        This function calculates the minimum section modulus requirements for ship structures according to the ABS Part 3 Ch 2 Sec 1 / 3.7 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or longer

        THIS CALCULATION IS FOR LONGITUDINAL GIRDERS 

        inputs:
        Mnsw = Maximum negative Bending Moment in kN*m from stillwater Distributed Loading
        Mpsw = Maximum positive Bending Moment in kN*m from stillwater Distributed Loading

        outputs:
        SM = Minimum Section Modulus in m^3
        """
        M_ws, M_wh = self.calc_WaveBending_Moments()


        Mtn = Mnsw + M_ws # Total Maximum Negative Bending Moment in kN*m
        Mtp = Mpsw + M_wh # Total Maximum Positive Bending Moment in kN*m
        Mt = max(abs(Mtn), abs(Mtp))

        fp = 17.5 # Allowable Stress in bending stress kN/cm^2 for steel 

        SM = Mt/(fp) # in cm^2-m

        SM_absolute_min = self.calc_Absolute_Minimum_Hull_Girder_SM()
        self.min_Hull_Girder_SM = max(SM, SM_absolute_min)
        return self.min_Hull_Girder_SM # 
    
    def calc_Hull_Girder_I_Constraint(self, SM):
        """
        This function calculates the minimum moment of inertia requirements for ship structures according to the ABS Part 3 Ch 2 Sec 1 / 3.7 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or longer

        THIS CALCULATION IS FOR LONGITUDINAL GIRDERS: all continuous longitudinal members. Check effective longitudinal section modulus for decks with large hatch openings

        inputs:
        SM = Minimum Section Modulus in cm^2-m

        outputs:
        I = Minimum Moment of Inertia in cm^4
        """
        I_min = SM*(self.Hull_Dict["L"])/(33.3) # in cm^2-m^2
        return I_min
    
    def calc_Shearing_Force_Constraint(self, Fnsw, Fpsw, I_hullGirder, Area_Moment_hullGirder, side_shell_thickness):
        """
        This function calculates the minimum shear area requirements for ship structures according to the ABS Part 3 Ch 2 Sec 1 / 3.9 Guidelines for Longitudinal Strength
        This is for ships with a length of 61m or longer

        THIS CALCULATION IS FOR LONGITUDINAL GIRDERS: Girders, decks, bottom shell, side shell, long. bullheads. 

        inputs:
        Fnsw = Maximum negative Shear Force in kN from stillwater Distributed Loading
        Fpsw = Maximum positive Shear Force in kN from stillwater Distributed Loading
        I_hullGirder = Moment of Inertia of the Hull Girder in cm^4
        Area_Moment_hullGirder = First Moment of Area of the Hull Girder in cm^3
        shell_thickness = thickness of the shell plating in mm -> re

        Igonring shear force correction for bulkheads per ABS Part 3 Ch 2 Sec 1 / 3.9.3

        outputs:
       
        """
        Fwp, Fwn = self.calc_Wave_Shear_Forces()

        Ftn = Fnsw + Fwn
        Ftp = Fpsw + Fwp
        Ft = max(abs(Ftn), abs(Ftp))

        fs = Ft*Area_Moment_hullGirder/(2*0.1*side_shell_thickness*I_hullGirder) # in kN/cm^2

        fs_Max_Allowable = 11.0 # ABS MVR 3-2-1/3.9.1, in kN/cm^2

        return fs, fs_Max_Allowable
    

    def calc_side_shell_pressure_thickness(self, spacing_mm):
        """Return the rounded MVR 3-2-2/3.9.3 side-shell threshold."""

        L = self.Hull_Dict["L"]
        D_s = self.Hull_Dict["D"]
        d_s = self.Hull_Dict["d"]
        d_D = max(d_s/D_s, 0.0433*L/D_s)

        if L <= 305.0:
            calculated = (spacing_mm/645.0)*np.sqrt((L-15.2)*d_D) + 2.5
        elif L <= 427.0:
            calculated = (spacing_mm/828.0)*np.sqrt((L+175.0)*d_D) + 2.5
        else:
            raise ValueError('ABS MVR 3-2-2/3.9.3 is limited to L <= 427 m')

        return self.round_rule_thickness(calculated)

    def calc_bottom_shell_pressure_thickness(self, spacing_mm):
        """Return the rounded MVR 3-2-2/3.15.1 bottom-shell threshold."""

        L = self.Hull_Dict["L"]
        draft_depth_ratio = self.Hull_Dict["d"]/self.Hull_Dict["D"]

        if L <= 122.0:
            calculated = ((spacing_mm/671.0)
                          * np.sqrt((L-18.3)*draft_depth_ratio) + 2.5)
        elif L <= 305.0:
            calculated = ((spacing_mm/508.0)
                          * np.sqrt((L-62.5)*draft_depth_ratio) + 2.5)
        elif L <= 427.0:
            calculated = ((spacing_mm/661.0)
                          * np.sqrt((L+105.0)*draft_depth_ratio) + 2.5)
        else:
            raise ValueError('ABS MVR 3-2-2/3.15.1 is limited to L <= 427 m')

        return self.round_rule_thickness(calculated)

    def calc_shell_end_thickness(self, spacing_mm=610.0):
        """Return the rounded MVR 3-2-2/5.1 end-shell threshold."""

        L = self.Hull_Dict["L"]

        if L < 90.0:
            calculated = 0.0455*L + 0.009*spacing_mm
        elif L <= 305.0:
            calculated = 0.035*(L+29.0) + 0.009*spacing_mm
        elif L <= 427.0:
            rule_depth = max(self.Hull_Dict["D"], 35.0)
            calculated = ((11.70 + 0.009*spacing_mm)
                          * np.sqrt(rule_depth/35.0))
        else:
            raise ValueError('ABS MVR 3-2-2/5.1 is limited to L <= 427 m')

        return self.round_rule_thickness(calculated)

    def calc_bottom_shell_minimum_thickness(self, spacing_mm,
                                            standard_frame_spacing_mm):
        """Return MVR 3-2-2/3.15.2 without taking optional Rb credit.

        The reduction factor may only be used when all three strength and
        buckling conditions in the clause are demonstrated.  GenISys does not
        yet demonstrate those conditions, so the unreduced factor Rb = 1.0 is
        the applicable result.
        """

        L = self.Hull_Dict["L"]
        lower_spacing_limit = min(0.88*standard_frame_spacing_mm, 813.0)
        rule_spacing = max(spacing_mm, lower_spacing_limit)
        calculated = rule_spacing*(L-18.3)/(42.0*L + 1070.0)

        return self.round_rule_thickness(calculated)

    def calc_inner_bottom_cargo_pressure_thickness(
        self,
        spacing_mm,
        cargo_depth_m,
        cargo_density_kg_m3=2400.0,
        gravity_m_s2=9.81,
    ):
        """Return the MVR 3-2-4/9.2 cargo-pressure plate threshold.

        The published plate equation uses a reference cargo load of
        7.01 kN/m^3 multiplied by the cargo-space height.  The optimized
        GenISys cargo state can have a different fill depth and density, so
        the source-rule head is adjusted in direct proportion to the actual
        vertical cargo pressure before the half-millimetre Rule rounding is
        applied.
        """

        cargo_specific_weight_kN_m3 = (
            cargo_density_kg_m3*gravity_m_s2/1000.0
        )
        equivalent_rule_head_m = (
            max(float(cargo_depth_m), 0.0)
            * cargo_specific_weight_kN_m3/7.01
        )
        calculated = (
            float(spacing_mm)*np.sqrt(equivalent_rule_head_m)/254.0
            + 1.5
        )

        return self.round_rule_thickness(calculated)

    def calc_Shell_Plating_Constraints(self,s_frames,s_side_stringers, s_bottom_stringers, isOSV = False):
        """
        This function calculates the minimum shell plating thicknesses and spacing requirements for ship structures according to the ABS Part 3 Ch 2 Sec 2 Guidelines for Shell Plating

        No Calculations for Side Fenders in this code (ABS Part 3 Ch 2 Sec 2 / 3.11)
        No Calculations for minimum shell plating at the ends of the vessel - Assume shell plating thickness is consistent across length of the ship

        inputs:
        isOSV = True if the ship is an Offshore Support Vessel, False if the ship is any other steel vessel
        s = spacing between transverse frames or longitudinals in milimeters

        """
        if isOSV:
            raise NotImplementedError(
                'The audited GenISys path is for a non-OSV bulk carrier'
            )

        side_spacing = max(s_frames, s_side_stringers)
        bottom_spacing = max(s_frames, s_bottom_stringers)

        side_pressure = self.calc_side_shell_pressure_thickness(side_spacing)
        bottom_pressure = self.calc_bottom_shell_pressure_thickness(
            bottom_spacing
        )
        end_minimum = self.calc_shell_end_thickness(610.0)

        if self.Hull_Dict["L"] <= 270.0:
            standard_frame_spacing = 2.08*self.Hull_Dict["L"] + 438.0
        else:
            standard_frame_spacing = 1000.0

        bottom_general_minimum = self.calc_bottom_shell_minimum_thickness(
            bottom_spacing,
            standard_frame_spacing,
        )

        # The general legacy result retains the governing envelope.  The
        # audited integration path also records every atomic source clause.
        min_t_side_shell = max(side_pressure, end_minimum)
        min_t_bottom_shell = max(
            bottom_pressure,
            bottom_general_minimum,
            end_minimum,
        )

        return min_t_side_shell, min_t_bottom_shell
        
    def calc_Deck_Constraints(self,s_frames, s_deck_stringers,h):
        """
        This function calculates the minimum shell plating thicknesses and spacing requirements for ship structures according to the ABS Part 3 Ch 2 Sec 3 Guidelines for Deck Strength

        inputs:
        s = spacing between transverse frames or longitudinals in milimeters
        h = height of the deck in meters

        Notes on h, ABS Part 3 Ch 2 Sec 3 / 5.1:
        	
            h = 3.66 m (12 ft) for an exposed deck intended to carry deck cargoes when uniform loading p does not exceed 25.66 kN/m2 (2617 kgf/m2, 536 lbf/ft2)
            h = p/7.01 m (p/715 m, p/44.7 ft) for an exposed deck intended to carry deck cargoes when load p exceeds 25.66 kN/m2 (2617 kgf/m2, 536 lbf/ft2)
            h = hcs, in m (ft), the cargo space height at side for lower decks intended to carry cargoes. Where the cargo load differs from 7.01 kN/m3 (715 kgf/m3, 44.7 lbf/ft3) multiplied by the cargo space height, the height is to be proportionately adjusted.

        
        """
        s = max(s_frames, s_deck_stringers) # in mm
        #ABS Part 3 Ch 2 Sec 3 / 5.1
        t_1 = s*np.sqrt(h)/254.0 +1.5 #mm
        
        t = max(t_1, 5.0) # ABS min thickness is 5mm

        return self.round_rule_thickness(t)

        #ABS Part 3 Ch 2 Sec 3 / 7.1
    
    def calc_Bottom_Structure_Constraints(self, s_bottom_girders,
                                          s_bottom_stringers,
                                          s_lower_hopper_stiffeners,
                                          s_webframes, s_frames, h):
        """
        This function calculates the minimum shell plating thicknesses and spacing requirements for ship structures according to the ABS Part 3 Ch 2 Sec 4 Guidelines for Bottom Structure

        inputs:
        s_bottom_girders = spacing between girder longitudinals in meters
        s_bottom_stringers = spacing between stringer longitudinals in milimeters
        s_lower_hopper_stiffeners = spacing between lower hopper stiffeners in milimeters
        s_webframes = spacing between transverse web frames in meters
        s_frames = spacing between transverse frames in milimeters
        h = height of the cargo hold in meters

        ASSUMPTIONS:
        1) The double bottom is a solid floor with no openings
        2) There are no tween decks
        3) The lowest deck is the Floor of the Double Bottom
        4) No calculation is needed for plate thickness of inner-bottom longitudinals - assume requirement is satisfied with 85% of the bottom longitudinal section modulus (does not consider ABS Part 3 Ch 2 Sec 4 / 11.5) 

        """
        bottom_reqs = {}
        #ABS Part 3 Ch 2 Sec 4 / 3.1.1
        t = 0.056*self.Hull_Dict["L"] + 5.5 #mm

        bottom_reqs["min_t_center_girder"] = self.round_rule_thickness(t)

        db = 32*self.Hull_Dict["B"] +190*np.sqrt(self.Hull_Dict["d"]) #mm

        bottom_reqs["min_double_bottom_height"] = db

        #ABS Part 3 Ch 2 Sec 4 / 3.7

        bottom_reqs["max_bottom_girder_spacing"] = 4570 #mm


        #ABS Part 3 Ch 2 Sec 4 / 5 - solid floors

        t = 0.036*self.Hull_Dict["L"] + 4.7+1.5 #mm - Longidutinally framed ships
        bottom_reqs["min_solid_floor_plate_t"] = self.round_rule_thickness(t)  #for solid floor plates, side girder plates, and stiffener plates

        bottom_reqs["max_solid_floor_spacing"] = 3660 #mm

        bottom_reqs["max_bottom_stiffener_spacing"] = 1530 #mm

        #ABS Part 3 Ch 2 Sec 4 / 7.3 - Transverse frame structures
        draft = max(self.Hull_Dict["d"], 0.67*self.Hull_Dict["D"])
        SM = 7.8*1.0*s_frames/1000.0*draft*(s_bottom_girders)**2.0 #in cm^3
        bottom_reqs["min_bottom_frame_SM"] = SM

        #ABS Part 3 Ch 2 Sec 4 / 9.2 - Inner Bottom Plating in cargo holds
        s = max(s_bottom_stringers, s_frames)
        t = (s*np.sqrt(h)/254) + 1.5 #mm
        bottom_reqs["min_inner_bottom_plating_t"] = self.round_rule_thickness(t) #Same as Deck plating thickness

        # ABS Part 3 Ch 2 Sec 4 / 9.5 - Grab-service plating
        # Use LOA for the rule length and ordinary-strength steel with Q = 1.0
        # under assumptions ASM-002, ASM-005, and ASM-009.
        Q = 1.0 # no steel strength reduction factor

        s = s_bottom_stringers
        t = (0.037*self.Hull_Dict["L"] + 0.009*s)*np.sqrt(Q) + 5.5 #mm
        bottom_reqs["min_inner_bottom_grab_t"] = self.round_rule_thickness(t)

        # The same rule applies to lower-wing sloping plating within 1.5 m of
        # the inner bottom, using the actual lower-hopper stiffener spacing.
        s = s_lower_hopper_stiffeners
        t = (0.037*self.Hull_Dict["L"] + 0.009*s)*np.sqrt(Q) + 5.5 #mm
        bottom_reqs["min_lower_hopper_grab_t"] = self.round_rule_thickness(t)

        #ABS Part 3 Ch 2 Sec 4 / 11.3 - Bottom Longitudinals
        ds = max(self.Hull_Dict["d"], 0.67*self.Hull_Dict["D"])
        l = max(s_webframes, 1.83) 
        SM = 7.8*1.3*ds*(s_bottom_stringers/1000)*l**2.0
        bottom_reqs["min_bottom_longitudinal_SM"] = SM

        #ABS Part 3 Ch 2 Sec 4 / 11.5 - Inner Bottom Longitudinals 
        SM = 0.85*bottom_reqs["min_bottom_longitudinal_SM"] # in cm^3


        bottom_reqs["min_inner_bottom_longitudinal_SM"] = SM 
        return bottom_reqs


    def calc_Frame_Constraints(self, s_frame, l_hold, h_hold, b):
        """
        This function calculates the minimum frame spacing requirements for ship structures according to the ABS Part 3 Ch 2 Sec 5 Guidelines for Frame Spacing
        s_frame = spacing between transverse frames in meters
        l_hold = height between the top of the floor-web bracket to the bottom of the deck-web brackets in meters, not to be less than 2.1m
        h_hold = height between the midpoint of l and the draft mark or 0.4l in meters, whichever is greater
        b = distance from the shell to the first longitudinal in meters (for this case, it is the width of the web frame)
        """
        l_hold = max(l_hold, 2.1)

        # ABS MVR 3-2-5/3.1.1 defines h as the distance from the middle of
        # the actual span to the scantling draft, but not less than 0.4*l.
        # StructureCalcs supplies that actual distance as h_hold.
        h_hold = max(h_hold, 0.4*l_hold)
     

        #ABS Part 3 Ch 2 Sec 5 / 1.7
        if self.Hull_Dict["L"] <= 270:
            max_s_frame = 2.08* self.Hull_Dict["L"] + 438 #mm
        
        elif self.Hull_Dict["L"] > 270 and self.Hull_Dict["L"] <= 427:
            max_s_frame = 1000.0 #mm

        else:
            # MVR 3-2-5/1.7 publishes no empirical spacing branch above
            # 427 m.  Return an unavailable threshold so the status-aware
            # caller can emit a failed rule-domain row while still evaluating
            # the independently defined 3.1 side-frame strength equation.
            max_s_frame = np.nan

        #ABS Part 3 Ch 2 Sec 5 3.1 Transverse Frame SM
        if self.Hull_Dict["L"] >= 90.0:
            C1 = 30.0
            C2 = 7.0
            C3 = 45.0
            # No deck cargo is carried.  The rule then requires the remaining
            # cargo/accommodation head sum h1 to be at least 2.44 m.
            h_1 = 2.44
            Q = 1.0 # no steel strength reduction factor

            SM = s_frame*l_hold**2.0 * (h_hold+(b*h_1)/C1)*(C2 + C3/l_hold**3.0)*Q # in cm^3
            #self.section_modulus_requirements["min_frame_SM"] = SM

        else:
            #ABS Part 3 Ch 2 Sec 5 3.1.2 Transverse Frame SM
            C1 = 7.8
            c = 0.915
            h = max(h_hold, 0.02*self.Hull_Dict["L"] + 0.46) #m
            Q = 1.0 # no steel strength reduction factor
            SM = C1*c*h*s_frame*l_hold**2.0 * Q # in cm^3

            #self.section_modulus_requirements["min_frame_SM"] = SM
        return max_s_frame, SM

    def calc_WebframeAndSideStringer_Constraints(self, s_web, l_web, h_web, b, K, s_side_stringer):
        """
        This function calculates the minimum web frame and side stringer thickness and spacing requirements for ship structures according to 
        the ABS Part 3 Ch 2 Sec 6 Guidelines for Web Frames and Side Stringers

        s_web = spacing between web frames in m
        l_web = height between the double bottom and the top of the deck in m
        h_web = height between the midpoint of l and the draft mark or 0.5l in meters, whichever is greater
        b = distance from the shell to the first longitudinal in meters (for this case, it is the width of the web frame)
        K = number of side frames between web frames

        s_side_stringer = spacing between side stringers in m


        """
        c = 1.5
        # With no deck cargo, the supporting-head term is the 2.44 m minimum
        # required by ABS MVR 3-2-5/3.1.1 and referenced by 3-2-6/3.1.1.
        h_e = 2.44
        Q = 1.0 # no steel strength reduction factor

        WebFrame_SM = 4.74*c*s_web*l_web**2.0 * (h_web+(b*h_e)/(45*K))*Q # in cm^3

        #self.section_modulus_requirements["min_webframe_SM"] = SM


        # ABS Part 3 Ch 2 Sec 6 / 3.5 - Web Frame Height and Thickness Proportions
        # Convert the required web-frame depth from meters to millimeters so it
        # can be compared directly with the parametric web-frame dimensions.
        min_webframe_height = 1000.0*0.125*l_web # in mm

        if self.Hull_Dict["L"] < 90.0:
            t = min(1000*0.01*b+3.5, 11.5) #in mm  Minimum web frame thickness based on the thickness of the web
            min_webframe_thickness = self.round_rule_thickness(t)
        else:
            t = min(1000*0.01*b+3.5, 14.0) #in mm  Minimum web frame thickness
            min_webframe_thickness = self.round_rule_thickness(t)

        #ABS Part 3 Ch 2 Sec 6 / 5.1 - Side Stringer Thickness
        # K from ABS Part 3 Ch 2 Sec 5 / 5.3 Type C calculation for simplicity 
        if self.Hull_Dict["L"] <= 180.0:
            K = 0.036*self.Hull_Dict["L"] -0.09
        else:
            K = 0.031*self.Hull_Dict["L"] + 0.83
            
        h = max(0.6667*self.Hull_Dict["D"], 1.8) #m 
        
        min_side_stringer_SM = 4.74*c*h*s_side_stringer * s_web**2.0 * Q

        #self.section_modulus_requirements["min_side_stringer_SM"] = SM 

        # ABS Part 3 Ch 2 Sec 6 / 5.3 - Side Stringer Height and Thickness Proportions
        min_side_stringer_height = 1000*0.125*s_web # in mm

        return WebFrame_SM, min_webframe_height, min_webframe_thickness, min_side_stringer_SM, min_side_stringer_height
    
    def calc_BeamsAndLongitudinal_Constraints(self, I_hullGirder, Y, structure_type, s_girders, s_beams, h_hold):
        """
        This function calculates the minimum beam and longitudinal thickness and spacing requirements for ship structures according to 
        the ABS Part 3 Ch 2 Sec 7 Guidelines for Beams and Longitudinals
        
        h = 3.66 #m exposed deck height per ABS Part 3 Ch 2 Sec 5 / 3.1

        inputs:
        I_hullGirder = Moment of Inertia of the Hull Girder in cm^2-m^2
        Y = Distance from the neutral axis to the measured deck in m
        structure_type options are 'longitudinal' or 'transverse'


        """
        #ABS Part 3 Ch 2 Sec 7 / 3.1 - Beam and Longitudinal Section Modulus
        k = self.min_Hull_Girder_SM*Y/I_hullGirder 
        
        if self.Hull_Dict["L"] < 90.0:
            if structure_type == 'longitudinal': 
                c = 1/(1.709 - 0.651*k)
            else:
                c = 0.6

        else:
            if structure_type == 'longitudinal':  
               c = 1/(1.709 - 0.651*k)
            else:
                c = 0.585

        
        SM = 7.8*c*h_hold*s_beams*s_girders**2.0 # in cm^3

        return SM

    def calc_DeckGirdersAndTransverses_Constraints(self, h_deck, bulkhead_cx_area, l_bulkhead, r_bulkhead, s_girder, s_bulkhead, s_beams, h_girder, h_beam):
        """
        This function calculates the minimum deck girder and transverse frame thickness and spacing requirements for ship structures according to 
        the ABS Part 3 Ch 2 Sec 8 Guidelines for Deck Girders and Transverses

        inputs:
        h_deck = height  of deck as required by ABS Part 3 Ch 2 Sec 8 / 5.1 in meters
        bulkhead_cx_area = cross-sectional area of the bulkhead in cm^2 This is the area of the bulkhead plate and vertical stiffeners. 
        l_bulkhead = height of bulkhead span, from double bottom to bottom of deck girders/beams, in meters
        r_bulkhead = radius of gyration of the bulkhead in cm
        s_girder = spacing between deck girders in m
        s_bulkhead = spacing between transverse bulkheads in m
        s_beams = spacing between beams in m

        h_girder = height of deck girder in mm
        h_beam = height of deck beam in mm
        """
        #ABS Part 3 Ch 2 Sec 8 /  3.1 Permissible vertical load of pillars on deck girders
        #Using transverse bulkheads as pillars on deck girders for simplicity
        k = 12.09 
        n = 4.44
      
        W_a = (k-n*l_bulkhead/r_bulkhead)*bulkhead_cx_area # in kN

        #self.general_constraints["max_permissible_vertical_load_on_transverse_bulkhead"] = W_a

        #ABS Part 3 Ch 2 Sec 8 / 3.3 - vertical load on bulkheads
        n = 7.01
        b = self.Hull_Dict["B"]
        h = 2.9 # Height specified by column a of Table 2 in ABS Part 3 Ch 2 Sec 7 / 3.1
        
        W_calc = n*b*s_bulkhead*h # in kN
        #self.general_constraints["calculated_vertical_load_on_transverse_bulkhead"] = W_calc

        #ABS Part 3 Ch 2 Sec 8 / 5.3 and 5.13 - Deck Girder Section Modulus
        c = 1.5 #Tank requirements are more stringent 
        b = s_girder
        min_Deck_Girder_SM = 4.74*c*b*h_deck*s_bulkhead**2.0 # in cm^3
        #self.section_modulus_requirements["min_deck_girder_SM"] = SM

        #ABS Part 3 Ch 2 Sec 8 / 5.5 and 5.13 - Transverse Frame Section Modulus
        c = 1.5 #Tank requirements are more stringent
        min_Deck_Beam_SM = 4.74*c*h_deck*s_girder**2.0 * s_beams # in cm^3
        #self.section_modulus_requirements["min_deck_Beam_SM"] = SM

        #ABS Part 3 Ch 2 Sec 8 / 5.7 and 5.13 - Deck Girder and Beam Height and Thickness Proportions
        h_girder_min = 1000.0*0.0833*s_bulkhead # in mm
        #self.section_modulus_requirements["min_deck_girder_height"] = h_girder_min
        h_beam_min = 1000.0*0.0833*s_girder # in mm
        #self.section_modulus_requirements["min_deck_beam_height"] = h_beam_min

        if self.Hull_Dict["L"] < 90.0:
            t_girder_min = 0.01*h_girder+4 # in mm
            t_beam_min = 0.01*h_beam+4 # in mm
        else:
            t_girder_min = min(0.01*h_girder+4, 15.0) # simplicity is Assuming face area of a girder is greater than 193.5 cm^2 per meter of length
            t_beam_min = max(0.01*h_beam+4, 15.0) # 15 mm is the minimum, not an upper limit
        
        #self.plate_thickness_requirements["min_deck_girder_thickness"] = t_girder_min
        #self.plate_thickness_requirements["min_deck_beam_thickness"] = t_beam_min

        return W_a, W_calc, min_Deck_Girder_SM, min_Deck_Beam_SM, h_girder_min, h_beam_min, t_girder_min, t_beam_min

    def calc_HatchSupport_Constraints(self, w_hatch, l_hatch, C, B, D, N, F, l):
        """
        This function calculates the minimum hatch support thickness and spacing requirements for ship structures according to 
        the ABS Part 3 Ch 2 Sec 8 Guidelines for Hatch Supports

        inputs:
        w_hatch = width of hatch in m
        l_hatch = length of hatch in m
        B = distance from Centerline to the midpoint between the hatch girder and beam knee in m
        C = distance from Girder split by hatch to midpoint between the hatch girder and beam knee in m
        D = distance from hatch beam to bulkhead in m
        N = distance from centerline to side of the ship in m
        F = distance from the side of the ship to the hatch opening in m
        l = distance fro centerline to the beam knee in m
        """
        #ABS Part 3 Ch 2 Sec 8 / 7.1 - Hatch Support Section Modulus
        h = 2.9 # Height specified by column a of Table 2 in ABS Part 3 Ch 2 Sec 7 / 3.1
        
        FN = F/N
        if FN <= 0.6:
            K = 2.2 + 1.29*FN
        else:
            K = 4.28-2.17*FN

        SM = K*(l_hatch*B + C*D)*h*(l+0.305)
        return SM

    def calc_Bulkhead_Constraints(self, s_bulk_vertical, s_bulk_horizontal):
        """
        This function calculates the minimum bulkhead thickness and spacing requirements for ship structures according to 
        the ABS Part 3 Ch 2 Sec 9 Guidelines for Bulkheads

        inputs:
        s_vertical = spacing between vertical stiffeners in mm
        s_horizontal = spacing between horizontal stiffeners in mm


        """
        #ABS Part 3 Ch 2 Sec 9 / 5.1 - Bulkhead platingThickness
        s = max(s_bulk_vertical, s_bulk_horizontal)
        k = 1.0 # more stringent for bulkheads
        h = self.Hull_Dict["D"] #m height of bulkhead
        c = 254.0 # Constant for collision bulkheads, but more stringent

        Q = 1.0 # no steel strength reduction factor

        t = s*k*np.sqrt(Q*h)/c + 1.5 #mm

        t_min = self.round_rule_thickness(max([t, 6.0, s/200.0+2.5])) #mm
        #self.plate_thickness_requirements["min_bulkhead_plating_t"] = t_min

        #ABS Part 3 Ch 2 Sec 9 / 5.3 - Bulkhead Stiffener Section Modulus
        k = 1.25 # more stringent for collision bulkheads
        c= 0.6  # constant assuming no end attachments for stiffeners (most stringent)
        h = self.Hull_Dict["D"] #m height of bulkhead
        SM = 7.8*k*c*s/1000.0*h*(s_bulk_horizontal/1000.0)**2.0 # in cm^3


        return t_min, SM


    




        

        
       







        










    




    



    
            
