## This script runs all of the hull design tools to create a ship hull design



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tools.HullParameterization import Hull_Parameterization as HP
import tools.ModifiedMichellCw as MWCw
from tools.EvaluationErrors import MethodDomainConstraintFailure
from tools.Powering_Constraints_v1p1 import (
    CONSTRAINT_COLUMNS as POWERING_CONSTRAINT_COLUMNS,
    HOLTROP_MINIMUM_RUN_LENGTH_RATIO,
    evaluate_holtrop_run_length_constraint,
    holtrop_run_length_ratio,
)
from tools.PoweringInputs import resolve_powering_inputs



'''
Key Functions: 
-Simulate Drag
-Calculate Engine Power
-Calculate Engine Room Length
-Caclulate Hull Constraints
'''

class HullDesign:
    @staticmethod
    def _HoltropBowBulbHeight(hbb_fraction, waterline_height):
        """Return the modeled height of the bow bulb's widest point."""

        return float(hbb_fraction)*float(waterline_height)

    @staticmethod
    def _RequirePositiveFinite(value, quantity_name):
        """Return a scalar after verifying a correlation denominator/input."""

        value = float(value)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{quantity_name} must be finite and positive; received "
                f"{value!r}."
            )

        return value

    @staticmethod
    def _RequireNonnegativeFinite(value, quantity_name):
        """Return a scalar valid for an optional zero-valued load input."""

        value = float(value)

        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{quantity_name} must be finite and nonnegative; received "
                f"{value!r}."
            )

        return value

    @staticmethod
    def _HoltropLCBPercent(lcb_x, forward_x, aft_x):
        """Convert the generated x-coordinate to Holtrop's LCB convention.

        HullParameterization increases x from the forward end toward the aft
        end.  Holtrop and Mennen define ``lcb`` as the longitudinal center of
        buoyancy forward of the waterline midpoint, expressed as a percentage
        of waterline length.  A center two metres forward of the midpoint of a
        100 m waterline therefore returns ``+2.0``, not ``+0.02``.
        """

        lcb_x = float(lcb_x)
        forward_x = float(forward_x)
        aft_x = float(aft_x)
        waterline_length = aft_x - forward_x

        if not np.all(np.isfinite([lcb_x, forward_x, aft_x])):
            raise ValueError(
                "LCB and waterline endpoints must be finite for the Holtrop "
                "conversion."
            )

        if waterline_length <= 0.0:
            raise ValueError(
                "The aft waterline endpoint must lie aft of the forward "
                "endpoint."
            )

        waterline_midpoint = 0.5*(forward_x + aft_x)

        return 100.0*(waterline_midpoint - lcb_x)/waterline_length

    def __init__(
        self,
        hull_params,
        Fn_ship,
        Draft_frac,
        Voyage_Range,
        Cw=None,
        powering_inputs=None,
    ):
        self.rho = 1025.0 # Density of Sea Water in kg/m^3
        self.g = 9.81 # Gravity in m/s^2

        # Resolve the existing seven global assumptions once so resistance,
        # installed power, and voyage fuel all consume the same explicit
        # scenario.  No propeller or certified-engine data are inferred.
        self.Powering_Inputs = resolve_powering_inputs(powering_inputs)

        self.hull_params = hull_params
        self.hull = HP(hull_params)
        cons = self.hull.input_Constraints()
        self.HullConstraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])
        cons_item = pd.read_csv('tools/Hull_constraints_list.csv')
        self.HullConstraints = pd.concat([self.HullConstraints, cons_item], ignore_index=True)
        self.HullConstraints['Constraint Value'] = cons
        self.HullConstraints['Constraint Eval'] = self.HullConstraints['Threshold Value'] + self.HullConstraints['Constraint Value'] # All thresholds are zero and Constraint Values are formatted to be negative if they are met
        self.HullConstraints['Constraint Bit'] = self.HullConstraints['Constraint Eval'] <= 0 # True if constraint met 
        
        
        
        self.Fn_ship = self._RequirePositiveFinite(
            Fn_ship,
            "Ship Froude number",
        )
        
        self.Draft = Draft_frac*self.hull.Dd # Draft of the ship in m
        self.Draft = self._RequirePositiveFinite(
            self.Draft,
            "Design draft",
        )

        self.Voyage_Range = self._RequirePositiveFinite(
            Voyage_Range,
            "Voyage range",
        ) # Voyage Range in nautical miles



        Z = self.hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
        self.Displacement = self._RequirePositiveFinite(
            HP.interp(self.hull.Volumes, Z, self.Draft),
            "Displacement volume",
        ) # Displacement in m^3
        self.WSA = self._RequirePositiveFinite(
            HP.interp(self.hull.Area_WS, Z, self.Draft),
            "Wetted surface area",
        ) # Wetted Surface Area in m^2
        self.WPA = self._RequirePositiveFinite(
            HP.interp(self.hull.Areas_WP, Z, self.Draft),
            "Waterplane area",
        ) # Waterplane Area in m^2
        self.WL_len = self._RequirePositiveFinite(
            HP.interp(self.hull.WL_Lengths, Z, self.Draft),
            "Waterline length",
        ) # Waterline Length in m

        # The existing design convention uses twice the modeled deck
        # half-breadth as B.  Preserve that convention here; changing the
        # principal-breadth definition would be an engineering policy change.
        self.B = self._RequirePositiveFinite(
            2.0*self.hull.Bd,
            "Design breadth",
        )

        coefficient_denominator = self._RequirePositiveFinite(
            self.hull.LOA*self.B*self.Draft,
            "Block-coefficient reference volume",
        )
        self.Cb = self._RequirePositiveFinite(
            self.Displacement/coefficient_denominator,
            "Block coefficient",
        )

        # Calculate the actual generated immersed midship area.  The legacy
        # formula subtracted a length-valued chine radius from an area and was
        # therefore dimensionally inconsistent.
        self.Midship_Section_Area = self.hull.Calc_MidshipSectionArea(
            self.Draft
        )
        self.Cm = self._RequirePositiveFinite(
            self.Midship_Section_Area/(self.B*self.Draft),
            "Midship-section coefficient",
        )

        self.Cp = self._RequirePositiveFinite(
            self.Cb/self.Cm,
            "Prismatic coefficient",
        )
        self.Cwp = self._RequirePositiveFinite(
            self.WPA/(self.hull.LOA*self.B),
            "Waterplane coefficient",
        )
        #print('Prismatic Coefficient: ', self.Cp, 'Waterplane Coefficient: ', self.Cwp)
        
        self.U_ship = self._RequirePositiveFinite(
            self.Fn_ship*np.sqrt(self.g*self.WL_len),
            "Ship speed",
        ) # Design Speed of the ship in m/s
        self.U_knots = self.U_ship*1.94384 # Design Speed in knots
        self.Re_ship = self._RequirePositiveFinite(
            self.U_ship*self.WL_len/(1e-6),
            "Ship Reynolds number",
        ) # Reynolds number of the ship at design speed, using kinematic viscosity of water at 15 degrees C
        

        self.LCB = HP.interp(self.hull.VolumeCentroids[:,0], Z, self.Draft)
        self.VCB = HP.interp(self.hull.VolumeCentroids[:,1], Z, self.Draft)

        # Derive Holtrop's signed percentage from the actual waterline
        # endpoints.  HullParameterization uses x increasing from bow to stern.
        waterline_points = self.hull.gen_WLPoints(
            self.Draft,
            PointsPerWL=101,
        )
        self.Holtrop_LCB_Percent = self._HoltropLCBPercent(
            self.LCB,
            waterline_points[0, 0],
            waterline_points[-1, 0],
        )


        self.Hull_Dict = {
            'L': self.hull.LOA, # Assume LOA = Scantling Length -> makes structural constraints more stringent
            'B': self.B,
            'D': self.hull.Dd,
            'd': self.Draft,
            'U_ship': self.U_ship,
            'U_knots': self.U_knots,
            # LCB retains the native absolute x-coordinate used by the rest of
            # GenISys.  The separate percentage is the converted Holtrop input.
            'LCB': self.LCB,
            'Holtrop_LCB_Percent_LWL': self.Holtrop_LCB_Percent,
            'VCB': self.VCB, # VCB at the design draft in meters from baseline
            'Fn': self.Fn_ship,
            'Cb': self.Cb, 
            'Cm': self.Cm,
            'Cp': self.Cp,
            'Midship_Section_Area': self.Midship_Section_Area,
            'Vol_Disp': self.Displacement,
            'Voyange_Range': self.Voyage_Range,
            'Re': self.Re_ship,
            'Wet_Surface_Area': self.WSA,
            'Waterline_Length': self.WL_len} 

        # Expose the exact normalized run-length algebra before resistance is
        # evaluated.  A failed method-domain value can then be preserved as a
        # normal constraint row instead of being reduced to an opaque Python
        # exception before the V1.1 constraint table exists.
        self.Holtrop_Run_Length_Ratio = holtrop_run_length_ratio(
            self.Cp,
            self.Holtrop_LCB_Percent,
        )
        self.Hull_Dict['Holtrop_Run_Length_Ratio'] = (
            self.Holtrop_Run_Length_Ratio
        )
        self.Hull_Dict['Holtrop_Run_Length_m'] = (
            self.Holtrop_Run_Length_Ratio*self.WL_len
        )


        #If Cw comes from a dataset, it should be a vector of length 32 with appropriate Fn and Draft scalings,  otherwise it will be calculated here.
        #self.calc_Hull_Drag(Cw = Cw) # Michell CW is not accurate for this experiment
        self.calc_Engine_Power()
        self.calc_Required_MinEngine_Room_Length()
        self.calc_Fuel_Consumption()        
        
    def calc_Hull_Drag_Michell(self, Cw = None):
        '''
        This function takes in a Ship Design Vector predicts the total drag of the hull at the design speed
        
        '''

        z_ref = np.array([0.25,0.33,0.5,0.67]) #This gets the Z values that are 25%,33%,50%, and 67% of Dd
        z_idx = np.array([1,2,4,6])
            
        Fn = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45])
        
        g = 9.81 # Gravity

        Rho = 1025 # Density of sea Water
        if Cw is None:
            N = 51 #Number of Sample Angles for Rw calculation

            Rw = np.zeros((len(z_ref)*len(Fn)))
            Cw = np.zeros((len(z_ref)*len(Fn)))
            
            for i in range(0,len(z_ref)):
                X,Z,Y,WL   = self.hull.gen_PC_for_Cw(self.hull.Dd*z_ref[i], NUM_WL=101, PointsPerWL=1001)
                
                for j in range(0,len(Fn)):
                    U = Fn[j] * np.sqrt(g*WL)
                    Rw[i*len(Fn)+j] = MWCw.ModMichell(Y,U,X,Z,Rho,N)

                    Cw[i*len(Fn)+j] = Rw[i*len(Fn)+j]/(0.5*Rho*(U**2.0)*self.hull.LOA**2.0) 
            #print('Cw ', Cw)
        else: 
            Cw = np.array(Cw)
            if len(Cw) != 32:
                raise ValueError("Cw input must be a vector of length 32")
        
        T = self.Draft/self.hull.Dd

        Rt,Fn = MWCw.CalcDrag(self.U_ship,self.hull.LOA,self.WL_len,Cw,T,self.WSA, rho=1025.0)

        self.Hull_Dict['Total_Resistance'] = Rt # in Newtons

    def calc_Hull_Drag_Holtrop(self):
        # Power prediction: Holtrop and Mennen 1982, An Approximate Power
        # Prediction Method.  These checks cover mathematical domains only;
        # they do not assert that a generated hull lies inside every empirical
        # applicability limit of the regression.
        if self.Cp >= 1.0:
            raise ValueError(
                "Holtrop resistance requires Cp < 1.0 for the implemented "
                "form-factor equation."
            )

        waterline_beam_ratio = self.WL_len/self.B

        if waterline_beam_ratio <= 2.0:
            raise ValueError(
                "Holtrop resistance requires LWL/B > 2.0 for the implemented "
                "c17 equation."
            )

        run_denominator = 4.0*self.Cp - 1.0

        if abs(run_denominator) <= np.finfo(float).eps:
            raise ValueError(
                "Holtrop run-length denominator 4*Cp - 1 is zero."
            )

        # Holtrop's lcb is a signed percentage of LWL forward of the
        # waterline midpoint.  It is not the fractional value used by the
        # previous implementation.
        lcb = self.Holtrop_LCB_Percent

        run_length_ratio = holtrop_run_length_ratio(self.Cp, lcb)
        self.Holtrop_Run_Length_Ratio = run_length_ratio
        self.Hull_Dict['Holtrop_Run_Length_Ratio'] = run_length_ratio
        self.Hull_Dict['Holtrop_Run_Length_m'] = (
            run_length_ratio*self.WL_len
        )

        # HDF-ASM-036 is an explicit method-domain predicate, not a
        # regulatory ship rule.  Preserve the failed row and the 49 preceding
        # hull-geometry results so stochastic repair can act on Cp/LCB's real
        # upstream hull controls.  Downstream power/fuel calculations stop
        # because the fractional form-factor equation is not valid here.
        if (
            not np.isfinite(run_length_ratio)
            or run_length_ratio < HOLTROP_MINIMUM_RUN_LENGTH_RATIO
        ):
            run_length_constraint = (
                evaluate_holtrop_run_length_constraint(run_length_ratio)
            )
            raise MethodDomainConstraintFailure(
                "Holtrop resistance requires Lr/LWL >= "
                f"{HOLTROP_MINIMUM_RUN_LENGTH_RATIO:.6g}; received "
                f"{run_length_ratio!r}.",
                name=str(run_length_constraint.iloc[0]['Name']),
                requirement=str(
                    run_length_constraint.iloc[0]['Requirement']
                ),
                threshold_value=HOLTROP_MINIMUM_RUN_LENGTH_RATIO,
                constraint_value=run_length_ratio,
                units='dimensionless',
                comparison='minimum',
                result_class='Powering Method Domain',
                constraint_rows=pd.concat(
                    [
                        self.HullConstraints[
                            POWERING_CONSTRAINT_COLUMNS
                        ],
                        run_length_constraint,
                    ],
                    ignore_index=True,
                ),
                diagnostics={
                    'Holtrop_Run_Length_Ratio': run_length_ratio,
                    'Holtrop_Run_Length_m': run_length_ratio*self.WL_len,
                    'Holtrop_Prismatic_Coefficient': self.Cp,
                    'Holtrop_LCB_Percent_LWL': lcb,
                },
            )

        Lr = run_length_ratio*self.WL_len

        # Calculate bulb resistance only when the generated hull actually has
        # a bow bulb.  Previously every hull inherited a bulb term from the
        # stored profile coefficients, even when bit_BB was false.
        if self.hull.bit_BB:
            Abt = np.pi*0.5*self.hull.BB_Prof[3]*self.hull.WL
            # CRF-010/HDF-ASM-037: Hbb is the parametric height of the
            # bulb's widest point as a fraction of the modeled waterline.
            # Holtrop's bulb height must therefore use that generated design
            # coordinate rather than the former fixed WL/2 assumption.
            hb = self._HoltropBowBulbHeight(
                self.hull.Hbb,
                self.hull.WL,
            )
        else:
            Abt = 0.0
            hb = 0.0

        self.Hull_Dict['Bow_Bulb_Height_m'] = float(hb)

        # Use the selected dataset-scenario inputs.  The backwards-compatible
        # base values are the old 1,000 m2 / factor-2 assumptions, but callers
        # can now replace them explicitly without editing this equation.
        # These are project inputs, not coefficients supplied by Holtrop.
        powering_inputs = getattr(
            self,
            "Powering_Inputs",
            resolve_powering_inputs(),
        )
        A_app = self._RequireNonnegativeFinite(
            powering_inputs["Appendage_Area_m2"],
            "Appendage wetted area",
        )
        A_rf = self._RequireNonnegativeFinite(
            powering_inputs["Appendage_Resistance_Factor"],
            "Appendage resistance factor",
        )
        self.Appendage_Area_m2 = A_app
        self.Appendage_Resistance_Factor = A_rf

        C_stern = float(self.hull.bit_EP_S)*10.0 #Ellipsoid stern gets coeffcient = 10

        # Entrance Angle: Use average angle from bow to half of bow taper
        WL_pts = self.hull.gen_WLPoints(self.Draft, PointsPerWL=101)
        bow_reference_x = self._RequirePositiveFinite(
            self.hull.Lb/2.0,
            "Bow entrance-angle reference length",
        )
        bow_reference_y = HP.interp(
            WL_pts[:, 1],
            WL_pts[:, 0],
            bow_reference_x,
        )
        ie = np.degrees(
            np.arctan2(bow_reference_y, bow_reference_x)
        )

        if not np.isfinite(ie) or ie >= 90.0:
            raise ValueError(
                "Holtrop entrance angle must be finite and less than 90 "
                "degrees."
            )

        # Calculate the frictional resistance coefficient using ITTC 1957 formula
        friction_denominator = np.log10(self.Re_ship) - 2.0

        if abs(friction_denominator) <= np.finfo(float).eps:
            raise ValueError(
                "ITTC-1957 friction denominator log10(Re) - 2 is zero."
            )

        Cf = 0.075/friction_denominator**2
        Rf = 0.5 * self.rho * self.U_ship**2 * self.WSA * Cf

        # Calculate Bulb Resistance and Appendage Resistance
        if Abt > 0.0:
            bulb_pressure_denominator = self.Draft - 1.5*hb

            if bulb_pressure_denominator <= 0.0:
                raise ValueError(
                    "Holtrop bulb-pressure denominator T - 1.5*hb must be "
                    "positive."
                )

            bulb_froude_radicand = (
                self.g*(self.Draft - hb - 0.25*np.sqrt(Abt))
                + 0.15*self.U_ship**2
            )

            if bulb_froude_radicand <= 0.0:
                raise ValueError(
                    "Holtrop bulb Froude-number radicand must be positive."
                )

            Pb = 0.56*np.sqrt(Abt)/bulb_pressure_denominator
            Fni = self.U_ship/np.sqrt(bulb_froude_radicand)
            Rb = (
                0.11
                * np.exp(-3.0/Pb**2)
                * Fni**3
                * Abt**1.5
                * self.rho
                * self.g
                / (1.0 + Fni**2)
            )
        else:
            Rb = 0.0

        Rapp = 0.5 * self.rho * self.U_ship**2 * A_app * A_rf * Cf

        # Wave resistance prediction using Holtrop and Mennen method:
        ratio_BL = self.B/self.hull.LOA

        if ratio_BL <= 0.11:
            c7 = 0.229577 * ratio_BL**(1/3)
        elif ratio_BL <= 0.25:
            c7 = ratio_BL
        else:
            c7 = 0.5 - 0.0625 / ratio_BL

        c1 = 2223105 * c7**3.78613 * (self.Draft/self.B)**1.07961 * (90 - ie)**(-1.37565)

        if Abt > 0.0:
            c3_denominator = (
                self.B
                * self.Draft
                * (0.31*np.sqrt(Abt) + self.Draft - hb)
            )
            c3_denominator = self._RequirePositiveFinite(
                c3_denominator,
                "Holtrop c3 denominator",
            )
            c3 = 0.56*Abt**1.5/c3_denominator
        else:
            c3 = 0.0

        c2 = np.exp(-1.89 * np.sqrt(c3))
        c5 = 1.0 # assume no transom submersion

        lambda_val = ((self.WL_len/self.B >= 12) * (1.446*self.Cp - 0.03*self.WL_len/self.B) +
              (self.WL_len/self.B < 12) * (1.446*self.Cp - 0.36))
        
        if self.Cp <= 0.8:
            c16 = 8.07981*self.Cp - 13.8673*self.Cp**2 + 6.984388*self.Cp**3
        else:
            c16 = 1.73014 - 0.7067*self.Cp

        m1 = (0.0140407*self.WL_len/self.Draft - 1.75254*self.Displacement**(1/3)/self.WL_len - 4.79323*self.B/self.WL_len - c16)

        c15 = -1.69385
        m2 = c15 * self.Cp**2 * np.exp(-0.1 / self.Fn_ship**2)
        d = -0.9

        c17 = 6919.3 * self.Cm**(-1.3346) * (self.Displacement/self.WL_len**3)**2.00977 * (waterline_beam_ratio - 2)**1.40692
        m3 = -7.2035 * (self.B/self.WL_len)**0.326869 * (self.Draft/self.B)**0.605375
        m4 = c15 * 0.4 * np.exp(-0.034 / self.Fn_ship**3.29)

        Rw04 = (c1 * c2 * c5 * self.Displacement * self.rho * self.g *np.exp(m1 * self.Fn_ship**d + m4 * np.cos(lambda_val / self.Fn_ship**2)))

        Rw055 = (c17 * c2 * c5 * self.Displacement * self.rho * self.g *np.exp(m3 * self.Fn_ship**d + m4 * np.cos(lambda_val / self.Fn_ship**2)))

        if self.Fn_ship < 0.4:
            Rw = Rw04
        elif self.Fn_ship >= 0.4 and self.Fn_ship < 0.55:
            Rw = Rw04 +(10.0*self.Fn_ship - 4.0)*(Rw055 - Rw04)/1.5
        else:
            Rw = Rw055

        #Corrections 
        c4 = min(self.Draft/self.WL_len, 0.04)
        Ca = (0.006*(self.WL_len + 100)**(-0.16) - 0.00205 + 0.003*np.sqrt(self.WL_len/7.5)*self.Cb**4*c2*(0.04 - c4))

        Ra = 0.5 * self.rho * self.U_ship**2 * self.WSA * Ca

        HullFactor = (0.93 + 0.487118*(1 + 0.011*C_stern) * (self.B/self.WL_len)**1.06806 * (self.Draft/self.WL_len)**0.46106 * (self.WL_len/Lr)**0.121563 * (self.WL_len**3/self.Displacement)**0.36486 * (1 - self.Cp)**(-0.604247))

        Rt = Rf * HullFactor + Rapp + Rw + Rb + Ra

        Rt = self._RequirePositiveFinite(
            Rt,
            "Holtrop total resistance",
        )

        self.Hull_Dict['Total_Resistance'] = Rt
        self.Hull_Dict['Appendage_Area_m2'] = self.Appendage_Area_m2
        self.Hull_Dict['Appendage_Resistance_Factor'] = (
            self.Appendage_Resistance_Factor
        )
        self.Hull_Dict['Waterline_Entrance_Half_Angle_deg'] = float(ie)
        self.Hull_Dict['Bulb_Transverse_Area_m2'] = float(Abt)
        return Rt


    

    def calc_Engine_Power(self, MCR=None, eta_est=None):
        '''
        This function calculates the required engine power to overcome the total drag of the hull at the design speed
        
        Parameters
        ----------
        eta_est = eta_o*eta_R*eta_h*eta_s
        
        eta_o : open water propeller efficiency = 0.637
        eta_R : relative rotative Efficiency = 1.0 
        eta_h : hull efficiency = 1.0
        eta_s : shaft efficiency = 0.98

        https://www.sciencedirect.com/science/article/pii/S0029801823011423 estimates of eta_r, eta_h, and eta_s are all close to 1.0 for high efficiency propellers on merchant ships.

        eta_o estimated by wageningen b series tests
        bernitsas et al 1981
        https://deepblue.lib.umich.edu/handle/2027.42/91702

        Van Lammeren et al 1969 
        https://trid.trb.org/View/354

        Power predicition: Holtrop and Mennen 1982 An approximate power prediction method
        Returns
        -------
        P_eng : Required installed Engine Power in kW
        '''

        # A direct method call may still provide explicit values for focused
        # studies.  Otherwise consume the scenario resolved at construction.
        powering_inputs = getattr(
            self,
            "Powering_Inputs",
            resolve_powering_inputs(),
        )

        if MCR is None:
            MCR = powering_inputs[
                "Maximum_Continuous_Rating_Fraction"
            ]

        if eta_est is None:
            eta_est = powering_inputs[
                "Aggregate_Propulsive_Efficiency"
            ]

        # Keep the current early-design aggregate-efficiency model, but reject
        # fractions outside their mathematical range.  A selected propulsor
        # and engine-matching calculation remains a separate future model.
        MCR = self._RequirePositiveFinite(MCR, "MCR fraction")
        eta_est = self._RequirePositiveFinite(
            eta_est,
            "Aggregate propulsive efficiency",
        )

        if MCR > 1.0:
            raise ValueError("MCR fraction cannot exceed 1.0.")

        if eta_est > 1.0:
            raise ValueError(
                "Aggregate propulsive efficiency cannot exceed 1.0."
            )
        self.Hull_Dict['Maximum_Continuous_Rating_Fraction'] = MCR
        self.Hull_Dict['Aggregate_Propulsive_Efficiency'] = eta_est

        # use the holtrop and mennen method:
        Rt = self.calc_Hull_Drag_Holtrop() # in Newtons
        P_required = Rt * self.U_ship # in Watts
        P_eng = self._RequirePositiveFinite(
            P_required/(MCR*eta_est),
            "Required installed engine power",
        ) # in Watts
        self.Hull_Dict['Engine_Power'] = P_eng/1000.0 # in kW
        self.Hull_Dict['Maximum_Continuous_Rating_Fraction'] = float(MCR)
        self.Hull_Dict['Aggregate_Propulsive_Efficiency'] = float(eta_est)
        self.Hull_Dict['Effective_Power_kW'] = float(P_required/1000.0)
        #return P_eng/1000.0
        print('Required Engine Power (kW): ', self.Hull_Dict['Engine_Power'])
        
        
    def calc_Required_MinEngine_Room_Length(self):
        '''
        Engine sizing Data

        Engine Length to Power Ratios from Wartsila RTA96C engine
        Len = 2.97E-4 * required Power (kW) + 1.04, R^2 = 0.997

        Min Engine Room Length = 1.1*Len + 2m for clearance and maintenance space
        
        https://swiss-ships.ch/berichte/waertsilae/Wartsila-RTA96C-engine-technology-review.pdf
        
        '''
        P_eng = self.Hull_Dict['Engine_Power'] # in kW
        Len = 0.000297 * P_eng + 1.04 # in meters
        self.Hull_Dict['Min_Engine_Room_Length'] = 1.1*Len + 2.0 # in meters # Add 10% and 2m for clearance and maintenance space

    def calc_Fuel_Consumption(
        self,
        specific_fuel_consumption_kg_per_kwh=None,
        auxiliary_fuel_tonnes_per_day=None,
        reserve_days=None,
    ):
        '''
        This function calculates the fuel consumption of the ship based on the installed engine power and specific fuel oil consumption

        based on Hart and Vlahopoulos 2008 https://link.springer.com/article/10.1007/s00158-009-0414-0

        '''

        ship_speed_knots = self._RequirePositiveFinite(
            self.Hull_Dict['U_knots'],
            "Ship speed in knots",
        )
        engine_power_kw = self._RequirePositiveFinite(
            self.Hull_Dict['Engine_Power'],
            "Installed engine power",
        )

        # Voyage range is in nautical miles and speed is in nautical miles per
        # hour.  Divide by 24 hours/day to retain the existing days-at-sea
        # convention explicitly.
        days_at_sea = self.Voyage_Range/(ship_speed_knots*24.0)
        days_at_sea = self._RequirePositiveFinite(
            days_at_sea,
            "Days at sea",
        )

        # Use explicit arguments for isolated studies or the resolved scenario
        # values for integrated evaluations.  The defaults remain the old
        # 0.19 kg/kWh, 0.2 t/day, and five-day reserve assumptions.
        powering_inputs = getattr(
            self,
            "Powering_Inputs",
            resolve_powering_inputs(),
        )

        if specific_fuel_consumption_kg_per_kwh is None:
            specific_fuel_consumption_kg_per_kwh = powering_inputs[
                "Main_Engine_SFOC_kg_per_kWh"
            ]

        if auxiliary_fuel_tonnes_per_day is None:
            auxiliary_fuel_tonnes_per_day = powering_inputs[
                "Auxiliary_Fuel_t_per_day"
            ]

        if reserve_days is None:
            reserve_days = powering_inputs["Fuel_Reserve_Days"]

        specific_fuel_consumption_kg_per_kwh = (
            self._RequirePositiveFinite(
                specific_fuel_consumption_kg_per_kwh,
                "Main-engine specific fuel consumption",
            )
        )
        auxiliary_fuel_tonnes_per_day = self._RequireNonnegativeFinite(
            auxiliary_fuel_tonnes_per_day,
            "Auxiliary fuel consumption",
        )
        reserve_days = self._RequireNonnegativeFinite(
            reserve_days,
            "Fuel reserve duration",
        )
        self.Hull_Dict['Main_Engine_SFOC_kg_per_kWh'] = (
            specific_fuel_consumption_kg_per_kwh
        )
        self.Hull_Dict['Auxiliary_Fuel_t_per_day'] = (
            auxiliary_fuel_tonnes_per_day
        )
        self.Hull_Dict['Fuel_Reserve_Days'] = reserve_days

        main_engine_fuel_kg_per_day = (
            specific_fuel_consumption_kg_per_kwh
            * 24.0
            * engine_power_kw
        )
        auxiliary_fuel_kg_per_day = (
            1000.0*auxiliary_fuel_tonnes_per_day
        )
        fuel_weight_kg = (
            main_engine_fuel_kg_per_day
            + auxiliary_fuel_kg_per_day
        )*(days_at_sea + reserve_days)
        fuel_weight_kg = self._RequirePositiveFinite(
            fuel_weight_kg,
            "Voyage and reserve fuel weight",
        )

        self.Hull_Dict['Days_at_Sea'] = days_at_sea
        self.Hull_Dict['Fuel_Weight_kg'] = fuel_weight_kg
        self.Hull_Dict['Main_Engine_SFOC_kg_per_kWh'] = float(
            specific_fuel_consumption_kg_per_kwh
        )
        self.Hull_Dict['Auxiliary_Fuel_t_per_day'] = float(
            auxiliary_fuel_tonnes_per_day
        )
        self.Hull_Dict['Fuel_Reserve_Days'] = float(reserve_days)
        self.Hull_Dict['Powering_Input_Basis'] = powering_inputs[
            'Powering_Input_Basis'
        ]
        self.Hull_Dict['Explicit_Powering_Input_Fields'] = (
            powering_inputs['Explicit_Powering_Input_Fields']
        )

    def generate_stl(self, path, file_name): 
        strpath =  path+file_name
        mesh = self.hull.gen_stl(NUM_WL=99, PointsPerWL=300, bit_AddTransom = 1, bit_AddDeckLid = 0, bit_RefineBowAndStern = 1,namepath = strpath)
        










   
