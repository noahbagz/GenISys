# This module utilizes the structural elements and bulkhead spacing to perform calculation on the ship structure

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import tools.ABS_Part3_Constraints as ABS_P3
from tools.HatchCovers import rounded_rectangle_area_m2


# ABS MVR Part 3, Chapter 2, Section 1/9.1 effective-section policy.
#
# GenISys intentionally generates a detailed three-hold structural segment,
# rather than a complete fore-to-aft structural model.  The classes below are
# the continuous longitudinal material that may contribute to hull-girder
# bending in the representative maximum-hatch section.  Transverse members,
# brackets, ordinary vertical frames, hatch covers, the deckhouse, and bilge
# keels are deliberately absent from this list.
MVR_BENDING_EFFECTIVE_MEMBER_CLASSES = frozenset({
    'Deck',
    'Side_Shell',
    'Bottom_Shell',
    'Shell_Strake',
    'Inner_Side_Shell',
    'Longitudinal_Bulkhead',
    'Hopper Pannel',
    'Deck_Longitudinal_Girder',
    'Deck_Longitudinal_Stiffener',
    'Bottom_Longitudinal_Girder',
    'Bottom_Longitudinal_Stiffener',
    'Inner_Bottom_Longitudinal_Stiffener',
    'Side_Shell_Longitudinal_Stiffener',
    'Inner_Side_Shell_Longitudinal_Stiffener',
    'Hopper_Pannel_Longitudinal_Stiffener',
})


class StructureCalcs:
    def __init__(self, Hull_Dict, structural_elements:pd.DataFrame, structure_params):
        
        self.Hull_Dict = Hull_Dict
        self.Structural_Elements = structural_elements
        self.structure_params = structure_params
        self.StructureConstraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])
        self.Struct_Dict = pd.DataFrame(columns = ['Parameter', 'Value', 'Units'])
        self.ABS = ABS_P3.ABS_Part3_Constraints(Hull_Dict)


    @staticmethod
    def _generated_member_positions(
        structural_elements,
        member_classes,
        coordinate,
    ):
        """Return sorted unique generated coordinates for selected members.

        The CadQuery element table is the geometry authority for V1.1.  A
        longitudinal member can be split into several pieces at hatch or hold
        boundaries, so the table can contain the same physical transverse
        coordinate more than once.  ``np.unique`` collapses those repeated
        generated values without recreating the layout from the parameter
        vector or changing the coordinates stored in the element table.
        """

        selected = structural_elements.loc[
            structural_elements['Class'].isin(member_classes),
            coordinate,
        ]
        coordinates = pd.to_numeric(selected, errors='coerce').dropna()

        return np.unique(coordinates.to_numpy(dtype=float))


    @classmethod
    def _generated_axis_spacing(
        cls,
        structural_elements,
        member_classes,
        coordinate,
        support_classes=(),
    ):
        """Return the greatest generated clear station spacing in metres.

        Rule loads and maximum-spacing checks are governed by the widest
        generated interval.  Primary supports are included for ordinary
        members: a transverse bulkhead closes a web-frame bay, while both a
        bulkhead and a web frame close an ordinary-frame bay.  Likewise, deck
        and bottom girders close the tributary breadth of the intervening
        longitudinal stiffeners.  This prevents a hold or girder boundary
        from appearing as one artificial double-width gap.
        """

        target_positions = cls._generated_member_positions(
            structural_elements,
            member_classes,
            coordinate,
        )

        if len(target_positions) == 0:
            raise ValueError(
                'Generated structural elements contain no station for '
                f'{tuple(member_classes)}.'
            )

        positions = cls._generated_member_positions(
            structural_elements,
            tuple(member_classes)+tuple(support_classes),
            coordinate,
        )

        if len(positions) < 2:
            raise ValueError(
                'Generated structural elements do not contain two distinct '
                f'{coordinate} stations for {tuple(member_classes)}.'
            )

        positive_intervals = np.diff(positions)
        positive_intervals = positive_intervals[positive_intervals > 1.0e-9]

        if len(positive_intervals) == 0:
            raise ValueError(
                'Generated structural-element stations collapse to one '
                f'location for {tuple(member_classes)}.'
            )

        return float(np.max(positive_intervals))


    def _generated_structural_spacings(self):
        """Measure every repeated structural spacing from generated geometry.

        Member-count coordinates remain the repair/design controls.  Their
        only route into a constraint, however, is through the structural
        elements which those counts actually generated.  Consequently a
        snapped bulkhead-stiffener count or aligned deck/bottom layout is
        evaluated at its exact CadQuery spacing rather than at an analytical
        reconstruction of the requested count.
        """

        elements = self.Structural_Elements

        return {
            'Bulkhead_Spacing': self._generated_axis_spacing(
                elements,
                ('Transverse_Bulkhead',),
                'x_loc',
            ),
            'WebFrame_Spacing': self._generated_axis_spacing(
                elements,
                ('Vertical_Web_Frame',),
                'x_loc',
                support_classes=('Transverse_Bulkhead',),
            ),
            'Frame_Spacing': self._generated_axis_spacing(
                elements,
                ('Vertical_Side_Frame',),
                'x_loc',
                support_classes=(
                    'Vertical_Web_Frame',
                    'Transverse_Bulkhead',
                ),
            ),
            'Bottom_Girder_Spacing': self._generated_axis_spacing(
                elements,
                ('Bottom_Longitudinal_Girder',),
                'y_loc',
            ),
            'Bottom_Stringer_Spacing': self._generated_axis_spacing(
                elements,
                ('Bottom_Longitudinal_Stiffener',),
                'y_loc',
                support_classes=('Bottom_Longitudinal_Girder',),
            ),
            'Deck_Girder_Spacing': self._generated_axis_spacing(
                elements,
                ('Deck_Longitudinal_Girder',),
                'y_loc',
            ),
            'Deck_Stringer_Spacing': self._generated_axis_spacing(
                elements,
                ('Deck_Longitudinal_Stiffener',),
                'y_loc',
                support_classes=('Deck_Longitudinal_Girder',),
            ),
            'Side_Stringer_Spacing': self._generated_axis_spacing(
                elements,
                ('Side_Shell_Longitudinal_Stiffener',),
                'z_loc',
            ),
            'VertBulkhead_Stringer_Spacing': (
                self._generated_axis_spacing(
                    elements,
                    ('Transverse_Bulkhead_Vertical_Stiffener',),
                    'y_loc',
                )
            ),
            'HorizBulkhead_Stringer_Spacing': (
                self._generated_axis_spacing(
                    elements,
                    ('Transverse_Bulkhead_Transverse_Stiffener',),
                    'z_loc',
                )
            ),
        }


    def _generated_hopper_stiffener_spacings(self):
        """Return actual lower/upper hopper stiffener spacings in millimetres.

        Hopper stiffeners are stored under their physical panel IDs.  Project
        each generated stiffener location onto its generated sloping panel and
        include both generated panel edges when finding the widest bay.  This
        captures the panel-stiffener alignment snap in the CadQuery generator;
        dividing panel length by a requested or generated count would miss a
        nonuniform bay created by that snap.
        """

        elements = self.Structural_Elements
        stiffeners = elements.loc[
            elements['Class'].eq('Hopper_Pannel_Longitudinal_Stiffener')
        ]
        panels = elements.loc[elements['Class'].eq('Hopper Pannel')]
        panel_spacings = []

        for parent_id, members in stiffeners.groupby('Parent_Struct'):
            panel = panels.loc[panels['Object_ID'].eq(parent_id)].iloc[0]
            panel_length_m = float(panel['L2'])
            panel_rotation_rad = np.deg2rad(float(panel['rot']))
            panel_direction_yz = np.array([
                np.sin(panel_rotation_rad),
                np.cos(panel_rotation_rad),
            ])
            panel_start_yz = np.array([
                float(panel['y_loc']),
                float(panel['z_loc']),
            ])

            member_locations_yz = members[['y_loc', 'z_loc']].to_numpy(
                dtype=float
            )
            generated_distances_m = (
                (member_locations_yz-panel_start_yz)
                @ panel_direction_yz
            )
            bay_boundaries_m = np.unique(np.concatenate((
                np.array([0.0, panel_length_m]),
                generated_distances_m,
            )))
            spacing_m = float(np.max(np.diff(np.sort(bay_boundaries_m))))
            panel_spacings.append((float(panel['z_loc']), spacing_m))

        if len(panel_spacings) != 2:
            raise ValueError(
                'The active bulk-carrier geometry requires one lower and one '
                'upper hopper panel with generated longitudinal stiffeners.'
            )

        panel_spacings.sort(key=lambda item: item[0])

        return (
            1000.0*panel_spacings[0][1],
            1000.0*panel_spacings[1][1],
        )


    def _generated_webframe_properties(self, webframe_spacing_m):
        """Calculate the weakest actual generated web-frame section.

        This restores the validated legacy section model: one web, its actual
        generated face plate when fitted, and one effective breadth of attached
        side-shell plating equal to the generated web-frame spacing.  The old
        active calculation instead formed a synthetic two-sided box using an
        unrelated plate thickness and never read the generated flange.
        """

        elements = self.Structural_Elements
        webframes = elements.loc[
            elements['Class'].eq('Vertical_Web_Frame')
        ].sort_values(['x_loc', 'Object_ID'])
        side_shell_thickness_mm = 1000.0*float(
            elements.loc[
                elements['Class'].eq('Side_Shell'),
                'Thickness',
            ].min()
        )
        section_results = []

        for _, webframe in webframes.iterrows():
            flange = elements.loc[
                elements['Object_ID'].eq(
                    f"{webframe['Object_ID']}_Flange"
                )
            ]
            flange_width_mm = (
                1000.0*float(flange.iloc[0]['L2'])
                if not flange.empty
                else 0.0
            )
            web_height_mm = 1000.0*float(webframe['L2'])
            web_thickness_mm = 1000.0*float(webframe['Thickness'])
            section_results.append(
                self.calc_stiffener_SM(
                    side_shell_thickness_mm,
                    1000.0*float(webframe_spacing_m),
                    web_height_mm,
                    web_thickness_mm,
                    flange_width_mm,
                )
                + (web_height_mm, web_thickness_mm, flange_width_mm)
            )

        if not section_results:
            raise ValueError(
                'Generated structural elements contain no Vertical_Web_Frame '
                'member for the MVR web-frame assessment.'
            )

        # The common section reported to the Rule evaluator is the weakest
        # actually generated web-frame section.  Current parametric designs use
        # one common section, but taking the minimum remains correct if a later
        # geometry generator varies it by station.
        weakest = min(section_results, key=lambda result: result[0])

        return {
            'section_modulus_cm3': float(weakest[0]),
            'effective_flange': bool(weakest[1]),
            'aspect_ratio': float(weakest[2]),
            'height_mm': float(weakest[3]),
            'thickness_mm': float(weakest[4]),
            'flange_width_mm': float(weakest[5]),
        }


    def calc_Structure_Values(self): 

        # Hull Girder Calculations:
        SM_hull_girder, AreaMoment_hull_girder, I_22, z_centroid_hull_girder = self.calc_Hull_Girder_SM()
        # ABS MVR 3-2-1/9.5.1 requires section modulus to be calculated
        # separately to the molded deck-at-side and to the baseline.  Keep
        # the historical governing value, but also retain both physical
        # values so the two required comparisons can be emitted atomically.
        deck_distance_m = self.Hull_Dict['D'] - z_centroid_hull_girder
        bottom_distance_m = z_centroid_hull_girder
        deck_sm_cm3 = I_22/deck_distance_m*100.0**3.0
        bottom_sm_cm3 = I_22/bottom_distance_m*100.0**3.0

        item = pd.DataFrame({
            'Parameter': [
                'Hull_Girder_SM',
                'Hull_Girder_Deck_SM',
                'Hull_Girder_Bottom_SM',
                'Hull_Girder_AreaMoment',
                'Hull_Girder_First_Area_Moment',
                'Hull_Girder_I',
                'Hull_Girder_Z_Centroid',
                'Hull_Girder_Effective_Member_Count',
                'Hull_Girder_Excluded_Member_Count',
                'Hull_Girder_Hatch_Deduction_Count',
                'Hull_Girder_Coaming_Credit_Count',
                'Hull_Girder_Minor_Opening_Deduction_Count',
            ],
            'Value': [
                SM_hull_girder,
                deck_sm_cm3,
                bottom_sm_cm3,
                AreaMoment_hull_girder,
                self.Hull_Girder_First_Area_Moment,
                I_22,
                z_centroid_hull_girder,
                self.Hull_Girder_Effective_Member_Count,
                self.Hull_Girder_Excluded_Member_Count,
                self.Hull_Girder_Hatch_Deduction_Count,
                self.Hull_Girder_Coaming_Credit_Count,
                self.Hull_Girder_Minor_Opening_Deduction_Count,
            ],
            'Units': [
                'cm^3', 'cm^3', 'cm^3', 'cm^3', 'cm^3', 'm^4', 'm',
                'members', 'members', 'openings', 'members', 'openings',
            ]})
        self.Struct_Dict = item.copy()

        # Shear Strake Calculations:
        h = self.structure_params[14] # Shear Strake Height in mm
        t = self.structure_params[13] # Shear Strake Thickness in mm
        w = 0.0 # Shear Strake Flange Width in mm
        plate_t = 0.0 # Assume no shell plate contribution
        spacing = 0.0 # Assume no spacing contribution
        SM_shear_strake, bit_shear_strake_flange, AR_shear_strake = self.calc_stiffener_SM(plate_t, spacing, h,t,w)

        item = pd.DataFrame({
            'Parameter': ['Shear_Strake_SM', 'Shear_Strake_Aspect_Ratio', 'Shear_Strake_Flange'], 
            'Value': [SM_shear_strake, AR_shear_strake, bit_shear_strake_flange], 
            'Units': ['cm^3', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

                                     
        # Extract Shell Thicknesses:
        item = pd.DataFrame({
            'Parameter': ['Bottom_Shell_Thickness', 'Side_Shell_Thickness', 'Top_Deck_Thickness', 'Inner_Bottom_Thickness'], 
            'Value': self.structure_params[7:11],
            'Units': ['mm', 'mm', 'mm', 'mm']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Extract every repeated-member spacing from the generated CadQuery
        # element table.  Parameter counts remain the design controls, but the
        # geometry they actually produced is the sole spacing authority used
        # by section calculations and Rule comparisons.
        generated_spacings = self._generated_structural_spacings()
        s_bulkheads = generated_spacings['Bulkhead_Spacing']
        s_webframes = generated_spacings['WebFrame_Spacing']
        s_frames = generated_spacings['Frame_Spacing']
        s_bottom_girders = generated_spacings['Bottom_Girder_Spacing']
        s_bottom_stringers = generated_spacings['Bottom_Stringer_Spacing']
        s_deck_girders = generated_spacings['Deck_Girder_Spacing']
        s_deck_stringers = generated_spacings['Deck_Stringer_Spacing']
        s_side_stringers = generated_spacings['Side_Stringer_Spacing']
        s_vertbulkhead_stringers = (
            generated_spacings['VertBulkhead_Stringer_Spacing']
        )
        s_horizbulkhead_stringers = (
            generated_spacings['HorizBulkhead_Stringer_Spacing']
        )

        # The hatch geometry below needs the actual generated deck-girder
        # locations as well as their spacing.
        y_deck_girders = self._generated_member_positions(
            self.Structural_Elements,
            ('Deck_Longitudinal_Girder',),
            'y_loc',
        )

        item = pd.DataFrame({
            'Parameter': ['Bulkhead_Spacing', 'WebFrame_Spacing', 'Frame_Spacing', 'Bottom_Girder_Spacing', 'Bottom_Stringer_Spacing', 'Deck_Girder_Spacing', 'Deck_Stringer_Spacing', 'Side_Stringer_Spacing', 'VertBulkhead_Stringer_Spacing', 'HorizBulkhead_Stringer_Spacing'], 
            'Value': [s_bulkheads, s_webframes, s_frames, s_bottom_girders, s_bottom_stringers, s_deck_girders, s_deck_stringers, s_side_stringers, s_vertbulkhead_stringers, s_horizbulkhead_stringers],
            'Units': ['m', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Bottom Structure:
        
        #Bottom Girder needs special treatment because it spans between bottom and inner bottom
        plate_t = self.structure_params[7] # Bottom Shell Thickness in mm
        spacing = s_bottom_girders*1000.0 # in mm
        h = self.structure_params[6] #double Bottom Height in mm
        h_double_bottom = h
        t = self.structure_params[23] # Bottom Girder plate Thickness in mm
        t_bottom_girder = t
        w = spacing
        t_w = self.structure_params[10] # Inner Bottom Thickness in mm

        Area = t_w*w + h*t + plate_t*spacing # in mm^2
        Z_centroid = (0*plate_t*spacing + t_w*w*(h) + h*t*(h/2.0))/Area # in mm
        I = (1/12*spacing* plate_t**3.0 + spacing*plate_t*(Z_centroid)**2.0 + 1/12*w*t_w**3.0 + w*t_w*(Z_centroid - h)**2.0 + 1/12*t*h**3.0 + h*t*(h/2.0 - Z_centroid)**2.0) # in mm^4
        z = max(Z_centroid, h - Z_centroid) # in mm
        SM_bottom_girders = (I/z)/1000.0 # in cm^3
        bit_bottom_girder_flange = True
        AR_bottom_girder = h/t # Aspect Ratio of the girder web

        # Bottom Floors similar: 
        t = self.structure_params[52] # Bottom floor plate Thickness in mm
        t_bottom_floor = t
        plate_t = self.structure_params[7] # Bottom Shell Thickness in mm
        spacing = s_webframes*1000.0 # in mm
        w = spacing
        t_w = self.structure_params[10] # Inner Bottom Thickness in mm

        Area = t_w*w + h*t + plate_t*spacing # in mm^2
        Z_centroid = (0*plate_t*spacing + t_w*w*(h) + h*t*(h/2.0))/Area # in mm
        I = (1/12*spacing* plate_t**3.0 + spacing*plate_t*(Z_centroid)**2.0 + 1/12*w*t_w**3.0 + w*t_w*(Z_centroid - h)**2.0 + 1/12*t*h**3.0 + h*t*(h/2.0 - Z_centroid)**2.0) # in mm^4
        z = max(Z_centroid, h - Z_centroid) # in mm
        SM_bottom_floors = (I/z)/1000.0 # in cm^3
        bit_bottom_floor_flange = True
        AR_bottom_floor = h/t # Aspect Ratio of the girder web

        # Bottom Stringers:
        plate_t = self.structure_params[7] # Bottom Shell Thickness in mm
        spacing = s_bottom_stringers*1000.0 # in mm
        h = self.structure_params[24] # Bottom Stringer Height in mm
        t = self.structure_params[25] # Bottom Stringer thickness in mm
        w = self.structure_params[26] # Bottom Stringer Flange Width in mm
        SM_bottom_stiffeners, bit_bottom_stiff_flange, AR_bottom_stiffener = self.calc_stiffener_SM(plate_t, spacing, h,t,w)


        #Inner Bottom Stringers:
        plate_t = self.structure_params[10] # Inner Bottom Thickness in mm
        SM_inner_bottom_stiffeners, bit_inner_bottom_stiff_flange, AR_inner_bottom_stiffener = self.calc_stiffener_SM(plate_t, spacing, h, t, w)  # Assume same as bottom stiffeners, different plate thickness

        #Bottom frames:
        plate_t = self.structure_params[7] # Bottom Shell Thickness in mm
        spacing = s_frames*1000.0 # in mm
        h = self.structure_params[93] # Bottom Frame Height in mm
        t = self.structure_params[94] # Bottom Frame Thickness in mm
        w = self.structure_params[95] # Bottom Frame Flange Width in mm
        SM_bottom_frames, bit_bottom_frame_flange, AR_bottom_frame = self.calc_stiffener_SM(plate_t, spacing, h,t,w)


        # inner bottom frames:
        plate_t = self.structure_params[10] # Inner Bottom Thickness in mm
        SM_inner_bottom_frames, bit_inner_bottom_frame_flange, AR_inner_bottom = self.calc_stiffener_SM(plate_t, spacing, h,t,w) # Assume same as bottom frames, different plate thickness

        #Combine all into an item and add to Struct_Dict
        item = pd.DataFrame({
            'Parameter': ['Bottom_Girder_SM', 'Bottom_Floor_SM', 'Bottom_Stringer_SM', 'Inner_Bottom_Stringer_SM', 'Bottom_Frame_SM', 'Inner_Bottom_Frame_SM'], 
            'Value': [SM_bottom_girders, SM_bottom_floors, SM_bottom_stiffeners, SM_inner_bottom_stiffeners, SM_bottom_frames, SM_inner_bottom_frames],
            'Units': ['cm^3', 'cm^3', 'cm^3', 'cm^3', 'cm^3', 'cm^3']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Double_Bottom_Height', 'Bottom_Girder_Plate_Thickness', 'Bottom_Floor_Plate_Thickness'], 
            'Value': [h_double_bottom, t_bottom_girder, t_bottom_floor],
            'Units': ['mm', 'mm', 'mm']})
        
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Bottom_Girder_Flange', 'Bottom_Floor_Flange', 'Bottom_Stringer_Flange', 'Inner_Bottom_Stringer_Flange', 'Bottom_Frame_Flange', 'Inner_Bottom_Frame_Flange'],
            'Value': [bit_bottom_girder_flange, bit_bottom_floor_flange, bit_bottom_stiff_flange, bit_inner_bottom_stiff_flange, bit_bottom_frame_flange, bit_inner_bottom_frame_flange],
            'Units': ['-', '-', '-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Bottom_Girder_Aspect_Ratio', 'Bottom_Floor_Aspect_Ratio', 'Bottom_Stringer_Aspect_Ratio', 'Inner_Bottom_Stringer_Aspect_Ratio', 'Bottom_Frame_Aspect_Ratio', 'Inner_Bottom_Frame_Aspect_Ratio'],
            'Value': [AR_bottom_girder, AR_bottom_floor, AR_bottom_stiffener, AR_inner_bottom_stiffener, AR_bottom_frame, AR_inner_bottom],
            'Units': ['-', '-', '-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        
    
        # Frames: 
        l_frames = self.structure_params[3] - (self.structure_params[6] + self.structure_params[61] + self.structure_params[64])/1000.0 # Height of the frames from bottom of double bottom to top deck in m
        h_frames = np.abs(self.structure_params[6]/1000.0 + l_frames/2.0 - self.structure_params[2]) #distance from mid-height of frame to draft in m

        h = self.structure_params[97] # Frame Height in mm
        t = self.structure_params[98] # Frame Thickness in mm
        w = self.structure_params[99] # Frame Flange Width in mm
        plate_t = self.structure_params[8] # Side Shell Thickness in mm

        SM_side, bit_side_frame_flange, AR_side_frame = self.calc_stiffener_SM(plate_t, s_frames*1000.0, h,t,w)

        item = pd.DataFrame({
            'Parameter': ['Side_Frame_SM', 'l_frames', 'h_frames'], 
            'Value': [SM_side, l_frames, h_frames],
            'Units': ['cm^3', 'm', 'm']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Frame_Flange', 'Frame_Aspect_Ratio'],
            'Value': [bit_side_frame_flange, AR_side_frame],
            'Units': ['-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Web Frames and Side Stringers
        #
        # Evaluate the exact generated web and its generated face plate with
        # the validated legacy attached-shell convention.  Do not reconstruct
        # a synthetic two-flange box from unrelated parameter slots.
        webframe_properties = self._generated_webframe_properties(
            s_webframes
        )
        SM_WebFrame = webframe_properties['section_modulus_cm3']
        bit_webframe_flange = webframe_properties['effective_flange']
        AR_webframe = webframe_properties['aspect_ratio']
        h_webframe = webframe_properties['height_mm']
        t_webframe = webframe_properties['thickness_mm']

        #Side Stringers:
        plate_t = self.structure_params[8] # Side Shell Thickness in mm
        spacing = s_side_stringers*1000.0 # in mm
        h = self.structure_params[53] # Side Stringer Height in mm
        h_side_stringer = h
        t = self.structure_params[54] # Side Stringer Thickness in mm
        w = self.structure_params[55] # Side Stringer Flange Width in mm

        SM_side_stiffeners, bit_side_stiffener_flange, AR_side_stiffener = self.calc_stiffener_SM(plate_t, spacing, h,t,w)

        item = pd.DataFrame({
            'Parameter': ['WebFrame_SM', 'WebFrame_Thickness', 'WebFrame_Height', 'Side_Stringer_SM', 'Side_Stringer_Height'], 
            'Value': [SM_WebFrame, t_webframe, h_webframe, SM_side_stiffeners, h_side_stringer],
            'Units': ['cm^3', 'mm', 'mm', 'cm^3', 'mm']})
        
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['WebFrame_Flange', 'WebFrame_Aspect_Ratio', 'Side_Stringer_Flange', 'Side_Stringer_Aspect_Ratio'],
            'Value': [bit_webframe_flange, AR_webframe, bit_side_stiffener_flange, AR_side_stiffener],  
            'Units': ['-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Beams and Girders

        # General Beam calc needs to be performed for all longitudinal and transverse stiffeners: ABS 

        # Deck Girders and Beams
        spacing = s_deck_stringers*1000.0 # in mm
        plate_t = self.structure_params[9] #deck Shell Thickness in mm
        h = self.structure_params[32] # Deck Girder Height in mm
        t = self.structure_params[33] # Deck Girder plate Thickness in mm
        t_deck_girder = t
        h_deck_girder = h
        w = self.structure_params[34] # Deck Girder Flange Width in mm
        SM_deck_girders, bit_deck_girder_flange, AR_deck_girder = self.calc_stiffener_SM(plate_t, spacing, h,t,w)


        # Deck Beams
        spacing = s_frames*1000.0 # in mm
        plate_t = self.structure_params[9] #deck Shell Thickness in mm
        h = self.structure_params[28] # Deck Beam Height in mm
        t = self.structure_params[29] # Deck Beam plate Thickness in mm
        t_deck_beam = t
        h_deck_beam = h
        w = self.structure_params[30] # Deck Beam Flange Width in mm
        SM_deck_beams, bit_deck_beam_flange, AR_deck_beam = self.calc_stiffener_SM(plate_t, spacing, h,t,w)
        item = pd.DataFrame({
            'Parameter': ['Deck_Girder_SM', 'Deck_Girder_Plate_Thickness', 'Deck_Girder_Height', 'Deck_Beam_SM', 'Deck_Beam_Height', 'Deck_Beam_Plate_Thickness'], 
            'Value': [SM_deck_girders, t_deck_girder, h_deck_girder, SM_deck_beams, h_deck_beam, t_deck_beam],
            'Units': ['cm^3', 'mm', 'mm', 'cm^3', 'mm', 'mm']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)   

        item = pd.DataFrame({
            'Parameter': ['Deck_Girder_Flange', 'Deck_Girder_Aspect_Ratio', 'Deck_Beam_Flange', 'Deck_Beam_Aspect_Ratio'],
            'Value': [bit_deck_girder_flange, AR_deck_girder, bit_deck_beam_flange, AR_deck_beam],
            'Units': ['-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Deck Stringers
        plate_t = self.structure_params[9] #deck Shell Thickness in mm
        spacing = s_deck_stringers*1000.0 # in mm
        h = self.structure_params[36] # Deck Stringer Height in mm
        t = self.structure_params[37] # Deck Stringer  Thickness in mm
        w = self.structure_params[38] # Deck Stringer Flange Width in mm
        SM_deck_stiffeners, bit_deck_stringer_flange, AR_deck_stringer = self.calc_stiffener_SM(plate_t, spacing, h,t,w)

        #intermediate deck beams
        spacing = s_frames*1000.0 # in mm
        plate_t = self.structure_params[9] #deck Shell Thickness in mm
        h = self.structure_params[101] # Intermediate Deck Beam Height in mm
        t = self.structure_params[102] # Intermediate Deck Beam Thickness in mm
        w = self.structure_params[103] # Intermediate Deck Beam Flange Width in mm
        SM_intermediate_deck_beams, bit_deck_frame_flange, AR_deck_frame = self.calc_stiffener_SM(plate_t, spacing, h,t,w)    

        item = pd.DataFrame({
            'Parameter': ['Deck_Stringer_SM', 'Intermediate_Deck_Beam_SM'], 
            'Value': [SM_deck_stiffeners, SM_intermediate_deck_beams],
            'Units': ['cm^3', 'cm^3']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Deck_Stringer_Flange', 'Deck_Stringer_Aspect_Ratio', 'Intermediate_Deck_Beam_Flange', 'Intermediate_Deck_Beam_Aspect_Ratio'],
            'Value': [bit_deck_stringer_flange, AR_deck_stringer, bit_deck_frame_flange, AR_deck_frame],
            'Units': ['-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Hatch Opening Reinforcements
        Hatch_L = self.structure_params[117]*self.structure_params[0] # Hatch Length in m
        Hatch_W = self.structure_params[118]*self.structure_params[1] # Hatch Width in m
        Hatch_R = self.structure_params[119] # Hatch Corner Radius in mm

        # The hatch dimensions are the outside dimensions of a rectangle
        # with four rounded corners.  Each corner removes one square-minus-
        # quarter-circle area, so the total deduction is ``(4-pi) r^2``.
        # Use the same helper as the hatch-cover mass model so weight and
        # structural opening deductions cannot silently disagree.
        Hatch_Area = rounded_rectangle_area_m2(
            Hatch_L,
            Hatch_W,
            Hatch_R/1000.0,
        )
        #Need get Deck Girder Positions 
        #First is the first deck girder outboard of the hatch: 
        y_deck_girder_edge = y_deck_girders[y_deck_girders >= Hatch_W/2.0][0]
        y_deck_girder_split = y_deck_girders[y_deck_girders < Hatch_W/2.0][-1]
        Hatch_F = 0.5*(self.structure_params[1] - y_deck_girder_split) #Per ABS ABS Part 3 Ch 2 Sec 8 / 7.1 - Hatch Support Area in m, distance from centerline to first deck girder inboard of hatch

        Hatch_B = y_deck_girder_edge + Hatch_F/2.0 
        Hatch_C = y_deck_girder_edge- y_deck_girder_split + Hatch_F/2.0
        Hatch_D = (s_bulkheads - Hatch_L)/2.0

        item = pd.DataFrame({
            'Parameter': ['Hatch_Length', 'Hatch_Width', 'Hatch_Corner_Radius', 'Hatch_Area', 'Hatch_F', 'Hatch_B', 'Hatch_C', 'Hatch_D'], 
            'Value': [Hatch_L, Hatch_W, Hatch_R, Hatch_Area, Hatch_F, Hatch_B, Hatch_C, Hatch_D],
            'Units': ['m', 'm', 'mm', 'm^2', 'm', 'm', 'm', 'm']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Bulkhead
        # Calculate bulkhead radius of gyration and moment of inertia
        radius_gyration_bulkhead, cross_sectional_area = self.calc_Bulkhead_gyration_radius() # in m
        
        t_bulkhead = self.structure_params[11] # Bulkhead Thickness in mm
        Area_bulkhead = self.structure_params[3]*self.structure_params[1] - 0.5*np.pi*(self.structure_params[4])**2.0 # in m^2, Area of the bulkhead 

        item = pd.DataFrame({
            'Parameter': ['Bulkhead_Radius_of_Gyration', 'Bulkhead_Structure_Cross_Sectional_Area', 'Bulkhead_Thickness', 'Bulkhead_Area'], 
            'Value': [radius_gyration_bulkhead, cross_sectional_area, t_bulkhead, Area_bulkhead],
            'Units': ['m', 'cm^2', 'mm', 'm^2']})
        
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        spacing = s_vertbulkhead_stringers*1000.0 # in mm
        plate_t = self.structure_params[11] # Bulkhead Thickness in mm
        h = self.structure_params[44] # Vertical Bulkhead Stringer Height in mm
        t = self.structure_params[45] # Vertical Bulkhead Stringer Thickness in mm
        w = self.structure_params[46] # Vertical Bulkhead Stringer Flange Width in

        SM_vertbulkhead_stiffeners, bit_vertbulkhead_stiffener_flange, AR_vertbulkhead_stiffener = self.calc_stiffener_SM(plate_t, spacing, h,t,w)

        spacing = s_horizbulkhead_stringers*1000.0 # in mm
        h = self.structure_params[40] # Horizontal Bulkhead Stringer Height in mm
        t = self.structure_params[41] # Horizontal Bulkhead Stringer Thickness in mm
        w = self.structure_params[42] # Horizontal Bulkhead Stringer Flange Width in
        SM_horizbulkhead_stiffeners, bit_horizbulkhead_stiffener_flange, AR_horizbulkhead_stiffener = self.calc_stiffener_SM(plate_t, spacing, h,t,w)

        item = pd.DataFrame({
            'Parameter': ['Vertical_Bulkhead_Stringer_SM', 'Horizontal_Bulkhead_Stringer_SM'], 
            'Value': [SM_vertbulkhead_stiffeners, SM_horizbulkhead_stiffeners],
            'Units': ['cm^3', 'cm^3']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Vertical_Bulkhead_Stringer_Flange', 'Vertical_Bulkhead_Stringer_Aspect_Ratio', 'Horizontal_Bulkhead_Stringer_Flange', 'Horizontal_Bulkhead_Stringer_Aspect_Ratio'],
            'Value': [bit_vertbulkhead_stiffener_flange, AR_vertbulkhead_stiffener, bit_horizbulkhead_stiffener_flange, AR_horizbulkhead_stiffener],
            'Units': ['-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        # Hopper and Bracket Calcs
        angle_lower_hopper_bracket = np.arctan2(self.structure_params[61], self.structure_params[60])*(180.0/np.pi) # in degrees
        t_lower_hopper_bracket = self.structure_params[62] # Lower Hopper Shell Thickness in mm
        L_lower_hopper_bracket = np.sqrt(self.structure_params[61]**2 + self.structure_params[60]**2) # Length of lower hopper bracket in mm
        

        angle_upper_hopper_bracket = np.arctan2(self.structure_params[64], self.structure_params[63])*(180.0/np.pi) # in degrees
        t_upper_hopper_bracket = self.structure_params[65] # Upper Hopper Shell Thickness in mm
        L_upper_hopper_bracket = np.sqrt(self.structure_params[64]**2 + self.structure_params[63]**2) # Length of upper hopper bracket in mm

        t_plate_lower_hopper = self.structure_params[80] # Hopper Shell Thickness in mm
        t_plate_upper_hopper = self.structure_params[81] # Hopper Shell Thickness in mm

        # Measure the two sloping-panel spacings from the generated panels and
        # their generated member counts.  This captures any count snapping or
        # geometry cleanup before a constraint consumes the value.
        (
            s_lower_hopper_stiffeners,
            s_upper_hopper_stiffeners,
        ) = self._generated_hopper_stiffener_spacings()

        h = self.structure_params[83] # Lower Hopper Stiffener Height in mm
        t = self.structure_params[84] # Lower Hopper Stiffener Thickness in mm
        w = self.structure_params[85] # Lower Hopper Stiffener Flange Width in
        SM_lower_hopper_stiffeners, bit_lower_hopper_stiff_flange, AR_lower_hopper_stiff = self.calc_stiffener_SM(t_plate_lower_hopper, s_lower_hopper_stiffeners, h,t,w)

        h = self.structure_params[88] # Upper Hopper Stiffener Height in mm
        t = self.structure_params[89] # Upper Hopper Stiffener Thickness in mm
        w = self.structure_params[90] # Upper Hopper Stiffener Flange Width in
        SM_upper_hopper_stiffeners, bit_upper_hopper_stiff_flange, AR_upper_hopper_stiff = self.calc_stiffener_SM(t_plate_upper_hopper, s_upper_hopper_stiffeners, h,t,w)

        item = pd.DataFrame({
            'Parameter': ['Lower_Hopper_Bracket_Angle', 'Lower_Hopper_Bracket_Thickness', 'Upper_Hopper_Bracket_Angle', 'Upper_Hopper_Bracket_Thickness', 'Lower_Hopper_Plate_Thickness', 'Upper_Hopper_Plate_Thickness', 'Lower_Hopper_Stiffener_Spacing', 'Lower_Hopper_Stiffener_SM', 'Upper_Hopper_Stiffener_Spacing', 'Upper_Hopper_Stiffener_SM'], 
            'Value': [angle_lower_hopper_bracket, t_lower_hopper_bracket, angle_upper_hopper_bracket, t_upper_hopper_bracket, t_plate_lower_hopper, t_plate_upper_hopper, s_lower_hopper_stiffeners, SM_lower_hopper_stiffeners, s_upper_hopper_stiffeners, SM_upper_hopper_stiffeners],
            'Units': ['degrees', 'mm', 'degrees', 'mm', 'mm', 'mm', 'mm', 'cm^3', 'mm', 'cm^3']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)

        item = pd.DataFrame({
            'Parameter': ['Lower_Hopper_Stiffener_Flange', 'Lower_Hopper_Stiffener_Aspect_Ratio', 'Upper_Hopper_Stiffener_Flange', 'Upper_Hopper_Stiffener_Aspect_Ratio'],
            'Value': [bit_lower_hopper_stiff_flange, AR_lower_hopper_stiff, bit_upper_hopper_stiff_flange, AR_upper_hopper_stiff],
            'Units': ['-', '-', '-', '-']})
        self.Struct_Dict = pd.concat([self.Struct_Dict, item], ignore_index=True)


    @staticmethod
    def _rectangular_member_first_moments(member, neutral_axis_m):
        """Return exact upper/lower first moments for one plate rectangle.

        Every effective longitudinal entry in the generated element table is
        stored as one rectangular cross-section component; a stiffener web
        and its flange are separate rows.  The component may be rotated in
        the transverse plane, so replacing it by an axis-aligned rectangle
        recovered from ``sqrt(12*I/A)`` is only approximate.  Clip its actual
        four-corner polygon at the horizontal neutral axis instead.

        The returned values are for the positive-y half model and have units
        of m^3.  ``calc_Hull_Girder_SM`` mirrors them across centerline after
        summing all components.
        """

        length_m = float(member['L2'])
        thickness_m = float(member['Thickness'])
        centroid_z_m = float(member['centroid_cx_11'])
        rotation_rad = np.deg2rad(float(member['rot']))

        if length_m <= 0.0 or thickness_m <= 0.0:
            return 0.0, 0.0

        # ``cad_query_tools.properties`` uses these same sine/cosine
        # projections for I_22.  Local u follows the plate breadth and local
        # v follows its thickness; this rotation preserves cross-section area.
        half_length_m = 0.5*length_m
        half_thickness_m = 0.5*thickness_m
        section_polygon = []

        for u_m, v_m in [
            (-half_length_m, -half_thickness_m),
            (half_length_m, -half_thickness_m),
            (half_length_m, half_thickness_m),
            (-half_length_m, half_thickness_m),
        ]:
            y_m = (
                u_m*np.sin(rotation_rad)
                - v_m*np.cos(rotation_rad)
            )
            z_m = (
                centroid_z_m
                + u_m*np.cos(rotation_rad)
                + v_m*np.sin(rotation_rad)
            )
            section_polygon.append((y_m, z_m))

        def clip_at_neutral_axis(polygon, keep_upper):
            """Clip one convex polygon to the requested NA half-plane."""

            clipped = []

            for start, end in zip(polygon, polygon[1:]+polygon[:1]):
                start_inside = (
                    start[1] >= neutral_axis_m
                    if keep_upper
                    else start[1] <= neutral_axis_m
                )
                end_inside = (
                    end[1] >= neutral_axis_m
                    if keep_upper
                    else end[1] <= neutral_axis_m
                )

                if start_inside != end_inside:
                    fraction = (
                        (neutral_axis_m-start[1])
                        / (end[1]-start[1])
                    )
                    clipped.append((
                        start[0]+fraction*(end[0]-start[0]),
                        neutral_axis_m,
                    ))

                if end_inside:
                    clipped.append(end)

            return clipped

        def polygon_first_moment(polygon):
            """Integrate ``A*|z-z_NA|`` from polygon area and centroid."""

            if len(polygon) < 3:
                return 0.0

            twice_signed_area = 0.0
            centroid_z_numerator = 0.0

            for start, end in zip(polygon, polygon[1:]+polygon[:1]):
                cross = start[0]*end[1] - end[0]*start[1]
                twice_signed_area += cross
                centroid_z_numerator += (start[1]+end[1])*cross

            signed_area_m2 = 0.5*twice_signed_area

            if abs(signed_area_m2) <= np.finfo(float).eps:
                return 0.0

            centroid_z_m = (
                centroid_z_numerator/(6.0*signed_area_m2)
            )

            return (
                abs(signed_area_m2)
                * abs(centroid_z_m-neutral_axis_m)
            )

        upper_polygon = clip_at_neutral_axis(
            section_polygon,
            keep_upper=True,
        )
        lower_polygon = clip_at_neutral_axis(
            section_polygon,
            keep_upper=False,
        )

        return (
            polygon_first_moment(upper_polygon),
            polygon_first_moment(lower_polygon),
        )

    def _mvr_three_hold_effective_member_table(self, modeled_length_m):
        """Build the approved gross MVR effective-section member table.

        GenISys uses the generated three-hold structural segment as its
        representative hull-girder section.  A member contributes only when:

        * its class is explicitly mapped as longitudinally effective; and
        * its x-directed span is continuous over the complete three-hold
          model length.

        A flange inherits the eligibility of its web through the generator's
        ``<web Object_ID>_Flange`` naming contract.  This prevents the former
        ``x_dir == 1`` shortcut from accidentally admitting a longitudinally
        oriented component belonging to an excluded structural system.

        Thicknesses are the gross, as-generated MVR scantlings.  CSR net
        scantlings and corrosion deductions are outside the GenISys scope
        under ASM-016/SCOPE-005.
        """

        assessed = self.Structural_Elements.copy()

        # Record the decision for every generated structural row.  Keeping
        # the excluded rows makes the rule interpretation inspectable during
        # dataset QA instead of hiding it inside a dataframe filter.
        assessed['MVR_Bending_Effective_Class'] = assessed['Class'].isin(
            MVR_BENDING_EFFECTIVE_MEMBER_CLASSES
        )
        assessed['MVR_Continuity_Status'] = 'not_continuous'

        is_x_directed = np.isclose(
            pd.to_numeric(assessed['x_dir'], errors='coerce'),
            1.0,
        )
        spans_three_holds = np.isclose(
            pd.to_numeric(assessed['L1'], errors='coerce'),
            float(modeled_length_m),
        )
        continuous_base = (
            assessed['MVR_Bending_Effective_Class']
            & is_x_directed
            & spans_three_holds
        )
        assessed.loc[continuous_base, 'MVR_Continuity_Status'] = (
            'continuous_over_three_hold_model'
        )

        # Stiffener flanges carry the same longitudinal stress as an eligible
        # continuous web.  The structural-elements schema names each flange
        # by appending ``_Flange`` to its web Object_ID.
        eligible_web_ids = set(
            assessed.loc[continuous_base, 'Object_ID'].astype(str)
        )
        flange_parent_ids = (
            assessed['Object_ID']
            .astype(str)
            .str.replace(r'_Flange$', '', regex=True)
        )
        continuous_flange = (
            assessed['Class'].eq('Stiffener_Flange')
            & flange_parent_ids.isin(eligible_web_ids)
            & is_x_directed
            & spans_three_holds
        )
        assessed.loc[continuous_flange, 'MVR_Bending_Effective_Class'] = True
        assessed.loc[continuous_flange, 'MVR_Continuity_Status'] = (
            'inherits_continuity_from_effective_web'
        )

        assessed['MVR_Bending_Effective'] = continuous_base | continuous_flange
        assessed['MVR_Effective_Area_Factor'] = np.where(
            assessed['MVR_Bending_Effective'],
            1.0,
            0.0,
        )
        assessed['MVR_Thickness_Basis'] = 'gross_as_generated'
        assessed['MVR_Rule_Section_ID'] = (
            'three_hold_representative_maximum_hatch_section'
        )
        assessed['MVR_Opening_Deduction_Breadth_m'] = 0.0

        # The CAD table stores the starboard half of the structure.  Most
        # rows therefore represent a mirrored pair, while a member whose
        # complete cross-section lies on y=0 is physically present only once.
        # This matters when the optional one- or three-bulkhead arrangement
        # includes an actual centerline longitudinal bulkhead.
        assessed['MVR_Mirror_Factor'] = np.where(
            np.isclose(
                pd.to_numeric(
                    assessed['centroid_cx_22'],
                    errors='coerce',
                ),
                0.0,
                atol=1.0e-9,
            ),
            1.0,
            2.0,
        )

        effective = assessed.loc[
            assessed['MVR_Bending_Effective']
        ].copy()

        # ABS MVR 3-2-1/9.3 and /9.5.1 opening treatment: deduct each modeled
        # principal hatch opening from the deck plate.  The generated table is
        # a starboard half-model, so one-half of the full hatch breadth is
        # deducted here and the completed section is mirrored below.
        hatch_deduction_count = 0
        deck_rows = effective.loc[effective['Class'].eq('Deck')]

        for deck_index, deck in deck_rows.iterrows():
            if pd.isna(deck['Lightening_Holes']):
                continue

            openings = ast.literal_eval(deck['Lightening_Holes'])

            # The opening list contains the successive hold hatches along x.
            # A transverse section cuts only one of them, so deduct the first
            # (representative) hatch breadth once rather than summing openings
            # that occur at different longitudinal stations.
            opening = openings[0]
            half_hatch_breadth_m = float(opening['h'])/(2.0*1000.0)
            effective_width_m = float(deck['L2']) - half_hatch_breadth_m

            effective.loc[deck_index, 'L2'] = effective_width_m
            effective.loc[
                deck_index,
                'MVR_Opening_Deduction_Breadth_m',
            ] = half_hatch_breadth_m

            effective_y_centroid_m = (
                float(deck['L2']) - effective_width_m/2.0
            )
            thickness_m = float(deck['Thickness'])

            effective.loc[deck_index, 'centroid_cx_22'] = (
                effective_y_centroid_m
            )
            effective.loc[deck_index, 'I_11'] = (
                thickness_m*effective_width_m**3.0/12.0
            )
            effective.loc[deck_index, 'I_22'] = (
                thickness_m**3.0*effective_width_m/12.0
            )
            effective.loc[deck_index, 'area_cx'] = (
                thickness_m*effective_width_m
            )

            hatch_deduction_count += 1

        self.Hull_Girder_Member_Assessment = assessed
        self.Hull_Girder_Effective_Members = effective
        self.Hull_Girder_Section_ID = (
            'three_hold_representative_maximum_hatch_section'
        )
        self.Hull_Girder_Thickness_Basis = 'gross_mvr'
        self.Hull_Girder_Effective_Member_Count = int(len(effective))
        self.Hull_Girder_Excluded_Member_Count = int(
            len(assessed) - len(effective)
        )
        self.Hull_Girder_Hatch_Deduction_Count = hatch_deduction_count

        # USER-029 explicitly denies coaming credit and assumes no minor
        # openings.  These zero counts expose those project assumptions in the
        # calculated dataset.
        self.Hull_Girder_Coaming_Credit_Count = 0
        self.Hull_Girder_Minor_Opening_Deduction_Count = 0

        return effective

    def calc_Hull_Girder_SM(self): 
        """
        Calculate gross section properties for the representative 3-hold model.

        The section includes only the explicitly mapped MVR-effective member
        classes that are continuous over the generated three-hold structure.
        Principal hatch breadth is deducted from the deck.  No coaming credit
        is taken and no unmodeled minor openings are assumed.
        """

        # Get the modeled three-hold length from its reference side-shell row.
        L = self.Structural_Elements.loc[self.Structural_Elements['Object_ID'] == 'Side_Shell', 'L1'].values[0] # m

        df = self._mvr_three_hold_effective_member_table(L)


        mirror_factor = df['MVR_Mirror_Factor'].astype(float)
        A_Cx = (mirror_factor*df['area_cx']).sum()  # m^2, full ship section
        z_centroid = (
            mirror_factor*df['area_cx']*df['centroid_cx_11']
        ).sum()/A_Cx  # m
        y_centroid = 0.0  # Assuming symmetry in y-direction
        I_11 = (
            mirror_factor
            * (
                df['I_11']
                + df['centroid_cx_22']**2.0*df['area_cx']
            )
        ).sum()  # m^4 about the full-section centerline
        I_22 = (
            mirror_factor
            * (
                df['I_22']
                + df['area_cx']*(df['centroid_cx_11']-z_centroid)**2.0
            )
        ).sum()  # m^4 about the full-section horizontal neutral axis

        # ABS MVR 3-2-1/3.9.2 uses the *true first moment of area* on one
        # side of the neutral axis.  The legacy ``A*z`` value below is a
        # whole-section envelope and is not that quantity.  Clip each actual
        # rotated rectangular component at the neutral axis so sloping hopper
        # plates and flanges are integrated without an axis-aligned surrogate.
        upper_first_moment_half_m3 = 0.0
        lower_first_moment_half_m3 = 0.0

        for _, member in df.iterrows():
            upper_member_m3, lower_member_m3 = (
                self._rectangular_member_first_moments(
                    member,
                    z_centroid,
                )
            )
            member_factor = float(member['MVR_Mirror_Factor'])
            upper_first_moment_half_m3 += member_factor*upper_member_m3
            lower_first_moment_half_m3 += member_factor*lower_member_m3

        self.Hull_Girder_First_Area_Moment = (
            max(
                upper_first_moment_half_m3,
                lower_first_moment_half_m3,
            )
            * 100.0**3.0
        )

        z = max(z_centroid, self.Hull_Dict['D'] - z_centroid)  # Distance from neutral axis to extreme fiber
        SM_hull_girder = (I_22/z)*(100.0**3)  # Section Modulus in cm^3
        AreaMoment_hull_girder = (A_Cx*z)*(100.0**3)  # Area Moment in cm^3

        

        return SM_hull_girder, AreaMoment_hull_girder, I_22, z_centroid  # cm^3, cm^3, m^4, m
    
    def calc_stiffener_SM(self,plate_t, spacing, h,t,w):

        '''
        This function calculates the section modulus of a stiffener based on its geometry and spacing
        '''

        Area = t*w + h*t + plate_t*spacing # in mm^2
        Z_centroid = (0 + t*w*(h) + h*t*(h/2.0))/Area # in mm
        I = (1/12*spacing* plate_t**3.0 + spacing*plate_t*(Z_centroid)**2.0 + 1/12*w*t**3.0 + w*t*(Z_centroid - h)**2.0 + 1/12*t*h**3.0 + h*t*(h/2.0 - Z_centroid)**2.0) # in mm^4
        z = max(Z_centroid, h - Z_centroid) # in mm
        SM = (I/z)/1000.0 # in cm^3

        bit_effective_flange = w > 2*t
        aspect_ratio = h/t

        return SM, bit_effective_flange, aspect_ratio # in cm^3, bool, dimensionless
    
    def calc_Bulkhead_gyration_radius(self): 

        '''
        This function calculates the radius of gyration of a transverse bulkhead based on its geometry and spacing
        '''
        b = self.structure_params[1]
        t_bulkhead = self.structure_params[11]/1000.0 # in m
        n_vertical_stringers = int(self.structure_params[21])*2 - 1
        h_stringer = self.structure_params[44]/1000.0 # in m
        t_stringer = self.structure_params[45]/1000.0 # in m
        w_stringer = self.structure_params[46]/1000.0 # in m

        A_bulkhead = b*t_bulkhead + n_vertical_stringers*(h_stringer*t_stringer + w_stringer*t_stringer) # in m^2
        X_centroid = (n_vertical_stringers*(h_stringer*t_stringer*(h_stringer/2.0) + w_stringer*t_stringer*(h_stringer)))/A_bulkhead # in m
        
        I_11 = 1/12*b*t_bulkhead**3.0 + b*t_bulkhead*(X_centroid)**2.0 + n_vertical_stringers*(1/12*w_stringer*t_stringer**3.0 + w_stringer*t_stringer*(X_centroid - h_stringer)**2.0 + 1/12*t_stringer*h_stringer**3.0 + h_stringer*t_stringer*(X_centroid - h_stringer/2.0)**2.0) # in m^4
        r_gyration = np.sqrt(I_11/A_bulkhead) # in m
        return r_gyration, A_bulkhead*100.0**2.0 # in cm^2

    def eval_MVR_Grab_Constraints(self, bottom_reqs):
        """
        Connect the MVR 3-2-4/9.5 grab-service thickness requirements to
        the modeled inner-bottom and lower-hopper plating.

        The calculation uses assumptions ASM-002, ASM-005, and ASM-009.
        """

        # Get the plate thickness assigned to each source-defined region.
        t_inner_bottom = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0]
        t_lower_hopper = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Lower_Hopper_Plate_Thickness']['Value'].values[0]

        min_t_inner_bottom = bottom_reqs['min_inner_bottom_grab_t']
        min_t_lower_hopper = bottom_reqs['min_lower_hopper_grab_t']

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure'],
            'Name': ['Min Inner Bottom Grab-Service Thickness',
                     'Min Lower Hopper Grab-Service Thickness'],
            'Requirement': ['ABS 3-2-4/9.5 (ASM-002, ASM-005, ASM-009)',
                            'ABS 3-2-4/9.5 (ASM-002, ASM-005, ASM-009)'],
            'Threshold Value': [min_t_inner_bottom, min_t_lower_hopper],
            'Constraint Value': [t_inner_bottom, t_lower_hopper],
            'Units': ['mm', 'mm'],
            'Constraint Eval': [min_t_inner_bottom - t_inner_bottom,
                                min_t_lower_hopper - t_lower_hopper],
            'Constraint Bit': [t_inner_bottom >= min_t_inner_bottom,
                               t_lower_hopper >= min_t_lower_hopper]
        })

        return item
    
    def eval_Structure_Constraints(self, Distibuted_Loading_Dict):
        """
        Evaluate the structure constraints based on the calculated structure values and the distributed loading dict

        Distibuted_Loading_Dict: Dictionary containing the distributed loading values for the ship

        self.StructureConstraints = pd.DataFrame(columns = ['Class', 'Name', 'Requirement', 'Threshold Value', 'Constraint Value', 'Units', 'Constraint Eval', 'Constraint Bit'])
        
        """
        # Hull Dim Constraints: 
        max_L, max_B, max_B_from_D = self.ABS.Hull_Dim_Constraints()
        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure'], 
            'Name': ['Max_LOA', 'Max_B', 'Max_Beam_from_Draft'], 
            'Requirement': ['ABS 3-1-1/3.1', 'ABS 3-1-2/7', 'ABS 3-1-2/7'], 
            'Threshold Value': [max_L, max_B, max_B_from_D], 
            'Constraint Value': [self.Hull_Dict['L'], self.Hull_Dict['B'], self.Hull_Dict['B']], 
            'Units': ['m', 'm', 'm'],
            'Constraint Eval': [self.Hull_Dict['L'] - max_L, self.Hull_Dict['B'] - max_B, self.Hull_Dict['B'] -  max_B_from_D],
            'Constraint Bit': [self.Hull_Dict['L'] <= max_L, self.Hull_Dict['B'] <= max_B, self.Hull_Dict['B'] <= max_B_from_D]
        })
        self.StructureConstraints = item.copy()


        # Bending Calculations: 
        SM_61 = self.ABS.calc_Min_Longitudinal_Section_Modulus_61m()
        Mnsw = Distibuted_Loading_Dict['Mnsw']
        Mpsw = Distibuted_Loading_Dict['Mpsw']

        SM_Wave = self.ABS.calc_Min_Longitudinal_Section_Modulus_Constraint(Mnsw, Mpsw)

        if self.Hull_Dict['L'] <= 61.0: 
            SM_req = SM_61*100.0 # convert to cm^3
            req = 'ABS 3-2-1/3.1'
            SM_type = 'Min Hull Girder Longitudinal Section Modulus for L <= 61m'
        else:
            SM_req = SM_Wave *100.0 # convert to cm^3
            req = 'ABS 3-2-1/3.7'
            SM_type = 'Min Hull Girder Longitudinal Section Modulus for Wave Bending'
        
        Hull_Girder_SM = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hull_Girder_SM']['Value'].values[0]
        item = pd.DataFrame({
            'Class': ['Structure'], 
            'Name': [SM_type], 
            'Requirement': [req], 
            'Threshold Value': [SM_req], 
            'Constraint Value': [Hull_Girder_SM], 
            'Units': ['cm^3'],
            'Constraint Eval': [SM_req - Hull_Girder_SM],
            'Constraint Bit': [Hull_Girder_SM >= SM_req]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Min Girder Section Modulus Calculations:
        I_min = self.ABS.calc_Hull_Girder_I_Constraint(Hull_Girder_SM/100.0) # in cm^3 -> convert to cm^2-m)
        Hull_Girder_I = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hull_Girder_I']['Value'].values[0] # in cm^2-m^2
        item = pd.DataFrame({
            'Class': ['Structure'], 
            'Name': ['Min Hull Girder Moment of Inertia'], 
            'Requirement': ['ABS 3-2-1/3.7.2'], 
            'Threshold Value': [I_min], 
            'Constraint Value': [Hull_Girder_I*(100.0**2.0)],
            'Units': ['cm^2-m^2'],
            'Constraint Eval': [I_min - Hull_Girder_I*(100.0**2.0)],
            'Constraint Bit': [Hull_Girder_I*(100.0**2.0) >= I_min]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        # Hull Girder Shear Force
        Fnsw = Distibuted_Loading_Dict['Fnsw']
        Fpsw = Distibuted_Loading_Dict['Fpsw']
        I = Hull_Girder_I*100.0**4.0 # in cm^4
        M = self.Struct_Dict[self.Struct_Dict['Parameter'] =='Hull_Girder_AreaMoment']['Value'].values[0] # in cm^3
        t = self.Struct_Dict[self.Struct_Dict['Parameter'] =='Side_Shell_Thickness']['Value'].values[0] # in mm
        fs, fs_max = self.ABS.calc_Shearing_Force_Constraint(Fnsw, Fpsw, I, M, t)
        item = pd.DataFrame({
            'Class': ['Structure'], 
            'Name': ['Max Hull Girder Shear Stress'], 
            'Requirement': ['ABS 3-2-1/3.9.2'], 
            'Threshold Value': [fs_max], 
            'Constraint Value': [fs],
            'Units': ['kN/cm^2'],
            'Constraint Eval': [fs - fs_max],
            'Constraint Bit': [fs <= fs_max]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Shell Plating 
        s_frames = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Frame_Spacing']['Value'].values[0]*1000.0 #in mm
        s_side_stringers = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_Spacing']['Value'].values[0]*1000.0 #in mm
        s_bottom_stringers = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_Spacing']['Value'].values[0]*1000.0 #in mm
        
        
        min_t_sideshell, min_t_bottomshell = self.ABS.calc_Shell_Plating_Constraints(s_frames, s_side_stringers, s_bottom_stringers)
        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure'], 
            'Name': ['Min Side Shell Thickness', 'Min Bottom Shell Thickness', 'Min Inner Side Shell Thickness'], 
            'Requirement': ['ABS 3-2-2/3.9', 'ABS 3-2-2/3.15', 'ABS 3-2-2/3.9'], 
            'Threshold Value': [min_t_sideshell, min_t_bottomshell, min_t_sideshell], 
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Shell_Thickness']['Value'].values[0], self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Shell_Thickness']['Value'].values[0], self.structure_params[12]],
            'Units': ['mm', 'mm', 'mm'],
            'Constraint Eval': [min_t_sideshell - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Shell_Thickness']['Value'].values[0], min_t_bottomshell - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Shell_Thickness']['Value'].values[0], min_t_sideshell - self.structure_params[12]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Shell_Thickness']['Value'].values[0] >= min_t_sideshell, self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Shell_Thickness']['Value'].values[0] >= min_t_bottomshell, self.structure_params[12] >= min_t_sideshell]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Top Deck Plating constraint
        s_deck_stringers = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Stringer_Spacing']['Value'].values[0]*1000.0 #in mm
        h = 3.66 # value for unloaded deck per ABS 3-2-3/5.1
        min_t_deck = self.ABS.calc_Deck_Constraints(s_frames, s_deck_stringers, h)

        s_inner_bottom_stringers = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_Spacing']['Value'].values[0]*1000.0 #in mm
        h_cargo = self.Hull_Dict['D'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0]/1000.0 # in m
        min_t_inner_bottom = self.ABS.calc_Deck_Constraints(s_frames, s_inner_bottom_stringers, h) # in mm

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure'], 
            'Name': ['Min Deck Shell Thickness', 'Min Inner Bottom as Deck Thickness'], 
            'Requirement': ['ABS 3-2-3/5.1', 'ABS 3-2-3/5.1'], 
            'Threshold Value': [min_t_deck, min_t_inner_bottom], 
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Top_Deck_Thickness']['Value'].values[0], self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0]],
            'Units': ['mm', 'mm'],
            'Constraint Eval': [min_t_deck - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Top_Deck_Thickness']['Value'].values[0], min_t_inner_bottom - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Top_Deck_Thickness']['Value'].values[0] >= min_t_deck, self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0] >= min_t_inner_bottom]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Bottom Requirements
        s_bottom_girders = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Spacing']['Value'].values[0] # in m
        s_bottom_stringers = s_inner_bottom_stringers # in mm
        s_webframes = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0] # in m
        s_lower_hopper_stiffeners = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Lower_Hopper_Stiffener_Spacing']['Value'].values[0] # in mm
        bottom_reqs = self.ABS.calc_Bottom_Structure_Constraints(s_bottom_girders, s_bottom_stringers, s_lower_hopper_stiffeners, s_webframes, s_frames, h_cargo)

        item = pd.DataFrame({
            'Class': ['Structure']*10,
            'Name': ['Min Center Bottom Girder Thickness', 
                     'Min Double Bottom Height', 
                     'Max Bottom Girder Spacing',
                     'Min Floor Thickness', 
                     'Max Floor Spacing', 
                     'Max Bottom Stiffener Spacing', 
                     'Min Bottom Frame SM', 
                     'Min Inner Bottom Thickness', 
                     'Min Bottom Stringer SM',
                     'Min Inner Bottom Stringer SM'],
            'Requirement': ['ABS 3-2-4/3.1',
                            'ABS 3-2-4/3.1',
                            'ABS 3-2-4/3.7',
                            'ABS 3-2-4/5',
                            'ABS 3-2-4/5',
                            'ABS 3-2-4/5',
                            'ABS 3-2-4/7.3', 
                            'ABS 3-2-4/9.2',
                            'ABS 3-2-4/11.3',
                            'ABS 3-2-4/11.5'],
            'Threshold Value': [bottom_reqs['min_t_center_girder'], 
                                bottom_reqs['min_double_bottom_height'], 
                                bottom_reqs['max_bottom_girder_spacing'], 
                                bottom_reqs['min_solid_floor_plate_t'], 
                                bottom_reqs['max_solid_floor_spacing'], 
                                bottom_reqs['max_bottom_stiffener_spacing'], 
                                bottom_reqs['min_bottom_frame_SM'], 
                                bottom_reqs['min_inner_bottom_plating_t'], 
                                bottom_reqs['min_bottom_longitudinal_SM'],
                                bottom_reqs['min_inner_bottom_longitudinal_SM']],
            'Constraint Value': [2.0*self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0], #need to double the girder thickness
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0],
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Spacing']['Value'].values[0]*1000.0,
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0],
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0]*1000.0,
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_Spacing']['Value'].values[0]*1000.0,
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Frame_SM']['Value'].values[0],
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0],
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_SM']['Value'].values[0],
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Stringer_SM']['Value'].values[0]],
            'Units': ['mm', 'mm', 'mm', 'mm', 'mm', 'mm', 'cm^3', 'mm', 'cm^3', 'cm^3'],
            'Constraint Eval': [bottom_reqs['min_t_center_girder'] - 2.0*self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0],
                                bottom_reqs['min_double_bottom_height'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0],
                                -bottom_reqs['max_bottom_girder_spacing'] + self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Spacing']['Value'].values[0]*1000.0,
                                bottom_reqs['min_solid_floor_plate_t'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0],
                                -bottom_reqs['max_solid_floor_spacing'] + self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0]*1000.0,
                                -bottom_reqs['max_bottom_stiffener_spacing'] + self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_Spacing']['Value'].values[0]*1000.0,
                                bottom_reqs['min_bottom_frame_SM'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Frame_SM']['Value'].values[0],
                                bottom_reqs['min_inner_bottom_plating_t'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0],
                                bottom_reqs['min_bottom_longitudinal_SM'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_SM']['Value'].values[0],
                                bottom_reqs['min_inner_bottom_longitudinal_SM'] - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Stringer_SM']['Value'].values[0]],
            'Constraint Bit': [2.0*self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0] >= bottom_reqs['min_t_center_girder'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0] >= bottom_reqs['min_double_bottom_height'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Spacing']['Value'].values[0]*1000.0 <= bottom_reqs['max_bottom_girder_spacing'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0] >= bottom_reqs['min_solid_floor_plate_t'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0]*1000.0 <= bottom_reqs['max_solid_floor_spacing'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_Spacing']['Value'].values[0]*1000.0 <= bottom_reqs['max_bottom_stiffener_spacing'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Frame_SM']['Value'].values[0] >= bottom_reqs['min_bottom_frame_SM'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0] >= bottom_reqs['min_inner_bottom_plating_t'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Stringer_SM']['Value'].values[0] >= bottom_reqs['min_bottom_longitudinal_SM'],
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Stringer_SM']['Value'].values[0] >= bottom_reqs['min_inner_bottom_longitudinal_SM']]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        # Grab-Service Plating Requirements
        item = self.eval_MVR_Grab_Constraints(bottom_reqs)
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Frame Requirements
        
        l_frames = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'l_frames']['Value'].values[0] # in m
        h_frames = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'h_frames']['Value'].values[0] # in m
        b = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Height']['Value'].values[0]/1000.0 # in m

        #print(s_frames, l_frames, h_frames, b)
        max_s_frames, min_SM_frame = self.ABS.calc_Frame_Constraints(s_frames/1000.0, l_frames, h_frames, b)

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure'], 
            'Name': ['Max Frame Spacing', 'Min Frame Section Modulus'], 
            'Requirement': ['ABS 3-2-5/1.7', 'ABS 3-2-5/3.1'], 
            'Threshold Value': [max_s_frames, min_SM_frame], 
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Frame_Spacing']['Value'].values[0]*1000.0, self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Frame_SM']['Value'].values[0]],
            'Units': ['mm', 'cm^3'],
            'Constraint Eval': [-max_s_frames + self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Frame_Spacing']['Value'].values[0]*1000.0, min_SM_frame - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Frame_SM']['Value'].values[0]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Frame_Spacing']['Value'].values[0]*1000.0 <= max_s_frames, self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Frame_SM']['Value'].values[0] >= min_SM_frame]

        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True) 

        #Web Frame Requirements
        s_webframes = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0] # in m
        l_webframes = h_cargo
        h_web = max(0.5*l_webframes, np.abs(h_cargo/2.0 + self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0]/1000.0 - self.Hull_Dict['d'])) # in m
        # ABS MVR 3-2-6/3.1.1: K = 1 for the generated longitudinal deck,
        # because a transverse deck beam is fitted at every web-frame station.
        # K is not the number of ordinary side frames between web frames.
        K = 1.0
        s_side_stringers = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_Spacing']['Value'].values[0] # in m
        min_webframe_SM, min_webframe_height, min_webframe_thickness, min_side_stringer_SM, min_side_stringer_height = self.ABS.calc_WebframeAndSideStringer_Constraints(s_webframes, l_webframes, h_web, b, K, s_side_stringers)

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure', 'Structure', 'Structure'],
            'Name': ['Min Web Frame Height', 'Min Web Frame Thickness', 'Min Web Frame Section Modulus', 'Min Side Stringer Section Modulus', 'Min Side Stringer Height'],
            'Requirement': ['ABS 3-2-6/3.5', 'ABS 3-2-6/3.5', 'ABS 3-2-6/3.1', 'ABS 3-2-6/5.1', 'ABS 3-2-6/5.3'],
            'Threshold Value': [min_webframe_height, min_webframe_thickness, min_webframe_SM, min_side_stringer_SM, min_side_stringer_height],
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Height']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_Height']['Value'].values[0]],
            'Units': ['mm', 'mm', 'cm^3', 'cm^3', 'mm'],
            'Constraint Eval': [min_webframe_height - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Height']['Value'].values[0], 
                                min_webframe_thickness - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Thickness']['Value'].values[0], 
                                min_webframe_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_SM']['Value'].values[0], 
                                min_side_stringer_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_SM']['Value'].values[0], 
                                min_side_stringer_height - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_Height']['Value'].values[0]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Height']['Value'].values[0] >= min_webframe_height, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Thickness']['Value'].values[0] >= min_webframe_thickness, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_SM']['Value'].values[0] >= min_webframe_SM,
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_SM']['Value'].values[0] >= min_side_stringer_SM, 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Side_Stringer_Height']['Value'].values[0] >= min_side_stringer_height]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        # ABS MVR 3-2-7/3.1 applies specifically to each deck beam and deck
        # longitudinal in association with its attached deck plating.  The
        # legacy evaluator reused this equation for seventeen unrelated member
        # types.  Those rows are removed rather than turned into fabricated
        # passes.  The two modeled members supported by the source wording are
        # the ordinary strength-deck longitudinal and the ordinary transverse
        # deck beam between deck girders.
        Y_top_deck = abs(
            self.Hull_Dict['D']
            - self.Struct_Dict.loc[
                self.Struct_Dict['Parameter'] == 'Hull_Girder_Z_Centroid',
                'Value',
            ].values[0]
        )
        Hull_Girder_I = (
            self.Struct_Dict.loc[
                self.Struct_Dict['Parameter'] == 'Hull_Girder_I',
                'Value',
            ].values[0]
            * 100.0**2.0
        )
        h_deck = self.ABS.calc_freeboard_deck_head()
        s_deck_stringers = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'Deck_Stringer_Spacing',
            'Value',
        ].values[0]
        l_deck_longitudinal = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'WebFrame_Spacing',
            'Value',
        ].values[0]
        s_deck_beams = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'Frame_Spacing',
            'Value',
        ].values[0]
        l_deck_beam = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'Deck_Girder_Spacing',
            'Value',
        ].values[0]
        min_SM_Long_Deck_Elements = (
            self.ABS.calc_BeamsAndLongitudinal_Constraints(
                Hull_Girder_I,
                Y_top_deck,
                'longitudinal',
                l_deck_longitudinal,
                s_deck_stringers,
                h_deck,
            )
        )
        min_SM_Trans_Deck_Elements = (
            self.ABS.calc_BeamsAndLongitudinal_Constraints(
                Hull_Girder_I,
                Y_top_deck,
                'transverse',
                l_deck_beam,
                s_deck_beams,
                h_deck,
            )
        )
        deck_longitudinal_sm = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'Deck_Stringer_SM',
            'Value',
        ].values[0]
        ordinary_deck_beam_sm = self.Struct_Dict.loc[
            self.Struct_Dict['Parameter'] == 'Intermediate_Deck_Beam_SM',
            'Value',
        ].values[0]

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure'],
            'Name': [
                'Min Relative Deck Stringer Section Modulus',
                'Min Relative Deck Frame Section Modulus',
            ],
            'Requirement': ['ABS 3-2-7/3.1']*2,
            'Threshold Value': [
                min_SM_Long_Deck_Elements,
                min_SM_Trans_Deck_Elements,
            ],
            'Constraint Value': [
                deck_longitudinal_sm,
                ordinary_deck_beam_sm,
            ],
            'Units': ['cm^3', 'cm^3'],
            'Constraint Eval': [
                min_SM_Long_Deck_Elements-deck_longitudinal_sm,
                min_SM_Trans_Deck_Elements-ordinary_deck_beam_sm,
            ],
            'Constraint Bit': [
                deck_longitudinal_sm >= min_SM_Long_Deck_Elements,
                ordinary_deck_beam_sm >= min_SM_Trans_Deck_Elements,
            ],
        })
        self.StructureConstraints = pd.concat(
            [self.StructureConstraints, item],
            ignore_index=True,
        )

        #Deck Constraints
        h_deck = 3.66 # in m
        bulkhead_cx_area = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Structure_Cross_Sectional_Area']['Value'].values[0] # in cm^2
        l_bulkhead = h_cargo
        r_bulkhead = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Radius_of_Gyration']['Value'].values[0]*100.0 # in cm
        s_girder = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Spacing']['Value'].values[0] # in m
        s_bulkhead = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Spacing']['Value'].values[0] # in m
        s_beams = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Spacing']['Value'].values[0] # in m
        h_girder = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Height']['Value'].values[0] # in mm
        h_beam = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Height']['Value'].values[0] # in mm

        W_a, W_calc, min_Deck_Girder_SM, min_Deck_Beam_SM, h_girder_min, h_beam_min, t_girder_min, t_beam_min = self.ABS.calc_DeckGirdersAndTransverses_Constraints(h_deck, bulkhead_cx_area, l_bulkhead, r_bulkhead, s_girder, s_bulkhead, s_beams, h_girder, h_beam)

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure', 'Structure', 'Structure', 'Structure', 'Structure'], 
            'Name': ['Min Deck Girder Section Modulus', 
                     'Min Deck Beam Section Modulus', 
                     'Min Deck Girder Height', 
                     'Min Deck Beam Height', 
                     'Min Deck Girder Thickness', 
                     'Min Deck Beam Thickness',
                     'Max Allowable Uniform Deck Loading'],
                     
            'Requirement': ['ABS 3-2-8/5.3',
                            'ABS 3-2-8/5.5',
                            'ABS 3-2-8/5.7',
                            'ABS 3-2-8/5.7',
                            'ABS 3-2-8/5.13',
                            'ABS 3-2-8/5.13',
                            'ABS 3-2-8/3.3'],
            'Threshold Value': [min_Deck_Girder_SM, 
                                min_Deck_Beam_SM, 
                                h_girder_min, 
                                h_beam_min, 
                                t_girder_min, 
                                t_beam_min, 
                                W_calc],
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Height']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Height']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Plate_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Plate_Thickness']['Value'].values[0],
                                 W_a],
            'Units': ['cm^3', 'cm^3', 'mm', 'mm', 'mm', 'mm', 'kN'],
            'Constraint Eval': [min_Deck_Girder_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_SM']['Value'].values[0], 
                                min_Deck_Beam_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0], 
                                h_girder_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Height']['Value'].values[0], 
                                h_beam_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Height']['Value'].values[0], 
                                t_girder_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Plate_Thickness']['Value'].values[0], 
                                t_beam_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Plate_Thickness']['Value'].values[0],
                                W_calc - W_a],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_SM']['Value'].values[0] >= min_Deck_Girder_SM, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0] >= min_Deck_Beam_SM, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Height']['Value'].values[0] >= h_girder_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Height']['Value'].values[0] >= h_beam_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Girder_Plate_Thickness']['Value'].values[0] >= t_girder_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_Plate_Thickness']['Value'].values[0] >= t_beam_min,
                               W_calc <= W_a]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        
        W_a, W_calc, min_bottom_girder_SM, min_bottom_Floor_SM, h_bottom_girder_min, h_bottom_floor_min, t_bottom_girder_min, t_bottom_floor_min = self.ABS.calc_DeckGirdersAndTransverses_Constraints(h_cargo, bulkhead_cx_area, l_bulkhead, r_bulkhead, s_girder, s_bulkhead, s_beams, h_girder, h_beam)
        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure', 'Structure', 'Structure', 'Structure', 'Structure'], 
            'Name': ['Min Bottom Girder Section Modulus', 
                     'Min Bottom Floor Section Modulus', 
                     'Min Bottom Girder Height', 
                     'Min Bottom Floor Height', 
                     'Min Bottom Girder Thickness', 
                     'Min Bottom Floor Thickness',
                     'Max Allowable Uniform Bottom Loading'],
            'Requirement': ['ABS 3-2-8/5.3',
                            'ABS 3-2-8/5.5',
                            'ABS 3-2-8/5.7',
                            'ABS 3-2-8/5.7',
                            'ABS 3-2-8/5.13',
                            'ABS 3-2-8/5.13',
                            'ABS 3-2-8/3.3'],
            'Threshold Value': [min_bottom_girder_SM, min_bottom_Floor_SM, h_bottom_girder_min, h_bottom_floor_min, t_bottom_girder_min, t_bottom_floor_min, W_calc],
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0],
                                 W_a],
            'Units': ['cm^3', 'cm^3', 'mm', 'mm', 'mm', 'mm', 'kN'],
            'Constraint Eval': [min_bottom_girder_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_SM']['Value'].values[0], 
                                min_bottom_Floor_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_SM']['Value'].values[0], 
                                h_bottom_girder_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0], 
                                h_bottom_floor_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0], 
                                t_bottom_girder_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0], 
                                t_bottom_floor_min - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0],
                                W_calc - W_a],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_SM']['Value'].values[0] >= min_bottom_girder_SM, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_SM']['Value'].values[0] >= min_bottom_Floor_SM, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0] >= h_bottom_girder_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Double_Bottom_Height']['Value'].values[0] >= h_bottom_floor_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Girder_Plate_Thickness']['Value'].values[0] >= t_bottom_girder_min, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bottom_Floor_Plate_Thickness']['Value'].values[0] >= t_bottom_floor_min,
                               W_calc <= W_a]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        #Hatch Opening Constraints: 
        #'Parameter': ['Hatch_Length', 'Hatch_Width', 'Hatch_Corner_Radius', 'Hatch_Area', 'Hatch_F', 'Hatch_B', 'Hatch_C', 'Hatch_D'], 
        w_hatch = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_Width']['Value'].values[0]
        l_hatch = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_Length']['Value'].values[0]    
        C = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_C']['Value'].values[0]
        B = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_B']['Value'].values[0]
        D = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_D']['Value'].values[0]
        N = self.Hull_Dict['B']/2.0
        F = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_F']['Value'].values[0]
        l = N - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Height']['Value'].values[0]/1000.0 # distance from centerline to web frame

        min_hatch_area = 3.0/8.0 * self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Spacing']['Value'].values[0] * self.Hull_Dict['B'] # in m^2

        min_Deck_Beam_SM = self.ABS.calc_HatchSupport_Constraints(w_hatch, l_hatch, C, B, D, N, F, l)

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure'], 
            'Name': ['Min Hatch Support Beam Section Modulus', 'Min Hatch Area'], 
            'Requirement': ['ABS 3-2-8/5.1', 'self imposed'], 
            'Threshold Value': [min_Deck_Beam_SM, min_hatch_area], 
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_Area']['Value'].values[0]],
            'Units': ['cm^3', 'm^2'],
            'Constraint Eval': [min_Deck_Beam_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0], 
                                min_hatch_area - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_Area']['Value'].values[0]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Deck_Beam_SM']['Value'].values[0] >= min_Deck_Beam_SM, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Hatch_Area']['Value'].values[0] >= min_hatch_area]
        })

        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        # Bulkhead Constriants
       
        s_horizontal_stiffeners = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'HorizBulkhead_Stringer_Spacing']['Value'].values[0]*1000.0 # in mm
        s_vertical_stiffeners = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'VertBulkhead_Stringer_Spacing']['Value'].values[0]*1000.0 # in mm
        min_t_bulkhead, min_Bulkhead_Stiffener_SM = self.ABS.calc_Bulkhead_Constraints(s_vertical_stiffeners, s_horizontal_stiffeners)

        #SM is for both vertical and horizontal stiffeners
        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure'],
            'Name': ['Min Bulkhead Plating Thickness', 'Min Vertical Bulkhead Stiffener Section Modulus', 'Min Horizontal Bulkhead Stiffener Section Modulus'],
            'Requirement': ['ABS 3-2-9/5.1', 'ABS 3-2-9/5.3', 'ABS 3-2-9/5.3'],
            'Threshold Value': [min_t_bulkhead, min_Bulkhead_Stiffener_SM, min_Bulkhead_Stiffener_SM],
            'Constraint Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Vertical_Bulkhead_Stringer_SM']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Horizontal_Bulkhead_Stringer_SM']['Value'].values[0]],
            'Units': ['mm', 'cm^3', 'cm^3'],
            'Constraint Eval': [min_t_bulkhead - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Thickness']['Value'].values[0], 
                                min_Bulkhead_Stiffener_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Vertical_Bulkhead_Stringer_SM']['Value'].values[0],
                                  min_Bulkhead_Stiffener_SM - self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Horizontal_Bulkhead_Stringer_SM']['Value'].values[0]],
            'Constraint Bit': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Bulkhead_Thickness']['Value'].values[0] >= min_t_bulkhead, 
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Vertical_Bulkhead_Stringer_SM']['Value'].values[0] >= min_Bulkhead_Stiffener_SM,
                               self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Horizontal_Bulkhead_Stringer_SM']['Value'].values[0] >= min_Bulkhead_Stiffener_SM]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

        # ABS MVR 3-1-2/13.5.2 web-aspect evaluation is intentionally
        # disabled for the GenISys project by user decision.  The member
        # aspect-ratio calculations remain in ``Struct_Dict`` as diagnostic
        # geometry, but they do not create constraint rows or affect design
        # feasibility.
        #
        # The former 17-row evaluation is retained below as commented source
        # so that its historical implementation remains available without
        # being executed.
        #
        # aspect_ratio_names = [
        #     'Deck_Girder_Aspect_Ratio',
        #     'Deck_Stringer_Aspect_Ratio',
        #     'Deck_Beam_Aspect_Ratio',
        #     'Shear_Strake_Aspect_Ratio',
        #     'Bottom_Girder_Aspect_Ratio',
        #     'Bottom_Floor_Aspect_Ratio',
        #     'WebFrame_Aspect_Ratio',
        #     'Side_Stringer_Aspect_Ratio',
        #     'Bottom_Stringer_Aspect_Ratio',
        #     'Inner_Bottom_Stringer_Aspect_Ratio',
        #     'Vertical_Bulkhead_Stringer_Aspect_Ratio',
        #     'Horizontal_Bulkhead_Stringer_Aspect_Ratio',
        #     'Upper_Hopper_Stiffener_Aspect_Ratio',
        #     'Lower_Hopper_Stiffener_Aspect_Ratio',
        #     'Intermediate_Deck_Beam_Aspect_Ratio',
        #     'Bottom_Frame_Aspect_Ratio',
        #     'Inner_Bottom_Frame_Aspect_Ratio',
        # ]
        # flange_bits_names = [
        #     'Deck_Girder_Flange',
        #     'Deck_Stringer_Flange',
        #     'Deck_Beam_Flange',
        #     'Shear_Strake_Flange',
        #     'Bottom_Girder_Flange',
        #     'Bottom_Floor_Flange',
        #     'WebFrame_Flange',
        #     'Side_Stringer_Flange',
        #     'Bottom_Stringer_Flange',
        #     'Inner_Bottom_Stringer_Flange',
        #     'Vertical_Bulkhead_Stringer_Flange',
        #     'Horizontal_Bulkhead_Stringer_Flange',
        #     'Upper_Hopper_Stiffener_Flange',
        #     'Lower_Hopper_Stiffener_Flange',
        #     'Intermediate_Deck_Beam_Flange',
        #     'Bottom_Frame_Flange',
        #     'Inner_Bottom_Frame_Flange',
        # ]
        #
        # for i in range(len(aspect_ratio_names)):
        #     name = aspect_ratio_names[i]
        #     flange_bit_name = flange_bits_names[i]
        #     aspect_ratio = self.Struct_Dict.loc[
        #         self.Struct_Dict['Parameter'] == name, 'Value'
        #     ].values[0]
        #     flange_bit = self.Struct_Dict.loc[
        #         self.Struct_Dict['Parameter'] == flange_bit_name, 'Value'
        #     ].values[0]
        #
        #     if flange_bit:
        #         max_aspect_ratio = 50.0
        #     else:
        #         max_aspect_ratio = 15.0
        #
        #     item = pd.DataFrame({
        #         'Class': ['Structure'],
        #         'Name': [name],
        #         'Requirement': ['ABS 3-1-2/13.5.2'],
        #         'Threshold Value': [max_aspect_ratio],
        #         'Constraint Value': [aspect_ratio],
        #         'Units': ['-'],
        #         'Constraint Eval': [aspect_ratio - max_aspect_ratio],
        #         'Constraint Bit': [aspect_ratio <= max_aspect_ratio],
        #     })
        #
        #     self.StructureConstraints = pd.concat(
        #         [self.StructureConstraints, item],
        #         ignore_index=True,
        #     )

        # Brackets
        # 'Lower_Hopper_Bracket_Angle', 'Lower_Hopper_Bracket_Thickness', 'Upper_Hopper_Bracket_Angle', 'Upper_Hopper_Bracket_Thickness', 'Lower_Hopper_Plate_Thickness', 'Upper_Hopper_Plate_Thickness', 
        top_bracket_t = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Upper_Hopper_Bracket_Thickness']['Value'].values[0] 
        bottom_bracket_t = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Lower_Hopper_Bracket_Thickness']['Value'].values[0]
        top_plate_t = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Upper_Hopper_Plate_Thickness']['Value'].values[0]
        bottom_plate_t = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Lower_Hopper_Plate_Thickness']['Value'].values[0]
        top_bracket_angle = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Upper_Hopper_Bracket_Angle']['Value'].values[0]
        bottom_bracket_angle = self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Lower_Hopper_Bracket_Angle']['Value'].values[0]

        item = pd.DataFrame({
            'Class': ['Structure', 'Structure', 'Structure', 'Structure', 'Structure','Structure', 'Structure', 'Structure'], 
            'Name': ['Min Upper Hopper Bracket Thickness', 
                     'Min Lower Hopper Bracket Thickness', 
                     'Min Upper Hopper Plate Thickness', 
                     'Min Lower Hopper Plate Thickness',
                     'Min Upper Hopper Bracket Angle',
                     'Min Lower Hopper Bracket Angle',
                     'Max Upper Hopper Bracket Angle', 
                    'Max Lower Hopper Bracket Angle'],
            'Requirement': ['self imposed']*8,
            'Threshold Value': [self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'WebFrame_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Top_Deck_Thickness']['Value'].values[0], 
                                 self.Struct_Dict[self.Struct_Dict['Parameter'] == 'Inner_Bottom_Thickness']['Value'].values[0],
                                 40.0,
                                 40.0, 
                                 50.0,
                                 50.0],
            'Constraint Value': [top_bracket_t, bottom_bracket_t, top_plate_t, bottom_plate_t, top_bracket_angle, bottom_bracket_angle, top_bracket_angle, bottom_bracket_angle],
            'Units': ['mm', 'mm', 'mm', 'mm', 'degrees', 'degrees', 'degrees', 'degrees'],
            'Constraint Eval': [self.Struct_Dict[self.Struct_Dict['Parameter'] =='WebFrame_Thickness']['Value'].values[0] - top_bracket_t, 
                                self.Struct_Dict[self.Struct_Dict['Parameter'] =='WebFrame_Thickness']['Value'].values[0] - bottom_bracket_t, 
                                self.Struct_Dict[self.Struct_Dict['Parameter'] =='Top_Deck_Thickness']['Value'].values[0] - top_plate_t, 
                                self.Struct_Dict[self.Struct_Dict['Parameter'] =='Inner_Bottom_Thickness']['Value'].values[0] - bottom_plate_t,
                                40.0 - top_bracket_angle,
                                40.0 - bottom_bracket_angle,
                                top_bracket_angle - 50.0,
                                bottom_bracket_angle - 50.0],
            'Constraint Bit': [top_bracket_t >= self.Struct_Dict[self.Struct_Dict['Parameter'] =='WebFrame_Thickness']['Value'].values[0], 
                               bottom_bracket_t >= self.Struct_Dict[self.Struct_Dict['Parameter'] =='WebFrame_Thickness']['Value'].values[0], 
                               top_plate_t >= self.Struct_Dict[self.Struct_Dict['Parameter'] =='Top_Deck_Thickness']['Value'].values[0], 
                               bottom_plate_t >= self.Struct_Dict[self.Struct_Dict['Parameter'] =='Inner_Bottom_Thickness']['Value'].values[0],
                               top_bracket_angle >= 40.0,
                               bottom_bracket_angle >= 40.0,
                               top_bracket_angle <= 50.0,
                               bottom_bracket_angle <= 50.0]
        })
        self.StructureConstraints = pd.concat([self.StructureConstraints, item], ignore_index=True)

    


       

                       
        


        



        




























        

    
    

        
        
