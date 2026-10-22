"""Interpret the legacy 120-value ship-structure parameter vector.

This module contains the parameter and spacing portion of
``Rhino_tools/Parametric_Structure_V2.py`` without importing Rhino.  Geometry
builders can therefore consume the same dictionaries and arrays in a normal
Python process.

The legacy generator intentionally adjusts several values in place.  Those
mutations are preserved here because generated geometry and the saved design
vector must continue to describe the same corrected structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_SCHEMA_PATH = PROJECT_ROOT / "StructuralParameterList_V2.csv"


class ParametricStructureDefinition:
    """Translate one legacy 120-parameter row into builder dictionaries.

    Parameters
    ----------
    params:
        A mutable, one-dimensional sequence in the order defined by
        :attr:`parameter_schema`.  As in the Rhino implementation, this
        sequence is adjusted in place to align related members and cap flange
        widths.  Use :meth:`clean_struct_params` on a two-dimensional sampled
        array before constructing definitions when raw samples have not yet
        been normalized.
    """

    # Indices retain the original groupings used by the sampling and cleanup
    # code.  Public aliases with the legacy capitalization are included below.
    param_idx_bit = np.array(
        [27, 31, 35, 39, 43, 47, 51, 56, 74, 79, 86, 91, 96, 100, 104,
         105, 111, 115, 116]
    )
    param_idx_cat = np.array([57, 58, 59])
    param_idx_int = np.array(
        [15, 16, 17, 18, 19, 20, 21, 22, 70, 75, 82, 87, 92, 106]
    )
    param_idx_plate_thicknesses = np.array(
        [7, 8, 9, 10, 11, 12, 23, 29, 33, 37, 41, 45, 49, 52, 54, 62,
         65, 68, 69, 72, 77, 80, 81, 84, 89, 94, 98, 102, 107, 109, 113]
    )
    param_idx_brackets = np.array([60, 61, 63, 64])
    param_idx_container = np.array(
        [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    )
    param_idx_bulkcarrier = np.array(
        [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    )

    # ``utf-8-sig`` removes the BOM attached to the historical ``idx`` header.
    parameter_schema = pd.read_csv(PARAMETER_SCHEMA_PATH, encoding="utf-8-sig")

    # Preserve the public name used by Structure_3H callers.
    Param_Dict = parameter_schema

    def __init__(self, params):
        # Retain the caller's mutable row.  Later corrections intentionally
        # update this same object, matching Structure_3H.
        self.params = params

        # Main hull dimensions are metres except for the dimensionless
        # overhang fraction.
        self.Hull_Dict = {
            "L_3h": params[0],
            "B": params[1],
            "T": params[2],
            "D": params[3],
            "R_b": params[4],
            "l_overhang": params[5],
        }

        # Structural dimensions remain in millimetres, as expected by the
        # Rhino builder contracts being reproduced by CadQuery.
        self.Struct_Params = {
            "Db": params[6],
            "Bottom_Shell_Thickness": params[7],
            "Side_Shell_Thickness": params[8],
            "Top_Deck_Thickness": params[9],
            "Inner_Bottom_Thickness": params[10],
            "Bulkhead_Thickness": params[11],
            "Inner_Side_Shell_Thickness": params[12],
        }

        self.Shear_Strake_Dict = {
            "t": params[13],
            "h": params[14],
            "1D_element": False,
            "Class": "Shear_Strake",
        }

        # Locate the four hold-bounding transverse bulkheads over the modeled
        # three-hold length.  These arrays are fractions of L_3h.
        self.x_bulkheads = np.linspace(
            self.Hull_Dict["l_overhang"],
            1 - self.Hull_Dict["l_overhang"],
            4,
        ).round(10)

        self.num_web_frames_per_hold = params[15]
        self.x_trans_web_frames = np.linspace(
            self.Hull_Dict["l_overhang"],
            1 - self.Hull_Dict["l_overhang"],
            int((self.num_web_frames_per_hold * 3) + 4),
        ).round(10)
        self.x_trans_web_frames = self.x_trans_web_frames[
            ~np.isin(self.x_trans_web_frames, self.x_bulkheads)
        ]

        # Intermediate transverse-frame locations also remain fractions of
        # L_3h.  Bulkhead and web-frame stations are removed.
        self.num_trans_frames = params[92]
        self.x_trans_frames = np.linspace(
            self.Hull_Dict["l_overhang"],
            1 - self.Hull_Dict["l_overhang"],
            int(
                (self.num_web_frames_per_hold * 3)
                + 4
                + (self.num_web_frames_per_hold + 1)
                * self.num_trans_frames
                * 3
            ),
        ).round(10)
        self.x_trans_frames = self.x_trans_frames[
            ~np.isin(self.x_trans_frames, self.x_bulkheads)
        ]
        self.x_trans_frames = self.x_trans_frames[
            ~np.isin(self.x_trans_frames, self.x_trans_web_frames)
        ]

        # The final girder lands at the inboard edge of the side web frame.
        # All y arrays below are fractions of the modeled half-breadth B/2.
        WF_frac = 1 - params[48] / (1000.0 * 0.5 * self.Hull_Dict["B"])
        self.WF_frac = WF_frac

        self.y_bottom_girders = np.linspace(
            0.0, WF_frac, int(params[16])
        ).round(10)
        self.y_bottom_stiffeners = np.linspace(
            0.0,
            WF_frac,
            int((params[16] - 1) * (params[17] + 1) + 1),
        ).round(10)

        delta_y = self.y_bottom_stiffeners[1] - self.y_bottom_stiffeners[0]
        num_stiffeners = int((1 - WF_frac) / delta_y)

        if num_stiffeners > 0:
            self.y_bottom_stiffeners = np.concatenate(
                (
                    self.y_bottom_stiffeners,
                    np.linspace(WF_frac, 1.0, num_stiffeners + 2)[1:-1],
                )
            )

        self.y_bottom_stiffeners = self.y_bottom_stiffeners[
            ~np.isin(self.y_bottom_stiffeners, self.y_bottom_girders)
        ]

        # The established generator forces deck members to align with the
        # bottom members.  Preserve both resulting mutations in ``params``.
        params[18] = params[16]
        params[19] = params[17]

        self.y_deck_girders = np.linspace(
            0.0, WF_frac, int(params[18])
        ).round(10)
        self.y_deck_stiffeners = np.linspace(
            0.0,
            WF_frac,
            int((params[18] - 1) * (params[19] + 1) + 1),
        ).round(10)

        delta_y = self.y_deck_stiffeners[1] - self.y_deck_stiffeners[0]
        num_stiffeners = int((1 - WF_frac) / delta_y)

        if num_stiffeners > 0:
            self.y_deck_stiffeners = np.concatenate(
                (
                    self.y_deck_stiffeners,
                    np.linspace(WF_frac, 1.0, num_stiffeners + 2)[1:-1],
                )
            )

        self.y_deck_stiffeners = self.y_deck_stiffeners[
            ~np.isin(self.y_deck_stiffeners, self.y_deck_girders)
        ]

        self.z_side_stiffeners = np.linspace(
            0.0, 1.0, int(params[20] + 2)
        ).round(10)[1:-1]

        # Snap the requested transverse-bulkhead stiffener counts to layouts
        # which divide the longitudinal grids evenly.
        num_y = int((params[18] - 1) * (params[19] + 1) + 1)
        y_options = []

        for i in range(2, num_y + 1):
            if int(num_y - 1) % i == 0:
                y_options.append(i + 1)

        y_options = sorted(y_options, reverse=True)
        diffs = np.abs(np.array(y_options) - params[21])
        params[21] = y_options[np.argmin(diffs)]

        self.y_vert_stiffeners = np.linspace(
            0.0, WF_frac, int(params[21])
        ).round(10)[:-1]

        num_z_options = []

        for i in range(2, int(params[20] + 2)):
            if int(params[20] + 1) % i == 0:
                num_z_options.append(i - 1)

        num_z_options = sorted(num_z_options, reverse=True)
        diffs = np.abs(np.array(num_z_options) - params[22])
        params[22] = num_z_options[np.argmin(diffs)]

        self.z_trans_stiffeners = np.linspace(
            0.0, 1.0, int(params[22] + 2)
        ).round(10)[1:-1]

        # Longitudinal-bulkhead locations are fractions of half-breadth and
        # retain the asymmetric half-model convention of the Rhino generator.
        self.bit_Long_Bulkheads = params[105]

        if self.bit_Long_Bulkheads == 1:
            self.num_long_bulkheads = params[106]

            if self.num_long_bulkheads == 1:
                self.y_long_bulkheads = np.array([0.0])
            elif self.num_long_bulkheads == 2:
                self.y_long_bulkheads = np.array([1.0 / 3.0])
            elif self.num_long_bulkheads == 3:
                self.y_long_bulkheads = np.array([0.0, 0.5])

            self.Long_Bulkhead_Params = {
                "y": self.y_long_bulkheads,
                "t": params[107],
            }

        # Longitudinal bottom girders and stiffeners.
        delta_y = (
            (self.y_bottom_stiffeners[1] - self.y_bottom_stiffeners[0])
            * 1000.0
            * 0.5
            * self.Hull_Dict["B"]
        )

        self.Bottom_Girder_Dict = {
            "y": self.y_bottom_girders,
            "h": self.Struct_Params["Db"],
            "t": params[23],
            "w": 0,
            "rot": 0,
            "dir": [1, 0, 0],
            "bit_TorC": 1,
            "1D_element": False,
            "Class": "Bottom_Longitudinal_Girder",
        }

        if params[26] > params[24]:
            params[26] = params[24]

        if params[26] > delta_y:
            params[26] = delta_y

        self.Bottom_Stiffener_Dict = {
            "y": self.y_bottom_stiffeners,
            "h": params[24],
            "t": params[25],
            "w": params[26],
            "rot": 0,
            "dir": [1, 0, 0],
            "bit_TorC": params[27],
            "1D_element": True,
            "Class": "Bottom_Longitudinal_Stiffener",
        }

        self.Inner_Bottom_Stiffener_Dict = [
            {
                "y": self.y_bottom_stiffeners,
                "h": params[24],
                "t": params[25],
                "w": params[26],
                "rot": 180,
                "dir": [1, 0, 0],
                "bit_TorC": params[27],
                "1D_element": True,
                "Class": "Inner_Bottom_Longitudinal_Stiffener",
            }
        ]

        # Intermediate bottom transverse frames.
        if params[95] > params[93]:
            params[95] = params[93]

        delta_x = (
            (self.x_trans_frames[1] - self.x_trans_frames[0])
            * 1000.0
            * self.Hull_Dict["L_3h"]
        )

        if params[95] > delta_x:
            params[95] = delta_x

        self.Inner_Bottom_Trans_Frame_Dict = [
            {
                "x": self.x_trans_frames,
                "h": params[93],
                "t": params[94],
                "w": params[95],
                "rot": -90,
                "dir": [0, 1, 0],
                "bit_TorC": params[96],
                "1D_element": True,
                "Class": "Inner_Bottom_Transverse_Beam",
            }
        ]

        if params[30] > params[28]:
            params[30] = params[28]

        if params[103] > params[101]:
            params[103] = params[101]

        if params[30] > delta_x:
            params[30] = delta_x

        if params[103] > delta_x:
            params[103] = delta_x

        self.Deck_Beam_Dict = [
            {
                "x": self.x_trans_web_frames,
                "h": params[28],
                "t": params[29],
                "w": params[30],
                "rot": -90,
                "dir": [0, 1, 0],
                "bit_TorC": params[31],
                "1D_element": False,
                "Class": "Deck_Transvere_Beam",
            },
            {
                "x": self.x_trans_frames,
                "h": params[101],
                "t": params[102],
                "w": params[103],
                "rot": -90,
                "dir": [0, 1, 0],
                "bit_TorC": params[104],
                "1D_element": True,
                "Class": "Deck_Transverse_Stiffener,",
            },
        ]

        # Deck longitudinal girders and stiffeners.
        if params[34] > params[32]:
            params[34] = params[32]

        if params[38] > params[36]:
            params[38] = params[36]

        delta_y = (
            (self.y_deck_stiffeners[1] - self.y_deck_stiffeners[0])
            * 1000.0
            * 0.5
            * self.Hull_Dict["B"]
        )

        if params[34] > delta_y:
            params[34] = delta_y

        if params[38] > delta_y:
            params[38] = delta_y

        self.Deck_Long_Dict = [
            {
                "y": self.y_deck_girders,
                "h": params[32],
                "t": params[33],
                "w": params[34],
                "rot": 180,
                "dir": [1, 0, 0],
                "bit_TorC": params[35],
                "1D_element": False,
                "Class": "Deck_Longitudinal_Girder",
            },
            {
                "y": self.y_deck_stiffeners,
                "h": params[36],
                "t": params[37],
                "w": params[38],
                "rot": 180,
                "dir": [1, 0, 0],
                "bit_TorC": params[39],
                "1D_element": True,
                "Class": "Deck_Longitudinal_Stiffener",
            },
        ]

        # One hatch is centered in each of the three holds.
        self.bit_Hatch = params[116]
        self.x_LH = [
            (self.x_bulkheads[0] + self.x_bulkheads[1]) / 2.0,
            (self.x_bulkheads[1] + self.x_bulkheads[2]) / 2.0,
            (self.x_bulkheads[2] + self.x_bulkheads[3]) / 2.0,
        ]
        self.Deck_Lightening_Hole_Dict = [
            {
                "x1": [float(value) for value in self.x_LH],
                "x2": 0.0,
                "l": float(params[117] * params[0] * 1000.0),
                "h": float(params[118] * params[1] * 1000.0),
                "r": float(params[119]),
            }
        ]
        self.Deck_Hatch_Half_width = (
            params[116] * params[118] * self.Hull_Dict["B"] * 0.5 * 1000.0
        )

        # Transverse-bulkhead stiffeners.
        if params[42] > params[40]:
            params[42] = params[40]

        delta_z = (
            (self.z_trans_stiffeners[1] - self.z_trans_stiffeners[0])
            * 1000.0
            * self.Hull_Dict["D"]
        )

        if params[42] > delta_z:
            params[42] = delta_z

        self.BLKHD_Trans_Stiff_Dict = [
            {
                "z": self.z_trans_stiffeners,
                "h": params[40],
                "t": params[41],
                "w": params[42],
                "rot": 0,
                "dir": [0, 1, 0],
                "bit_TorC": params[43],
                "1D_element": True,
                "Class": "Transverse_Bulkhead_Transverse_Stiffener",
            }
        ]

        if params[46] > params[44]:
            params[46] = params[44]

        delta_y = (
            (self.y_vert_stiffeners[1] - self.y_vert_stiffeners[0])
            * 1000.0
            * 0.5
            * self.Hull_Dict["B"]
        )

        if params[46] > delta_y:
            params[46] = delta_y

        self.BLKHD_Vert_Stiff_Dict = [
            {
                "y": self.y_vert_stiffeners,
                "h": params[44],
                "t": params[45],
                "w": params[46],
                "rot": 0,
                "dir": [0, 0, 1],
                "bit_TorC": params[47],
                "1D_element": True,
                "Class": "Transverse_Bulkhead_Vertical_Stiffener",
            }
        ]

        # Longitudinal-bulkhead vertical and longitudinal stiffeners.
        if params[114] > params[112]:
            params[114] = params[112]

        delta_x = (
            (self.x_trans_frames[1] - self.x_trans_frames[0])
            * 1000.0
            * self.Hull_Dict["L_3h"]
        )

        if params[114] > delta_x:
            params[114] = delta_x

        self.LONG_BLKHD_Vert_Stiff_Dict = [
            {
                "x": self.x_trans_frames,
                "h": params[112],
                "t": params[113],
                "w": params[114],
                "rot": 90,
                "dir": [0, 0, 1],
                "bit_TorC": params[115],
                "1D_element": True,
                "Class": "Longitudinal_Bulkhead_Vertical_Stiffener",
            },
            {
                "x": self.x_trans_web_frames,
                "h": params[112],
                "t": params[113],
                "w": params[114],
                "rot": 90,
                "dir": [0, 0, 1],
                "bit_TorC": params[115],
                "1D_element": True,
                "Class": "Longitudinal_Bulkhead_Vertical_Stiffener",
            },
        ]

        if params[110] > params[108]:
            params[110] = params[108]

        delta_z = (
            (self.z_side_stiffeners[1] - self.z_side_stiffeners[0])
            * 1000.0
            * self.Hull_Dict["D"]
        )

        if params[110] > delta_z:
            params[110] = delta_z

        self.LONG_BLKHD_Long_Stiff_Dict = [
            {
                "z": self.z_side_stiffeners,
                "h": params[108],
                "t": params[109],
                "w": params[110],
                "rot": 90,
                "dir": [1, 0, 0],
                "bit_TorC": params[111],
                "1D_element": True,
                "Class": "Longitudinal_Bulkhead_Longitudinal_Stiffener",
            }
        ]

        # Major transverse web frames and floors.
        if params[50] > params[48]:
            params[50] = params[48]

        delta_x = (
            (self.x_trans_frames[1] - self.x_trans_frames[0])
            * 1000.0
            * self.Hull_Dict["L_3h"]
        )

        if params[50] > delta_x:
            params[50] = delta_x

        self.Web_Frame_Dict = {
            "x": self.x_trans_web_frames,
            "Side_Dict": {
                "h": params[48],
                "t": params[49],
                "w": params[50],
                "rot": -90,
                "dir": [0, 0, 1],
                "bit_TorC": params[51],
                "1D_element": False,
                "Class": "Vertical_Web_Frame",
            },
            "Bottom_Dict": {
                "t": params[52],
                "1D_element": False,
                "Class": "Transverse_Floor",
            },
        }

        # Intermediate side and bottom frames.
        if params[99] > params[97]:
            params[99] = params[97]

        if params[99] > delta_x:
            params[99] = delta_x

        self.Trans_Frame_Dict = {
            "x": self.x_trans_frames,
            "h": params[97],
            "t": params[98],
            "w": params[99],
            "rot": -90,
            "dir": [0, 0, 1],
            "bit_TorC": params[100],
            "1D_element": True,
            "Class": "Vertical_Side_Frame",
        }

        if params[95] > params[93]:
            params[95] = params[93]

        if params[95] > delta_x:
            params[95] = delta_x

        self.Trans_Frame_Bottom_Dict = {
            "x": self.x_trans_frames,
            "h": params[93],
            "t": params[94],
            "w": params[95],
            "rot": 90,
            "dir": [0, 1, 0],
            "bit_TorC": params[96],
            "1D_element": True,
            "Class": "Transverse_Bottom_Frame",
        }

        # Side-shell and inner-side-shell longitudinal stiffeners.
        if params[55] > params[53]:
            params[55] = params[53]

        delta_z = (
            (self.z_side_stiffeners[1] - self.z_side_stiffeners[0])
            * 1000.0
            * self.Hull_Dict["D"]
        )

        if params[55] > delta_z:
            params[55] = delta_z

        self.Inner_Side_Shell_Long_Stiff_Dict = [
            {
                "z": self.z_side_stiffeners,
                "h": params[53],
                "t": params[54],
                "w": params[55],
                "rot": 90,
                "dir": [1, 0, 0],
                "bit_TorC": params[56],
                "1D_element": True,
                "Class": "Inner_Side_Shell_Longitudinal_Stiffener",
            }
        ]
        self.Inner_Side_Shell_Vert_Stiff_Dict = [
            {
                "x": self.x_trans_frames,
                "h": params[97],
                "t": params[98],
                "w": params[99],
                "rot": 90,
                "dir": [0, 0, 1],
                "bit_TorC": params[100],
                "1D_element": True,
                "Class": "Inner_Side_Shell_Vertical_Stiffener",
            }
        ]
        self.Side_Shell_Stiff_Dict = {
            "z": self.z_side_stiffeners,
            "h": params[53],
            "t": params[54],
            "w": params[55],
            "rot": -90,
            "dir": [1, 0, 0],
            "bit_TorC": params[56],
            "1D_element": True,
            "Class": "Side_Shell_Longitudinal_Stiffener",
        }

        # Select exactly the same ship-class branch as the legacy one-hot row.
        self.Ship_Class = params[57:60]

        if self.Ship_Class[0] == 1:
            self._set_tanker_parameters(params)
        elif self.Ship_Class[1] == 1:
            self._set_container_parameters(params)
        elif self.Ship_Class[2] == 1:
            self._set_bulk_carrier_parameters(params)

    def _web_frame_vertices(self):
        """Return the lower and upper bracket vertices in model metres."""

        fw_vertices = np.zeros((len(self.x_trans_web_frames), 3))
        fw_vertices[:, 0] = self.x_trans_web_frames * self.Hull_Dict["L_3h"]
        fw_vertices[:, 1] = (
            self.Hull_Dict["B"] / 2
            - self.Web_Frame_Dict["Side_Dict"]["h"] / 1000.0
        )
        fw_vertices[:, 2] = self.Struct_Params["Db"] / 1000.0

        dw_vertices = np.zeros((len(self.x_trans_web_frames), 3))
        dw_vertices[:, 0] = self.x_trans_web_frames * self.Hull_Dict["L_3h"]
        dw_vertices[:, 1] = fw_vertices[:, 1]

        return fw_vertices, dw_vertices

    def _snap_lower_bracket(self, params, fw_vertices):
        """Snap the lower bracket ends to bottom and side stiffeners."""

        y_struct = np.concatenate(
            (self.y_bottom_stiffeners, self.y_bottom_girders)
        )
        y_struct.sort()
        y_struct = y_struct * 1000.0 * 0.5 * self.Hull_Dict["B"]
        y_L1_end = fw_vertices[0, 1] * 1000.0 - params[60]

        try:
            idx_stiff = np.where(y_struct <= y_L1_end)[0][-1]
        except IndexError:
            idx_stiff = 0

        params[60] = fw_vertices[0, 1] * 1000.0 - y_struct[idx_stiff]

        z_struct = self.z_side_stiffeners * 1000.0 * self.Hull_Dict["D"]
        z_L2_end = fw_vertices[0, 2] * 1000.0 + params[61]

        try:
            idx_stiff = np.where(z_struct >= z_L2_end)[0][0]
        except IndexError:
            idx_stiff = len(z_struct) - 1

        params[61] = z_struct[idx_stiff] - fw_vertices[0, 2] * 1000.0

    def _snap_upper_bracket(self, params, dw_vertices):
        """Snap the upper bracket ends, respecting an active hatch opening."""

        y_struct = np.concatenate(
            (self.y_deck_stiffeners, self.y_deck_girders)
        )
        y_struct = y_struct * 1000.0 * 0.5 * self.Hull_Dict["B"]
        y_struct.sort()
        y_L1_end = dw_vertices[0, 1] * 1000.0 - params[63]

        try:
            idx_stiff = np.where(y_struct <= y_L1_end)[0][-1]
        except IndexError:
            idx_stiff = 0

        if y_struct[idx_stiff] < self.Deck_Hatch_Half_width:
            idx_stiff = np.where(
                y_struct >= self.Deck_Hatch_Half_width
            )[0][0]

        params[63] = dw_vertices[0, 1] * 1000.0 - y_struct[idx_stiff]

        z_struct = self.z_side_stiffeners * 1000.0 * self.Hull_Dict["D"]
        z_L2_end = dw_vertices[0, 2] * 1000.0 - params[64]

        try:
            idx_stiff = np.where(z_struct <= z_L2_end)[0][-1]
        except IndexError:
            idx_stiff = 0

        params[64] = dw_vertices[0, 2] * 1000.0 - z_struct[idx_stiff]

    def _set_tanker_parameters(self, params):
        """Create tanker bracket dictionaries and preserve snapped lengths."""

        fw_vertices, dw_vertices = self._web_frame_vertices()
        dw_vertices[:, 2] = (
            self.Hull_Dict["D"] - self.Deck_Beam_Dict[0]["h"] / 1000.0
        )

        self._snap_lower_bracket(params, fw_vertices)
        self._snap_upper_bracket(params, dw_vertices)

        self.Bracket_Dict = [
            {
                "Vertex": fw_vertices,
                "L1": np.array([0, -params[60], 0]),
                "L2": np.array([0, 0, params[61]]),
                "t": params[62],
                "num_sides": 3,
                "Class": "Bracket_Floor_Web",
            },
            {
                "Vertex": dw_vertices,
                "L1": np.array([0, -params[63], 0]),
                "L2": np.array([0, 0, -params[64]]),
                "t": params[65],
                "num_sides": 3,
                "Class": "Bracket_Deck_Web",
            },
        ]

    def _set_container_parameters(self, params):
        """Create container step, bracket, panel, and deck dictionaries."""

        fw_vertices, _ = self._web_frame_vertices()

        # The container step width snaps to a bottom longitudinal member.
        L1_brack = params[67]
        y_struct = np.concatenate(
            (self.y_bottom_stiffeners, self.y_bottom_girders)
        )
        y_struct.sort()
        y_struct = y_struct * 1000.0 * 0.5 * self.Hull_Dict["B"]
        y_L1_end = fw_vertices[0, 1] * 1000.0 - L1_brack

        try:
            idx_stiff = np.where(y_struct <= y_L1_end)[0][-1]
        except IndexError:
            idx_stiff = 0

        L1_brack = fw_vertices[0, 1] * 1000.0 - y_struct[idx_stiff]
        params[67] = L1_brack

        # The container step height snaps to a side-shell longitudinal.
        L2_brack = params[66]
        z_struct = self.z_side_stiffeners * 1000.0 * self.Hull_Dict["D"]
        z_L2_end = fw_vertices[0, 2] * 1000.0 + L2_brack

        try:
            idx_stiff = np.where(z_struct >= z_L2_end)[0][0]
        except IndexError:
            idx_stiff = len(z_struct) - 1

        L2_brack = z_struct[idx_stiff] - fw_vertices[0, 2] * 1000.0
        params[66] = L2_brack

        L1_pannel = params[67] + self.Web_Frame_Dict["Side_Dict"]["h"]
        L2_pannel = params[66] + self.Struct_Params["Db"]

        self.Bracket_Dict = [
            {
                "Vertex": fw_vertices,
                "L1": np.array([0, -L1_brack, 0]),
                "L2": np.array([0, 0, L2_brack]),
                "t": self.Web_Frame_Dict["Side_Dict"]["t"],
                "num_sides": 4,
                "Class": "Bracket_Floor_Web",
            }
        ]

        y_pos = (
            self.Hull_Dict["B"] / 2
            - self.Web_Frame_Dict["Side_Dict"]["h"] / 1000.0
            - params[67] / 1000.0
        )
        z_pos = self.Struct_Params["Db"] / 1000.0 + params[66] / 1000.0
        start_horiz = np.array([0, y_pos, z_pos])
        start_vert = np.array([0, y_pos, 0])

        if params[73] > params[71]:
            params[73] = params[71]

        if params[78] > params[76]:
            params[78] = params[76]

        delta_s = L1_pannel / (params[70] + 1)

        if params[73] > delta_s:
            params[73] = delta_s

        delta_s = L2_pannel / (params[75] + 1)

        if params[78] > delta_s:
            params[78] = delta_s

        self.ContainerShip_Pannel_Dict = [
            {
                "t_pan": params[68],
                "start": start_horiz,
                "L1": L1_pannel,
                "L2": 0,
                "num_stiffeners": params[70],
                "h": params[71],
                "t": params[72],
                "w": params[73],
                "rot": 90,
                "bit_TorC": params[74],
                "1D_element": True,
                "Class": "Stringer_Deck_Pannel",
            },
            {
                "t_pan": params[69],
                "start": start_vert,
                "L1": 0,
                "L2": L2_pannel,
                "num_stiffeners": params[75],
                "h": params[76],
                "t": params[77],
                "w": params[78],
                "rot": 90,
                "bit_TorC": params[79],
                "1D_element": True,
                "Class": "Stringer_Deck_Side_Pannel",
            },
        ]
        self.Container_Deck_Dict = {
            "z": np.array([1.0]),
            "h": params[48],
            "t": params[9],
            "w": 0,
            "rot": -90,
            "dir": [1, 0, 0],
            "bit_TorC": 0,
            "1D_element": False,
            "Class": "Deck",
        }

    def _set_bulk_carrier_parameters(self, params):
        """Create bulk-carrier hopper, bracket, and stiffener dictionaries."""

        fw_vertices, dw_vertices = self._web_frame_vertices()
        dw_vertices[:, 2] = self.Hull_Dict["D"]

        self._snap_lower_bracket(params, fw_vertices)

        # The legacy bulk-carrier branch uses the nearest lower side stiffener
        # when the requested lower-bracket height extends past all stiffeners.
        z_struct = self.z_side_stiffeners * 1000.0 * self.Hull_Dict["D"]
        requested_z_end = fw_vertices[0, 2] * 1000.0 + params[61]

        if not np.any(z_struct >= requested_z_end):
            idx_stiff = np.where(z_struct <= requested_z_end)[0][-1]
            params[61] = z_struct[idx_stiff] - fw_vertices[0, 2] * 1000.0

        self._snap_upper_bracket(params, dw_vertices)

        start_fw = np.array(
            [
                0,
                self.Hull_Dict["B"] / 2
                - self.Web_Frame_Dict["Side_Dict"]["h"] / 1000.0
                - params[60] / 1000.0,
                self.Struct_Params["Db"] / 1000.0,
            ]
        )
        start_dw = np.array(
            [
                0,
                self.Hull_Dict["B"] / 2
                - self.Web_Frame_Dict["Side_Dict"]["h"] / 1000.0
                - params[63] / 1000.0,
                self.Hull_Dict["D"],
            ]
        )

        self.Bracket_Dict = [
            {
                "Vertex": fw_vertices,
                "L1": np.array([0, -params[60], 0]),
                "L2": np.array([0, 0, params[61]]),
                "t": params[62],
                "num_sides": 3,
                "Class": "Bracket_Floor_Web",
            },
            {
                "Vertex": dw_vertices,
                "L1": np.array([0, -params[63], 0]),
                "L2": np.array([0, 0, -params[64]]),
                "t": params[65],
                "num_sides": 3,
                "Class": "Bracket_Deck_Web",
            },
        ]

        if params[85] > params[83]:
            params[85] = params[83]

        if params[90] > params[88]:
            params[90] = params[88]

        delta_s = np.sqrt(params[60] ** 2 + params[61] ** 2) / (
            params[82] + 1
        )

        if params[85] > delta_s:
            params[85] = delta_s

        delta_s = np.sqrt(params[63] ** 2 + params[64] ** 2) / (
            params[87] + 1
        )

        if params[90] > delta_s:
            params[90] = delta_s

        self.BulkCarrier_Hopper_Dict = [
            {
                "t_pan": params[80],
                "start": start_fw,
                "L1": params[60],
                "L2": params[61],
                "num_stiffeners": params[82],
                "h": params[83],
                "t": params[84],
                "w": params[85],
                "rot": 90,
                "bit_TorC": params[86],
                "1D_element": True,
                "Class": "Hopper_Pannel",
            },
            {
                "t_pan": params[81],
                "start": start_dw,
                "L1": params[63],
                "L2": -params[64],
                "num_stiffeners": params[87],
                "h": params[88],
                "t": params[89],
                "w": params[90],
                "rot": -90,
                "bit_TorC": params[91],
                "1D_element": True,
                "Class": "Hopper_Pannel",
            },
        ]

    def parameters_by_name(self):
        """Return the corrected parameter row keyed by the CSV schema names."""

        return dict(zip(self.parameter_schema["name"], self.params, strict=True))

    @classmethod
    def clean_struct_params(cls, params):
        """Normalize sampled parameter rows in place using the Rhino rules.

        ``params`` is the same two-dimensional NumPy array accepted by
        ``Structure_3H.clean_Struct_Params``.  The original object is returned
        after mutation for drop-in use in sampling pipelines.
        """

        for i in range(len(params)):
            params[i, cls.param_idx_bit] = params[i, cls.param_idx_bit] >= 0.5
            params[i, cls.param_idx_int] = np.int32(
                params[i, cls.param_idx_int] + 0.5
            )
            params[i, cls.param_idx_plate_thicknesses] = np.int32(
                params[i, cls.param_idx_plate_thicknesses] + 0.5
            )

            ship_class = np.argmax(params[i, cls.param_idx_cat])
            params[i, 57:60] = 0
            params[i, 57 + ship_class] = 1

            lower_limits = cls.parameter_schema["LL"].to_numpy()
            upper_limits = cls.parameter_schema["UL"].to_numpy()
            strategic_lower = cls.parameter_schema["Strategic_LL"].to_numpy()
            strategic_upper = cls.parameter_schema["Strategic_UL"].to_numpy()

            if params[i, 6] < 1000.0 * params[i, 4]:
                params[i, 4] = params[i, 6] / 1000.0

            if params[i, 57] == 1:
                params[i, cls.param_idx_bulkcarrier] = 0
                params[i, cls.param_idx_container] = 0

                for j in cls.param_idx_brackets:
                    params[i, j] = np.clip(
                        params[i, j], lower_limits[j], upper_limits[j]
                    )

            elif params[i, 58] == 1:
                params[i, cls.param_idx_bulkcarrier] = 0
                params[i, cls.param_idx_brackets] = 0
                params[i, 66] = np.clip(
                    params[i, 66], lower_limits[66], upper_limits[66]
                )
                params[i, 67] = np.clip(
                    params[i, 67], lower_limits[67], upper_limits[67]
                )

            elif params[i, 59] == 1:
                params[i, cls.param_idx_container] = 0

                for j in cls.param_idx_brackets:
                    params[i, j] = np.clip(
                        params[i, j], strategic_lower[j], strategic_upper[j]
                    )

            # Every GenISys bulk carrier now includes an inner-side shell.
            # Ten millimetres was already the legacy activation threshold, so
            # it is retained as the minimum modeled thickness instead of using
            # values below it as an absent-member sentinel.
            params[i, 12] = max(10.0, params[i, 12])
            params[i, 50] = 0

            if params[i, 105] <= 0.5:
                params[i, 105] = 0
                params[i, 106:116] = 0.0
            elif params[i, 106] <= 0.5:
                params[i, 105:116] = 0.0
            else:
                params[i, 105] = 1

            if (
                params[i, 116] <= 0.5
                or params[i, 117] < 1e-2
                or params[i, 118] < 1e-2
            ):
                params[i, 116] = 0
                params[i, 117:120] = 0.0
            else:
                params[i, 116] = 1

        return params

    # Compatibility alias for code using the historical method spelling.
    clean_Struct_Params = clean_struct_params


# Shorter compatibility name for incremental ports from ``Structure_3H``.
Structure3HParameters = ParametricStructureDefinition


__all__ = [
    "PARAMETER_SCHEMA_PATH",
    "ParametricStructureDefinition",
    "Structure3HParameters",
]
