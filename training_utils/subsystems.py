"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
from __future__ import annotations


from collections import OrderedDict


import numpy as np


PARAMETERS_BY_SUBSYSTEM = OrderedDict(
    [
        (
            "Hull_Principal",
            ("LOA", "Lb", "Ls", "Bd", "Dd", "Bs", "WL"),
        ),
        ("Hull_Midbody", ("Rc", "Rk")),
        (
            "Hull_Bow",
            (
                "Bow_A", "Bow_B", "Bk", "Kappa_Bow", "Delta_Bow_A",
                "Delta_Bow_B", "Drift_A", "Drift_B", "Drift_C",
            ),
        ),
        (
            "Hull_Stern",
            (
                "bit_EP_S", "Trans_A", "SK", "Kappa_Stern",
                "Delta_Stern_A", "Delta_Stern_B", "Beta_Trans",
                "Bc_Trans", "Rc_Trans", "Rk_Trans",
            ),
        ),
        (
            "Hull_Bulbous_Bow",
            ("bit_BB", "Lbb", "Hbb", "Bbb", "Lbbm", "Rbb"),
        ),
        (
            "Hull_Stern_Bulb",
            ("bit_SB", "Kappa_SB", "Lsb", "HSBOA", "Hsb", "Bsb", "Lsbm", "Rsb"),
        ),
        ("Powering", ("Draft_frac", "Fn_ship", "Voyage_Range")),
        (
            "Compartment_Arrangement",
            (
                "frac_CrashBlkhd", "frac_AftPeakBlkhd", "frac_lazarette_L",
                "frac_engineRoom_L", "frac_fuel_tank_L", "Num_Cargo_Holds",
            ),
        ),
        (
            "Bottom_Shell_and_Double_Bottom",
            ("Db", "bottom shell thickness", "Inner Bottom Thickness"),
        ),
        (
            "Structure_Longitudinal_Bottom",
            (
                "num bottom girders", "num intermediate bottom stiffeners",
                "bottom girder thickness", "intermediate bottom stiffener h",
                "intermediate bottom stiffener t", "intermediate bottom stiffener w",
                "intermediate bottom stiffener torc",
            ),
        ),
        (
            "Structure_Transverse_Bottom",
            (
                "floor t", "intermediate bottom frame h", "intermediate bottom frame t",
                "intermediate bottom frame w", "intermediate bottom frame torc",
            ),
        ),
        ("Side_Shell", ("side shell thickness", "Inner Side Shell Thickness")),
        (
            "Structure_Longitudinal_Side",
            (
                "Shear Strake thickness", "Shear Strake Height",
                "num side shell stiffeners", "side shell Long stiff h",
                "side shell long stiff t", "side shell long stiff w",
                "side shell long stiff torc",
            ),
        ),
        (
            "Structure_Transverse_Side",
            (
                "web frames per hold", "web frame h", "web frame t", "web frame torc",
                "num intermediate trans frames", "intermediate side frame h",
                "intermediate side frame t", "intermediate side frame w",
                "intermediate side frame torc",
            ),
        ),
        ("Top_Deck_Plating", ("Top Deck Thickness",)),
        (
            "Structure_Longitudinal_Deck",
            (
                "deck girder h", "deck girder t", "deck girder w", "deck girder torc",
                "deck stiff h", "deck stiff t", "deck stiff w", "deck stiff torc",
            ),
        ),
        (
            "Structure_Transverse_Deck",
            (
                "trans deck beam h", "trans deck beam t", "trans deck beam w",
                "trans deck beam torc", "intermediate deck frame h",
                "intermediate deck frame t", "intermediate deck frame w",
                "intermediatedeck frame torc",
            ),
        ),
        (
            "Bulkhead_Structure",
            (
                "Trans Bulkhead Thickness", "num vertical bulkhead stiffeners",
                "blkhd vert stiff h", "blkhd vert stiff t", "blkhd vert stiff w",
                "blkhd vert stiff torc", "num transverse bulkhead stiffeners",
                "blkhd trans stiff h", "blkhd trans stiff t", "blkhd trans stiff w",
                "blkhd trans stiff torc",
            ),
        ),
        (
            "Hopper_Structure",
            (
                "floor web bracket L1", "floor web bracket L2", "floor web bracket t",
                "deck web bracket L1", "deck web bracket L2", "deck web bracke t",
                "bulk car bottom hopper t", "bulk car top hopper t",
                "num bulk car bottom hopper stiff", "bulk car bottom hopper stiff h",
                "bulk car bottom hopper stiff t", "bulk car bottom hopper stiff w",
                "bulk car bottom hopper stiff torc", "num bulk car top hopper stiff",
                "bulk car top hopper stiff h", "bulk car top hopper stiff t",
                "bulk car top hopper stiff w", "bulk car top hopper stiff torc",
            ),
        ),
        ("Hatch_Structure", ("Hatch Opening L", "Hatch Opening W", "Hatch Opening R")),
    ]
)


SUBSYSTEM_ORDER = tuple(PARAMETERS_BY_SUBSYSTEM)


LOCAL_LABEL_COLUMN = {
    name: f"Feasible::Subsystem::{name}" for name in SUBSYSTEM_ORDER
}


SYSTEM_LABEL_COLUMN = "Feasible::System_Level"


GLOBAL_LABEL_COLUMN = "Feasible::Global"


LEGACY_MODEL_TO_SUBSYSTEM = {
    "Hull_Principal": "Hull_Principal",
    "Hull_Midbody": "Hull_Midbody",
    "Hull_Bow": "Hull_Bow",
    "Hull_Stern": "Hull_Stern",
    "Hull_Bulbous_Bow": "Hull_Bulbous_Bow",
    "Hull_Stern_Bulb": "Hull_Stern_Bulb",
    "Powering": "Powering",
    "Compartment_Arrangement": "Compartment_Arrangement",
    "Structure_Longitudinal_Bottom": "Structure_Longitudinal_Bottom",
    "Structure_Transverse_Bottom": "Structure_Transverse_Bottom",
    "Structure_Longitudinal_Side": "Structure_Longitudinal_Side",
    "Structure_Transverse_Side": "Structure_Transverse_Side",
    "Structure_Longitudinal_Deck": "Structure_Longitudinal_Deck",
    "Structure_Transverse_Deck": "Structure_Transverse_Deck",
    "Structure_Vertical_Bulkhead": "Bulkhead_Structure",
    "Structure_Transverse_Bulkhead": "Bulkhead_Structure",
    "Structure_Hopper": "Hopper_Structure",
    "Structure_Longitudinal_Hopper": "Hopper_Structure",
    "Structure_Hatch": "Hatch_Structure",
}


SHELL_LABEL_TO_SUBSYSTEM = {
    "Structure_Bottom": "Bottom_Shell_and_Double_Bottom",
    "Structure_Side": "Side_Shell",
    "Structure_Top_Deck": "Top_Deck_Plating",
    "Structure_Bulkhead": "Bulkhead_Structure",
}


def parameter_subsets(model_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Return the frozen subsystem coordinate positions."""

    model_index = {name: index for index, name in enumerate(model_names)}
    declared = [name for names in PARAMETERS_BY_SUBSYSTEM.values() for name in names]

    missing = sorted(set(model_names) - set(declared))
    extra = sorted(set(declared) - set(model_names))
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    if missing or extra or duplicates:
        raise ValueError(
            "The 20-subsystem parameter map does not match the 133-coordinate "
            f"schema: missing={missing}, extra={extra}, duplicates={duplicates}."
        )

    return {
        subsystem: np.asarray([model_index[name] for name in names], dtype=int)
        for subsystem, names in PARAMETERS_BY_SUBSYSTEM.items()
    }


__all__ = [
    "GLOBAL_LABEL_COLUMN",
    "LEGACY_MODEL_TO_SUBSYSTEM",
    "LOCAL_LABEL_COLUMN",
    "PARAMETERS_BY_SUBSYSTEM",
    "SHELL_LABEL_TO_SUBSYSTEM",
    "SUBSYSTEM_ORDER",
    "SYSTEM_LABEL_COLUMN",
    "parameter_subsets",
]


