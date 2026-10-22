"""Stochastic equation-based repair tools for GenISys V1.1.

The package deliberately wraps the existing CadQuery generator and complete
GenISys evaluator.  It does not contain a second definition of ship
feasibility; repaired vectors are accepted only after the normal V1.1
constraint path has evaluated them.
"""

from .configuration import RepairConfiguration
from .parameters import ParameterRegistry


__all__ = [
    "ParameterRegistry",
    "RepairConfiguration",
]
