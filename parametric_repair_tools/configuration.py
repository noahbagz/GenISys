"""Configuration and immutable numerical policy for parametric repair.

The random-margin intervals match the maintained Ship_Structures released-
dataset driver.  Randomness changes how far a repair moves onto the passing
side; it never invents a constraint-to-parameter relationship.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "Ship_Systems_Designs_v1p1"
    / "Laptop_Pipeline_Test_20x20_Reduced_0_399"
)
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT / "Ship_Systems_Designs_v1p1" / "repair_experiments"
)


@dataclass(frozen=True)
class RepairConfiguration:
    """Hold every setting needed to reproduce one repair experiment."""

    source_root: str = str(DEFAULT_SOURCE_ROOT)
    experiment_root: str = str(DEFAULT_EXPERIMENT_ROOT)

    # The development seed is frozen before repair tuning.  A validation run
    # must use a separately saved batch and a different seed.
    batch_seed: int = 20_260_914
    validation_batch_seed: int = 20_261_003
    batch_size: int = 10

    # The population repair uses every completed, partially feasible source
    # below the completed-population median.  Each source receives two
    # independent stochastic repair-margin streams so it produces two
    # alternative one-pass designs without any repair/evaluation feedback.
    below_median_batch_seed: int = 20_260_915
    alternatives_per_below_median_source: int = 2

    # Every repair family uses the approved 25-50 percent passing-side margin.
    # One seeded draw is retained per design and rule family.
    direct_margin_range: tuple[float, float] = (0.25, 0.50)
    section_margin_range: tuple[float, float] = (0.25, 0.50)
    spacing_margin_range: tuple[float, float] = (0.25, 0.50)

    # Coupled system repairs retain a separately named step because their
    # parameter response is not a one-to-one threshold equation. The selected
    # step still uses the same approved 25-50 percent range.
    system_step_range: tuple[float, float] = (0.25, 0.50)

    # Development readiness is measured after one repair pass and one final
    # evaluation for each of the ten frozen source designs.
    development_feasible_target: int = 7

    # HDF-083 is a non-counted diagnostic saved during the single repaired-
    # design evaluation. It does not direct a second repair.
    run_flooding_diagnostic: bool = True
    flooding_step: int = 15
    flooding_permeabilities: tuple[float, ...] = (0.95,)
    save_mode: str = "reduced"

    # Explicit repair-domain limits.  Sampling bounds remain metadata only;
    # these wider limits protect mathematical/CAD domains during repair.
    repair_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "LOA": (80.0, 500.0),
            # Lb is a nondimensional bow-taper length, not the rule length.
            # The repair domain may extend below the original 0.075 random-
            # sampling bound because the user explicitly permits repaired
            # designs outside sampling bounds.  Keep a positive lower guard
            # so the bow equations never receive a zero-length taper; the
            # complete hull constraint set remains authoritative afterward.
            "Lb": (0.02, 0.25),
            "Bd": (0.091, 0.20),
            "Dd": (0.075, 0.20),
            "Draft_frac": (0.20, 0.92),
            "Fn_ship": (0.05, 0.35),
            "Voyage_Range": (100.0, 10_000.0),
            "frac_CrashBlkhd": (0.03, 0.12),
            "frac_AftPeakBlkhd": (0.02, 0.12),
            "frac_lazarette_L": (0.02, 0.12),
            "frac_engineRoom_L": (0.03, 0.30),
            "frac_fuel_tank_L": (0.005, 0.20),
            # The 189-coordinate interface has exactly fifteen cargo-fraction
            # fields, which is a hard representational upper limit.
            "Num_Cargo_Holds": (3.0, 15.0),
            "Db": (500.0, 8_000.0),
            "Hatch Opening L": (0.10, 0.45),
            "Hatch Opening W": (0.15, 0.90),
            "Hatch Opening R": (25.0, 2_000.0),
        }
    )

    def to_dict(self) -> dict:
        """Return a JSON-serializable form with stable path strings."""

        payload = asdict(self)
        payload["source_root"] = str(Path(self.source_root).resolve())
        payload["experiment_root"] = str(
            Path(self.experiment_root).resolve()
        )
        return payload

    def fingerprint(self) -> str:
        """Hash the exact settings used to generate repair candidates."""

        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_EXPERIMENT_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "PROJECT_ROOT",
    "RepairConfiguration",
]
