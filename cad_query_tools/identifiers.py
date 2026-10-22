"""Deterministic logical identifiers for generated structural elements."""

from __future__ import annotations

import re
from collections import defaultdict


def _slug(value: object) -> str:
    """Convert a user or class label into a stable identifier component."""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return text.strip("-") or "unnamed"


class IdentifierFactory:
    """Issue repeatable logical IDs in element insertion order.

    ``make_rhino`` is the complete ship builder's compatibility path. It keeps
    the coordinate-derived semantic root constructed by the Rhino script.
    ``make`` retains the earlier generic convention for standalone primitives
    that do not provide enough context to construct that semantic root::

        {ship_id}:{element_type}:x-{metres}:{side}:{sequence}

    The same inputs and build order produce byte-for-byte identical identifiers
    through either path.
    """

    def __init__(self, ship_id: str):
        self.ship_id = _slug(ship_id)
        self._counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._rhino_root_counts: defaultdict[str, int] = defaultdict(int)

    def make(
        self,
        element_type: str,
        x_location: float,
        side: str = "starboard-half",
    ) -> tuple[str, int]:
        """Return the next ID and its zero-based sequence number."""

        element_slug = _slug(element_type)
        side_slug = _slug(side)
        key = (element_slug, side_slug)
        sequence = self._counts[key]
        self._counts[key] += 1

        object_id = (
            f"{self.ship_id}:{element_slug}:"
            f"x-{float(x_location):.6f}:{side_slug}:{sequence:05d}"
        )

        return object_id, sequence

    def make_rhino(self, object_id_root: str) -> tuple[str, int]:
        """Return a unique ID based on the legacy Rhino semantic root.

        Rhino constructs semantic names such as ``Deck_Z_12.0`` and
        ``Side_Shell_frame_X_8.0``.  Its stiffener helper can reuse one name
        for several disconnected pieces, however, which prevents a reliable
        one-to-one lookup after CAD export.  Preserve the exact root for the
        first piece and use the legacy ``_part_n`` vocabulary only when that
        root is encountered again.
        """

        root = str(object_id_root)
        sequence = self._rhino_root_counts[root]
        self._rhino_root_counts[root] += 1

        if sequence == 0:
            return root, sequence

        return f"{root}_part_{sequence}", sequence
