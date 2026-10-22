"""Canonical identities for V1.1 constraint counting.

The GenISys long-form result files deliberately retain historical and
member-specific names.  A small number of those names are compatibility
aliases for the same physical equation/member pair.  Keeping both rows is
useful traceability evidence, but counting both rows would overstate the
number of independent constraints and failures.

This module therefore owns a second, count-only identity.  It never deletes
or rewrites a long-form result row.  Callers may collapse a copy of the table
before building feasibility totals or wide ML tables.  When aliased rows ever
disagree, the canonical result uses logical ``all``: one failed spelling makes
the shared physical constraint fail.  That conservative rule prevents an
older passing alias from hiding a newer failed evaluation.
"""

from __future__ import annotations

import pandas as pd


# Only explicitly reviewed aliases belong here.  Similar-looking rules are
# intentionally absent: shared geometry, a common citation, or equal numeric
# values in one design do not by themselves establish an alias.
_CANONICAL_ALIASES_BY_NAME = {
    # ABS MVR 3-2-4/3.1.1(a): the legacy and audit-qualified names use the
    # same rounded threshold and twice the ordinary girder plate thickness.
    'Min Center Bottom Girder Thickness': (
        'CANONICAL::MVR_3-2-4_3.1.1(a)::center_girder_plate_thickness'
    ),
    'MVR center-girder plate thickness amidships': (
        'CANONICAL::MVR_3-2-4_3.1.1(a)::center_girder_plate_thickness'
    ),

    # ABS MVR 3-2-4/5.1-b: both names evaluate the same generated solid-floor
    # plate against the same half-millimetre-rounded minimum.
    'Min Floor Thickness': (
        'CANONICAL::MVR_3-2-4_5.1-b::solid_floor_plate_thickness'
    ),
    'MVR solid-floor plate thickness': (
        'CANONICAL::MVR_3-2-4_5.1-b::solid_floor_plate_thickness'
    ),

    # ABS MVR 3-2-5/1.7: the current legacy and audit paths call the same
    # frame-spacing method with the same evaluated design inputs.
    'Max Frame Spacing': (
        'CANONICAL::MVR_3-2-5_1.7::transverse_frame_spacing'
    ),
    'MVR maximum transverse-frame spacing': (
        'CANONICAL::MVR_3-2-5_1.7::transverse_frame_spacing'
    ),

    # Compatibility aliases retained alongside the clearer atomic V1.1
    # plate names.  Logical-all aggregation remains conservative if an old
    # and new implementation ever diverge.
    'Min Inner Bottom Grab-Service Thickness': (
        'CANONICAL::MVR_3-2-4_9.5::inner_bottom_grab_thickness'
    ),
    'Inner bottom plating - grab loading': (
        'CANONICAL::MVR_3-2-4_9.5::inner_bottom_grab_thickness'
    ),
    'Min Lower Hopper Grab-Service Thickness': (
        'CANONICAL::MVR_3-2-4_9.5::lower_hopper_grab_thickness'
    ),
    'Lower hopper plating - grab loading': (
        'CANONICAL::MVR_3-2-4_9.5::lower_hopper_grab_thickness'
    ),

    # These two active rows are an exact runtime pair.  Both call the same
    # ABS 3-2-8/3.3 transverse-bulkhead composite-support calculation.
    'Max Allowable Uniform Deck Loading': (
        'CANONICAL::MVR_3-2-8_3.3::transverse_bulkhead_uniform_load'
    ),
    'Max Allowable Uniform Bottom Loading': (
        'CANONICAL::MVR_3-2-8_3.3::transverse_bulkhead_uniform_load'
    ),

    # Audit-qualified/atomic names for the same two plate comparisons.  The
    # long-form tables keep both descriptions for rule-audit traceability.
    'MVR inner-bottom optimized-cargo pressure thickness': (
        'CANONICAL::MVR_3-2-4_9.2::inner_bottom_cargo_pressure'
    ),
    'Inner bottom plating - cargo pressure': (
        'CANONICAL::MVR_3-2-4_9.2::inner_bottom_cargo_pressure'
    ),
    'MVR general strength-deck minimum thickness': (
        'CANONICAL::MVR_3-2-3_5.1-b::strength_deck_absolute_minimum'
    ),
    'Strength deck plating - absolute minimum': (
        'CANONICAL::MVR_3-2-3_5.1-b::strength_deck_absolute_minimum'
    ),

    # CRF-003/CRF-009: the two corrected legacy Section 7 rows use the same
    # generated spacings, freeboard-deck head, coefficients, offered section
    # properties, and residual direction as their clearer atomic V1.1 names.
    # Keep both descriptions in long-form files and count each physical
    # member/equation pair once.
    'Min Relative Deck Stringer Section Modulus': (
        'CANONICAL::MVR_3-2-7_3.1::strength_deck_longitudinal_SM'
    ),
    'Strength-deck longitudinal section modulus': (
        'CANONICAL::MVR_3-2-7_3.1::strength_deck_longitudinal_SM'
    ),
    'Min Relative Deck Frame Section Modulus': (
        'CANONICAL::MVR_3-2-7_3.1::ordinary_strength_deck_beam_SM'
    ),
    'Ordinary strength-deck transverse-beam section modulus': (
        'CANONICAL::MVR_3-2-7_3.1::ordinary_strength_deck_beam_SM'
    ),
}


def canonical_constraint_keys(
    constraints: pd.DataFrame,
    *,
    audited: bool = False,
) -> pd.Series:
    """Return one stable, count-only identity for every result row.

    ``audited=True`` retains the established ``Audit ID::Name`` default.
    The expanded legacy-format table instead uses ``Requirement::Name``.
    Explicit aliases replace either default with a shared key.
    """

    names = constraints['Name'].astype(str)

    if audited and 'Audit ID' in constraints.columns:
        default = constraints['Audit ID'].astype(str) + '::' + names
    else:
        default = constraints['Requirement'].astype(str) + '::' + names

    aliases = names.map(_CANONICAL_ALIASES_BY_NAME)

    return aliases.fillna(default).rename('Canonical Constraint Key')


def collapse_constraint_aliases(
    constraints: pd.DataFrame,
    *,
    audited: bool = False,
) -> pd.DataFrame:
    """Collapse evaluated aliases in a copy used only for aggregation.

    Rows with a blank ``Constraint Bit`` are metrics or incomplete results and
    do not enter feasibility counting. Not-applicable constraints carry a True
    bit and therefore enter as passes. ``Index`` is included whenever a
    compiled multi-design table supplies it, so an identity is collapsed only
    within one system design. The returned representative retains the first
    row's descriptive columns and exposes the count-only identity through
    ``Canonical Constraint Key``.
    """

    if constraints.empty:
        empty = constraints.copy()
        empty['Canonical Constraint Key'] = pd.Series(dtype=object)
        empty['Alias Row Count'] = pd.Series(dtype=int)
        return empty

    evaluated = constraints.loc[
        constraints['Constraint Bit'].notna()
    ].copy()

    if evaluated.empty:
        evaluated['Canonical Constraint Key'] = pd.Series(dtype=object)
        evaluated['Alias Row Count'] = pd.Series(dtype=int)
        return evaluated

    evaluated['Canonical Constraint Key'] = canonical_constraint_keys(
        evaluated,
        audited=audited,
    )
    # CSV round trips may present Booleans as ``True``/``False`` strings or
    # as numeric 1/0 values.  Normalize all established encodings before
    # applying the conservative logical-all alias rule.
    normalized_bits = (
        evaluated['Constraint Bit'].astype(str).str.strip().str.lower()
    )
    evaluated['_Constraint_Bit_Bool'] = normalized_bits.isin(
        {'true', '1', '1.0'}
    )

    group_columns = ['Canonical Constraint Key']

    if 'Index' in evaluated.columns:
        group_columns.insert(0, 'Index')

    # Keep the first complete descriptive row, then replace only the Boolean
    # with the conservative shared result and record how many aliases it owns.
    first_indices = evaluated.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ).head(1).index
    collapsed = evaluated.loc[first_indices].copy()
    grouped_bits = evaluated.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )['_Constraint_Bit_Bool'].all()
    grouped_sizes = evaluated.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ).size()

    collapsed_index = pd.MultiIndex.from_frame(collapsed[group_columns])

    if len(group_columns) == 1:
        # A one-level Index avoids pandas' tuple/scalar ambiguity when only a
        # single design table is being collapsed.
        collapsed_index = pd.Index(
            collapsed[group_columns[0]],
            name=group_columns[0],
        )

    collapsed['Constraint Bit'] = grouped_bits.reindex(
        collapsed_index
    ).to_numpy(dtype=bool)
    collapsed['Alias Row Count'] = grouped_sizes.reindex(
        collapsed_index
    ).to_numpy(dtype=int)
    collapsed = collapsed.drop(columns=['_Constraint_Bit_Bool'])

    return collapsed.reset_index(drop=True)


def canonical_constraint_bits(
    constraints: pd.DataFrame,
    *,
    audited: bool = False,
) -> pd.Series:
    """Return the Boolean series used for unique constraint totals."""

    collapsed = collapse_constraint_aliases(
        constraints,
        audited=audited,
    )

    if collapsed.empty:
        return pd.Series(dtype=bool, name='Constraint Bit')

    return collapsed['Constraint Bit'].astype(bool).reset_index(drop=True)


__all__ = [
    'canonical_constraint_bits',
    'canonical_constraint_keys',
    'collapse_constraint_aliases',
]
