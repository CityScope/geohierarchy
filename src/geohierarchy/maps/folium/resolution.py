"""Zoom-resolution assignment for hierarchy levels.

Each level of a :class:`~geohierarchy.core.GeoHierarchy` is shown at exactly
one, level-specific range of Leaflet zoom levels (0-25 inclusive), so that at
any given zoom exactly one level's vector tiles are visible. This module
computes that assignment automatically from feature size (area/length), and
validates manually-specified ranges for gaps/overlaps.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

MIN_ZOOM = 0
MAX_ZOOM = 25


def assign_resolutions(
    hierarchy,
    levels: Optional[Iterable[str]] = None,
    method: str = "area",
) -> Dict[str, Tuple[int, int]]:
    """Automatically assign a contiguous, non-overlapping zoom range to each level.

    Levels are sorted by median geoweight (area or length, whichever the
    level already stores) descending -- coarsest (biggest features) first.
    Bin edges are placed at the midpoints between consecutive levels'
    ``log10(median size)`` values, then the full edge range is linearly
    rescaled onto integer zooms ``[0, 25]``: the coarsest level's range
    always starts at 0 and the finest level's range always ends at 25, so
    the returned ranges partition ``[0, 25]`` with no gaps or overlaps
    regardless of how many levels are involved.

    Args:
        hierarchy: A :class:`~geohierarchy.core.GeoHierarchy` instance.
        levels: Level names to assign. Defaults to every level in
            ``hierarchy.geometries``.
        method: Unused placeholder for future weighting methods; kept for
            API stability (geoweight column is always whatever
            ``hierarchy.geoweight_by[level]`` already is, "area" or
            "length").

    Returns:
        Mapping of level name to an inclusive ``(min_zoom, max_zoom)`` tuple.

    Raises:
        ValueError: If ``levels`` is empty or a named level is missing
            geometry.
    """
    level_names: List[str] = (
        list(levels) if levels is not None else list(hierarchy.geometries.keys())
    )
    if not level_names:
        raise ValueError("assign_resolutions requires at least one level")

    medians = _medians(hierarchy, level_names)
    return _bin_by_size(medians, MIN_ZOOM, MAX_ZOOM)


def _medians(hierarchy, level_names: List[str]) -> Dict[str, float]:
    medians: Dict[str, float] = {}
    for name in level_names:
        if name not in hierarchy.geometries:
            raise ValueError(f"Unknown level '{name}'")
        weight_col = hierarchy.geoweight_by[name]
        gdf = hierarchy.geometries[name]
        median = float(gdf[weight_col].median())
        medians[name] = median if median > 0 else 1e-12
    return medians


def _bin_by_size(
    medians: Dict[str, float], zoom_lo: int, zoom_hi: int
) -> Dict[str, Tuple[int, int]]:
    """Log-scale median-size binning of ``medians.keys()`` onto ``[zoom_lo, zoom_hi]``.

    Coarsest (largest median size) gets ``zoom_lo``; finest gets ``zoom_hi``.
    Shared by :func:`assign_resolutions` (binning onto the full [0, 25]) and
    the "fill around manual pins" logic in :meth:`HierarchyMap.resolve_zoom_ranges`
    (binning onto whatever contiguous sub-range of [0, 25] is left over).
    """
    ordered = sorted(medians, key=lambda n: medians[n], reverse=True)

    if len(ordered) == 1:
        return {ordered[0]: (zoom_lo, zoom_hi)}

    log_sizes = [math.log10(medians[n]) for n in ordered]

    # Bin edges: one before the first level, one between each consecutive
    # pair (at the midpoint), one after the last level.
    edges = [log_sizes[0] + (log_sizes[0] - log_sizes[1]) / 2]
    for i in range(len(log_sizes) - 1):
        edges.append((log_sizes[i] + log_sizes[i + 1]) / 2)
    edges.append(log_sizes[-1] - (log_sizes[-2] - log_sizes[-1]) / 2)

    # Rescale edges (descending, since size decreases as we go finer) onto
    # [zoom_lo, zoom_hi] ascending.
    lo, hi = edges[-1], edges[0]
    span = hi - lo
    if span <= 0:
        # Degenerate: all levels have (numerically) identical size. Split
        # the zoom range evenly instead of dividing by zero.
        n = len(ordered)
        result = {}
        bounds = [round(zoom_lo + (zoom_hi - zoom_lo) * i / n) for i in range(n + 1)]
        bounds[0], bounds[-1] = zoom_lo, zoom_hi
        for i, name in enumerate(ordered):
            result[name] = (
                bounds[i],
                bounds[i + 1] if i == n - 1 else bounds[i + 1] - 1,
            )
        result[ordered[-1]] = (result[ordered[-1]][0], zoom_hi)
        return result

    def to_zoom(edge_value: float) -> float:
        frac = (hi - edge_value) / span  # 0 at coarsest edge, 1 at finest edge
        return zoom_lo + frac * (zoom_hi - zoom_lo)

    raw_zooms = [to_zoom(e) for e in edges]
    raw_zooms[0] = zoom_lo
    raw_zooms[-1] = zoom_hi

    # Round to integers, then repair to guarantee strict contiguity:
    # boundary[i+1] (rounded) must equal boundary[i] (rounded) + 1 at the
    # min/max seams shared between adjacent levels.
    zoom_bounds = [round(z) for z in raw_zooms]
    for i in range(1, len(zoom_bounds)):
        if zoom_bounds[i] <= zoom_bounds[i - 1]:
            zoom_bounds[i] = zoom_bounds[i - 1] + 1
    zoom_bounds[-1] = max(zoom_bounds[-1], zoom_hi)
    zoom_bounds[0] = zoom_lo

    # Build contiguous ranges: level i covers [prev_end + 1, zoom_bounds[i+1]]
    result: Dict[str, Tuple[int, int]] = {}
    prev_end = zoom_lo - 1
    for i, name in enumerate(ordered):
        start = prev_end + 1 if i > 0 else zoom_lo
        end = zoom_bounds[i + 1] if i < len(ordered) - 1 else zoom_hi
        if end < start:
            end = start
        result[name] = (start, end)
        prev_end = end

    return result


def fill_around_manual(
    hierarchy,
    all_levels: List[str],
    manual: Dict[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """Auto-assign zoom ranges for the levels not in ``manual``, fit around it.

    ``manual`` entries are kept exactly as given. The remaining levels are
    grouped into the maximal runs of consecutive positions (in coarsest-first
    size order over *all* levels) that fall between two manual pins (or the
    ends of ``[0, 25]``), and each run is independently log-size-binned via
    :func:`_bin_by_size` onto its own leftover zoom sub-range -- so the
    overall result still partitions ``[0, 25]`` with no gaps/overlaps as
    long as ``manual`` itself doesn't already overlap/gap (validated by the
    caller with :func:`validate_resolutions`).

    Args:
        hierarchy: A :class:`~geohierarchy.core.GeoHierarchy`.
        all_levels: Every level name that needs a zoom range (manual + auto).
        manual: Manually pinned ``{level: (min_zoom, max_zoom)}`` subset.

    Returns:
        Mapping covering every name in ``all_levels`` (manual entries
        unchanged, others auto-filled).
    """
    auto_levels = [name for name in all_levels if name not in manual]
    if not auto_levels:
        return dict(manual)

    medians = _medians(hierarchy, all_levels)
    order = sorted(all_levels, key=lambda n: medians[n], reverse=True)

    result: Dict[str, Tuple[int, int]] = dict(manual)

    # Walk the size-ordered sequence, collecting maximal runs of consecutive
    # auto levels, bounded by whatever manual ranges (or the ends of
    # [MIN_ZOOM, MAX_ZOOM]) flank them.
    i = 0
    n = len(order)
    while i < n:
        if order[i] in manual:
            i += 1
            continue
        run_start = i
        while i < n and order[i] not in manual:
            i += 1
        run = order[run_start:i]

        zoom_lo = manual[order[run_start - 1]][1] + 1 if run_start > 0 else MIN_ZOOM
        zoom_hi = manual[order[i]][0] - 1 if i < n else MAX_ZOOM

        run_medians = {name: medians[name] for name in run}
        if zoom_hi < zoom_lo:
            zoom_hi = zoom_lo
        result.update(_bin_by_size(run_medians, zoom_lo, zoom_hi))

    return result


def validate_resolutions(resolutions: Dict[str, Tuple[int, int]]) -> None:
    """Validate a full set of manually-specified zoom ranges.

    Checks that the given ranges, taken together, partition
    ``[MIN_ZOOM, MAX_ZOOM]`` with no overlaps and no gaps.

    Args:
        resolutions: Mapping of level name to ``(min_zoom, max_zoom)``.

    Raises:
        ValueError: If any range is invalid, ranges overlap, or ranges
            leave a gap anywhere inside ``[MIN_ZOOM, MAX_ZOOM]``.
    """
    if not resolutions:
        raise ValueError("No resolutions to validate")

    items = sorted(resolutions.items(), key=lambda kv: kv[1][0])

    for name, (lo, hi) in items:
        if lo > hi:
            raise ValueError(f"Level '{name}' has min_zoom > max_zoom ({lo} > {hi})")
        if lo < MIN_ZOOM or hi > MAX_ZOOM:
            raise ValueError(
                f"Level '{name}' zoom range ({lo}, {hi}) falls outside "
                f"[{MIN_ZOOM}, {MAX_ZOOM}]"
            )

    first_lo = items[0][1][0]
    if first_lo != MIN_ZOOM:
        raise ValueError(
            f"Zoom ranges leave a gap: coverage starts at {first_lo}, "
            f"expected {MIN_ZOOM}"
        )

    for (name_a, (lo_a, hi_a)), (name_b, (lo_b, hi_b)) in zip(items, items[1:]):
        if lo_b <= hi_a:
            raise ValueError(
                f"Zoom ranges for '{name_a}' ({lo_a}-{hi_a}) and '{name_b}' "
                f"({lo_b}-{hi_b}) overlap"
            )
        if lo_b != hi_a + 1:
            raise ValueError(
                f"Zoom ranges leave a gap between '{name_a}' (ends {hi_a}) "
                f"and '{name_b}' (starts {lo_b})"
            )

    last_hi = items[-1][1][1]
    if last_hi != MAX_ZOOM:
        raise ValueError(
            f"Zoom ranges leave a gap: coverage ends at {last_hi}, "
            f"expected {MAX_ZOOM}"
        )
