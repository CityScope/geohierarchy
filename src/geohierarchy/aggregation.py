"""
Aggregation strategies for hierarchical spatial resampling.

Rules:
- NaN → null
- null values ignored everywhere
- geoweight column is ALWAYS "_geoweight"
- IMPORTANT: if all values are null → result is null (NOT 0)
- IMPORTANT: weighted ops drop rows where value OR weight are null
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import polars as pl
from copy import copy

# ================================================================
# HELPERS
# ================================================================


def _valid(v: pl.Expr, w: pl.Expr | None = None) -> pl.Expr:
    """
    Row validity mask:
    - NaN → null first
    - value must be non-null
    - if weight provided → weight must also be non-null
    """
    mask = v.fill_nan(None).is_not_null()

    if w is not None:
        mask = mask & w.fill_nan(None).is_not_null()

    return mask


def sum_agg(column_name, weight_column=None, geoweighted=False):
    if weight_column is None:
        w = pl.lit(1)
    else:
        w = pl.col(weight_column)

    if geoweighted:
        geo_w = pl.col("_geoweight")
    else:
        geo_w = pl.lit(1)

    eff_w = w * geo_w

    v = pl.col(column_name)

    if geoweighted or (weight_column is not None):
        mask = _valid(v, eff_w)
    else:
        mask = _valid(v)

    n_rows = mask.sum()

    # Bug fix (2026-09-04, live user report + verified: a synthetic 2-cell
    # example with a cell split 50/50 across two destination polygons
    # returned totals of 26.67 and 20 for true values of 20 and 10 -- a
    # ~55%/100% inflation, and NOT a coincidence of that example: this
    # extra `* (n_rows / sum_w)` factor has no correct interpretation for
    # an area-weighted SUM. It looks like a stray leftover from a weighted
    # MEAN's normalization (`mean_agg` correctly divides by `sum_w`, no
    # `n_rows` involved) accidentally applied here too. The correct
    # geoweighted/weighted sum is simply `sum(v * eff_w)` -- each row's
    # value already scaled by its own effective weight (e.g. the real
    # area-overlap fraction for `geoweighted=True`), nothing more. This was
    # the actual `Sum(geoweighted=True, intersection_mode="exact")` path
    # `add_vector_data` uses for every real "area-weighted population"
    # census-polygon aggregation in this codebase -- verified live,
    # `sum_agg` (not `sum_expr`, which is unused in production) is exactly
    # what `Sum.upscale_aggs` -> `GeoHierarchy.add_vector_data` calls.
    return (
        pl.when(n_rows > 0)
        .then((v * eff_w).filter(mask).sum())
        .otherwise(None)
        .alias(column_name)
    )


def sum_expr(column_name, group_col, weight_column=None, geoweighted=False):
    if weight_column is None:
        w = pl.lit(1)
    else:
        w = pl.col(weight_column)

    if geoweighted:
        geo_w = pl.col("_geoweight")
    else:
        geo_w = pl.lit(1)

    eff_w = w * geo_w

    v = pl.col(column_name)

    if geoweighted or (weight_column is not None):
        mask = _valid(v, eff_w)
    else:
        mask = _valid(v)

    n_rows = mask.sum().over(group_col)

    # Same bug fix as `sum_agg` above -- see its comment. `sum_expr` is
    # unused elsewhere in this codebase currently, but kept consistent.
    return (
        pl.when(n_rows > 0)
        .then((v * eff_w).filter(mask).sum().over(group_col))
        .otherwise(None)
        .alias(column_name)
    )


def mean_agg(column_name, weight_column=None, geoweighted=False):
    if weight_column is None:
        w = pl.lit(1)
    else:
        w = pl.col(weight_column)

    if geoweighted:
        geo_w = pl.col("_geoweight")
    else:
        geo_w = pl.lit(1)

    eff_w = w * geo_w

    v = pl.col(column_name)

    if geoweighted or (weight_column is not None):
        mask = _valid(v, eff_w)
    else:
        mask = _valid(v)

    num = (v * eff_w).filter(mask).sum()
    den = eff_w.filter(mask).sum()

    return pl.when(den > 0).then(num / den).otherwise(None).alias(column_name)


def mean_expr(column_name, group_col, weight_column=None, geoweighted=False):
    if weight_column is None:
        w = pl.lit(1)
    else:
        w = pl.col(weight_column)

    if geoweighted:
        geo_w = pl.col("_geoweight")
    else:
        geo_w = pl.lit(1)

    eff_w = w * geo_w

    v = pl.col(column_name)

    if geoweighted or (weight_column is not None):
        mask = _valid(v, eff_w)
    else:
        mask = _valid(v)

    num = (v * eff_w).filter(mask).sum().over(group_col)
    den = eff_w.filter(mask).sum().over(group_col)

    return pl.when(den > 0).then(num / den).otherwise(None).alias(column_name)


def divide_expr(column_name, group_col, weight_column=None, geoweighted=False):
    if weight_column is None:
        w = pl.lit(1)
    else:
        w = pl.col(weight_column)

    if geoweighted:
        geo_w = pl.col("_geoweight")
    else:
        geo_w = pl.lit(1)

    eff_w = w * geo_w

    v = pl.col(column_name)

    if geoweighted or (weight_column is not None):
        mask = _valid(v, eff_w)
    else:
        mask = _valid(v)

    w_sum = pl.when(mask).then(eff_w).otherwise(0).sum().over(group_col)

    return (
        pl.when(w_sum > 0).then((v * eff_w) / w_sum).otherwise(None).alias(column_name)
    )


# ================================================================
# BASE CLASS
# ================================================================


class AggregationStrategy(ABC):
    """Base class describing how a column resamples between hierarchy levels.

    Subclasses implement both directions of resampling: ``upscale_aggs``
    combines many fine-grained rows into one coarse row (a Polars group-by
    aggregation), while ``downscale_exprs`` splits or broadcasts a coarse
    row's value across its matching fine-grained rows.

    Attributes:
        geoweighted: If ``True``, row contributions are additionally
            weighted by the fraction of geometric overlap (area or length)
            between the source and destination geometries, using the
            ``"_geoweight"`` column produced by
            :func:`geohierarchy.utils.get_id_mapping`.
        mapping: Which id-mapping function pairs source and destination
            rows: ``"overlap"`` (:func:`geohierarchy.utils.get_id_mapping`,
            geometric intersection) or ``"knn"``
            (:func:`geohierarchy.utils.get_knn_mapping`, inverse-distance
            nearest neighbors -- see :class:`SmoothMean`).
        preserve_total: Whether a column using this strategy has a
            meaningful total that downscaling must reproduce exactly (an
            absolute/additive quantity, e.g. population) as opposed to a
            relative one with no sensible total (a rate, ratio, or median,
            e.g. income). ``True`` for :class:`Sum`. Strategies whose
            downscale isn't already exactly total-preserving by
            construction (currently just :class:`SmoothMean`, whose
            blending is spatial rather than proportional) use this flag to
            trigger a corrective rescale in
            :meth:`GeoHierarchy._aggregate_edge_batch` -- see
            :meth:`SmoothMean.__init__`.
    """

    def __init__(
        self,
        geoweighted: bool = False,
        mapping: str = "overlap",
        preserve_total: bool = False,
        intersection_mode: Optional[str] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            geoweighted: Whether to weight rows by geometric overlap
                fraction during aggregation.
            mapping: Id-mapping mode, ``"overlap"`` or ``"knn"``.
            preserve_total: Whether this column's downscaled total should
                exactly match its source total (see class docstring).
            intersection_mode: How source/destination geometries are paired
                -- ``"centroid"``, ``"touches"``, or ``"exact"`` (see
                :func:`geohierarchy.utils.get_id_mapping`). ``None`` (the
                default) falls back to legacy behavior derived from
                ``geoweighted`` (``True`` -> ``"exact"``, ``False`` ->
                ``"centroid"``).
        """
        # "touches"/"exact" only matter if the resulting `_geoweight` column
        # is actually consumed by the aggregation math (`sum_agg`/`mean_agg`
        # etc. only read `_geoweight` when `geoweighted=True`) -- so picking
        # either mode implies geoweighting unless the caller explicitly
        # disabled it. "centroid" (the legacy default) leaves `geoweighted`
        # as given, since every matched row's weight is uniformly 1.0 there
        # anyway.
        if intersection_mode in ("touches", "exact"):
            geoweighted = True
        self.geoweighted = geoweighted
        self.mapping = mapping
        self.preserve_total = preserve_total
        self.intersection_mode = intersection_mode

    @abstractmethod
    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        """Build group-by aggregation expressions for fine -> coarse resampling.

        Args:
            columns: Column names to build aggregation expressions for.

        Returns:
            One Polars expression per column, suitable for use inside
            ``DataFrame.group_by(...).agg(...)``.
        """

    @abstractmethod
    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        """Build row-wise expressions for coarse -> fine resampling.

        Args:
            columns: Column names to build expressions for.
            group_col: Name of the column identifying which source
                (coarse) row each destination row belongs to.

        Returns:
            One Polars expression per column, suitable for use inside
            ``DataFrame.with_columns(...)``. May be empty if the strategy
            has no meaningful downscale behavior.
        """

    def consolidate_downscale(self, columns: List[str]) -> List[pl.Expr]:
        """Combine multiple downscaled fragments landing on the same destination row.

        A destination geometry can overlap more than one source row (e.g.
        a street crossing two tracts during a tract -> street downscale),
        which produces one fragment row per source it overlaps. This
        collapses those fragments -- already grouped by destination id --
        back into a single row per destination.

        The default keeps an arbitrary single fragment's value, which is
        correct for strategies that broadcast the same (or a directly
        comparable) value to every fragment. :class:`Sum` overrides this
        to add the fragments instead, since its downscaled shares are
        specifically built to sum back to the original total.

        Args:
            columns: Column names to consolidate.

        Returns:
            One Polars expression per column, suitable for use inside
            ``DataFrame.group_by(dst_id).agg(...)``.
        """
        return [pl.col(c).first().alias(c) for c in columns]

    def upscale(
        self, df: pl.DataFrame, group_col: str, columns: List[str]
    ) -> pl.DataFrame:
        """Aggregate ``df`` from fine to coarse rows.

        Args:
            df: Row-level data joined against the destination id mapping.
            group_col: Name of the destination (coarse) id column to
                group by.
            columns: Column names to aggregate.

        Returns:
            One row per distinct ``group_col`` value with the aggregated
            columns.
        """
        return df.group_by(group_col).agg(self.upscale_aggs(columns))

    def downscale(
        self, df: pl.DataFrame, group_col: str, columns: List[str]
    ) -> pl.DataFrame:
        """Disaggregate ``df`` from coarse to fine rows.

        Args:
            df: Row-level data joined against the source id mapping.
            group_col: Name of the source (coarse) id column each row
                belongs to.
            columns: Column names to disaggregate.

        Returns:
            ``df`` with the disaggregated columns added or replaced.
        """
        exprs = self.downscale_exprs(columns, group_col)
        return df.with_columns(exprs) if exprs else df


def aggregation_strategy(
    upscale: AggregationStrategy, downscale: Optional[AggregationStrategy] = None
) -> AggregationStrategy:
    """Combine one strategy's upscale behavior with another's downscale behavior.

    Useful when the natural way to aggregate a column upward (e.g. a
    weighted sum) differs from how it should be split back downward.

    Args:
        upscale: Strategy whose ``upscale_aggs`` is used as-is.
        downscale: Strategy whose ``downscale_exprs`` is grafted onto a
            copy of ``upscale``. If ``None``, ``upscale``'s own
            ``downscale_exprs`` is kept.

    Returns:
        A copy of ``upscale`` with ``downscale_exprs`` replaced by
        ``downscale``'s implementation, if provided.
    """
    combined = copy(upscale)

    if downscale is not None:
        # Assigned as a plain instance attribute (not via MethodType), so
        # no implicit `self` is inserted -- callers already pass exactly
        # (columns, group_col), and this closes over `downscale` so its
        # own configuration (e.g. weight_column) is used, not upscale's.
        def _downscale_exprs(columns: List[str], group_col: str) -> List[pl.Expr]:
            return downscale.downscale_exprs(columns, group_col)

        def _consolidate_downscale(columns: List[str]) -> List[pl.Expr]:
            return downscale.consolidate_downscale(columns)

        combined.downscale_exprs = _downscale_exprs
        combined.consolidate_downscale = _consolidate_downscale
        # The downscale strategy's id-mapping mode (e.g. SmoothMean's "knn")
        # governs the downscale pass, not upscale's own "overlap" default.
        combined.mapping = downscale.mapping
        combined.knn_k = getattr(downscale, "knn_k", None)
        combined.knn_power = getattr(downscale, "knn_power", None)
        # Likewise, whether the downscaled total should be corrected back
        # to match the source total is a property of the downscale
        # strategy actually used, not upscale's.
        combined.preserve_total = downscale.preserve_total

    return combined


# ================================================================
# SUM
# ================================================================


class Sum(AggregationStrategy):
    """Additive quantities such as population or vehicle counts.

    Upscaling sums the (optionally weighted) values; downscaling splits a
    coarse total proportionally across destination rows using the same
    weights, so totals are preserved in both directions.
    """

    def __init__(
        self,
        weight_column: Optional[str] = None,
        geoweighted: bool = False,
        intersection_mode: Optional[str] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            weight_column: Optional column whose values weight each row's
                contribution to the sum, on top of any geoweighting.
            geoweighted: Whether to additionally weight rows by geometric
                overlap fraction.
            intersection_mode: ``"centroid"``, ``"touches"``, or ``"exact"``
                -- see :class:`AggregationStrategy`.
        """
        super().__init__(
            geoweighted, preserve_total=True, intersection_mode=intersection_mode
        )
        self.weight_column = weight_column

    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        exprs = []
        for c in columns:
            exprs.append(sum_agg(c, self.weight_column, self.geoweighted))

        return exprs

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        exprs = []
        for c in columns:
            exprs.append(
                divide_expr(c, group_col, self.weight_column, self.geoweighted)
            )

        return exprs

    def consolidate_downscale(self, columns: List[str]) -> List[pl.Expr]:
        # Proportional shares are built to sum back to the original total,
        # so a destination row split across multiple sources adds its
        # fragments together rather than keeping just one.
        return [pl.col(c).sum().alias(c) for c in columns]


# ================================================================
# MEAN
# ================================================================


class Mean(AggregationStrategy):
    """Intensive quantities such as rates, incomes, or accessibility scores.

    Upscaling computes a (optionally weighted) average; downscaling
    broadcasts that average back down, weighted the same way, unless the
    strategy is unweighted (plain average), in which case there is no
    meaningful per-row split and the value is left for the caller's
    fallback (typically a plain broadcast join).
    """

    def __init__(
        self,
        weight_column: Optional[str] = None,
        geoweighted: bool = False,
        intersection_mode: Optional[str] = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            weight_column: Optional column whose values weight each row in
                the average, on top of any geoweighting.
            geoweighted: Whether to additionally weight rows by geometric
                overlap fraction.
            intersection_mode: ``"centroid"``, ``"touches"``, or ``"exact"``
                -- see :class:`AggregationStrategy`.
        """
        super().__init__(geoweighted, intersection_mode=intersection_mode)
        self.weight_column = weight_column

    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        exprs = []

        for c in columns:
            exprs.append(mean_agg(c, self.weight_column, self.geoweighted))

        return exprs

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        exprs = []
        for c in columns:
            if self.geoweighted or (self.weight_column is not None):
                exprs.append(
                    mean_expr(c, group_col, self.weight_column, self.geoweighted)
                )

        return exprs


# ================================================================
# SMOOTH MEAN (KNN-BLENDED DOWNSCALE)
# ================================================================


class SmoothMean(AggregationStrategy):
    """A downscale that blends several nearby source cells instead of copying one.

    :class:`Mean` and :class:`Sum`'s downscale strictly follows the source
    geometry's boundaries: every destination row gets a value derived only
    from whichever source row(s) it geometrically overlaps, so the result
    is blocky wherever the source grid is much coarser than the
    destination grid (e.g. downscaling from H3 resolution 8 to 9). This
    strategy instead uses :func:`geohierarchy.utils.get_knn_mapping` to
    blend each destination row's value from its ``k`` nearest source
    centroids, inverse-distance weighted, which smooths out those
    boundaries.

    Only its downscale behavior is meaningful -- combine it with a normal
    upscale strategy via :func:`aggregation_strategy`, e.g.::

        set_aggregation("population", upscale=Sum(), downscale=SmoothMean(k=6, density=True))

    :meth:`upscale_aggs` still works standalone (plain unweighted mean), so
    the strategy is usable directly for a column that's only ever
    downscaled.

    ``density`` matters for any additive (count-like) column: a source
    cell's raw count isn't comparable across neighbors of different sizes,
    and blending raw counts directly to a much smaller destination cell
    (as with H3 resolution 8 -> 9, where each finer cell is ~1/7 the area)
    would hand it a value sized for the *source's* area, inflating its
    ``.density`` several-fold. With ``density=True``, each neighbor's value
    is divided by that neighbor's own area before blending, and the
    blended density is scaled back up by the destination row's own
    (smaller) area, so counts stay proportionate to cell size. Leave it
    ``False`` for columns that are already an intensive quantity (medians,
    rates, per-capita figures), which don't need this conversion.

    Warning:
        Only verified for the downscale direction (its intended use). If a
        column combining this strategy is ever upscaled through the same
        edge, the id-mapping mode this strategy selects (``"knn"``)
        currently applies uniformly to both directions, which has not been
        validated against ``upscale_aggs``'s expectations.

    Attributes:
        k: Number of nearest source neighbors blended per destination row.
        power: Inverse-distance weighting exponent.
        density: Whether to blend in per-area density space, converting
            back to a magnitude sized for the destination row afterward.
    """

    def __init__(self, k: int = 6, power: float = 2.0, density: bool = False) -> None:
        """Initialize the strategy.

        Args:
            k: Number of nearest source neighbors to blend.
            power: Inverse-distance weighting exponent; higher values
                concentrate more weight on the nearest neighbor(s).
            density: Whether ``columns`` hold additive counts that should
                be blended as densities (see class docstring) rather than
                blended as-is.
        """
        # An absolute count (density=True) has a real total to preserve;
        # a relative/intensive column (density=False, e.g. a median) doesn't.
        super().__init__(geoweighted=True, mapping="knn", preserve_total=density)
        self.knn_k = k
        self.knn_power = power
        self.density = density

    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).fill_nan(None).mean().alias(c) for c in columns]

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        # The actual blend happens in consolidate_downscale, which needs
        # the raw value, "_geoweight", and (if density=True) the area
        # columns from get_knn_mapping -- so there is nothing to
        # precompute per row here.
        return []

    def consolidate_downscale(self, columns: List[str]) -> List[pl.Expr]:
        # Null neighbors are ignored and the remaining weights renormalized
        # (mirroring mean_agg/mean_expr's null handling elsewhere in this
        # module), so a destination cell with at least one non-null
        # neighbor gets a proper weighted average of just those, and a
        # destination cell whose neighbors are *all* null gets null back --
        # not 0, which is what a plain ``.sum()`` over an all-null group
        # would otherwise silently produce.
        exprs = []
        for c in columns:
            # Each joined row's value belongs to a neighbor (the source of
            # propagation), so density conversion divides by *that* row's
            # area; the blended density is then scaled by the destination
            # row's own area (the same value for every row in this group).
            v = pl.col(c) / pl.col("_neighbor_area") if self.density else pl.col(c)
            w = pl.col("_geoweight")
            mask = v.is_not_null()
            sum_w = w.filter(mask).sum()
            weighted_sum = (v * w).filter(mask).sum()
            avg = pl.when(sum_w > 0).then(weighted_sum / sum_w).otherwise(None)
            if self.density:
                avg = avg * pl.col("_query_area").first()
            exprs.append(avg.alias(c))
        return exprs


# ================================================================
# MAX
# ================================================================


class Max(AggregationStrategy):
    """Take the maximum value across source rows; has no downscale behavior."""

    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).fill_nan(None).max().alias(c) for c in columns]

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        return []

    def consolidate_downscale(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).max().alias(c) for c in columns]


# ================================================================
# MIN
# ================================================================


class Min(AggregationStrategy):
    """Take the minimum value across source rows; has no downscale behavior."""

    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).fill_nan(None).min().alias(c) for c in columns]

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        return []

    def consolidate_downscale(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).min().alias(c) for c in columns]
