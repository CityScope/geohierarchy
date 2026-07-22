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

    sum_w = eff_w.filter(mask).sum()
    n_rows = mask.sum()

    return (
        pl.when(sum_w > 0)
        .then((v * eff_w * (n_rows / sum_w)).filter(mask).sum())
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

    sum_w = eff_w.filter(mask).sum().over(group_col)
    n_rows = mask.sum().over(group_col)

    return (
        pl.when(sum_w > 0)
        .then((v * eff_w * (n_rows / sum_w)).filter(mask).sum().over(group_col))
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

    w_sum = pl.when(mask).then(w).otherwise(0).sum().over(group_col)

    return pl.when(w_sum > 0).then((v * w) / w_sum).otherwise(None).alias(column_name)


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
    """

    def __init__(self, geoweighted: bool = False) -> None:
        """Initialize the strategy.

        Args:
            geoweighted: Whether to weight rows by geometric overlap
                fraction during aggregation.
        """
        self.geoweighted = geoweighted

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
        self, weight_column: Optional[str] = None, geoweighted: bool = False
    ) -> None:
        """Initialize the strategy.

        Args:
            weight_column: Optional column whose values weight each row's
                contribution to the sum, on top of any geoweighting.
            geoweighted: Whether to additionally weight rows by geometric
                overlap fraction.
        """
        super().__init__(geoweighted)
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
        self, weight_column: Optional[str] = None, geoweighted: bool = False
    ) -> None:
        """Initialize the strategy.

        Args:
            weight_column: Optional column whose values weight each row in
                the average, on top of any geoweighting.
            geoweighted: Whether to additionally weight rows by geometric
                overlap fraction.
        """
        super().__init__(geoweighted)
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
