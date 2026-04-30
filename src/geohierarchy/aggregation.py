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
from typing import List
import polars as pl


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
    def __init__(self, geoweighted: bool = False):
        self.geoweighted = geoweighted

    @abstractmethod
    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        pass

    @abstractmethod
    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        pass

    def upscale(
        self, df: pl.DataFrame, group_col: str, columns: List[str]
    ) -> pl.DataFrame:
        return df.group_by(group_col).agg(self.upscale_aggs(columns))

    def downscale(
        self, df: pl.DataFrame, group_col: str, columns: List[str]
    ) -> pl.DataFrame:
        exprs = self.downscale_exprs(columns, group_col)
        return df.with_columns(exprs) if exprs else df


# ================================================================
# SUM
# ================================================================


class Sum(AggregationStrategy):
    def __init__(self, weight_column: str | None = None, geoweighted: bool = False):
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


# ================================================================
# MEAN
# ================================================================


class Mean(AggregationStrategy):
    def __init__(self, weight_column: str | None = None, geoweighted: bool = False):
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
    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).fill_nan(None).max().alias(c) for c in columns]

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        return []


# ================================================================
# MIN
# ================================================================


class Min(AggregationStrategy):
    def upscale_aggs(self, columns: List[str]) -> List[pl.Expr]:
        return [pl.col(c).fill_nan(None).min().alias(c) for c in columns]

    def downscale_exprs(self, columns: List[str], group_col: str) -> List[pl.Expr]:
        return []
