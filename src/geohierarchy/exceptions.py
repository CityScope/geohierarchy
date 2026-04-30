"""Custom exceptions for geohierarchy."""


class ColumnNotFoundError(Exception):
    """Raised when a requested column does not exist in the dataset."""

    pass


class AggregationStrategyError(Exception):
    """Raised when an aggregation strategy is configured incorrectly."""

    pass
