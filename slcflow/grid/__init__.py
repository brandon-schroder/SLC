"""Grid topology, streamline initialization, and metric evaluation (G-5, G-6)."""
from .core import (
    GridMetrics,
    GridTopology,
    MetricsConfig,
    evaluate_metrics,
    initialize_positions,
)
from .quadrature import cumulative, invert_cumulative

__all__ = [
    "GridTopology",
    "MetricsConfig",
    "GridMetrics",
    "initialize_positions",
    "evaluate_metrics",
    "cumulative",
    "invert_cumulative",
]
