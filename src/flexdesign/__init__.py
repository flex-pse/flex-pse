"""FlexDesign: size equipment across representative months of a multi-year study."""

from importlib.metadata import version as _dist_version

from flexdesign.build import (
    build_design_model,
    build_horizons,
    expand_horizons,
    present_worth_factors,
)
from flexdesign.results import CapacityResult, DesignResults, design_results

__version__ = _dist_version("flex-pse")

__all__ = [
    "CapacityResult",
    "DesignResults",
    "build_design_model",
    "build_horizons",
    "design_results",
    "expand_horizons",
    "present_worth_factors",
]
