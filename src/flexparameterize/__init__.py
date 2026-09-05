"""FlexParameterize: parameterize a FlexOps model from tabular plant data.

The pipeline is **tabular data -> tag aliasing -> sufficiency validation ->
regression -> {apply to model | emit config}**. The two
terminal stages are the two directions of the FlexParameterize <-> FlexOps
coupling and they agree by construction:

* :func:`~flexparameterize.apply.apply_to_model` mutates a live model in place;
* :func:`~flexparameterize.emit.emit_model_config` produces the serializable
  config that rebuilds the same parameterized model.

Any tabular source works — historian export, spreadsheet, CSV, database query —
provided its columns are mapped to model aliases by a
:class:`~flexparameterize.tags.TagMap`.
"""

from importlib.metadata import version as _dist_version

from flexparameterize.apply import ApplyReport, apply_to_model
from flexparameterize.emit import emit_model_config
from flexparameterize.regression import (
    ConstantIntensityRegressor,
    FitResult,
    LinearRegressor,
    Regressor,
    get_regressor,
)
from flexparameterize.tags import TagMap, TagReport, model_alias
from flexparameterize.validate import (
    IOPairStatus,
    SufficiencyReport,
    check_sufficiency,
)

__version__ = _dist_version("flex-pse")

__all__ = [
    "ApplyReport",
    "ConstantIntensityRegressor",
    "FitResult",
    "IOPairStatus",
    "LinearRegressor",
    "Regressor",
    "SufficiencyReport",
    "TagMap",
    "TagReport",
    "__version__",
    "apply_to_model",
    "check_sufficiency",
    "emit_model_config",
    "get_regressor",
    "model_alias",
]
