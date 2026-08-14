"""FlexParameterize: parameterize a FlexOps model from tabular plant data.

The pipeline is **tabular data -> tag aliasing -> sufficiency validation ->
regression -> {apply to model | emit config}** (architecture §5). The two
terminal stages are the two directions of the FlexParameterize <-> FlexOps
coupling and they agree by construction:

* :func:`~flexparameterize.apply.apply_to_model` mutates a live model in place;
* :func:`~flexparameterize.emit.emit_model_config` produces the serializable
  config that rebuilds the same parameterized model.

Any tabular source works — historian export, spreadsheet, CSV, database query —
provided its columns are mapped to model aliases by a
:class:`~flexparameterize.tags.TagMap`.
"""

from flexparameterize.apply import ApplyReport, apply_to_model
from flexparameterize.emit import emit_model_config
from flexparameterize.regression import ConstantIntensityRegressor
from flexparameterize.tags import TagMap, TagReport, model_alias
from flexparameterize.validate import (
    IOPairStatus,
    SufficiencyReport,
    check_sufficiency,
)

__all__ = [
    "ApplyReport",
    "ConstantIntensityRegressor",
    "IOPairStatus",
    "SufficiencyReport",
    "TagMap",
    "TagReport",
    "apply_to_model",
    "check_sufficiency",
    "emit_model_config",
    "model_alias",
]
