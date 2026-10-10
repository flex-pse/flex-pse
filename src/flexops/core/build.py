"""build_model: construct a whole flex-pse model from one config.

The config-driven entry point. A single validated
:class:`~flexcore.config.spec.FlowsheetSpec` (or a legacy nested
:class:`~flexcore.config.schema.ModelConfig`, converted to one) yields the whole
model by running the ordered stages in :mod:`flexops.core.stages`. The
frozen-API fixtures ``api_freeze.py`` and ``api_freeze_config.json`` (under
``flexops/tests/fixtures/api_freeze/``) are the same model built each way.

**Units in a persisted config are data, not code.** A units-carrying quantity
is written as ``{"value": 15, "units": "min"}`` (or, for the time step, the
string ``"15 min"``) and :func:`parse_quantity` turns it into a Pyomo
expression at build time.
"""

import pyomo.environ as pyo

from flexcore.config.io import load_spec
from flexops.core.stages import (
    STAGE_FUNCTIONS,
    STAGES,
    BuildContext,
    parse_quantity,
)
from flexops.core.units import parse_units

__all__ = ["build_model", "parse_quantity", "parse_units"]


def build_model(config, *, expand_arcs: bool = False) -> pyo.ConcreteModel:
    """Build the whole Pyomo model described by a config.

    Args:
        config: Anything :func:`~flexcore.config.io.load_spec` accepts: a flat
            spec, a path or mapping for either a flat spec or a legacy nested
            config, or a nested :class:`~flexcore.config.schema.ModelConfig`.
            Raw input always goes through the validated schema.
        expand_arcs: Whether to apply ``network.expand_arcs`` after the
            topology is built.

    Returns:
        The constructed ``ConcreteModel``, carrying ``time_block``, the named
        property packages and costing blocks, the network, plant and unit
        tree, and the objective if the spec has one. Arcs are expanded only if
        ``expand_arcs`` is true.

    Raises:
        FlexConfigError: If the config fails validation (the message names the
            offending field path, and ``__cause__`` is the underlying pydantic
            ``ValidationError``), or names an unknown unit-model class.
    """
    spec = load_spec(config)
    model = pyo.ConcreteModel(name=spec.name)
    ctx = BuildContext(base_dir=spec._base_dir, expand_arcs=expand_arcs)
    for name in STAGES:
        STAGE_FUNCTIONS[name](model, spec, ctx)
    model._flex_build_context = ctx  # used by apply_stages
    model._flex_spec = spec  # the spec this model was built from
    return model
