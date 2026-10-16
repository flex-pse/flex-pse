"""build_model: construct a whole flex-pse model from one flat spec.

The config-driven entry point. A validated
:class:`~flexcore.config.spec.FlowsheetSpec` yields the whole model by running
the ordered stages in :mod:`flexops.core.stages`. A units-carrying quantity is
written as ``{"value": 15, "units": "min"}`` (or, for the time step, the string
``"15 min"``) and :func:`parse_quantity` turns it into a Pyomo expression.
"""

import pyomo.environ as pyo

from flexcore.config.io import load_spec
from flexops.core.stages import STAGES, BuildContext, parse_quantity
from flexops.core.units import parse_units

__all__ = ["build_model", "parse_quantity", "parse_units"]


def build_model(config, *, expand_arcs: bool = True) -> pyo.ConcreteModel:
    """Build the whole Pyomo model described by a flat spec.

    Args:
        config: Anything :func:`~flexcore.config.io.load_spec` accepts: a
            ``FlowsheetSpec``, or a ``.json`` path or mapping validated into one.
        expand_arcs: Whether to apply ``network.expand_arcs`` after the
            topology is built, so logic and extensions can use the expanded
            equalities.

    Returns:
        The constructed ``ConcreteModel``, carrying ``time_block``, the named
        property packages and costing blocks, the network, plant and unit
        tree, its logic and extensions, and the objective if the spec has one.

    Raises:
        FlexConfigError: If the config fails validation (the message names the
            offending field path) or does not fit the model it builds.
    """
    spec = load_spec(config)
    model = pyo.ConcreteModel(name=spec.name)
    ctx = BuildContext(base_dir=spec._base_dir, expand_arcs=expand_arcs)
    for stage in STAGES:
        stage(model, spec, ctx)
    model._flex_spec = spec  # the spec this model was built from
    return model
