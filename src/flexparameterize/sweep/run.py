"""Run a parameter sweep of a Pyomo model with the ``parameter-sweep`` package.

Available only with the ``[sweep]`` extra: ``parameter_sweep`` is imported
inside the functions here, never at module import time.
"""

import warnings
from collections.abc import Callable
from functools import partial
from types import ModuleType

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError

from flexcore.exceptions import FlexConfigError
from flexops.core.units import parse_units
from flexparameterize.config import ParameterizeConfig, SamplerType


def run_sweep(
    config: ParameterizeConfig,
    build_model: Callable[[], pyo.ConcreteModel],
    initialize: Callable[[pyo.ConcreteModel], None],
    optimize: Callable[[pyo.ConcreteModel, object], object],
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Sweep the model over the config's inputs and return one row per point.

    Args:
        config: Inputs, outputs, sampler and solver of the sweep.
        build_model: Builds a fresh model; called again before every point.
        initialize: Initializes a built model at the point's input values.
        optimize: Solves the model with the given solver and returns the
            solver results.
        bounds: Optional ``{input: (low, high)}`` box replacing the config
            ranges of those inputs for this sweep.

    Returns:
        A frame with ``index``, one column per input and output, and
        ``solve_successful``.

    Raises:
        FlexConfigError: If ``bounds`` names an unknown input or leaves the
            config range, ``parameter-sweep`` is not installed, or a path or
            units does not match the built model.
    """
    bounds = bounds or {}
    for name, (low, high) in bounds.items():
        if name not in config.inputs:
            raise FlexConfigError(
                f"bounds names {name!r}, which is not a config input.",
                field="bounds",
                value=name,
            )
        config_low, config_high = config.inputs[name].range
        if not config_low <= low < high <= config_high:
            raise FlexConfigError(
                f"bounds for {name!r} must be an increasing pair inside the config "
                f"range [{config_low}, {config_high}], got ({low}, {high}).",
                field="bounds",
                value=(low, high),
            )
    box = {name: bounds.get(name, spec.range) for name, spec in config.inputs.items()}
    ps = _import_parameter_sweep()
    sampler = config.sampler
    if sampler.type is SamplerType.GRID:
        return _sweep(
            ps,
            config,
            build_model,
            initialize,
            optimize,
            lambda component, name: ps.LinearSample(component, *box[name], sampler.n),
            None,
        )
    return _sweep(
        ps,
        config,
        build_model,
        initialize,
        optimize,
        lambda component, name: ps.LatinHypercubeSample(
            component, *box[name], sampler.n
        ),
        sampler.n,
    )


def solve_point(
    config: ParameterizeConfig,
    build_model: Callable[[], pyo.ConcreteModel],
    initialize: Callable[[pyo.ConcreteModel], None],
    optimize: Callable[[pyo.ConcreteModel, object], object],
    point: dict[str, float],
) -> pd.DataFrame:
    """Solve the model at one point exactly as a sweep would and return one row.

    Args:
        config: Inputs, outputs and solver of the sweep.
        build_model: Builds a fresh model.
        initialize: Initializes a built model at the point's input values.
        optimize: Solves the model with the given solver and returns the
            solver results.
        point: A value for every config input, inside its range.

    Returns:
        A one-row frame with the same columns as :func:`run_sweep`.

    Raises:
        FlexConfigError: If ``point`` misses, adds or leaves the range of an
            input, ``parameter-sweep`` is not installed, or a path or units
            does not match the built model.
    """
    if set(point) != set(config.inputs):
        raise FlexConfigError(
            f"point must give exactly the inputs {sorted(config.inputs)}, "
            f"got {sorted(point)}.",
            field="point",
            value=point,
        )
    for name, value in point.items():
        low, high = config.inputs[name].range
        if not low <= value <= high:
            raise FlexConfigError(
                f"point value {value} for {name!r} is outside [{low}, {high}].",
                field="point",
                value=point,
            )
    ps = _import_parameter_sweep()
    return _sweep(
        ps,
        config,
        build_model,
        initialize,
        optimize,
        lambda component, name: ps.PredeterminedFixedSample(component, [point[name]]),
        None,
    )


def _import_parameter_sweep() -> ModuleType:
    """Return the ``parameter_sweep`` module or say how to install it."""
    try:
        import parameter_sweep
    except ImportError as exc:
        raise FlexConfigError(
            "Sweeps require the parameter-sweep package. Install it with "
            "`pip install 'flex-pse[sweep]'`."
        ) from exc
    return parameter_sweep


def _sweep(
    ps: ModuleType,
    config: ParameterizeConfig,
    build_model: Callable[[], pyo.ConcreteModel],
    initialize: Callable[[pyo.ConcreteModel], None],
    optimize: Callable[[pyo.ConcreteModel, object], object],
    sample: Callable[[object, str], object],
    num_samples: int | None,
) -> pd.DataFrame:
    """Run ``ParameterSweep`` with a rebuild before every point."""
    solver = pyo.SolverFactory(config.solver.name)
    solver.options.update(config.solver.options)
    sweep = ps.ParameterSweep(
        optimize_function=optimize,
        optimize_kwargs={"solver": solver},
        initialize_function=initialize,
        initialize_before_sweep=True,
        update_sweep_params_before_init=True,
        number_of_subprocesses=1,
    )
    with warnings.catch_warnings():
        # Not writing files is intended; the caller saves the returned frame.
        warnings.filterwarnings("ignore", message="No results will be writen")
        _, results = sweep.parameter_sweep(
            partial(_build_with_context, build_model, config.context),
            lambda m: {
                name: sample(_resolve(m, spec.path, spec.units, name, fixed=True), name)
                for name, spec in config.inputs.items()
            },
            build_outputs=lambda m: {
                name: _resolve(m, spec.path, spec.units, name, fixed=False)
                for name, spec in config.outputs.items()
            },
            num_samples=num_samples,
            seed=config.sampler.seed,
        )
    columns = {
        name: results["sweep_params"][name]["value"] for name in config.inputs
    } | {name: results["outputs"][name]["value"] for name in config.outputs}
    successful = np.asarray(results["solve_successful"], dtype=bool)
    return pd.DataFrame(
        {"index": np.arange(len(successful))}
        | columns
        | {"solve_successful": successful}
    )


def _build_with_context(
    build_model: Callable[[], pyo.ConcreteModel], context: dict[str, float]
) -> pyo.ConcreteModel:
    """Build the model and fix every context component at its value."""
    m = build_model()
    for path, value in context.items():
        component = m.find_component(path)
        if component is None:
            raise FlexConfigError(
                f"context path {path!r} is not a component of the built model.",
                field="context",
                value=path,
            )
        if component.is_variable_type():
            component.fix(value)
        else:
            component.set_value(value)
    return m


def _resolve(m: pyo.ConcreteModel, path: str, units: str, name: str, *, fixed: bool):
    """Return the component at ``path`` after checking it exists and its units."""
    component = m.find_component(path)
    if component is None:
        raise FlexConfigError(
            f"{name!r} path {path!r} is not a component of the built model.",
            field=name,
            value=path,
        )
    if fixed and not (
        (component.is_variable_type() and component.fixed)
        or (component.is_parameter_type() and component.mutable)
    ):
        raise FlexConfigError(
            f"input {name!r} at {path!r} must be a fixed Var or mutable Param; "
            "sweeping anything else would over-specify the model.",
            field=name,
            value=path,
        )
    model_units = pyo.units.get_units(component)
    if model_units is None:
        model_units = pyo.units.dimensionless
    try:
        factor = pyo.units.convert_value(
            1.0, from_units=model_units, to_units=parse_units(units)
        )
    except UnitsError:
        factor = None
    if factor is None or abs(factor - 1.0) > 1e-12:
        raise FlexConfigError(
            f"{name!r} is in units {units!r} in the config but {model_units} on "
            "the model.",
            field=name,
            value=units,
        )
    return component
