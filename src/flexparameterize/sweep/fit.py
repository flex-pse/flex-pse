"""Fit one surrogate spec per configured output to sweep results."""

import hashlib
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec
from flexcore.exceptions import FlexDataError
from flexops.core.units import parse_units
from flexparameterize.config import ParameterizeConfig, SweepOutput, config_hash
from flexparameterize.regression import get_regressor


def fit_sweep(
    config: ParameterizeConfig, data: pd.DataFrame
) -> dict[str, dict[str, SurrogateSpec]]:
    """Fit one surrogate spec per configured output to sweep results.

    Args:
        config: The config the data was swept with.
        data: Rows from ``run_sweep`` and ``solve_point``, concatenated in any
            order, with a column per input and output and ``solve_successful``.

    Returns:
        ``{unit: {relation: spec}}``, the shape ``apply_to_model`` takes as
        ``surrogates``.

    Raises:
        FlexDataError: If a column is missing, more than
            ``max_failed_fraction`` of the rows failed, fewer than two holdout
            rows solved, a holdout output is zero, or the holdout misses an
            acceptance limit.
    """
    data = data.reset_index(drop=True)
    missing = [
        name
        for name in [*config.inputs, *config.outputs, "solve_successful"]
        if name not in data.columns
    ]
    if missing:
        raise FlexDataError(f"sweep data has no column for {missing}.", field="data")

    solved = data["solve_successful"].astype(bool)
    n_failed = int((~solved).sum())
    if n_failed > config.max_failed_fraction * len(data):
        raise FlexDataError(
            f"{n_failed} of {len(data)} points failed, more than "
            f"max_failed_fraction={config.max_failed_fraction} allows.",
            field="max_failed_fraction",
        )
    held_out = data.index % config.holdout.k == config.holdout.r
    train, holdout = data[solved & ~held_out], data[solved & held_out]
    if len(holdout) < 2:
        raise FlexDataError(
            f"only {len(holdout)} holdout point(s) solved; the holdout check "
            "needs at least 2.",
            field="holdout",
        )

    shared = {
        "config_hash": config_hash(config),
        "data_hash": hashlib.sha256(
            data.to_csv(index=False, float_format="%.17g").encode()
        ).hexdigest(),
        "versions": _versions(),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "n_failed": n_failed,
    }
    specs: dict[str, dict[str, SurrogateSpec]] = {}
    for name, output in config.outputs.items():
        spec = _fit_output(config, name, output, train, holdout)
        specs.setdefault(output.unit, {})[output.relation] = spec.model_copy(
            update={"provenance": shared | spec.provenance}
        )
    return specs


def _fit_output(
    config: ParameterizeConfig,
    name: str,
    output: SweepOutput,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> SurrogateSpec:
    """Fit one output and check it on the holdout rows."""
    factors = list(dict.fromkeys(f for term in output.terms for f in term.split("*")))
    regressor = get_regressor(output.surrogate)().fit(
        _terms(output, train),
        _converted(output, train, name),
        input_units={output.input_names[f]: config.inputs[f].units for f in factors},
        output_units=output.output_units,
    )

    actual = _converted(output, holdout, name)[name]
    predicted = pd.Series(regressor.coefficients["intercept"], index=holdout.index)
    for column, values in _terms(output, holdout).items():
        predicted += regressor.coefficients[column] * values
    if (actual == 0).any():
        raise FlexDataError(
            f"output {name!r} is zero at a holdout point, so its relative error "
            "is undefined.",
            field="holdout",
        )
    residual = float(((predicted - actual) ** 2).sum())
    holdout_r2 = 1.0 - residual / float(((actual - actual.mean()) ** 2).sum())
    max_rel_error = float(((predicted - actual) / actual).abs().max())
    limits = config.acceptance
    if not (holdout_r2 >= limits.r2_min and max_rel_error <= limits.max_rel_error):
        raise FlexDataError(
            f"output {name!r} misses the holdout limits: R^2 {holdout_r2:.6g} "
            f"(min {limits.r2_min}), max relative error {max_rel_error:.3g} "
            f"(max {limits.max_rel_error}).",
            field="acceptance",
        )

    return regressor.to_surrogate_spec().model_copy(
        update={
            "provenance": {
                "input_ranges": {
                    output.input_names[f]: [
                        float(train[f].min()),
                        float(train[f].max()),
                    ]
                    for f in factors
                },
                "metrics": regressor.metrics
                | {"holdout_r2": holdout_r2, "holdout_max_rel_error": max_rel_error},
            }
        }
    )


def _terms(output: SweepOutput, rows: pd.DataFrame) -> pd.DataFrame:
    """Return one column per term, named with flex-pse variable names."""
    return pd.DataFrame(
        {
            "*".join(output.input_names[f] for f in term.split("*")): rows[
                term.split("*")
            ].prod(axis=1)
            for term in output.terms
        },
        index=rows.index,
    )


def _converted(output: SweepOutput, rows: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return the output column converted from its model units to output_units."""
    source, target = parse_units(output.units), parse_units(output.output_units)
    return pd.DataFrame(
        {
            name: [
                pyunits.convert_value(float(v), from_units=source, to_units=target)
                for v in rows[name]
            ]
        },
        index=rows.index,
    )


def _versions() -> dict[str, str]:
    """Return the installed versions of the packages a fit depends on."""
    versions = {}
    for package in (
        "flex-pse",
        "pyomo",
        "idaes-pse",
        "parameter-sweep",
        "watertap",
        "numpy",
        "scikit-learn",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions
