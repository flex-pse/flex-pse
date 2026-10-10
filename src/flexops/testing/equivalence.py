"""Structural comparison of two Pyomo models."""

import hashlib
import json

import pyomo.environ as pyo

from flexops.core.compare import _collect, _name, _repn_key, model_differences

MAX_REPORTED_DIFFERENCES = 50
"""Number of differences shown in a failure message; the rest are counted."""


def assert_models_equivalent(
    a, b, *, rel_tol: float = 1e-9, check_values: bool = False
) -> None:
    """Raise AssertionError listing every difference between two Pyomo models.

    See :func:`~flexops.core.compare.model_differences` for what is compared.

    Args:
        a: The first model.
        b: The second model.
        rel_tol: Relative tolerance for numeric comparisons.
        check_values: Also compare values of unfixed Vars (useful after a solve).

    Raises:
        AssertionError: If the models differ; the message lists every difference
            (the first ``MAX_REPORTED_DIFFERENCES``, then a count of the rest).
    """
    diffs = model_differences(a, b, rel_tol=rel_tol, check_values=check_values)
    if diffs:
        shown = diffs[:MAX_REPORTED_DIFFERENCES]
        rest = len(diffs) - len(shown)
        if rest:
            shown.append(f"... and {rest} more")
        raise AssertionError(f"Models differ ({len(diffs)}):\n" + "\n".join(shown))


def model_fingerprint(model) -> dict:
    """Summarize a model's components, Vars and Constraints as small plain data.

    Args:
        model: The Pyomo model to summarize.

    Returns:
        A JSON-serializable dict: sorted component names, plus a digest per Var
        and active Constraint component covering every entry's bounds, fixed
        state and body terms.
    """
    entries: dict[str, dict] = {"vars": {}, "constraints": {}}
    for name, var in _collect(model, pyo.Var).items():
        state = [var.lb, var.ub, var.fixed, var.value if var.fixed else None]
        entries["vars"].setdefault(_name(var.parent_component(), model), {})[
            name
        ] = state
    for name, con in _collect(model, pyo.Constraint, active=True).items():
        constant, linear, quadratic, nonlinear = _repn_key(con.body, model)
        state = [con.lower, con.upper, constant, linear, quadratic, nonlinear]
        entries["constraints"].setdefault(_name(con.parent_component(), model), {})[
            name
        ] = state
    digest = {
        kind: {
            component: hashlib.sha1(
                json.dumps(states, sort_keys=True, default=str).encode()
            ).hexdigest()
            for component, states in components.items()
        }
        for kind, components in entries.items()
    }
    return {
        "components": sorted(
            _name(c, model) for c in model.component_objects(descend_into=True)
        ),
        **digest,
    }
