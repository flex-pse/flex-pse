"""Structural comparison of two Pyomo models."""

import math

import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.repn import generate_standard_repn

from flexops.core.time_block import find_time_block


def _name(component, root) -> str:
    """Return a component's fully qualified name relative to ``root``."""
    return component.getname(fully_qualified=True, relative_to=root)


def _close(x, y, rel_tol: float) -> bool:
    """Return whether two numbers (or Nones) agree within tolerance."""
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-12)


def _repn_key(expr, root):
    """Reduce an expression to (constant, linear, quadratic, nonlinear) terms."""
    r = generate_standard_repn(expr, compute_values=True, quadratic=True)
    lin = sorted(
        (_name(v, root), c) for v, c in zip(r.linear_vars, r.linear_coefs, strict=True)
    )
    quad = sorted(
        (tuple(sorted(_name(x, root) for x in pair)), c)
        for pair, c in zip(r.quadratic_vars, r.quadratic_coefs, strict=True)
    )
    nonlin = None if r.nonlinear_expr is None else str(r.nonlinear_expr)
    return r.constant, lin, quad, nonlin


def _compare_repn(key_a, key_b, label: str, rel_tol: float, diffs: list) -> None:
    """Append the differences between two ``_repn_key`` results to ``diffs``."""
    const_a, lin_a, quad_a, nonlin_a = key_a
    const_b, lin_b, quad_b, nonlin_b = key_b
    if not _close(const_a, const_b, rel_tol):
        diffs.append(f"{label}.constant: {const_a} != {const_b}")
    for kind, terms_a, terms_b in (
        ("linear", lin_a, lin_b),
        ("quadratic", quad_a, quad_b),
    ):
        coefs_a, coefs_b = dict(terms_a), dict(terms_b)
        if coefs_a.keys() != coefs_b.keys():
            diffs.append(
                f"{label}.{kind} terms differ: "
                f"{sorted(map(str, coefs_a.keys() ^ coefs_b.keys()))}"
            )
            continue
        for term, coef in coefs_a.items():
            if not _close(coef, coefs_b[term], rel_tol):
                diffs.append(f"{label}.{kind}[{term}]: {coef} != {coefs_b[term]}")
    if nonlin_a != nonlin_b:
        diffs.append(f"{label}.nonlinear: {nonlin_a} != {nonlin_b}")


def _compare_sets(kind: str, names_a: set, names_b: set, diffs: list) -> set:
    """Record names found in only one model and return those in both."""
    diffs.extend(f"Only in a: {n} ({kind})" for n in sorted(names_a - names_b))
    diffs.extend(f"Only in b: {n} ({kind})" for n in sorted(names_b - names_a))
    return names_a & names_b


def _collect(model, ctype, **kwargs) -> dict:
    """Map component path to data object for every ``ctype`` on ``model``."""
    return {
        _name(data, model): data
        for data in model.component_data_objects(ctype, descend_into=True, **kwargs)
    }


def model_differences(
    a, b, *, rel_tol: float = 1e-9, check_values: bool = False
) -> list[str]:
    """Return every difference between two Pyomo models, sorted.

    Two models are equivalent when, keyed by component path relative to the model:
    they have the same Var, Param, active Constraint, active Objective and Arc
    names; each Var has the same domain, bounds and fixed flag (and value when
    fixed); each mutable Param has the same value; each active Constraint has
    the same bounds and body; the objective has the same sense and expression;
    and both TimeBlocks have the same time_index and dt.

    Args:
        a: The first model.
        b: The second model.
        rel_tol: Relative tolerance for numeric comparisons.
        check_values: Also compare values of unfixed Vars (useful after a solve).

    Returns:
        One line per difference; empty when the models are equivalent.
    """
    diffs: list[str] = []

    vars_a, vars_b = _collect(a, pyo.Var), _collect(b, pyo.Var)
    for name in sorted(_compare_sets("Var", set(vars_a), set(vars_b), diffs)):
        va, vb = vars_a[name], vars_b[name]
        if str(va.domain) != str(vb.domain):
            diffs.append(f"{name}.domain: {va.domain} != {vb.domain}")
        for attr in ("lb", "ub"):
            if not _close(getattr(va, attr), getattr(vb, attr), rel_tol):
                diffs.append(
                    f"{name}.{attr}: {getattr(va, attr)} != {getattr(vb, attr)}"
                )
        if va.fixed != vb.fixed:
            diffs.append(f"{name}.fixed: {va.fixed} != {vb.fixed}")
        elif (va.fixed or check_values) and not _close(va.value, vb.value, rel_tol):
            diffs.append(f"{name}.value: {va.value} != {vb.value}")

    params_a, params_b = _collect(a, pyo.Param), _collect(b, pyo.Param)
    for name in sorted(_compare_sets("Param", set(params_a), set(params_b), diffs)):
        pa, pb = params_a[name], params_b[name]
        if pa.parent_component().mutable and not _close(
            pyo.value(pa), pyo.value(pb), rel_tol
        ):
            diffs.append(f"{name}.value: {pyo.value(pa)} != {pyo.value(pb)}")

    arcs_a, arcs_b = _collect(a, Arc, active=None), _collect(b, Arc, active=None)
    for name in sorted(_compare_sets("Arc", set(arcs_a), set(arcs_b), diffs)):
        if arcs_a[name].active != arcs_b[name].active:
            diffs.append(
                f"{name}.active: {arcs_a[name].active} != {arcs_b[name].active}"
            )

    cons_a = _collect(a, pyo.Constraint, active=True)
    cons_b = _collect(b, pyo.Constraint, active=True)
    for name in sorted(_compare_sets("Constraint", set(cons_a), set(cons_b), diffs)):
        ca, cb = cons_a[name], cons_b[name]
        for attr in ("lower", "upper"):
            bound_a, bound_b = getattr(ca, attr), getattr(cb, attr)
            value_a = None if bound_a is None else pyo.value(bound_a)
            value_b = None if bound_b is None else pyo.value(bound_b)
            if not _close(value_a, value_b, rel_tol):
                diffs.append(f"{name}.{attr}: {value_a} != {value_b}")
        _compare_repn(
            _repn_key(ca.body, a), _repn_key(cb.body, b), name, rel_tol, diffs
        )

    objs_a = _collect(a, pyo.Objective, active=True)
    objs_b = _collect(b, pyo.Objective, active=True)
    for name in sorted(_compare_sets("Objective", set(objs_a), set(objs_b), diffs)):
        oa, ob = objs_a[name], objs_b[name]
        if oa.sense != ob.sense:
            diffs.append(f"{name}.sense: {oa.sense} != {ob.sense}")
        _compare_repn(
            _repn_key(oa.expr, a), _repn_key(ob.expr, b), name, rel_tol, diffs
        )

    time_a, time_b = find_time_block(a), find_time_block(b)
    if list(time_a.time_index) != list(time_b.time_index):
        diffs.append("time_block.time_index differs")
    if not _close(pyo.value(time_a.dt), pyo.value(time_b.dt), rel_tol):
        diffs.append(f"time_block.dt: {pyo.value(time_a.dt)} != {pyo.value(time_b.dt)}")

    return sorted(diffs)
