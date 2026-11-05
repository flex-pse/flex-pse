"""MultilinearSurrogate: a constant plus each input's own linear terms plus
their pairwise cross terms -- the expanded multilinear form that also covers
what used to be called ``linear`` (no cross terms) and ``bilinear`` (one).
"""

from typing import ClassVar

import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import CoefficientRegistry
from flexops.core.units import parse_units
from flexops.surrogates.base import Surrogate

_INTERCEPT = "intercept"
"""str: reserved coefficient key naming the relationship's constant term."""

_DATA_KEYS = ("input_variables", "output_variables", "coefficients")


class MultilinearSurrogate(Surrogate):
    """A constant plus a sum of coefficient * (product of distinct inputs).

    ``data`` is::

        {"input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
         "output_variables": {"power_electrical": "kW"},
         "coefficients": {"intercept": 5.0, "flow_out": 0.4,
                           "flow_out*outlet_state.pressure": 1.2e-5}}

    A coefficient key is a ``*``-separated product of names from
    ``input_variables``, each appearing at most once (no ``^`` exponent, no
    repeated factor -- that is what makes it *multi*\\ linear rather than
    polynomial); the reserved key ``"intercept"`` is the constant term, read
    in the declared output units. Each variable is resolved on the unit at
    build time and converted from its own units into the units declared here
    before use, so a coefficient is read in the basis the fit declared, not
    whatever units the model happens to carry.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.MULTILINEAR

    def _validate(self) -> None:
        """Validate the multilinear ``data`` contract (see class docstring).

        Raises:
            FlexConfigError: If a key is missing, an extra key is present, a
                units string does not parse, ``output_variables`` does not
                have exactly one entry, a coefficient key names a factor not
                in ``input_variables``, or a coefficient key carries a ``^``
                exponent or repeats a factor.
        """
        unknown = sorted(set(self.data) - set(_DATA_KEYS))
        if unknown:
            raise FlexConfigError(
                f"multilinear surrogate data carries unknown key(s) {unknown}; "
                f"it may only have {_DATA_KEYS}.",
                field="data",
                value=unknown,
            )
        missing = sorted(set(_DATA_KEYS) - set(self.data))
        if missing:
            raise FlexConfigError(
                f"multilinear surrogate data is missing key(s) {missing}; it "
                f"must have {_DATA_KEYS}.",
                field="data",
                value=missing,
            )

        inputs = self.data["input_variables"]
        if not isinstance(inputs, dict) or not inputs:
            raise FlexConfigError(
                "multilinear surrogate 'input_variables' must be a non-empty "
                f"{{name: units}} mapping, got {inputs!r}.",
                field="input_variables",
                value=inputs,
            )
        outputs = self.data["output_variables"]
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise FlexConfigError(
                "multilinear surrogate 'output_variables' must be a single "
                f"{{name: units}} entry -- a registered relation has one "
                f"target -- got {outputs!r}.",
                field="output_variables",
                value=outputs,
            )
        declared_fields = (("input_variables", inputs), ("output_variables", outputs))
        for field, mapping in declared_fields:
            for name, units in mapping.items():
                if not name or not isinstance(units, str) or not units:
                    raise FlexConfigError(
                        f"multilinear surrogate {field!r} entry {name!r} must "
                        f"map to a non-empty units string, got {units!r}.",
                        field=field,
                        value=units,
                    )
                parse_units(units)

        coefficients = self.data["coefficients"]
        if not isinstance(coefficients, dict) or not coefficients:
            raise FlexConfigError(
                "multilinear surrogate 'coefficients' must be a non-empty "
                f"mapping, got {coefficients!r}.",
                field="coefficients",
                value=coefficients,
            )
        for key, value in coefficients.items():
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise FlexConfigError(
                    f"multilinear surrogate coefficient {key!r} must be a "
                    f"number, got {value!r}.",
                    field="coefficients",
                    value=key,
                ) from exc
            if key == _INTERCEPT:
                continue
            factors = key.split("*")
            if any("^" in factor for factor in factors):
                raise FlexConfigError(
                    f"Coefficient term {key!r} carries a '^' exponent; "
                    "multilinear terms are degree 1 in each factor. Use a "
                    "surrogate_type that admits higher degree (e.g. "
                    "'quadratic').",
                    field="coefficients",
                    value=key,
                )
            if len(set(factors)) != len(factors):
                raise FlexConfigError(
                    f"Coefficient term {key!r} repeats a factor; multilinear "
                    "terms admit each input at most once. Use a "
                    "surrogate_type that admits higher degree (e.g. "
                    "'quadratic').",
                    field="coefficients",
                    value=key,
                )
            unknown_factors = [factor for factor in factors if factor not in inputs]
            if unknown_factors:
                raise FlexConfigError(
                    f"Coefficient term {key!r} names {unknown_factors}, not in "
                    f"'input_variables' ({sorted(inputs)}).",
                    field="coefficients",
                    value=key,
                )

    @property
    def intercept(self) -> float:
        """Return the ``"intercept"`` coefficient, ``0.0`` if it is not given."""
        return float(self.data["coefficients"].get(_INTERCEPT, 0.0))

    @property
    def input_variables(self) -> dict[str, str]:
        """Return the declared input variable names and their units."""
        return dict(self.data["input_variables"])

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the one declared output variable name and its units."""
        return dict(self.data["output_variables"])

    def build(self, unit, target):
        """Return ``(block, body(t))`` in this surrogate's declared output units.

        The returned block is a ``pyo.Block(concrete=True)`` carrying a
        ``coefficient_vars`` indexed ``pyo.Var`` (keyed by coefficient name
        plus the reserved ``"intercept"`` key) and a ``coefficients``
        ``CoefficientRegistry`` over that indexed Var. The block is returned
        un-added; ``swap_relation`` attaches it to ``unit`` itself.

        Args:
            unit: The unit the relationship is built on.
            target: Unused -- a multilinear body reads its own declared
                units -- but part of every builder's signature.

        Returns:
            A tuple ``(block, body)`` where ``block`` is the surrogate's
            coefficient block (or ``None`` if no coefficients are present),
            and ``body`` is a callable taking a time index and returning a
            Pyomo expression in the declared output units.
        """
        del target
        output_units = parse_units(next(iter(self.output_variables.values())))
        declared = {
            name: (
                unit.resolve_variable(name, field="input_variables"),
                parse_units(units),
            )
            for name, units in self.input_variables.items()
        }

        coefficients_data = self.data["coefficients"]
        all_keys = list(coefficients_data.keys())
        if _INTERCEPT not in all_keys:
            all_keys.append(_INTERCEPT)

        index_set = pyo.Set(initialize=all_keys)
        coefficient_vars = pyo.Var(index_set, initialize=1.0)
        block = pyo.Block(concrete=True)
        block.coefficient_vars = coefficient_vars

        for key in all_keys:
            value = float(coefficients_data.get(key, 0.0))
            coefficient_vars[key].set_value(value)
            coefficient_vars[key].fix()

        block.coefficients = CoefficientRegistry()
        block.coefficients.register_coefficients(coefficient_vars)

        def body(t):
            total = coefficient_vars[_INTERCEPT]
            for key, _ in coefficients_data.items():
                if key == _INTERCEPT:
                    continue
                term = coefficient_vars[key]
                for name in key.split("*"):
                    var, units = declared[name]
                    try:
                        converted = pyunits.convert(var[t], units)
                    except UnitsError as exc:
                        raise FlexConfigError(
                            f"multilinear surrogate declares {name!r} in "
                            f"{units!s}, incompatible with its actual units "
                            f"{pyunits.get_units(var[t])!s} on the unit.",
                            field="input_variables",
                            value=name,
                        ) from exc
                    term = term * (converted / units)
                total = total + term
            return total * output_units

        return block, body
