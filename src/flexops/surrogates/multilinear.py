"""MultilinearSurrogate: a constant plus each input's own linear terms plus
their pairwise cross terms -- the expanded multilinear form that also covers
what used to be called ``linear`` (no cross terms) and ``bilinear`` (one).
"""

from typing import Any, ClassVar

import pyomo.environ as pyo
from pydantic import field_validator, model_validator
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import CoefficientRegistry
from flexops.core.units import parse_units
from flexops.surrogates._config import VariablesConfig
from flexops.surrogates.base import Surrogate

INTERCEPT = "intercept"
"""str: the parameter naming the relationship's constant term."""


class MultilinearConfig(VariablesConfig):
    """A multilinear surrogate's config: its variables and its product terms.

    Each term is a ``*``-separated product of distinct input names, e.g.
    ``"flow_out*outlet_state.pressure"``. Each input appears at most once per
    term and carries no exponent; that is what makes the form *multi*-linear
    rather than polynomial.
    """

    terms: list[str]

    @field_validator("input_variables")
    @classmethod
    def _some_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("must declare at least one input")
        return value

    @model_validator(mode="after")
    def _terms_are_multilinear(self) -> "MultilinearConfig":
        if len(set(self.terms)) != len(self.terms):
            raise ValueError(f"terms repeat an entry: {self.terms}")
        for term in self.terms:
            if term == INTERCEPT:
                raise ValueError(
                    f"{INTERCEPT!r} is always a parameter; don't list it in terms"
                )
            factors = term.split("*")
            if any("^" in factor for factor in factors):
                raise ValueError(
                    f"term {term!r} carries a '^' exponent; multilinear terms are "
                    "degree 1 in each factor. Use a surrogate_type that admits "
                    "higher degree (e.g. 'quadratic')."
                )
            if len(set(factors)) != len(factors):
                raise ValueError(
                    f"term {term!r} repeats a factor; multilinear terms admit each "
                    "input at most once. Use a surrogate_type that admits higher "
                    "degree (e.g. 'quadratic')."
                )
            unknown = [
                factor for factor in factors if factor not in self.input_variables
            ]
            if unknown:
                raise ValueError(
                    f"term {term!r} names {unknown}, not in input_variables "
                    f"({sorted(self.input_variables)})"
                )
        return self


class MultilinearSurrogate(Surrogate):
    """A constant plus a sum of coefficient * (product of distinct inputs).

    The spec is::

        {"surrogate_type": "multilinear",
         "config": {
             "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
             "output_variables": {"power_electrical": "kW"},
             "terms": ["flow_out", "flow_out*outlet_state.pressure"]},
         "parameters": {
             "intercept": {"value": 5.0, "fixed": true},
             "flow_out": {"value": 0.4, "fixed": true},
             "flow_out*outlet_state.pressure": {"value": 1.2e-5, "fixed": true}}}

    The parameters are ``"intercept"`` (the constant term, read in the
    declared output units) plus one scalar per term, named by the term. A
    free parameter without a value starts at ``0.0``.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.MULTILINEAR
    config_model: ClassVar[type[MultilinearConfig]] = MultilinearConfig

    @classmethod
    def parameter_shapes(cls, config: MultilinearConfig) -> dict[str, tuple]:
        """Return ``intercept`` plus one scalar per term."""
        return {INTERCEPT: (), **{term: () for term in config.terms}}

    @classmethod
    def make_spec(
        cls,
        *,
        input_variables: dict[str, str],
        output_variables: dict[str, str],
        coefficients: dict[str, float],
        fixed: dict[str, bool] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> SurrogateSpec:
        """Return a validated multilinear spec.

        Args:
            input_variables: ``{name: units}`` of every input.
            output_variables: ``{name: units}`` of the one output.
            coefficients: ``{term: value}``; each key other than
                ``"intercept"`` becomes a term, in this order. A missing
                ``"intercept"`` is ``0.0``.
            fixed: ``{name: False}`` for each parameter to leave free; every
                other parameter is fixed.
            provenance: Free-form fit metadata.

        Returns:
            The spec.

        Raises:
            FlexConfigError: If the result does not match the contract.
        """
        values = {INTERCEPT: 0.0, **coefficients}
        return cls._make(
            config={
                "input_variables": dict(input_variables),
                "output_variables": dict(output_variables),
                "terms": [key for key in coefficients if key != INTERCEPT],
            },
            values=values,
            fixed=fixed,
            provenance=provenance,
        )

    @property
    def input_variables(self) -> dict[str, str]:
        """Return the declared input variable names and their units."""
        return dict(self.config.input_variables)

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the one declared output variable name and its units."""
        return dict(self.config.output_variables)

    def _build(self, unit, target):
        """Return ``(block, body(t))`` in this surrogate's declared output units.

        The returned block is a ``pyo.Block(concrete=True)`` carrying a
        ``coefficient_vars`` Var indexed by parameter name and a
        ``coefficients`` registry with one scalar parameter per entry.

        Args:
            unit: The unit the relationship is built on.
            target: Unused -- a multilinear body reads its own declared
                units -- but part of every builder's signature.

        Returns:
            ``(block, body)``, ``body`` taking a time index and returning a
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

        names = list(self.parameter_shapes(self.config))
        block = pyo.Block(concrete=True)
        block.coefficient_vars = pyo.Var(
            pyo.Set(initialize=names, ordered=True),
            initialize={name: float(self.value_of(name)) for name in names},
        )
        coefficient_vars = block.coefficient_vars
        block.coefficients = CoefficientRegistry()
        block.coefficients.register_coefficients(coefficient_vars)
        terms = list(self.config.terms)

        def body(t):
            total = coefficient_vars[INTERCEPT]
            for term in terms:
                product = coefficient_vars[term]
                for name in term.split("*"):
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
                    product = product * (converted / units)
                total = total + product
            return total * output_units

        return block, body
