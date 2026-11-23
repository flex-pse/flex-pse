"""ConstantIntensitySurrogate: power proportional to flow, via the unit's own Var.

Unlike the other surrogates it swaps no relation. A unit built with
:meth:`~flexops.core.ops_block.OpsBlockData.add_constant_intensity_relation`
already has the relationship ``power_electrical == energy_intensity * flow``
and registers ``energy_intensity`` as a regressable process parameter; this
surrogate sets that parameter's value and fixed flag, and reads them back.
"""

from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateSpec, SurrogateType, _StrictModel
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import same_variable
from flexops.surrogates._config import _check_units
from flexops.surrogates.base import Surrogate

INTENSITY = nm.INTENSITY_VARS[nm.PowerKind.ELECTRICAL]
"""str: the one parameter, the unit's ``energy_intensity``."""


class ConstantIntensityConfig(_StrictModel):
    """A constant-intensity surrogate's config: the variables, for reference.

    The relationship's variables are the unit's own, so these record which
    columns a fit used rather than wiring anything. They are still checked
    when the spec is applied: each must name the unit's intensity basis flow
    or its electrical power draw.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_variables: dict[str, str] = {}
    output_variables: dict[str, str] = {}

    @model_validator(mode="after")
    def _units_parse(self) -> "ConstantIntensityConfig":
        # Units are documentation here, so an empty string (unknown) is allowed.
        for mapping in (self.input_variables, self.output_variables):
            _check_units({name: units for name, units in mapping.items() if units})
        return self


class ConstantIntensitySurrogate(Surrogate):
    """Sets a unit's ``energy_intensity`` rather than swapping a relation.

    The spec is::

        {"surrogate_type": "constant_intensity",
         "config": {"input_variables": {"flow_out": "m^3/hr"},
                    "output_variables": {"power_electrical": "kW"}},
         "parameters": {"energy_intensity": {"value": 0.9, "fixed": true}}}
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.CONSTANT_INTENSITY
    config_model: ClassVar[type[ConstantIntensityConfig]] = ConstantIntensityConfig
    replaces_relation: ClassVar[bool] = False

    @classmethod
    def parameter_shapes(cls, config: ConstantIntensityConfig) -> dict[str, tuple]:
        """Return the one scalar parameter, ``energy_intensity``."""
        del config
        return {INTENSITY: ()}

    @classmethod
    def make_spec(
        cls,
        *,
        energy_intensity: float | None,
        input_variables: dict[str, str] | None = None,
        output_variables: dict[str, str] | None = None,
        fixed: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> SurrogateSpec:
        """Return a validated constant-intensity spec.

        Args:
            energy_intensity: The intensity, in the unit's declared units, or
                ``None`` to leave a free parameter at its current value.
            input_variables: The fitted input column and its units.
            output_variables: The fitted output column and its units.
            fixed: Whether the unit holds the intensity fixed.
            provenance: Free-form fit metadata.

        Returns:
            The spec.
        """
        config: dict[str, Any] = {}
        if input_variables:
            config["input_variables"] = dict(input_variables)
        if output_variables:
            config["output_variables"] = dict(output_variables)
        values = {} if energy_intensity is None else {INTENSITY: energy_intensity}
        return cls._make(
            config=config,
            values=values,
            fixed={INTENSITY: fixed},
            provenance=provenance,
        )

    @property
    def input_variables(self) -> dict[str, str]:
        """Return the recorded input variable names; not resolved at build."""
        return dict(self.config.input_variables)

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the recorded output variable names; not resolved at build."""
        return dict(self.config.output_variables)

    @staticmethod
    def parameter_of(unit):
        """Return ``unit``'s regressable ``energy_intensity`` component.

        Raises:
            FlexConfigError: If ``unit`` does not register one.
        """
        regressable = {
            record.name: record.param
            for record in unit._io_registry.parameters
            if record.regressable
        }
        if INTENSITY not in regressable:
            raise FlexConfigError(
                f"A 'constant_intensity' relationship determines {INTENSITY!r}, "
                f"which {unit.name!r} does not register as a regressable process "
                f"parameter (it registers {sorted(regressable)}). Give this unit "
                "a surrogate that swaps its relation instead.",
                field=INTENSITY,
                value=unit.name,
            )
        return regressable[INTENSITY]

    def apply_to_unit(self, unit) -> Any:
        """Set ``unit``'s ``energy_intensity`` value and fixed flag from the spec.

        Args:
            unit: The built unit.

        Returns:
            The intensity component.

        Raises:
            FlexConfigError: If a named variable is not on the unit or is not
                the one its intensity relates, if the unit has no regressable
                intensity, or if the spec frees an intensity that is a Param
                (which can't be free).
        """
        self._check_variables(unit)
        parameter = self.parameter_of(unit)
        state = self.spec.parameters[INTENSITY]
        if state.value is not None:
            unit.update_parameters({INTENSITY: state.value})
        if parameter.is_variable_type():
            if state.fixed:
                parameter.fix()
            else:
                parameter.unfix()
        elif not state.fixed:
            raise FlexConfigError(
                f"{unit.name}.{INTENSITY} is a Param, so it can't be left free; set "
                "fixed to true.",
                field=f"parameters.{INTENSITY}.fixed",
                value=False,
            )
        return parameter

    def _check_variables(self, unit) -> None:
        """Raise unless each named variable is on ``unit`` and is the right one.

        The names aren't used to build anything, since the relation's
        variables are the unit's own, but a mistyped or wrong name would
        record a fit against the wrong column. Each must resolve on the unit
        (or, as :func:`~flexparameterize.apply.apply_to_model` writes them,
        from the model root). The input must be the flow the intensity is
        metered on, and the output the unit's electrical power draw.
        """
        kind = nm.PowerKind.ELECTRICAL
        registry = unit._io_registry
        basis = registry.intensity_basis.get(kind)
        powers = [record.var for record in registry.power if record.kind is kind]
        expected = {
            "input_variables": unit.find_component(basis) if basis else None,
            "output_variables": powers[0] if len(powers) == 1 else None,
        }
        for field, names in (
            ("input_variables", self.config.input_variables),
            ("output_variables", self.config.output_variables),
        ):
            for name in names:
                var = unit.resolve_variable(name, field=field, model_wide=True)
                target = expected[field]
                if target is not None and not same_variable(var, target):
                    role = field.split("_")[0]
                    raise FlexConfigError(
                        f"constant_intensity names {role} {name!r}, but "
                        f"{unit.name!r}'s energy intensity relates "
                        f"{target.local_name!r}. Name that variable as the "
                        f"{role}.",
                        field=field,
                        value=name,
                    )

    def read_from_unit(self, unit) -> SurrogateSpec:
        """Return this surrogate's spec with ``unit``'s live intensity.

        Args:
            unit: The built unit.

        Returns:
            The spec, which re-applies the intensity as it stands.
        """
        parameter = self.parameter_of(unit)
        fixed = parameter.fixed if parameter.is_variable_type() else True
        value = parameter.value
        state = (
            {"fixed": False}
            if value is None
            else {"value": float(value), "fixed": bool(fixed)}
        )
        return SurrogateSpec.model_validate(
            {
                **self.spec.model_dump(),
                "parameters": {INTENSITY: state},
            }
        )

    def _build(self, unit, target):
        """Raise: a constant intensity swaps no relation; use apply_to_unit."""
        del unit, target
        raise FlexConfigError(
            "A constant_intensity surrogate swaps no relation; it sets the unit's "
            f"{INTENSITY!r}. Apply it with apply_relation_spec or apply_to_unit.",
            field="surrogate_type",
            value=self.surrogate_type,
        )
