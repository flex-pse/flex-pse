r"""Mixer(OpsBlockData): N named inlets joined into one outlet (§3.2, §3.4).

A mixing junction: an arbitrary number of streams joined into one. Its port
count is a config option, so no fixed-arity IO-topology base fits — it
subclasses :class:`~flexops.core.ops_block.OpsBlockData` directly and
hand-writes its ports and balance, the way
:class:`~flexops.unit_models.powergeneration.combustor.Combustor` does.

**Density is constant across the junction**, so volume is conserved directly
and the whole flow model is one linear equality:

.. math::

    \dot{V}_{out}[t] = \sum_i \dot{V}_{in,i}[t]

There is no energy input: a mixer declares neither ``power_electrical`` nor
``power_thermal``, and registers no power.

Every stream gets its **own port**. Pyomo's ``Port`` members are equalities
here, not extensive quantities apportioned across several arcs, so N inlets
means N ports each carrying exactly one arc — never one inlet port fed by N
arcs.

Pressure (and any other intensive state the property package carries) is
treated as it is at a physical mixing node: every inlet is held at the
**reference inlet**'s value — the first name in ``inlet_names`` — and that
value passes through to the outlet. Temperature is the one intensive state with
a choice, ``config.temperature_mixing``:

* :attr:`MixerTemperatureRule.EQUAL` (the default) treats temperature like
  pressure — every inlet equals the reference inlet's, which passes through to
  the outlet. Isothermal mixing; the block stays linear.
* :attr:`MixerTemperatureRule.FLOW_WEIGHTED` leaves the inlet temperatures
  independent and blends them by volume:

  .. math::

      T_{out}[t] \cdot \dot{V}_{out}[t] = \sum_i \dot{V}_{in,i}[t] \cdot
      T_{in,i}[t]

  which is the correct mixing rule under this unit's constant density plus a
  constant heat capacity.

.. note::
   ``FLOW_WEIGHTED`` is **bilinear**, so a model containing one is an NLP and
   needs IPOPT rather than HiGHS — the same caveat
   :class:`~flexops.unit_models.storage.tank.Tank` documents for
   ``capacity * level``. It also leaves ``T_out`` undetermined at a time point
   where every inlet flow is zero (the equation degenerates to 0 == 0).

Because the model is *only* this volumetric balance, two configurations reach
past what it represents, and each raises a ``UserWarning`` at construction
rather than failing:

* **A property package carrying more than one component.** Total volumetric
  flow is summed and no per-component mass balance is written, so the outlet
  composition is not tracked. Mix streams that share one composition.
* **A vapor phase under** :attr:`MixerTemperatureRule.FLOW_WEIGHTED`. Volumes
  are additive only at equal temperature and pressure; otherwise the equation
  of state sets the mixed volume, and this unit conserves volume directly
  instead. Pressure is held equal across inlets by construction under either
  rule, and so is temperature under
  :attr:`MixerTemperatureRule.EQUAL` — so this is the one configuration where a
  gas can actually mix at unequal conditions.
"""

import enum
import warnings

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue

from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.unit_models._multiport import single_flow_phase, validate_port_names


class MixerTemperatureRule(enum.StrEnum):
    """How a :class:`Mixer` sets its outlet temperature."""

    EQUAL = "equal"
    FLOW_WEIGHTED = "flow_weighted"


def _inlet_names_domain(value) -> tuple[str, ...]:
    """ConfigValue domain: coerce to a tuple.

    Only coerces the type; emptiness/uniqueness/non-empty-string checks happen
    in ``build()`` via
    :func:`~flexops.unit_models._multiport.validate_port_names`, so they raise
    :class:`~flexcore.exceptions.FlexConfigError` directly rather than the
    ``ValueError`` Pyomo's ``ConfigValue`` wraps every domain-raised exception
    into.
    """
    return tuple(value)


def _temperature_mixing_domain(value) -> MixerTemperatureRule:
    """ConfigValue domain: coerce to a :class:`MixerTemperatureRule`."""
    try:
        return MixerTemperatureRule(value)
    except ValueError as exc:
        allowed = ", ".join(repr(rule.value) for rule in MixerTemperatureRule)
        raise FlexConfigError(
            f"temperature_mixing must be one of {allowed}, got {value!r}.",
            field="temperature_mixing",
            value=value,
        ) from exc


@declare_process_block_class("Mixer")
class MixerData(OpsBlockData):
    r"""N named inlet streams summed into one outlet, at constant density.

    See the module docstring for the balance, the intensive-state treatment,
    and the documented simplifications. ``inlet_names`` sets the inlet count and
    their port names (``f"inlet_{name}"``); its first entry is the reference
    inlet every other inlet's intensive states are tied to.

    Config:
        ``property_package`` (inherited): a single-phase package shared by
        every port. ``inlet_names`` (default ``("a", "b")``): the inlets'
        role/port names, which must be unique and non-empty.
        ``temperature_mixing`` (default
        :attr:`MixerTemperatureRule.EQUAL`): how the outlet temperature is set,
        and only meaningful on a property package that carries a temperature
        state — requesting :attr:`MixerTemperatureRule.FLOW_WEIGHTED` on one
        that does not is rejected.

    Warns:
        UserWarning: If ``property_package`` carries more than one component
            (the outlet composition is not weighted), or if it is a vapor phase
            and ``temperature_mixing`` is
            :attr:`MixerTemperatureRule.FLOW_WEIGHTED` (the mixed volume is
            approximate, since the equation of state is not applied). See the
            module docstring.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Mixer
        >>> m = dummy_time_block(3)
        >>> m.junction = Mixer(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     inlet_names=("sludge", "recycle", "makeup"),
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "inlet_names",
        ConfigValue(
            default=("a", "b"),
            domain=_inlet_names_domain,
            description="Role names of the mixer's inlets; inlet i is built as "
            "port f'inlet_{name}'. Must be unique and non-empty. The first "
            "name is the reference inlet, whose intensive states every other "
            "inlet is held at and which passes through to the outlet.",
        ),
    )
    CONFIG.declare(
        "temperature_mixing",
        ConfigValue(
            default=MixerTemperatureRule.EQUAL,
            domain=_temperature_mixing_domain,
            description="How the outlet temperature is set: 'equal' (every "
            "inlet is held at the reference inlet's temperature, which passes "
            "through to the outlet — isothermal and linear) or "
            "'flow_weighted' (the inlet temperatures stay independent and the "
            "outlet is their volume-weighted blend — bilinear, so the model "
            "becomes an NLP). Only meaningful when the property package "
            "carries a temperature state.",
        ),
    )

    def build(self) -> None:
        """Validate the config, then build the ports, balance, and outlet state."""
        super().build()
        validate_port_names(self.config.inlet_names, "inlet_names")
        self._phase = single_flow_phase(self.config.property_package, "Mixer")
        self._warn_if_multicomponent()
        self.add_stream_ports(
            inlet_ports=self._inlet_port_names(), outlet_ports=("outlet",)
        )
        self._blend_temperature = self._resolve_temperature_rule()
        self._warn_if_gas_blended_at_unequal_temperature()
        self._register_stream_states()
        self._build_mass_balance()
        self._build_outlet_state()

    # -- simplification warnings --------------------------------------------
    # TODO: these two warnings go through the standard library's `warnings`
    # module because flex-pse has no logging facility of its own yet. Move them
    # onto the project logging class when it lands (issue #61), so a caller can
    # silence or route them alongside every other flex-pse diagnostic.

    def _warn_if_multicomponent(self) -> None:
        """Warn that a multi-component package's composition is not tracked.

        The balance sums *total* volumetric flow and writes no per-component
        mass balance, so streams of differing composition blend into an outlet
        whose composition this unit simply does not carry. Not an error: a
        caller who knows the composition is uniform (or who does not care about
        it) still gets a correct total-volume balance.
        """
        components = list(self.config.property_package.component_list)
        if len(components) < 2:
            return
        warnings.warn(
            f"Mixer {self.name} was built on a property package carrying "
            f"multiple components ({', '.join(components)}), but it models "
            "simple volumetric mixing only: the balance sums total volumetric "
            "flow and writes no per-component mass balance, so the outlet "
            "composition is not tracked. Use it only where the inlet streams "
            "share one composition.",
            stacklevel=2,
        )

    def _warn_if_gas_blended_at_unequal_temperature(self) -> None:
        """Warn that additive volumes need equal temperature for a vapor phase.

        Volumes are only additive at equal temperature and pressure; otherwise
        the equation of state sets the mixed volume, and this unit conserves
        volume directly instead. Inlet pressures are held equal by construction
        and so are inlet temperatures -- except under
        :attr:`MixerTemperatureRule.FLOW_WEIGHTED`, which deliberately leaves
        them independent. That leaves exactly one configuration to flag, known
        entirely from the config and the package's metadata.
        """
        phase = self.config.property_package.get_phase(self._phase)
        if not (phase.is_vapor_phase() and self._blend_temperature):
            return
        warnings.warn(
            f"Mixer {self.name} blends a vapor-phase stream under "
            "temperature_mixing='flow_weighted', so its inlets may enter at "
            "different temperatures. This unit models simple volumetric "
            "mixing: volume is conserved directly and the equation of state is "
            "not applied, so the mixed volume is only approximate. Use "
            "temperature_mixing='equal' for an exact volumetric balance.",
            stacklevel=2,
        )

    # -- config resolution --------------------------------------------------

    def _inlet_port_names(self) -> tuple[str, ...]:
        """Return the ``f"inlet_{name}"`` port names, in ``inlet_names`` order."""
        return tuple(f"inlet_{name}" for name in self.config.inlet_names)

    def _reference_inlet_name(self) -> str:
        """Return the first configured inlet name -- the mixed-stream reference."""
        return self.config.inlet_names[0]

    def _flow_basis_name(self) -> str:
        """Return the property package's extensive flow state-variable name."""
        return self.config.property_package.get_flow_basis_var_name()

    def _inlet_state(self, name: str):
        """Return the state block behind the inlet named ``name``."""
        return self.find_component(f"inlet_{name}_state")

    def _resolve_temperature_rule(self) -> bool:
        """Return whether the outlet temperature is a flow-weighted blend.

        Returns:
            True when ``temperature_mixing`` selected
            :attr:`MixerTemperatureRule.FLOW_WEIGHTED`.

        Raises:
            FlexConfigError: If that rule was requested on a property package
                carrying no ``temperature`` state variable — there would be
                nothing to blend, and silently ignoring the option would hide
                the mismatch.
        """
        if self.config.temperature_mixing is not MixerTemperatureRule.FLOW_WEIGHTED:
            return False
        if "temperature" not in self.outlet_state.define_state_vars():
            raise FlexConfigError(
                "temperature_mixing='flow_weighted' needs a property_package "
                "carrying a temperature state variable; this one carries "
                f"{sorted(self.outlet_state.define_state_vars())}. Build the "
                "package with has_temperature=True, or leave "
                "temperature_mixing at its 'equal' default.",
                field="temperature_mixing",
                value=self.config.temperature_mixing,
            )
        return True

    # -- ports, balance, outlet state ---------------------------------------

    def _register_stream_states(self) -> None:
        """Register the non-flow states beyond the flows ``add_stream_ports`` did.

        Only the **reference** inlet's intensive states are inputs: the mixing
        equalities below pin every other inlet's to it, so registering them all
        would over-fix the model. The one exception is temperature under
        :attr:`MixerTemperatureRule.FLOW_WEIGHTED`, where the inlet
        temperatures are deliberately left untied and each is its own input.
        """
        flow_name = self._flow_basis_name()
        ref_state = self._inlet_state(self._reference_inlet_name())
        for name, var in ref_state.define_state_vars().items():
            if name != flow_name:
                self.register_io_variable(var, role="input")
        if self._blend_temperature:
            for inlet_name in self.config.inlet_names[1:]:
                self.register_io_variable(
                    self._inlet_state(inlet_name).temperature, role="input"
                )
        for name, var in self.outlet_state.define_state_vars().items():
            if name != flow_name:
                self.register_io_variable(var, role="output")

    def _build_mass_balance(self) -> None:
        """Build the per-stream flow References, conservation, and state ties."""
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        inlet_names = self.config.inlet_names

        flows = {}
        for name in inlet_names:
            state = self._inlet_state(name)
            self.add_component(
                f"flow_in_{name}",
                pyo.Reference(state.find_component(flow_name)[:, self._phase]),
            )
            flows[name] = self.find_component(f"flow_in_{name}")
        self.add_component(
            "flow_out",
            pyo.Reference(self.outlet_state.find_component(flow_name)[:, self._phase]),
        )
        flow_out = self.flow_out

        @self.Constraint(
            tb.time_index,
            doc="Conservation: outlet flow == the sum of the inlet flows. "
            "Density is constant across the junction, so volume is conserved "
            "directly with no energy input.",
        )
        def mixing_mass_balance(b, t):
            return flow_out[t] == sum(flows[name][t] for name in inlet_names)

        self._tie_inlet_states()

    def _tie_inlet_states(self) -> None:
        """Hold every non-reference inlet's intensive states at the reference's.

        The linear stand-in for a blended intensive state, and what makes a
        multi-inlet mixer well-posed: without it each extra inlet's pressure
        and temperature would be an unconstrained degree of freedom. Under
        :attr:`MixerTemperatureRule.FLOW_WEIGHTED`, temperature is excluded —
        it is governed by ``outlet_temperature_eq`` instead.
        """
        other_names = self.config.inlet_names[1:]
        if not other_names:
            return
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        ref_name = self._reference_inlet_name()
        excluded = {flow_name}
        if self._blend_temperature:
            excluded.add("temperature")
        tied = [
            name
            for name in self._inlet_state(ref_name).define_state_vars()
            if name not in excluded
        ]
        for state_var in tied:

            def _equality_rule(b, t, name, _v=state_var, _ref=ref_name):
                other = b._inlet_state(name).find_component(_v)
                reference = b._inlet_state(_ref).find_component(_v)
                return other[t] == reference[t]

            self.add_component(
                f"inlet_state_equality_{state_var}",
                pyo.Constraint(
                    tb.time_index,
                    other_names,
                    rule=_equality_rule,
                    doc=f"Mixing node: inlet {state_var} equals the reference "
                    f"inlet's {state_var}.",
                ),
            )

    def _build_outlet_state(self) -> None:
        """Pass the reference inlet's states through; blend temperature if asked."""
        exclude_vars = [self._flow_basis_name()]
        if self._blend_temperature:
            exclude_vars.append("temperature")
        self.add_pass_through_constraints(
            self.find_component(f"inlet_{self._reference_inlet_name()}"),
            self.outlet,
            exclude_vars=exclude_vars,
        )
        if not self._blend_temperature:
            return

        tb = self._find_time_block()
        inlet_names = self.config.inlet_names
        flows = {name: self.find_component(f"flow_in_{name}") for name in inlet_names}
        flow_out = self.flow_out

        @self.Constraint(
            tb.time_index,
            doc="Flow-weighted mixing: outlet temperature * outlet flow == the "
            "sum over inlets of inlet flow * inlet temperature. Written "
            "multiplied through rather than as a ratio so it stays a "
            "polynomial equality (and is well-defined at zero flow, where it "
            "leaves the outlet temperature undetermined).",
        )
        def outlet_temperature_eq(b, t):
            return b.outlet_state.temperature[t] * flow_out[t] == sum(
                flows[name][t] * b._inlet_state(name).temperature[t]
                for name in inlet_names
            )
