r"""Digestor(OpsBlockData): multi-feed anaerobic digester with biogas and optional
sludge outlets.

A wastewater anaerobic digester: an arbitrary number of feed streams, each
carrying its own property package, are converted into one biogas outlet and
(optionally) one treated-sludge outlet. Because every feed can have a different
property package, the unit subclasses :class:`~flexops.core.ops_block.OpsBlockData`
directly and hand-builds its ports and state blocks, the way
:class:`~flexops.unit_models.powergeneration.combustor.Combustor` does.

**Volume balance.** Total inlet volumetric flow is conserved across the two
outlets:

.. math::

    \dot{V}_{biogas}[t] + \dot{V}_{sludge}[t] = \sum_i \dot{V}_{feed,i}[t]

Biogas volume is driven by a **fixed-fraction correlation**:

.. math::

    \dot{V}_{biogas}[t] = f \cdot \sum_i \dot{V}_{feed,i}[t]

where ``f`` is :attr:`biogas_fraction`, a fixed (regressable) scalar Var in
``(0, 1)``. When :attr:`has_sludge_outlet` is ``True``, sludge volume is the
residual; when it is ``False``, only the biogas relation is built.

The biogas correlation is **surrogate-replaceable**: when the unit is built
from a config whose ``flexops_config.surrogate.functional_form`` is
``"linear"``, the plain :attr:`biogas_relation` constraint is deactivated in
place and a ``biogas_relation_fitted`` constraint is added with the same
swap contract :class:`~flexops.unit_models.powergeneration.combustor.Combustor`
uses for its power relation.

**Intensive states.** The first inlet in ``inlet_packages`` is the reference
inlet. Its non-flow states are passed through to both outlets; every other
inlet's non-flow states are tied to the reference inlet's. This keeps the
model linear and well-posed without requiring the caller to fix every feed's
temperature and pressure separately.

Config:
    ``inlet_packages``: a ``dict[str, property_package]`` mapping each feed's
    role name to its property package. The keys set the inlet count and port
    names (``f"inlet_{name}"``); values must be single-phase packages.
    ``biogas_fraction`` (default ``0.08``): the dimensionless fraction of
    total inlet volume that becomes biogas, bounded in ``(0, 1)``.
    ``has_sludge_outlet`` (default ``True``): whether a treated-sludge outlet
    is built. ``biogas_property_package`` (default ``SimpleGasFlow``): the
    property package for the biogas outlet. ``sludge_property_package``
    (default ``SimpleAqueousFlow``): the property package for the sludge
    outlet, required when ``has_sludge_outlet`` is ``True``.

Example:
    >>> from flexops.testing import dummy_time_block
    >>> from flexops.unit_models.watertreatment import Digestor
    >>> from flexops.properties.simple_gas import SimpleGasFlow
    >>> m = dummy_time_block(3)
    >>> m.unit = Digestor(  # doctest: +SKIP
    ...     inlet_packages={"sludge": m.properties},
    ...     biogas_property_package=SimpleGasFlow(),
    ... )
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.core.ops_block import OpsBlockData
from flexops.surrogates import surrogate_from_spec

_log = get_logger(__name__)


def _inlet_packages_domain(value):
    """ConfigValue domain: accept ``None`` or a dict of name -> property package."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise FlexConfigError(
            "inlet_packages must be a dict of inlet name to property "
            f"package, got {type(value).__name__}.",
            field="inlet_packages",
            value=value,
        )
    return dict(value)


def _biogas_fraction_domain(value):
    """ConfigValue domain: biogas_fraction must be a float in (0, 1)."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise FlexConfigError(
                f"biogas_fraction must be a float in (0, 1), got {value!r}.",
                field="biogas_fraction",
                value=value,
            ) from exc
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (0 < value < 1)
    ):
        return float(value)
    raise FlexConfigError(
        f"biogas_fraction must be a float in (0, 1), got {value!r}.",
        field="biogas_fraction",
        value=value,
    )


@declare_process_block_class("Digestor")
class DigestorData(OpsBlockData):
    """Multi-feed anaerobic digester with biogas and optional sludge outlets.

    See the module docstring for the balance, the intensive-state treatment,
    and the documented simplifications.
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "inlet_packages",
        ConfigValue(
            default={},
            domain=_inlet_packages_domain,
            description="Mapping of inlet name to its property package. "
            "Each key becomes one inlet port named f'inlet_{key}'; every "
            "value must be a single-phase property package.",
        ),
    )
    CONFIG.declare(
        "biogas_fraction",
        ConfigValue(
            default=0.08,
            domain=_biogas_fraction_domain,
            description="Fraction of total inlet volume converted to biogas "
            "(dimensionless, a fixed, regressable Var once built, in (0, 1)).",
        ),
    )
    CONFIG.declare(
        "has_sludge_outlet",
        ConfigValue(
            default=True,
            domain=bool,
            description="Whether the digester exposes a treated-sludge outlet. "
            "When False, only the biogas outlet is built and the mass balance "
            "reduces to biogas_volume == biogas_fraction * total_inlet.",
        ),
    )
    CONFIG.declare(
        "biogas_property_package",
        ConfigValue(
            default=None,
            description="Property package for the biogas outlet. Defaults to "
            "a plain SimpleGasFlow() when None.",
        ),
    )
    CONFIG.declare(
        "sludge_property_package",
        ConfigValue(
            default=None,
            description="Property package for the treated-sludge outlet. "
            "Defaults to a plain SimpleAqueousFlow() when None. Required "
            "when has_sludge_outlet is True.",
        ),
    )
    CONFIG.declare(
        "operating_temperature",
        ConfigValue(
            default=288.706 * pyunits.K,
            description="Reactor operating temperature (K). Defaults to 15.55 C "
            "(288.706 K), the standard condition for SCFM measurement. Fixed "
            "at construction and passed through to the biogas outlet.",
        ),
    )
    CONFIG.declare(
        "operating_pressure",
        ConfigValue(
            default=101325.0 * pyunits.Pa,
            description="Reactor operating pressure (Pa). Defaults to 1 bar "
            "(101325 Pa), the standard condition for SCFM measurement. Fixed "
            "at construction and passed through to the biogas outlet.",
        ),
    )

    def build(self) -> None:
        """Validate config, build per-inlet state blocks, outlets, balances."""
        super().build()
        self._validate_inlet_packages()
        self._resolve_outlet_packages()
        self._build_reactor_state()
        self._build_inlets()
        self._build_outlets()
        self._build_mass_balance()
        self._build_outlet_state()
        self._build_biogas_relation()
        self._register_stream_states()

    # -- reactor operating conditions -----------------------------------------

    def _build_reactor_state(self) -> None:
        """Create reactor T/P Vars fixed from config, tied to the biogas outlet."""
        tb = self._find_time_block()
        self.reactor_temperature = pyo.Var(
            tb.time_index,
            initialize=pyo.value(self.config.operating_temperature),
            units=pyunits.K,
            doc="Digestor reactor operating temperature (K).",
        )
        self.reactor_temperature.fix()
        self.reactor_pressure = pyo.Var(
            tb.time_index,
            initialize=pyo.value(self.config.operating_pressure),
            units=pyunits.Pa,
            doc="Digestor reactor operating pressure (Pa).",
        )
        self.reactor_pressure.fix()

    # -- config validation ---------------------------------------------------

    def _validate_inlet_packages(self) -> None:
        """Reject empty ``inlet_packages`` and ensure every value is single-phase."""
        packages = self.config.inlet_packages
        if not packages:
            raise FlexConfigError(
                "inlet_packages must contain at least one inlet; got an empty " "dict.",
                field="inlet_packages",
                value=packages,
            )
        for name, pkg in packages.items():
            if not isinstance(name, str) or not name:
                raise FlexConfigError(
                    "inlet_packages keys must be non-empty strings, got " f"{name!r}.",
                    field="inlet_packages",
                    value=packages,
                )
            phases = list(pkg.phase_list)
            if len(phases) != 1:
                raise FlexConfigError(
                    "Digestor requires each inlet property package to carry "
                    f"exactly one phase; inlet '{name}' has "
                    f"phase_list={phases!r}.",
                    field="inlet_packages",
                    value=packages,
                )
        names = list(packages.keys())
        if len(set(names)) != len(names):
            raise FlexConfigError(
                "inlet_packages keys must be unique, got " f"{names!r}.",
                field="inlet_packages",
                value=packages,
            )
        self._inlet_names = tuple(names)

    def _resolve_outlet_packages(self) -> None:
        """Validate that required outlet property packages are present."""
        if self.config.biogas_property_package is None:
            raise FlexConfigError(
                "biogas_property_package is required; pass a single-phase "
                "property package for the biogas outlet.",
                field="biogas_property_package",
                value=None,
            )
        if (
            self.config.has_sludge_outlet
            and self.config.sludge_property_package is None
        ):
            raise FlexConfigError(
                "sludge_property_package is required when has_sludge_outlet is "
                "True; pass a single-phase property package for the sludge outlet.",
                field="sludge_property_package",
                value=None,
            )

    # -- inlet construction --------------------------------------------------

    def _build_inlets(self) -> None:
        """Build one state block + port per inlet, using its own property package."""
        tb = self._find_time_block()
        self._phase_map = {}
        for name in self._inlet_names:
            pkg = self.config.inlet_packages[name]
            phase = list(pkg.phase_list)[0]
            self._phase_map[name] = phase
            state_name = f"inlet_{name}_state"
            self.add_component(
                state_name,
                pkg.build_state_block(time_index=tb.time_index),
            )
            state = self.find_component(state_name)
            self.add_inlet_port(
                name=f"inlet_{name}",
                block=state,
                doc=f"{name} feed stream",
            )

    # -- outlet construction -------------------------------------------------

    def _build_outlets(self) -> None:
        """Build the biogas outlet and, optionally, the sludge outlet."""
        tb = self._find_time_block()

        biogas_pkg = self.config.biogas_property_package
        biogas_phase = list(biogas_pkg.phase_list)[0]
        self.add_component(
            "outlet_biogas_state",
            biogas_pkg.build_state_block(time_index=tb.time_index),
        )
        self.add_outlet_port(
            name="outlet_biogas",
            block=self.outlet_biogas_state,
            doc="Biogas product stream",
        )

        self._biogas_phase = biogas_phase

        if self.config.has_sludge_outlet:
            sludge_pkg = self.config.sludge_property_package
            sludge_phase = list(sludge_pkg.phase_list)[0]
            self.add_component(
                "outlet_sludge_state",
                sludge_pkg.build_state_block(time_index=tb.time_index),
            )
            self.add_outlet_port(
                name="outlet_sludge",
                block=self.outlet_sludge_state,
                doc="Treated sludge stream",
            )
            self._sludge_phase = sludge_phase

    # -- stream-state registration -------------------------------------------

    def _register_stream_states(self) -> None:
        """Register inlet state vars as inputs;
        reactor T/P and biogas volume as outputs."""
        for name in self._inlet_names:
            state = self.find_component(f"inlet_{name}_state")
            for _var_name, var in state.define_state_vars().items():
                self.register_io_variable(var, role="input")

        self.register_io_variable(self.biogas_volume, role="output")
        self.register_io_variable(self.reactor_temperature, role="output")
        self.register_io_variable(self.reactor_pressure, role="output")

    # -- mass balance --------------------------------------------------------

    def _build_mass_balance(self) -> None:
        """Build per-feed flow References, biogas relation, and optional residual."""
        tb = self._find_time_block()
        inlet_names = self._inlet_names

        flows = {}
        for name in inlet_names:
            state = self.find_component(f"inlet_{name}_state")
            pkg = self.config.inlet_packages[name]
            flow_name = pkg.get_flow_basis_var_name()
            phase = self._phase_map[name]
            self.add_component(
                f"flow_in_{name}",
                pyo.Reference(state.find_component(flow_name)[:, phase]),
            )
            flows[name] = getattr(self, f"flow_in_{name}")

        biogas_state = self.outlet_biogas_state
        self.add_component(
            "biogas_volume",
            pyo.Reference(
                biogas_state.find_component("flow_vol_phase")[:, self._biogas_phase]
            ),
        )
        self.biogas_volume.doc = "Biogas volumetric flow rate."

        if self.config.has_sludge_outlet:
            sludge_state = self.outlet_sludge_state
            self.add_component(
                "sludge_volume",
                pyo.Reference(
                    sludge_state.find_component("flow_vol_phase")[:, self._sludge_phase]
                ),
            )
            self.sludge_volume.doc = "Treated-sludge volumetric flow rate."

            @self.Constraint(
                tb.time_index,
                doc="Mass balance: sludge mass flow equals total inlet mass "
                "flow minus biogas mass flow.",
            )
            def sludge_mass_eq(b, t):
                return (
                    sludge_state.flow_mass_phase[t, self._sludge_phase]
                    == sum(
                        self.find_component(f"inlet_{name}_state").flow_mass_phase[
                            t, self._phase_map[name]
                        ]
                        for name in inlet_names
                    )
                    - biogas_state.flow_mass_phase[t, self._biogas_phase]
                )

    # -- outlet intensive states ----------------------------------------------

    def _build_outlet_state(self) -> None:
        """Pass the reference inlet's non-flow states through to both outlets."""
        ref_name = self._inlet_names[0]
        ref_port = self.find_component(f"inlet_{ref_name}")
        ref_state = self.find_component(f"inlet_{ref_name}_state")
        flow_name = self.config.inlet_packages[ref_name].get_flow_basis_var_name()
        ref_vars = ref_state.define_state_vars()

        tb = self._find_time_block()

        for outlet_name in ("outlet_biogas", "outlet_sludge"):
            outlet_port = self.find_component(outlet_name)
            if outlet_port is None:
                continue
            outlet_state = self.find_component(f"{outlet_name}_state")
            outlet_vars = outlet_state.define_state_vars()
            exclude = [flow_name] + [
                v
                for v in ref_vars
                if v not in outlet_vars or v in ("pressure", "temperature")
            ]
            self.add_pass_through_constraints(
                ref_port,
                outlet_port,
                exclude_vars=exclude,
                name_prefix=f"pass_through_{outlet_name}",
            )

        biogas_state = self.outlet_biogas_state
        if hasattr(biogas_state, "pressure") and hasattr(biogas_state, "temperature"):
            self.add_component(
                "reactor_pressure_eq",
                pyo.Constraint(
                    tb.time_index,
                    rule=lambda b, t: b.reactor_pressure[t]
                    == b.outlet_biogas_state.pressure[t],
                ),
            )
            self.add_component(
                "reactor_temperature_eq",
                pyo.Constraint(
                    tb.time_index,
                    rule=lambda b, t: b.reactor_temperature[t]
                    == b.outlet_biogas_state.temperature[t],
                ),
            )

        if self.config.has_sludge_outlet:
            sludge_state = self.outlet_sludge_state
            if hasattr(sludge_state, "pressure") and hasattr(
                sludge_state, "temperature"
            ):
                self.add_component(
                    "sludge_pressure_eq",
                    pyo.Constraint(
                        tb.time_index,
                        rule=lambda b, t: b.reactor_pressure[t]
                        == b.outlet_sludge_state.pressure[t],
                    ),
                )
                self.add_component(
                    "sludge_temperature_eq",
                    pyo.Constraint(
                        tb.time_index,
                        rule=lambda b, t: b.reactor_temperature[t]
                        == b.outlet_sludge_state.temperature[t],
                    ),
                )

    # -- biogas relation -----------------------------------------------------

    def _build_biogas_relation(self) -> None:
        """Build the fixed-fraction biogas relation."""
        tb = self._find_time_block()
        biogas_fraction = self.declare_process_parameter(
            "biogas_fraction",
            self.config.biogas_fraction,
            pyunits.dimensionless,
            "Fraction of total inlet volumetric flow converted to biogas.",
            bounds=(0.0, 1.0),
        )

        inlet_names = self._inlet_names

        @self.Constraint(
            tb.time_index,
            doc="Biogas volume equals biogas_fraction times the sum of inlet "
            "volumetric flows.",
        )
        def biogas_relation(b, t):
            return b.biogas_volume[t] == biogas_fraction * sum(
                getattr(b, f"flow_in_{name}")[t] for name in inlet_names
            )

        self.register_relation(self.biogas_relation, target=self.biogas_volume)

        spec = getattr(self.config.flexops_config, "surrogate", None)
        if (
            spec is not None
            and spec.surrogate_type is not SurrogateType.CONSTANT_INTENSITY
        ):
            self.swap_relation("biogas_relation", surrogate_from_spec(spec))
