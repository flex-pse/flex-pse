"""The surrogate contract, checked the same way for every registered type.

Every class in ``SURROGATES`` must: build a spec with ``make_spec`` that
validates without Pyomo; register exactly the parameters (names and shapes)
its contract declares; read back the spec it was built from; apply each
parameter's fixed flag; and reject parameters that don't match. A new
surrogate type gets these checks by adding an example to ``EXAMPLES``.
"""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import (
    SURROGATES,
    ArimaSurrogate,
    ConstantIntensitySurrogate,
    MultilinearSurrogate,
    surrogate_from_spec,
)
from flexops.surrogates._stub import StubSurrogate

EXAMPLES = {
    SurrogateType.MULTILINEAR: lambda: MultilinearSurrogate.make_spec(
        input_variables={"flow_out": "m^3/hr"},
        output_variables={"power_electrical": "kW"},
        coefficients={"intercept": 1.0, "flow_out": 0.4},
    ),
    SurrogateType.ARIMA: lambda: ArimaSurrogate.make_spec(
        order=(2, 0, 1),
        deterministic="intercept",
        input_variables={"flow_out": "m^3/hr"},
        output_variables={"power_electrical": "kW"},
        values={
            "intercept": 1.0,
            "ar.L1": 0.5,
            "ar.L2": 0.1,
            "ma.L1": 0.2,
            "exog.flow_out": 0.4,
        },
        history={
            "start_date": "2025-01-01T00:00:00",
            "time_step_seconds": 900.0,
            "eta_values": [0.0, 0.0],
            "eps_values": [0.0, 0.0],
        },
    ),
    SurrogateType.CONSTANT_INTENSITY: lambda: ConstantIntensitySurrogate.make_spec(
        energy_intensity=0.7
    ),
}
IMPLEMENTED = sorted(EXAMPLES, key=str)
SWAPPING = [t for t in IMPLEMENTED if SURROGATES[t].replaces_relation]


_MODELS = []
"""Models the units below belong to. A Pyomo block only weakly references its
parent, so each model is kept alive for the test session."""


def _unit():
    """A bare unit with flow_out and a registered constant-intensity relation."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-01-01T01:00",
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_constant_intensity_relation(
        m.unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    _MODELS.append(m)
    return m.unit


@pytest.mark.unit
def test_every_registered_type_is_implemented_or_a_stub():
    """A registered type either has a contract example here or is a stub."""
    for surrogate_type, cls in SURROGATES.items():
        assert surrogate_type in EXAMPLES or issubclass(cls, StubSurrogate)


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", IMPLEMENTED, ids=str)
def test_make_spec_validates_without_pyomo(surrogate_type):
    """make_spec's output is a valid, current-version, all-fixed spec."""
    spec = EXAMPLES[surrogate_type]()
    cls = SURROGATES[surrogate_type]
    assert isinstance(spec, SurrogateSpec)
    assert spec.config_version == cls.config_version
    assert cls.validate_spec(spec) == spec
    assert list(spec.parameters) == list(
        cls.parameter_shapes(cls.config_model.model_validate(spec.config))
    )
    assert spec.free_parameters() == []


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", SWAPPING, ids=str)
def test_build_registers_exactly_the_contracts_parameters(surrogate_type):
    """The built registry's names and shapes are parameter_shapes'."""
    unit = _unit()
    surrogate = surrogate_from_spec(EXAMPLES[surrogate_type]())
    block, _ = surrogate.build(unit, unit.power_electrical)
    registry = block.coefficients
    assert {name: registry.shape(name) for name in registry} == (
        surrogate.parameter_shapes(surrogate.config)
    )


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", SWAPPING, ids=str)
def test_read_spec_returns_the_spec_it_was_built_from(surrogate_type):
    """Reading an untouched block back gives the spec it was built from."""
    unit = _unit()
    spec = EXAMPLES[surrogate_type]()
    surrogate = surrogate_from_spec(spec)
    block, _ = surrogate.build(unit, unit.power_electrical)
    assert surrogate.read_spec(block, unit.power_electrical) == spec


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", SWAPPING, ids=str)
def test_build_applies_each_parameters_fixed_flag(surrogate_type):
    """A parameter the spec frees is free on the block; the rest are fixed."""
    unit = _unit()
    spec = EXAMPLES[surrogate_type]()
    first = next(iter(spec.parameters))
    spec.parameters[first] = spec.parameters[first].model_copy(update={"fixed": False})
    surrogate = surrogate_from_spec(spec)
    block, _ = surrogate.build(unit, unit.power_electrical)
    registry = block.coefficients
    assert [name for name in registry if not registry.is_fixed(name)] == [first]
    assert surrogate.read_spec(block, unit.power_electrical).free_parameters() == [
        first
    ]


@pytest.mark.unit
def test_constant_intensity_applies_to_and_reads_from_the_unit():
    """The non-swapping type sets the unit's intensity and reads it back."""
    unit = _unit()
    spec = EXAMPLES[SurrogateType.CONSTANT_INTENSITY]()
    surrogate = surrogate_from_spec(spec)
    surrogate.apply_to_unit(unit)
    assert pyo.value(unit.energy_intensity) == pytest.approx(0.7)
    assert surrogate.read_from_unit(unit) == spec


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", IMPLEMENTED, ids=str)
def test_unknown_and_missing_parameters_are_rejected(surrogate_type):
    """Parameters must name exactly the contract's set."""
    spec = EXAMPLES[surrogate_type]()
    first = next(iter(spec.parameters))
    missing = spec.model_copy(
        update={"parameters": {k: v for k, v in spec.parameters.items() if k != first}}
    )
    unknown = spec.model_copy(
        update={"parameters": {**spec.parameters, "bogus": spec.parameters[first]}}
    )
    with pytest.raises(FlexConfigError, match=rf"missing=\['{first}'\]"):
        surrogate_from_spec(missing)
    with pytest.raises(FlexConfigError, match=r"unknown=\['bogus'\]"):
        surrogate_from_spec(unknown)


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", IMPLEMENTED, ids=str)
def test_unknown_config_keys_are_rejected(surrogate_type):
    """Every config model is strict: an undocumented key is an error."""
    spec = EXAMPLES[surrogate_type]()
    bad = spec.model_copy(update={"config": {**spec.config, "bogus": 1}})
    with pytest.raises(FlexConfigError, match="config.bogus"):
        surrogate_from_spec(bad)


@pytest.mark.unit
@pytest.mark.parametrize("surrogate_type", IMPLEMENTED, ids=str)
def test_a_swapping_surrogate_rejects_a_mismatched_build(surrogate_type, monkeypatch):
    """A class whose _build registers something other than its contract fails."""
    cls = SURROGATES[surrogate_type]
    if not cls.replaces_relation:
        pytest.skip("constant_intensity builds no block")
    unit = _unit()
    surrogate = surrogate_from_spec(EXAMPLES[surrogate_type]())
    real_build = cls._build

    def drop_one(self, unit, target):
        block, body = real_build(self, unit, target)
        name = next(iter(block.coefficients))
        del block.coefficients._vars[name]
        del block.coefficients._shapes[name]
        return block, body

    monkeypatch.setattr(cls, "_build", drop_one)
    with pytest.raises(FlexConfigError, match="disagree"):
        surrogate.build(unit, unit.power_electrical)
