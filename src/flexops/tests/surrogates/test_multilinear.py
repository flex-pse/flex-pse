"""Tests for MultilinearSurrogate: the expanded multilinear surrogate class."""

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import MultilinearSurrogate

_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"intercept": 1.0, "flow_out": 2.0},
}


def _ml(data):
    """A MultilinearSurrogate from an old-style data dict, via make_spec."""
    return MultilinearSurrogate(MultilinearSurrogate.make_spec(**data))


def _unit(has_pressure: bool = False):
    """A bare OpsBlock carrying ``flow_out`` and a constant-intensity relation."""
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date="2025-01-01T01:00",
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=has_pressure)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_constant_intensity_relation(
        m.unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    return m, m.unit


# -- build() ---------------------------------------------------------------


@pytest.mark.unit
def test_builds_the_intercept_and_one_linear_term():
    """A single-input term matches the old 'linear' form's numeric behavior."""
    _, unit = _unit()
    surrogate = _ml(_DATA)

    _, body = surrogate.build(unit, unit.power_electrical)
    unit.flow_out[0].set_value(3.0)

    assert pyo.value(body(0)) == pytest.approx(1.0 + 2.0 * 3.0)


@pytest.mark.unit
def test_builds_the_cross_term_of_two_distinct_inputs():
    """Outlet flow, outlet pressure, and their cross term in one relationship.

    Covers resolving a dotted name into a state sub-block
    (``outlet_state.pressure``), the one case a bare local name does not
    exercise.
    """
    _, unit = _unit(has_pressure=True)
    surrogate = _ml(
        {
            "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {
                "intercept": 1.0,
                "flow_out": 2.0,
                "outlet_state.pressure": 1e-3,
                "flow_out*outlet_state.pressure": 5e-4,
            },
        }
    )

    _, body = surrogate.build(unit, unit.power_electrical)
    unit.flow_out[0].set_value(3.0)
    unit.outlet_state.pressure[0].set_value(2.0e5)

    # 1 + 2*3 + 1e-3*2e5 + 5e-4*3*2e5 == 1 + 6 + 200 + 300 == 507
    assert pyo.value(body(0)) == pytest.approx(507.0)


@pytest.mark.unit
def test_build_reads_a_declared_basis_differing_from_the_model():
    """A coefficient fitted in m^3/s attaches to a m^3/hr model variable."""
    _, unit = _unit()
    surrogate = _ml(
        {
            "input_variables": {"flow_out": "m^3/s"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"intercept": 0.0, "flow_out": 1.0},
        }
    )

    _, body = surrogate.build(unit, unit.power_electrical)
    unit.flow_out[0].set_value(36.0)  # 36 m^3/hr == 0.01 m^3/s

    assert pyo.value(body(0)) == pytest.approx(0.01)


@pytest.mark.unit
def test_build_raises_on_a_dimensionally_incompatible_declaration():
    """A declared basis with the wrong dimension is a config error, not a
    silent (wrong) rescale."""
    _, unit = _unit()
    surrogate = _ml(
        {
            "input_variables": {"flow_out": "kW"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"intercept": 0.0, "flow_out": 1.0},
        }
    )
    _, body = surrogate.build(unit, unit.power_electrical)
    unit.flow_out[0].set_value(1.0)

    with pytest.raises(FlexConfigError, match="incompatible"):
        body(0)


@pytest.mark.unit
def test_build_places_coefficient_vars_on_the_block_with_sanitized_names():
    """Coefficient keys are preserved as index entries on an indexed Var.

    No component-name sanitization is needed because each coefficient is an
    index entry on a single ``coefficient_vars`` ``pyo.Var``.
    """
    _, unit = _unit(has_pressure=True)
    surrogate = _ml(
        {
            "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {
                "intercept": 1.0,
                "flow_out*outlet_state.pressure": 5e-4,
            },
        }
    )

    block, body = surrogate.build(unit, unit.power_electrical)
    assert block is not None
    assert block.find_component("coefficient_vars") is not None
    assert block.find_component("intercept") is None
    assert block.find_component("flow_out_outlet_state_pressure") is None
    from flexops.core.registration import CoefficientRegistry

    assert isinstance(block.coefficients, CoefficientRegistry)

    assert "intercept" in block.coefficients
    assert "flow_out*outlet_state.pressure" in block.coefficients
    assert pyo.value(block.coefficients["intercept"]) == pytest.approx(1.0)
    assert pyo.value(
        block.coefficients["flow_out*outlet_state.pressure"]
    ) == pytest.approx(5e-4)


# -- contract validation ------------------------------------------------------


def _spec(config=None, parameters=None):
    """A multilinear spec dict, with _DATA's contract unless overridden."""
    return {
        "config": {
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "terms": ["flow_out"],
            **(config or {}),
        },
        "parameters": parameters
        or {"intercept": {"value": 1.0}, "flow_out": {"value": 2.0}},
    }


@pytest.mark.unit
def test_make_spec_records_terms_and_fixes_every_parameter():
    """make_spec turns coefficient keys into terms; intercept defaults to 0."""
    spec = MultilinearSurrogate.make_spec(
        input_variables={"flow_out": "m^3/hr"},
        output_variables={"power_electrical": "kW"},
        coefficients={"flow_out": 2.0},
    )
    assert spec.config["terms"] == ["flow_out"]
    assert {n: (s.value, s.fixed) for n, s in spec.parameters.items()} == {
        "intercept": (0.0, True),
        "flow_out": (2.0, True),
    }


@pytest.mark.unit
def test_rejects_an_unknown_config_key():
    with pytest.raises(FlexConfigError, match="config.extra"):
        MultilinearSurrogate(_spec({"extra": 1}))


@pytest.mark.unit
def test_rejects_a_missing_config_key():
    spec = _spec()
    del spec["config"]["output_variables"]
    with pytest.raises(FlexConfigError, match="config.output_variables"):
        MultilinearSurrogate(spec)


@pytest.mark.unit
def test_rejects_empty_input_variables():
    with pytest.raises(FlexConfigError, match="input_variables"):
        MultilinearSurrogate(_spec({"input_variables": {}, "terms": []}))


@pytest.mark.unit
def test_rejects_output_variables_with_more_than_one_entry():
    with pytest.raises(FlexConfigError, match="output_variables"):
        MultilinearSurrogate(
            _spec(
                {"output_variables": {"power_electrical": "kW", "flow_out": "m^3/hr"}}
            )
        )


@pytest.mark.unit
def test_rejects_an_unparsable_units_string():
    with pytest.raises(FlexConfigError, match="bogus_unit"):
        MultilinearSurrogate(_spec({"input_variables": {"flow_out": "bogus_unit"}}))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"flow_out": {"value": 2.0}}, r"missing=\['intercept'\]"),
        (
            {
                "intercept": {"value": 1.0},
                "flow_out": {"value": 2.0},
                "x": {"value": 1},
            },
            r"unknown=\['x'\]",
        ),
        ({"intercept": {"value": 1.0}}, r"missing=\['flow_out'\]"),
    ],
)
def test_rejects_parameters_that_dont_match_the_terms(parameters, message):
    """Parameters are exactly the intercept plus one per term."""
    with pytest.raises(FlexConfigError, match=message):
        MultilinearSurrogate(_spec(parameters=parameters))


@pytest.mark.unit
def test_rejects_a_non_numeric_coefficient():
    with pytest.raises(FlexConfigError, match="valid number"):
        _ml({**_DATA, "coefficients": {"intercept": 1.0, "flow_out": "fast"}})


@pytest.mark.unit
def test_rejects_an_array_value_for_a_scalar_parameter():
    with pytest.raises(FlexConfigError, match=r"shape \(\)"):
        _ml({**_DATA, "coefficients": {"intercept": 1.0, "flow_out": [2.0]}})


@pytest.mark.unit
def test_rejects_a_squared_term():
    """The old 'quadratic' spec shape is now a construction-time rejection."""
    with pytest.raises(FlexConfigError, match=r"flow_out\^2"):
        _ml({**_DATA, "coefficients": {"intercept": 1.0, "flow_out^2": 0.5}})


@pytest.mark.unit
def test_rejects_a_repeated_factor():
    with pytest.raises(FlexConfigError, match="repeats a factor"):
        _ml({**_DATA, "coefficients": {"intercept": 1.0, "flow_out*flow_out": 0.5}})


@pytest.mark.unit
def test_rejects_a_coefficient_factor_not_in_input_variables():
    with pytest.raises(FlexConfigError, match="not in"):
        _ml({**_DATA, "coefficients": {"intercept": 1.0, "flow_out*nope": 0.5}})


@pytest.mark.unit
def test_free_parameter_without_a_value_starts_at_zero():
    """A free parameter left without a value starts at 0.0, and stays free."""
    _, unit = _unit()
    spec = _spec(parameters={"intercept": {"value": 1.0}, "flow_out": {"fixed": False}})
    block, _ = MultilinearSurrogate(spec).build(unit, unit.power_electrical)
    assert block.coefficients.value("flow_out") == 0.0
    assert not block.coefficients.is_fixed("flow_out")
    assert block.coefficients.is_fixed("intercept")


@pytest.mark.unit
def test_read_spec_returns_live_values_and_fixed_flags():
    """read_spec reflects changes made on the built block."""
    _, unit = _unit()
    surrogate = _ml(_DATA)
    block, _ = surrogate.build(unit, unit.power_electrical)
    block.coefficients.set_value("flow_out", 3.5)
    block.coefficients.unfix("intercept")
    spec = surrogate.read_spec(block, unit.power_electrical)
    assert {n: (s.value, s.fixed) for n, s in spec.parameters.items()} == {
        "intercept": (1.0, False),
        "flow_out": (3.5, True),
    }
    assert spec.config == surrogate.spec.config


@pytest.mark.component
def test_regression_recovers_multilinear_coefficients():
    """A manual least-squares Pyomo regression recovers known multilinear coefficients.

    Uses a separate *data-generating* surrogate (true coefficients) and a
    *fit* surrogate whose coefficients are initialized away from the truth,
    so the regression must discover the true values from scratch. The
    regression model uses the fit surrogate's own ``body(t)`` (stored on
    the block by ``swap_relation``) so the test exercises the actual
    surrogate object end-to-end.
    """
    n_flow = 8
    n_pressure = 8
    n_points = n_flow * n_pressure

    m = pyo.ConcreteModel()
    end_hour = n_points // 4
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date=f"2025-01-01T{end_hour:02d}:00",
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=True)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "flow_out", pyo.Reference(m.unit.outlet_state.flow_vol_phase[:, "Liq"])
    )
    m.unit.add_constant_intensity_relation(
        m.unit.flow_out, intensity=0.5 * pyunits.kWh / pyunits.m**3
    )
    unit = m.unit

    true_coeffs = {
        "intercept": 1.0,
        "flow_out": 2.0,
        "outlet_state.pressure": 1e-3,
        "flow_out*outlet_state.pressure": 5e-4,
    }
    true_spec = _ml(
        {
            "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": true_coeffs,
        }
    )

    _, true_body = true_spec.build(unit, unit.power_electrical)

    flow_grid = np.linspace(1.0, 10.0, n_flow)
    pressure_grid = np.linspace(1.0e5, 3.0e5, n_pressure)
    flow_mesh, pressure_mesh = np.meshgrid(flow_grid, pressure_grid)
    flows = flow_mesh.flatten()
    pressures = pressure_mesh.flatten()

    observed = []
    for i in range(n_points):
        unit.flow_out[i].set_value(float(flows[i]))
        unit.outlet_state.pressure[i].set_value(float(pressures[i]))
        observed.append(pyo.value(true_body(i)))
    observed = np.array(observed)

    unknown_coeffs = {
        "intercept": 1,
        "flow_out": 1.0,
        "outlet_state.pressure": 1,
        "flow_out*outlet_state.pressure": 1,
    }
    fit_spec = _ml(
        {
            "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": unknown_coeffs,
        }
    )

    block = unit.swap_relation("power_electrical_relation", fit_spec)
    unit.register_surrogate_coefficients("power_electrical_relation")

    coefs = block.coefficients
    unit.unfix_surrogate_coefficients()

    fit_body = block.body

    for t in range(m.time_block.n_points):
        unit.flow_out[t].set_value(float(flows[t]))
        unit.flow_out[t].fix()
        unit.outlet_state.pressure[t].set_value(float(pressures[t]))
        unit.outlet_state.pressure[t].fix()

    m.reg_T = pyo.RangeSet(0, m.time_block.n_points - 1)
    m.reg_y_obs = pyo.Param(
        m.reg_T, initialize={t: float(observed[t]) for t in m.reg_T}
    )

    def _pred(m, t):
        return fit_body(t)

    m.reg_pred = pyo.Expression(m.reg_T, rule=_pred)

    def _obj(m):
        return sum((m.reg_pred[t] - m.reg_y_obs[t]) ** 2 for t in m.reg_T)

    m.reg_obj = pyo.Objective(rule=_obj)

    solver = pyo.SolverFactory("ipopt")
    m.unit.inlet_state.flow_vol_phase.fix()
    m.unit.inlet_state.pressure.fix()
    m.display()
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    unit.fix_surrogate_coefficients()
    assert pyo.value(coefs["intercept"]) == pytest.approx(
        true_coeffs["intercept"], rel=1e-3
    )
    assert pyo.value(coefs["flow_out"]) == pytest.approx(
        true_coeffs["flow_out"], rel=1e-3
    )
    assert pyo.value(coefs["outlet_state.pressure"]) == pytest.approx(
        true_coeffs["outlet_state.pressure"], rel=1e-3
    )
    assert pyo.value(coefs["flow_out*outlet_state.pressure"]) == pytest.approx(
        true_coeffs["flow_out*outlet_state.pressure"], rel=1e-3
    )
