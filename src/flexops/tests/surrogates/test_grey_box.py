"""Tests for the external-model grey-box adapter and ExternalModelSurrogate.

The PyNumero-adapter tests use a hand-written stub driver so they run without
torch and pin the sparse-matrix contract independently of any framework.
"""

import subprocess
import sys

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent

from flexcore.exceptions import FlexConfigError
from flexcore.solvers import check_training_range, get_solver
from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import MultilinearSurrogate, grey_box
from flexops.surrogates.grey_box import (
    ExternalFramework,
    ExternalModelDriver,
    ExternalModelSurrogate,
    _ExternalModelGreyBox,
    get_driver,
)


def _unit(has_pressure: bool = False):
    """A bare OpsBlock carrying flow_out and a constant-intensity relation."""
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


def _dummy_model(x):
    """A trivial, always-resolvable model for tests that stub out the driver."""
    return float(np.sum(x))


class _StubDriver(ExternalModelDriver):
    """Hand-written driver: fixed gradient/Hessian, no framework involved."""

    framework = ExternalFramework.PYTORCH

    def evaluate(self, x):
        return float(np.sum(x))

    def jacobian(self, x):
        return np.arange(1, len(x) + 1, dtype=float)

    def hessian(self, x):
        n = len(x)
        return np.ones((n, n))

    def check_differentiable(self, x):
        pass


def _adapter(n=3, probe=None):
    """Build an ``_ExternalModelGreyBox`` directly, over a stub driver."""
    input_names = [f"x{i}" for i in range(n)]
    probe = probe if probe is not None else {name: 1.0 for name in input_names}
    driver = _StubDriver(model=_dummy_model, n_inputs=n)
    return _ExternalModelGreyBox(driver, input_names, "y", probe), driver


_EXTERNAL_MODEL_DATA = {
    "framework": "pytorch",
    "model_path": f"{__name__}._dummy_model",
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
}


# -- framework/driver registry (no framework, no solver) ---------------------


@pytest.mark.unit
@pytest.mark.needs_torch
def test_get_driver_resolves_pytorch():
    """The implemented framework resolves to its driver class."""
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    assert get_driver(ExternalFramework.PYTORCH) is TorchDriver
    assert get_driver("pytorch") is TorchDriver


@pytest.mark.unit
def test_get_driver_unknown_framework_raises_config_error():
    """An unknown framework name raises FlexConfigError listing known values."""
    with pytest.raises(FlexConfigError, match="pytorch"):
        get_driver("not_a_real_framework")


@pytest.mark.unit
def test_importing_flexops_surrogates_does_not_import_torch():
    """A bare install imports flexops.surrogates with no torch import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flexops.surrogates; "
            "assert 'torch' not in sys.modules, sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# -- validation (no framework, no solver) ------------------------------------


@pytest.mark.unit
def test_validate_rejects_multiple_output_variables():
    """A registered relation has exactly one target."""
    with pytest.raises(FlexConfigError, match="output_variables"):
        ExternalModelSurrogate(
            {
                **_EXTERNAL_MODEL_DATA,
                "output_variables": {"power_electrical": "kW", "extra": "kW"},
            }
        )


@pytest.mark.unit
def test_validate_rejects_probe_point_key_not_in_inputs():
    """probe_point keys must be a subset of input_variables."""
    with pytest.raises(FlexConfigError, match="probe_point"):
        ExternalModelSurrogate(
            {**_EXTERNAL_MODEL_DATA, "probe_point": {"not_an_input": 1.0}}
        )


@pytest.mark.unit
@pytest.mark.needs_torch
def test_validate_resolves_model_path_eagerly():
    """A bad dotted path raises FlexConfigError at construction."""
    with pytest.raises(FlexConfigError, match="model_path"):
        ExternalModelSurrogate(
            {**_EXTERNAL_MODEL_DATA, "model_path": "not.a.real.path"}
        )


# -- PyNumero adapter, stub driver (unit, no solver) -------------------------


@pytest.mark.unit
def test_jacobian_has_fixed_sparsity_pattern_including_zeros():
    """Every entry is present, even an all-zero gradient (pitfall 5)."""
    adapter, driver = _adapter(n=3)
    driver.jacobian = lambda x: np.zeros(3)
    adapter.set_input_values([0.0, 0.0, 0.0])

    jac = adapter.evaluate_jacobian_outputs()

    assert jac.nnz == 3
    assert list(jac.row) == [0, 0, 0]
    assert list(jac.col) == [0, 1, 2]


@pytest.mark.unit
def test_hessian_has_fixed_sparsity_pattern_including_zeros():
    """Every lower-triangular entry is present, even an all-zero Hessian."""
    adapter, driver = _adapter(n=3)
    driver.hessian = lambda x: np.zeros((3, 3))
    adapter.set_output_constraint_multipliers([1.0])
    adapter.set_input_values([1.0, 2.0, 3.0])

    hess = adapter.evaluate_hessian_outputs()

    assert hess.nnz == 3 * 4 // 2


@pytest.mark.unit
def test_hessian_is_lower_triangular_only():
    """Guards pitfall 4: PyNumero rejects any entry with row < col."""
    adapter, _ = _adapter(n=3)
    adapter.set_output_constraint_multipliers([1.0])
    adapter.set_input_values([1.0, 2.0, 3.0])

    hess = adapter.evaluate_hessian_outputs()

    assert np.all(hess.row >= hess.col)


@pytest.mark.unit
def test_hessian_scales_with_output_multiplier():
    """PyNumero's output residual Hessian in the inputs is lambda * grad^2 f."""
    adapter, _ = _adapter(n=2)
    adapter.set_input_values([1.0, 2.0])

    adapter.set_output_constraint_multipliers([1.0])
    base = adapter.evaluate_hessian_outputs().toarray()
    adapter.set_output_constraint_multipliers([3.0])
    scaled = adapter.evaluate_hessian_outputs().toarray()

    assert np.allclose(scaled, base * 3.0)


@pytest.mark.unit
def test_finalize_block_construction_initializes_inputs_and_outputs():
    """Grey-box Vars carry the probe point and are never fixed (pitfall 6)."""
    probe = {"x0": 2.0, "x1": 3.0}
    adapter, _ = _adapter(n=2, probe=probe)

    m = pyo.ConcreteModel()
    m.block = ExternalGreyBoxBlock(external_model=adapter)

    assert pyo.value(m.block.inputs["x0"]) == 2.0
    assert pyo.value(m.block.inputs["x1"]) == 3.0
    assert pyo.value(m.block.outputs["y"]) == pytest.approx(5.0)
    assert not m.block.inputs["x0"].fixed
    assert not m.block.outputs["y"].fixed


# -- ExternalModelSurrogate.build (stub driver, monkeypatched get_driver) ---


@pytest.mark.unit
def test_grey_box_models_are_distinct_per_time_index(monkeypatch):
    """Guards pitfall 1: a callable rule, never one shared instance."""
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    m, unit = _unit()
    surrogate = ExternalModelSurrogate(_EXTERNAL_MODEL_DATA)

    block, _ = surrogate.build(unit, unit.power_electrical)
    m.unit.add_component("surrogate_power", block)

    times = list(unit.power_electrical.index_set())
    assert len(times) >= 2
    models = [block.egb[t].get_external_model() for t in times]
    assert len({id(model) for model in models}) == len(models)


@pytest.mark.unit
def test_input_link_constraints_are_units_consistent(monkeypatch):
    """Guards pitfall 8: egb.inputs carries no units; the link divides them out."""
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    m, unit = _unit()
    surrogate = ExternalModelSurrogate(
        {**_EXTERNAL_MODEL_DATA, "input_variables": {"flow_out": "m^3/s"}}
    )

    block, _ = surrogate.build(unit, unit.power_electrical)
    m.unit.add_component("surrogate_power", block)

    # assert_units_consistent does not support the ExternalGreyBoxBlock ctype
    # itself (its Vars carry no units by design -- pitfall 8); the constraint
    # that converts into/out of them is what must be checked.
    assert_units_consistent(block.input_links)


# -- needs_torch (unit, no solver) ------------------------------------------


@pytest.mark.unit
@pytest.mark.needs_torch
def test_validate_raises_on_non_differentiable_model():
    """A .detach() break in the forward pass raises at construction."""
    pytest.importorskip("torch")
    globals()["_broken_model"] = lambda x: (x[0] * 2).detach()

    with pytest.raises(FlexConfigError, match="differentiable"):
        ExternalModelSurrogate(
            {
                "framework": "pytorch",
                "model_path": f"{__name__}._broken_model",
                "input_variables": {"x": "dimensionless"},
                "output_variables": {"y": "dimensionless"},
            }
        )


@pytest.mark.unit
@pytest.mark.needs_torch
def test_validate_accepts_model_with_infinite_gradient_at_probe():
    """An infinite gradient (sqrt at 0) is not a false-positive rejection."""
    torch = pytest.importorskip("torch")
    globals()["_sqrt_model"] = lambda x: torch.sqrt(x[0])

    surrogate = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._sqrt_model",
            "input_variables": {"x": "dimensionless"},
            "output_variables": {"y": "dimensionless"},
            "probe_point": {"x": 0.0},
        }
    )
    assert surrogate is not None


@pytest.mark.unit
@pytest.mark.needs_torch
def test_float32_module_is_coerced_and_evaluates():
    """A default-float32 nn.Module is coerced to float64 (pitfall 3)."""
    torch = pytest.importorskip("torch")

    class _LinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x.reshape(1))

    globals()["_float32_model"] = _LinearModule()

    surrogate = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._float32_model",
            "input_variables": {"x": "dimensionless"},
            "output_variables": {"y": "dimensionless"},
            "probe_point": {"x": 1.0},
        }
    )
    value = surrogate._driver.evaluate(np.array([2.0]))
    assert isinstance(value, float)


@pytest.mark.unit
@pytest.mark.needs_torch
def test_jacobian_matches_analytic_derivative():
    """f(x) = x**3 has a known closed-form derivative, evaluated away from 0."""
    pytest.importorskip("torch")
    globals()["_cube_model"] = lambda x: x[0] ** 3

    surrogate = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._cube_model",
            "input_variables": {"x": "dimensionless"},
            "output_variables": {"y": "dimensionless"},
            "probe_point": {"x": 2.0},
        }
    )
    jac = surrogate._driver.jacobian(np.array([2.0]))
    assert jac == pytest.approx([12.0])


@pytest.mark.unit
@pytest.mark.needs_torch
def test_hessian_matches_analytic_second_derivative():
    """f(x) = x**3 has second derivative 6x, evaluated away from 0."""
    pytest.importorskip("torch")
    globals()["_cube_model_2"] = lambda x: x[0] ** 3

    surrogate = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._cube_model_2",
            "input_variables": {"x": "dimensionless"},
            "output_variables": {"y": "dimensionless"},
            "probe_point": {"x": 2.0},
        }
    )
    hess = surrogate._driver.hessian(np.array([2.0]))
    assert hess[0, 0] == pytest.approx(12.0)


# -- swap_relation + real solve (component, needs_cyipopt, needs_torch) -----


def _swapped_unit(model_name: str):
    """A unit with power_electrical_relation swapped to a linear grey box."""
    globals()[model_name] = lambda x: 2.0 * x[0] + 1.0
    m, unit = _unit()
    for t in unit.flow_out.index_set():
        unit.flow_out[t].fix(3.0)
    surrogate = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}.{model_name}",
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "probe_point": {"flow_out": 3.0},
        }
    )
    unit.swap_relation("power_electrical_relation", surrogate)
    m.obj = pyo.Objective(expr=1)
    return m, unit


@pytest.mark.component
@pytest.mark.needs_cyipopt
@pytest.mark.needs_torch
def test_swap_relation_attaches_grey_box_and_solves():
    """A swapped external_model relation solves via cyipopt to the model's
    own hand-computed value: power = 2 * flow_out + 1."""
    pytest.importorskip("torch")
    m, unit = _swapped_unit("_swap_solve_model")

    solver = pyo.SolverFactory("cyipopt")
    results = solver.solve(m, tee=False)

    assert str(results.solver.status) == "ok"
    t0 = next(iter(unit.power_electrical.index_set()))
    assert pyo.value(unit.power_electrical[t0]) == pytest.approx(7.0, rel=1e-5)


@pytest.mark.component
@pytest.mark.needs_cyipopt
@pytest.mark.needs_torch
def test_reswapping_a_grey_box_relation_solves_to_the_same_optimum():
    """Guards pitfall 2: re-swapping must deactivate the stale grey box
    itself, not just its parent surrogate block -- PyomoNLPWithGreyBoxBlocks
    checks the grey-box data object's own active flag, ignoring its parent's,
    so a flag assertion on the wrong object would pass while the bug is
    live. This checks the exact nested flag *and* that the model still
    solves to the same, correct optimum."""
    pytest.importorskip("torch")
    m, unit = _swapped_unit("_swap_resolve_model_1")
    record = next(
        r for r in unit._io_registry.relations if r.name == "power_electrical_relation"
    )
    stale_block = record.surrogate_block

    second = ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._swap_resolve_model_1",
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "probe_point": {"flow_out": 3.0},
        }
    )
    unit.swap_relation("power_electrical_relation", second)

    t0 = next(iter(unit.power_electrical.index_set()))
    assert stale_block.egb[t0].active is False

    solver = pyo.SolverFactory("cyipopt")
    results = solver.solve(m, tee=False)

    assert str(results.solver.status) == "ok"
    assert pyo.value(unit.power_electrical[t0]) == pytest.approx(7.0, rel=1e-5)


# -- evaluation cache ----------------------------------------------------------


class _CountingDriver(_StubDriver):
    """Stub driver counting how often each evaluation runs."""

    def __init__(self, model, n_inputs):
        super().__init__(model, n_inputs)
        self.calls = {"value": 0, "jacobian": 0, "hessian": 0}

    def evaluate(self, x):
        self.calls["value"] += 1
        return super().evaluate(x)

    def jacobian(self, x):
        self.calls["jacobian"] += 1
        return super().jacobian(x)

    def hessian(self, x):
        self.calls["hessian"] += 1
        return super().hessian(x)


def _counting_adapter():
    """An adapter over a :class:`_CountingDriver`, with its call counts reset."""
    driver = _CountingDriver(model=_dummy_model, n_inputs=2)
    adapter = _ExternalModelGreyBox(driver, ["x0", "x1"], "y", {"x0": 1.0, "x1": 1.0})
    return adapter, driver


@pytest.mark.unit
def test_each_derivative_is_evaluated_once_per_point():
    adapter, driver = _counting_adapter()
    adapter.set_input_values(np.array([2.0, 3.0]))
    for _ in range(3):
        adapter.evaluate_outputs()
        adapter.evaluate_jacobian_outputs()
        adapter.evaluate_hessian_outputs()
        adapter.set_input_values(np.array([2.0, 3.0]))
    assert driver.calls == {"value": 1, "jacobian": 1, "hessian": 1}


@pytest.mark.unit
def test_changing_the_point_recomputes():
    adapter, driver = _counting_adapter()
    adapter.set_input_values(np.array([2.0, 3.0]))
    assert adapter.evaluate_outputs()[0] == pytest.approx(5.0)
    adapter.set_input_values(np.array([4.0, 3.0]))
    assert adapter.evaluate_outputs()[0] == pytest.approx(7.0)
    assert driver.calls["value"] == 2


@pytest.mark.unit
def test_mutating_the_input_buffer_does_not_return_stale_values():
    adapter, _ = _counting_adapter()
    buffer = np.array([2.0, 3.0])
    adapter.set_input_values(buffer)
    assert adapter.evaluate_outputs()[0] == pytest.approx(5.0)
    buffer[0] = 4.0
    adapter.set_input_values(buffer)
    assert adapter.evaluate_outputs()[0] == pytest.approx(7.0)


@pytest.mark.unit
def test_multiplier_change_rescales_a_cached_hessian():
    adapter, driver = _counting_adapter()
    adapter.set_input_values(np.array([2.0, 3.0]))
    adapter.set_output_constraint_multipliers([2.0])
    first = adapter.evaluate_hessian_outputs().data.copy()
    adapter.set_output_constraint_multipliers([4.0])
    second = adapter.evaluate_hessian_outputs().data
    np.testing.assert_allclose(second, 2.0 * first)
    assert driver.calls["hessian"] == 1


# -- validation ----------------------------------------------------------------


def _torch_data(model_name, **extra):
    """Surrogate data for a model stored on this test module."""
    return {
        "framework": "pytorch",
        "model_path": f"{__name__}.{model_name}",
        "input_variables": {"x": "dimensionless"},
        "output_variables": {"y": "dimensionless"},
        **extra,
    }


@pytest.mark.unit
@pytest.mark.needs_torch
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_output_at_the_probe_raises_and_suggests_probe_point(bad):
    torch = pytest.importorskip("torch")
    globals()["_non_finite_model"] = lambda x: x[0] * torch.tensor(bad).double()
    with pytest.raises(FlexConfigError, match="probe_point"):
        ExternalModelSurrogate(_torch_data("_non_finite_model"))


@pytest.mark.unit
@pytest.mark.needs_torch
def test_wrong_input_count_raises_config_error():
    torch = pytest.importorskip("torch")
    globals()["_three_input_model"] = torch.nn.Linear(3, 1)
    with pytest.raises(FlexConfigError, match="1 input"):
        ExternalModelSurrogate(_torch_data("_three_input_model"))


def _stub_surrogate(monkeypatch, **extra):
    """A surrogate over the stub driver, with two inputs."""
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    return ExternalModelSurrogate(
        {
            "framework": "pytorch",
            "model_path": f"{__name__}._dummy_model",
            "input_variables": {"a": "dimensionless", "b": "dimensionless"},
            "output_variables": {"y": "dimensionless"},
            **extra,
        }
    )


@pytest.mark.unit
def test_start_point_uses_probe_then_range_midpoint_then_one(monkeypatch):
    surrogate = _stub_surrogate(
        monkeypatch,
        probe_point={"a": 5.0},
        training_range={"a": [0.0, 2.0], "b": [10.0, 20.0]},
    )
    assert surrogate._probe_point == {"a": 5.0, "b": 15.0}
    assert _stub_surrogate(monkeypatch)._probe_point == {"a": 1.0, "b": 1.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "training_range",
    [
        {"nope": [0.0, 1.0]},
        {"a": ["low", "high"]},
        {"a": [1.0]},
        {"a": 3.0},
        {"a": [2.0, 1.0]},
        {"a": [1.0, 1.0]},
        {"a": [float("nan"), 1.0]},
        ["a"],
    ],
)
def test_invalid_training_range_raises(monkeypatch, training_range):
    with pytest.raises(FlexConfigError, match="training_range"):
        _stub_surrogate(monkeypatch, training_range=training_range)


@pytest.mark.unit
def test_training_range_reaches_the_grey_box_model(monkeypatch):
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    m, unit = _unit()
    surrogate = ExternalModelSurrogate(
        {**_EXTERNAL_MODEL_DATA, "training_range": {"flow_out": [0.0, 2.0]}}
    )
    block = unit.swap_relation("power_electrical_relation", surrogate)
    t0 = next(iter(unit.flow_out.index_set()))
    external = block.egb[t0].get_external_model()
    assert external.training_range == {"flow_out": (0.0, 2.0)}
    block.egb[t0].inputs["flow_out"].set_value(3.0)
    assert [v.input_name for v in check_training_range(m)] == ["flow_out"]


@pytest.mark.unit
def test_unknown_data_key_error_lists_the_allowed_keys(monkeypatch):
    with pytest.raises(FlexConfigError, match="training_range"):
        _stub_surrogate(monkeypatch, nope=1)


# -- switching and swapping away -----------------------------------------------


def _power_multilinear():
    """A closed-form surrogate on the same relation as :func:`_swapped_unit`."""
    return MultilinearSurrogate(
        {
            "input_variables": {"flow_out": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "coefficients": {"flow_out": 2.0, "intercept": 1.0},
        }
    )


@pytest.mark.unit
def test_swapping_to_a_grey_box_drops_the_previous_coefficients(monkeypatch):
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    _, unit = _unit()
    unit.swap_relation("power_electrical_relation", _power_multilinear())
    assert any(
        p.relation_name == "power_electrical_relation"
        for p in unit._io_registry.parameters
    )
    unit.swap_relation(
        "power_electrical_relation", ExternalModelSurrogate(_EXTERNAL_MODEL_DATA)
    )
    assert not any(
        p.relation_name == "power_electrical_relation"
        for p in unit._io_registry.parameters
    )


@pytest.mark.unit
def test_switching_to_a_grey_box_block_sets_flags_and_drops_coefficients(monkeypatch):
    monkeypatch.setattr(grey_box, "get_driver", lambda framework: _StubDriver)
    _, unit = _unit()
    grey = unit.swap_relation(
        "power_electrical_relation", ExternalModelSurrogate(_EXTERNAL_MODEL_DATA)
    )
    grey_name = unit.current_surrogate_block("power_electrical_relation")
    closed = unit.swap_relation("power_electrical_relation", _power_multilinear())
    closed_name = unit.current_surrogate_block("power_electrical_relation")
    t0 = next(iter(unit.flow_out.index_set()))
    assert not grey.egb[t0].active

    unit.switch_surrogate_block(grey_name)

    assert grey.active
    assert grey.egb[t0].active
    assert not closed.active
    assert not any(
        p.relation_name == "power_electrical_relation"
        for p in unit._io_registry.parameters
    )

    unit.switch_surrogate_block(closed_name)

    assert closed.active
    assert not grey.egb[t0].active
    assert {
        p.name
        for p in unit._io_registry.parameters
        if p.relation_name == "power_electrical_relation"
    } == {"flow_out", "intercept"}


@pytest.mark.component
@pytest.mark.needs_cyipopt
@pytest.mark.needs_torch
def test_switching_away_from_a_grey_box_and_back_solves():
    pytest.importorskip("torch")
    m, unit = _swapped_unit("_switch_back_model")
    grey_name = unit.current_surrogate_block("power_electrical_relation")
    unit.swap_relation("power_electrical_relation", _power_multilinear())
    closed_name = unit.current_surrogate_block("power_electrical_relation")
    t0 = next(iter(unit.power_electrical.index_set()))

    unit.switch_surrogate_block(grey_name)
    solver = get_solver(model=m)
    solver.solve(m)
    assert solver.name == "cyipopt"
    assert pyo.value(unit.power_electrical[t0]) == pytest.approx(7.0, rel=1e-5)

    unit.switch_surrogate_block(closed_name)
    assert solver.name == "cyipopt"
    assert get_solver(model=m).name != "cyipopt"
