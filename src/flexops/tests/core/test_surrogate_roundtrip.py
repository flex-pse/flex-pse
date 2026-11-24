"""End-to-end surrogate specs: build, swap, update, emit, dump, load, rebuild.

Every round trip here goes through a JSON file: ``emit_model`` -> ``dump_spec``
-> ``load_spec`` -> ``build_model``. It then checks each active surrogate's
type, parameter values and fixed flags against the live model, and, where the
two models should match exactly, ``assert_models_equivalent``.
"""

import copy
import json
import logging
import warnings

import pyomo.environ as pyo
import pytest

from flexcore.config.io import dump_spec, load_spec
from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.config.spec import SurrogateElement
from flexcore.exceptions import FlexConfigError, FlexMigrationWarning
from flexops import apply_spec, build_model, emit_model
from flexops.core.compare import assert_models_equivalent
from flexops.core.registration import CoefficientRegistry
from flexops.core.stages import apply_relation_spec
from flexops.core.units import parse_units
from flexops.surrogates import (
    SURROGATES,
    ArimaSurrogate,
    ConstantIntensitySurrogate,
    MultilinearSurrogate,
    surrogate_from_spec,
)
from flexops.surrogates._config import VariablesConfig
from flexops.surrogates.base import Surrogate
from flexops.tests.core.test_assembler import pump, spec_dict, unit

HISTORY = {
    "start_date": "2025-01-01T00:00:00",
    "time_step_seconds": 3600.0,
    "eta_values": [1.0],
    "eps_values": [0.0],
}


def arima(inp, inp_units, out, out_units, *, ar=0.5, exog=0.8, fixed=None):
    """An ARIMA(1, 0, 1) spec with an intercept and one input."""
    return ArimaSurrogate.make_spec(
        order=(1, 0, 1),
        deterministic="intercept",
        input_variables={inp: inp_units},
        output_variables={out: out_units},
        values={"intercept": 2.0, "ar.L1": ar, "ma.L1": 0.2, f"exog.{inp}": exog},
        fixed=fixed,
        history=HISTORY,
    )


def multilinear(inp, inp_units, out, out_units, *, slope=0.4, fixed=None):
    """A one-input multilinear spec."""
    return MultilinearSurrogate.make_spec(
        input_variables={inp: inp_units},
        output_variables={out: out_units},
        coefficients={inp: slope, "intercept": 0.1},
        fixed=fixed,
    )


RO_SPLIT = multilinear("feed", "m^3/hr", "permeate", "m^3/hr")
RO_POWER = arima("permeate", "m^3/hr", "power_electrical", "kW")
PUMP_POWER = arima("flow_out", "m^3/hr", "power_electrical", "kW", ar=0.3, exog=0.6)


def element(unit_name, relation, spec):
    """A surrogate element dict placing ``spec`` on ``unit_name.relation``."""
    return SurrogateElement.from_surrogate_spec(
        spec, unit=unit_name, relation=relation
    ).model_dump(mode="json")


BASE = [{"kind": "plant", "name": "p"}, unit("p.ro", "ReverseOsmosis"), pump("p.pump")]
FULL = [
    element("p.ro", "split_definition", RO_SPLIT),
    element("p.ro", "power_electrical_relation", RO_POWER),
    element("p.pump", "power_electrical_relation", PUMP_POWER),
]


def build(*elements):
    return build_model(spec_dict(*BASE, *elements))


def states(model):
    """``{unit.relation: (type, {name: (value, fixed)})}`` of every surrogate."""
    found = {}
    for u in (model.p.ro, model.p.pump):
        for record in u._io_registry.relations:
            block = record.surrogate_block
            if block is None or not block.active:
                continue
            registry = block.coefficients
            found[f"{u.local_name}.{record.name}"] = (
                record.surrogate_instance.surrogate_type,
                {
                    name: (registry.value(name), registry.is_fixed(name))
                    for name in registry
                },
            )
        for relation, surrogate in u._io_registry.applied_surrogates.items():
            spec = surrogate.read_from_unit(u)
            found[f"{u.local_name}.{relation}"] = (
                spec.surrogate_type,
                {n: (s.value, s.fixed) for n, s in spec.parameters.items()},
            )
    return found


def file_round_trip(model, tmp_path, name="spec.json"):
    """Emit ``model`` to a JSON file, load it back, and build it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emitted = emit_model(model)
    path = tmp_path / name
    dump_spec(emitted, path)
    loaded = load_spec(path)
    return emitted, json.loads(path.read_text()), build_model(loaded)


def surrogate_elements(written):
    return {
        (el["unit"], el.get("relation", "power_electrical_relation")): el
        for el in written["elements"]
        if el["kind"] == "surrogate"
    }


def surrogate_patches(written):
    """Raw-path fix/unfix elements on surrogate blocks; there should be none."""
    return [
        el
        for el in written["elements"]
        if el["kind"] in ("fix", "unfix") and ".surrogate_" in el["variable"]
    ]


def assert_round_trips(model, tmp_path, *, exact=True):
    """Check that a file round trip keeps every surrogate and its state."""
    emitted, written, rebuilt = file_round_trip(model, tmp_path)
    assert states(rebuilt) == states(model)
    assert surrogate_patches(written) == []
    if exact:
        assert_models_equivalent(model, rebuilt)
    _, rewritten, _ = file_round_trip(rebuilt, tmp_path, "again.json")
    assert surrogate_elements(rewritten) == surrogate_elements(written)
    return written, rebuilt


@pytest.fixture
def surrogate_log():
    """Capture the surrogate parameter log, at DEBUG and above."""
    logger = logging.getLogger("flexops.surrogates.base")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield records
    logger.removeHandler(handler)
    logger.setLevel(level)


# -- several surrogates ---------------------------------------------------------


@pytest.mark.component
def test_several_surrogates_per_unit_and_model_round_trip(tmp_path):
    """Two surrogates on one unit plus one on another survive a JSON round trip."""
    model = build(*FULL)
    written, rebuilt = assert_round_trips(model, tmp_path)

    found = states(rebuilt)
    assert {key: kind for key, (kind, _) in found.items()} == {
        "ro.split_definition": SurrogateType.MULTILINEAR,
        "ro.power_electrical_relation": SurrogateType.ARIMA,
        "pump.power_electrical_relation": SurrogateType.ARIMA,
    }
    assert all(fixed for _, params in found.values() for _, fixed in params.values())
    ro_power = surrogate_elements(written)[("p.ro", "power_electrical_relation")]
    assert ro_power["parameters"]["exog.permeate"] == {"value": 0.8, "fixed": True}
    assert ro_power["config"]["history"] == HISTORY


@pytest.mark.component
@pytest.mark.parametrize(
    "ro_power",
    [
        pytest.param(
            multilinear("permeate", "m^3/hr", "power_electrical", "kW"), id="same-type"
        ),
        pytest.param(RO_POWER, id="mixed-types"),
    ],
)
def test_same_and_mixed_types_on_one_unit_round_trip(ro_power, tmp_path):
    """Two multilinear surrogates, or a multilinear and an ARIMA, on the RO."""
    model = build(
        element("p.ro", "split_definition", RO_SPLIT),
        element("p.ro", "power_electrical_relation", ro_power),
    )
    assert_round_trips(model, tmp_path)


@pytest.mark.component
def test_same_type_on_two_units_round_trips(tmp_path):
    """ARIMA on the RO and the pump keep their own values."""
    model = build(FULL[1], FULL[2])
    assert_round_trips(model, tmp_path)
    found = states(model)
    assert found["ro.power_electrical_relation"][1]["ar.L1"] == (0.5, True)
    assert found["pump.power_electrical_relation"][1]["ar.L1"] == (0.3, True)


# -- each implemented type, three ways ------------------------------------------


@pytest.mark.component
@pytest.mark.parametrize("how", ["spec", "surrogate_from_spec", "constructor"])
@pytest.mark.parametrize(
    "spec",
    [PUMP_POWER, multilinear("flow_out", "m^3/hr", "power_electrical", "kW")],
    ids=["arima", "multilinear"],
)
def test_each_swapping_type_round_trips_however_it_was_attached(spec, how, tmp_path):
    """A spec element, surrogate_from_spec and the class constructor all emit."""
    if how == "spec":
        model = build(element("p.pump", "power_electrical_relation", spec))
    else:
        model = build()
        surrogate = (
            surrogate_from_spec(spec)
            if how == "surrogate_from_spec"
            else type(surrogate_from_spec(spec))(spec)
        )
        model.p.pump.swap_relation("power_electrical_relation", surrogate)
    written, _ = assert_round_trips(model, tmp_path)
    emitted = surrogate_elements(written)[("p.pump", "power_electrical_relation")]
    assert emitted["surrogate_type"] == spec.surrogate_type.value
    assert emitted["config_version"] == 1


@pytest.mark.component
@pytest.mark.parametrize("how", ["spec", "apply_relation_spec", "apply_spec"])
def test_constant_intensity_round_trips_however_it_was_applied(how, tmp_path):
    """A constant intensity sets energy_intensity and emits it with its state."""
    spec = ConstantIntensitySurrogate.make_spec(energy_intensity=0.9)
    el = element("p.pump", "power_electrical_relation", spec)
    if how == "spec":
        model = build(el)
    else:
        model = build()
        if how == "apply_relation_spec":
            apply_relation_spec(model.p.pump, spec)
        else:
            apply_spec(model, [el])
    assert pyo.value(model.p.pump.energy_intensity) == pytest.approx(0.9)
    written, rebuilt = assert_round_trips(model, tmp_path)
    emitted = surrogate_elements(written)[("p.pump", "power_electrical_relation")]
    assert emitted["parameters"] == {"energy_intensity": {"value": 0.9, "fixed": True}}
    assert pyo.value(rebuilt.p.pump.energy_intensity) == pytest.approx(0.9)


@pytest.mark.component
def test_constant_intensity_live_change_is_emitted(tmp_path):
    """Changing the intensity after the build is what the emitted spec says."""
    model = build(
        element(
            "p.pump",
            "power_electrical_relation",
            ConstantIntensitySurrogate.make_spec(energy_intensity=0.9),
        )
    )
    model.p.pump.update_parameters({"energy_intensity": 1.4})
    model.p.pump.energy_intensity.unfix()
    written, _ = assert_round_trips(model, tmp_path)
    emitted = surrogate_elements(written)[("p.pump", "power_electrical_relation")]
    assert emitted["parameters"] == {"energy_intensity": {"value": 1.4, "fixed": False}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "surrogate_type", ["quadratic", "exponential", "neural_network"]
)
def test_stub_types_raise_not_implemented_from_a_spec(surrogate_type):
    """A declared-but-unimplemented type fails at build, naming an alternative."""
    stub = {
        "kind": "surrogate",
        "unit": "p.pump",
        "surrogate_type": surrogate_type,
        "config": {},
        "parameters": {},
    }
    with pytest.raises(NotImplementedError, match="MultilinearSurrogate"):
        build(stub)


# -- swapping ---------------------------------------------------------------------


@pytest.mark.component
def test_swap_twice_keeps_the_second_surrogate(tmp_path):
    """The emitted spec describes the active block, not the first one."""
    model = build(*FULL)
    second = arima("permeate", "m^3/hr", "power_electrical", "kW", ar=0.9, exog=0.1)
    model.p.ro.swap_relation("power_electrical_relation", surrogate_from_spec(second))
    _, rebuilt = assert_round_trips(model, tmp_path, exact=False)
    assert states(rebuilt)["ro.power_electrical_relation"][1]["ar.L1"] == (0.9, True)


@pytest.mark.component
@pytest.mark.parametrize(
    "second",
    [
        pytest.param(
            arima("permeate", "m^3/hr", "power_electrical", "kW", ar=0.9),
            id="same-type",
        ),
        pytest.param(
            multilinear("permeate", "m^3/hr", "power_electrical", "kW"),
            id="different-type",
        ),
    ],
)
def test_switching_back_emits_the_reactivated_surrogate(second, tmp_path):
    """After switch_surrogate_block, the spec is the block switched to."""
    model = build(*FULL)
    model.p.ro.swap_relation("power_electrical_relation", surrogate_from_spec(second))
    model.p.ro.switch_surrogate_block("surrogate_power_electrical")
    written, rebuilt = assert_round_trips(model, tmp_path, exact=False)
    emitted = surrogate_elements(written)[("p.ro", "power_electrical_relation")]
    assert emitted["surrogate_type"] == "arima"
    assert emitted["parameters"]["ar.L1"]["value"] == 0.5
    assert states(rebuilt) == states(build(*FULL))


@pytest.mark.component
def test_swap_twice_then_unfix_keeps_the_free_coefficients(tmp_path):
    """Unfixed coefficients on a second (renamed) block come back unfixed."""
    model = build(*FULL)
    second = arima("flow_out", "m^3/hr", "power_electrical", "kW", ar=0.7)
    model.p.pump.swap_relation("power_electrical_relation", surrogate_from_spec(second))
    assert model.p.pump.current_surrogate_block("power_electrical_relation") == (
        "surrogate_power_electrical_1"
    )
    model.p.pump.unfix_surrogate_coefficients()
    _, rebuilt = assert_round_trips(model, tmp_path, exact=False)
    params = states(rebuilt)["pump.power_electrical_relation"][1]
    assert not any(fixed for _, fixed in params.values())
    assert params["ar.L1"] == (0.7, False)


# -- live changes ----------------------------------------------------------------


@pytest.mark.component
def test_live_updates_and_fixed_state_are_emitted(tmp_path):
    """update_parameters, unfix and fix again all show in the emitted spec."""
    model = build(*FULL)
    pump_unit = model.p.pump
    pump_unit.update_parameters({"ar.L1": 0.7})
    written, _ = assert_round_trips(model, tmp_path)
    params = surrogate_elements(written)[("p.pump", "power_electrical_relation")][
        "parameters"
    ]
    assert params["ar.L1"] == {"value": 0.7, "fixed": True}

    pump_unit.unfix_surrogate_coefficients()
    written, _ = assert_round_trips(model, tmp_path)
    params = surrogate_elements(written)[("p.pump", "power_electrical_relation")][
        "parameters"
    ]
    assert not any(state["fixed"] for state in params.values())

    pump_unit.fix_surrogate_coefficients()
    written, _ = assert_round_trips(model, tmp_path)
    params = surrogate_elements(written)[("p.pump", "power_electrical_relation")][
        "parameters"
    ]
    assert all(state["fixed"] for state in params.values())


@pytest.mark.component
def test_update_parameters_refuses_a_name_two_surrogates_share():
    """Both RO surrogates have an intercept, so the bare name is ambiguous."""
    model = build(*FULL)
    with pytest.raises(FlexConfigError, match="several relations"):
        model.p.ro.update_parameters({"intercept": 1.0})
    model.p.ro.update_surrogate_parameters(
        {"intercept": 1.0}, relation_name="split_definition"
    )
    found = states(model)
    assert found["ro.split_definition"][1]["intercept"] == (1.0, True)
    assert found["ro.power_electrical_relation"][1]["intercept"] == (2.0, True)


@pytest.mark.component
def test_update_surrogate_parameters_rejects_unknown_names():
    """A name the surrogate doesn't register is an error listing its names."""
    model = build(*FULL)
    with pytest.raises(FlexConfigError, match="ar.L1"):
        model.p.pump.update_surrogate_parameters({"ar.L2": 0.1})


# -- apply_spec ---------------------------------------------------------------


@pytest.mark.component
def test_apply_spec_updates_in_place_when_the_config_matches(tmp_path):
    """Same type and config: values and fixed flags change, no new block."""
    model = build(*FULL)
    updated = arima(
        "flow_out", "m^3/hr", "power_electrical", "kW", ar=0.25, fixed={"ar.L1": False}
    )
    apply_spec(model, [element("p.pump", "power_electrical_relation", updated)])
    assert model.p.pump.list_surrogate_blocks() == ["surrogate_power_electrical"]
    params = states(model)["pump.power_electrical_relation"][1]
    assert params["ar.L1"] == (0.25, False)
    assert params["ma.L1"] == (0.2, True)
    assert_round_trips(model, tmp_path)


@pytest.mark.component
def test_apply_spec_swaps_when_the_config_differs(tmp_path):
    """A different config (here, a type change) swaps a new block in."""
    model = build(*FULL)
    other = multilinear("flow_out", "m^3/hr", "power_electrical", "kW")
    apply_spec(model, [element("p.pump", "power_electrical_relation", other)])
    assert model.p.pump.list_surrogate_blocks() == [
        "surrogate_power_electrical",
        "surrogate_power_electrical_1",
    ]
    assert (
        states(model)["pump.power_electrical_relation"][0] is SurrogateType.MULTILINEAR
    )
    assert_round_trips(model, tmp_path, exact=False)


# -- free parameters and logging ---------------------------------------------------


@pytest.mark.component
def test_free_parameters_in_a_spec_survive_and_are_logged(tmp_path, surrogate_log):
    """A spec freeing a coefficient builds it free, logs it at INFO, emits it free."""
    free = arima("flow_out", "m^3/hr", "power_electrical", "kW", fixed={"ma.L1": False})
    model = build(FULL[0], element("p.pump", "power_electrical_relation", free))
    messages = [(r.levelno, r.getMessage()) for r in surrogate_log]
    assert (
        logging.INFO,
        "p.pump.power_electrical_relation: arima surrogate swapped in, 1 of 4 "
        "parameters free: ma.L1",
    ) in messages
    assert (
        logging.DEBUG,
        "p.ro.split_definition: multilinear surrogate swapped in, all 2 parameters "
        "fixed",
    ) in messages
    assert not any(
        level == logging.INFO and "split_definition" in text for level, text in messages
    )

    surrogate_log.clear()
    written, _ = assert_round_trips(model, tmp_path)
    assert written_free(written) == {("p.pump", "power_electrical_relation"): ["ma.L1"]}
    assert any(
        r.levelno == logging.INFO and "surrogate written, 1 of 4" in r.getMessage()
        for r in surrogate_log
    )


def written_free(written):
    return {
        key: [name for name, state in el["parameters"].items() if not state["fixed"]]
        for key, el in surrogate_elements(written).items()
        if any(not state["fixed"] for state in el["parameters"].values())
    }


@pytest.mark.component
def test_a_free_parameter_without_a_value_starts_from_the_guess(tmp_path):
    """Omitting a free parameter's value builds it at the documented guess."""
    spec = PUMP_POWER.model_copy(deep=True)
    spec.parameters["ma.L1"] = spec.parameters["ma.L1"].model_copy(
        update={"value": None, "fixed": False}
    )
    model = build(element("p.pump", "power_electrical_relation", spec))
    assert states(model)["pump.power_electrical_relation"][1]["ma.L1"] == (1.0, False)
    assert_round_trips(model, tmp_path)


# -- indexed units ------------------------------------------------------------------


def indexed_pump():
    return [
        {"kind": "set", "name": "p.trains", "values": [1, 2]},
        pump("p.pump", index="p.trains"),
    ]


@pytest.mark.component
def test_indexed_unit_shares_one_surrogate(tmp_path):
    """Every member gets the surrogate; one element is written for the unit."""
    model = build_model(
        spec_dict(
            {"kind": "plant", "name": "p"},
            *indexed_pump(),
            element("p.pump", "power_electrical_relation", PUMP_POWER),
        )
    )
    emitted, written, rebuilt = file_round_trip(model, tmp_path)
    assert list(surrogate_elements(written)) == [
        ("p.pump", "power_electrical_relation")
    ]
    assert surrogate_patches(written) == []
    assert_models_equivalent(model, rebuilt)


@pytest.mark.component
def test_indexed_members_with_different_values_round_trip(tmp_path):
    """Member 0's values are written; the rebuild check patches the others."""
    model = build_model(
        spec_dict(
            {"kind": "plant", "name": "p"},
            *indexed_pump(),
            element("p.pump", "power_electrical_relation", PUMP_POWER),
        )
    )
    model.p.pump[2].update_parameters({"ar.L1": 0.45})
    emitted, written, rebuilt = file_round_trip(model, tmp_path)
    element_written = surrogate_elements(written)[
        ("p.pump", "power_electrical_relation")
    ]
    assert element_written["parameters"]["ar.L1"]["value"] == 0.3
    registry = rebuilt.p.pump[2].surrogate_power_electrical.coefficients
    assert registry.value("ar.L1") == pytest.approx(0.45)
    assert_models_equivalent(model, rebuilt)


@pytest.mark.component
def test_indexed_members_with_different_types_cannot_be_written():
    """Members of one unit element can't carry different surrogate types."""
    model = build_model(
        spec_dict(
            {"kind": "plant", "name": "p"},
            *indexed_pump(),
            element("p.pump", "power_electrical_relation", PUMP_POWER),
        )
    )
    model.p.pump[2].swap_relation(
        "power_electrical_relation",
        surrogate_from_spec(
            multilinear("flow_out", "m^3/hr", "power_electrical", "kW")
        ),
    )
    with pytest.raises(FlexConfigError, match="differ"):
        emit_model(model)


# -- receding-horizon state -------------------------------------------------------


def _pretend_solved(model, unit_name="pump"):
    """Give the ARIMA target and innovations solved-looking values."""
    u = getattr(model.p, unit_name)
    block = u.surrogate_power_electrical
    for position, t in enumerate(block._time_values):
        u.power_electrical[t].set_value(3.0 + position)
        block.eps[t].set_value(0.01 * position)
    return u, block


@pytest.mark.component
def test_get_surrogate_spec_is_a_pure_read():
    """Reading the spec twice, before and after values change, changes nothing."""
    model = build(*FULL)
    first = model.p.pump.get_surrogate_spec()
    _pretend_solved(model)
    assert model.p.pump.get_surrogate_spec() == first
    assert isinstance(first, SurrogateSpec)
    assert first.config["history"] == HISTORY


@pytest.mark.component
def test_update_surrogate_state_records_the_solved_history(tmp_path):
    """After update_surrogate_state the spec's history ends at the horizon."""
    model = build(*FULL)
    u, block = _pretend_solved(model)
    before = [pyo.value(block.initial_eta_history[i]) for i in block.eta_history_index]

    u.update_surrogate_state()
    history = u.get_surrogate_spec().config["history"]
    horizon = len(block._time_values)
    assert len(history["eta_values"]) == 1 + horizon
    assert history["eta_values"][:1] == pytest.approx(before)
    assert history["eps_values"][1:] == pytest.approx(
        [0.01 * i for i in range(horizon)]
    )
    assert history["start_date"].startswith("2025-01-01")

    # The same horizon rebuilt from the recorded state starts from the same lags.
    _, written, rebuilt = file_round_trip(model, tmp_path)
    rebuilt_block = rebuilt.p.pump.surrogate_power_electrical
    assert [
        pyo.value(rebuilt_block.initial_eta_history[i])
        for i in rebuilt_block.eta_history_index
    ] == pytest.approx(before)
    emitted = surrogate_elements(written)[("p.pump", "power_electrical_relation")]
    assert emitted["config"]["history"] == history


@pytest.mark.component
def test_next_horizon_initializes_from_the_solved_values():
    """A horizon starting inside the solved window uses the solved lags."""
    model = build(*FULL)
    u, block = _pretend_solved(model)
    u.update_surrogate_state()
    spec = u.get_surrogate_spec()
    history = spec.config["history"]

    later = spec_dict(*BASE, element("p.pump", "power_electrical_relation", spec))
    time = next(el for el in later["elements"] if el["kind"] == "time")
    time["start_date"] = "2025-01-01 05:00"
    next_model = build_model(later)
    next_block = next_model.p.pump.surrogate_power_electrical
    # Five steps in, the one AR lag is the solved disturbance at step 4.
    assert pyo.value(next_block.initial_eta_history[0]) == pytest.approx(
        history["eta_values"][1 + 4]
    )
    assert pyo.value(next_block.initial_eps_history[0]) == pytest.approx(0.04)


@pytest.mark.component
def test_update_surrogate_state_refuses_an_unsolved_block():
    """A target with no values can't be recorded as solved state."""
    model = build(*FULL)
    for t in model.p.pump.power_electrical:
        model.p.pump.power_electrical[t].set_value(None)
    with pytest.raises(FlexConfigError, match="after a successful solve"):
        model.p.pump.update_surrogate_state()


@pytest.mark.component
def test_update_surrogate_state_is_a_no_op_without_state():
    """A multilinear surrogate has no time-dependent state to record."""
    model = build(*FULL)
    before = model.p.ro.get_surrogate_spec("split_definition")
    model.p.ro.update_surrogate_state("split_definition")
    assert model.p.ro.get_surrogate_spec("split_definition") == before


# -- ARIMA input key order -------------------------------------------------------


@pytest.mark.component
def test_arima_exog_pairs_by_name_after_a_json_key_reorder(tmp_path):
    """Reordering input_variables in the JSON file keeps each exog.<name> paired."""
    two_inputs = ArimaSurrogate.make_spec(
        order=(0, 0, 0),
        input_variables={"feed": "m^3/hr", "permeate": "m^3/hr"},
        output_variables={"power_electrical": "kW"},
        values={"exog.feed": 0.1, "exog.permeate": 0.9},
        history=HISTORY,
    )
    model = build(element("p.ro", "power_electrical_relation", two_inputs))
    _, written, _ = file_round_trip(model, tmp_path)
    el = surrogate_elements(written)[("p.ro", "power_electrical_relation")]
    el["config"]["input_variables"] = dict(
        reversed(list(el["config"]["input_variables"].items()))
    )
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(written))

    rebuilt = build_model(load_spec(path))
    block = rebuilt.p.ro.surrogate_power_electrical
    assert block._exog_names == ["permeate", "feed"]
    assert block.coefficients.value("exog.feed") == pytest.approx(0.1)
    assert block.coefficients.value("exog.permeate") == pytest.approx(0.9)
    assert pyo.value(block.exog_coefs[1]) == pytest.approx(0.9)


# -- 0.1.1 specs ---------------------------------------------------------------------


@pytest.mark.component
def test_a_0_1_1_spec_builds_the_same_model_as_its_migration():
    """The old data layout loads, migrates and builds what the new layout builds."""
    old = spec_dict(
        *BASE,
        {
            "kind": "surrogate",
            "unit": "p.ro",
            "relation": "split_definition",
            "surrogate_type": "multilinear",
            "data": {
                "input_variables": {"feed": "m^3/hr"},
                "output_variables": {"permeate": "m^3/hr"},
                "coefficients": {"feed": 0.4, "intercept": 0.1},
            },
        },
        {
            "kind": "surrogate",
            "unit": "p.pump",
            "surrogate_type": "arima",
            "data": {
                "input_variables": {"flow_out": "m^3/hr"},
                "output_variables": {"power_electrical": "kW"},
                "coefficients": {
                    "order": [1, 0, 1],
                    "intercept": 2.0,
                    "ar_coefs": [0.3],
                    "ma_coefs": [0.2],
                    "exog_coefs": {"flow_out": 0.6},
                },
                "history": HISTORY,
            },
        },
    )
    old["schema_version"] = "0.1.1"
    with pytest.warns(FlexMigrationWarning):
        migrated = build_model(load_spec(old))
    assert_models_equivalent(migrated, build(FULL[0], FULL[2]))


# -- array parameters ---------------------------------------------------------------


class _LinearFeaturesConfig(VariablesConfig):
    """Test-only config: a weight row over the inputs."""


class LinearFeaturesSurrogate(Surrogate):
    """Test-only surrogate with array parameters: y = W @ x + b.

    ``W`` is a ``(1, n)`` matrix and ``b`` a ``(1,)`` vector, so a spec carries
    nested lists, and one fixed flag covers each whole array.
    """

    surrogate_type = SurrogateType.NEURAL_NETWORK
    config_model = _LinearFeaturesConfig

    @classmethod
    def parameter_shapes(cls, config):
        return {"W": (1, len(config.input_variables)), "b": (1,)}

    @property
    def input_variables(self):
        return dict(self.config.input_variables)

    @property
    def output_variables(self):
        return dict(self.config.output_variables)

    def _build(self, unit, target):
        del target
        names = list(self.input_variables)
        out_units = parse_units(next(iter(self.output_variables.values())))
        inputs = [
            (unit.resolve_variable(n, field="input_variables"), parse_units(u))
            for n, u in self.input_variables.items()
        ]
        block = pyo.Block(concrete=True)
        block.W = pyo.Var([0], range(len(names)), initialize=0.0)
        block.b = pyo.Var([0], initialize=0.0)
        block.coefficients = CoefficientRegistry()
        block.coefficients.register_coefficient("W", block.W, shape=(1, len(names)))
        block.coefficients.register_coefficient("b", block.b)

        def body(t):
            total = block.b[0]
            for j, (var, units) in enumerate(inputs):
                total = total + block.W[0, j] * (
                    pyo.units.convert(var[t], units) / units
                )
            return total * out_units

        return block, body


@pytest.fixture
def array_surrogate(monkeypatch):
    monkeypatch.setitem(
        SURROGATES, SurrogateType.NEURAL_NETWORK, LinearFeaturesSurrogate
    )
    return LinearFeaturesSurrogate


def features_spec(W, b, *, fixed=None):
    return LinearFeaturesSurrogate._make(
        config={
            "input_variables": {"feed": "m^3/hr", "permeate": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
        },
        values={"W": W, "b": b},
        fixed=fixed,
    )


@pytest.mark.component
def test_array_parameters_round_trip_as_nested_lists(array_surrogate, tmp_path):
    """A matrix and a vector parameter survive a JSON round trip, with one flag each."""
    spec = features_spec([[0.25, -0.5]], [0.1], fixed={"b": False})
    model = build(element("p.ro", "power_electrical_relation", spec))
    block = model.p.ro.surrogate_power_electrical
    assert pyo.value(block.W[0, 1]) == pytest.approx(-0.5)
    assert all(block.W[i].fixed for i in block.W)
    assert not block.b[0].fixed

    written, _ = assert_round_trips(model, tmp_path)
    params = surrogate_elements(written)[("p.ro", "power_electrical_relation")][
        "parameters"
    ]
    assert params == {
        "W": {"value": [[0.25, -0.5]], "fixed": True},
        "b": {"value": [0.1], "fixed": False},
    }


@pytest.mark.component
def test_array_parameters_update_fix_and_reject_wrong_shapes(array_surrogate):
    """update_parameters takes a list; fix/unfix act on the whole array."""
    model = build(
        element("p.ro", "power_electrical_relation", features_spec([[1.0, 2.0]], [0.0]))
    )
    ro = model.p.ro
    ro.update_surrogate_parameters({"W": [[3.0, 4.0]]})
    registry = ro.surrogate_power_electrical.coefficients
    assert registry.value("W") == [[3.0, 4.0]]
    registry.unfix("W")
    assert not registry.is_fixed("W")
    with pytest.raises(FlexConfigError, match=r"shape \(1, 2\)"):
        ro.update_surrogate_parameters({"W": [3.0, 4.0]})
    with pytest.raises(FlexConfigError, match=r"must have shape \(1, 2\)"):
        features_spec([[1.0], [2.0]], [0.0])
    with pytest.raises(FlexConfigError, match="ragged"):
        features_spec([[1.0, 2.0], [3.0]], [0.0])


@pytest.mark.unit
def test_copy_of_full_is_untouched():
    """The module-level elements are never mutated by the tests."""
    assert copy.deepcopy(FULL) == FULL


# --- Named variables must exist, and be the right ones ---------------------


@pytest.mark.component
@pytest.mark.parametrize("make", [arima, multilinear], ids=["arima", "multilinear"])
@pytest.mark.parametrize(
    ("inp", "out", "match"),
    [
        ("flow_outt", "power_electrical", "input_variables names 'flow_outt'"),
        ("flow_out", "power_electrcal", "output_variables names 'power_electrcal'"),
        ("flow_out", "flow_in", "names output 'flow_in', but"),
        ("outlet_state", "power_electrical", "which is a Block"),
    ],
    ids=["input-typo", "output-typo", "wrong-output", "input-not-a-variable"],
)
def test_a_surrogate_naming_a_missing_or_wrong_variable_is_not_built(
    make, inp, out, match
):
    """A spec whose variable names don't resolve to the relation's is refused."""
    spec = make(inp, "m^3/hr", out, "kW")
    el = element("p.pump", "power_electrical_relation", spec)
    with pytest.raises(FlexConfigError, match=match):
        build(el)
    model = build()
    with pytest.raises(FlexConfigError, match=match):
        apply_spec(model, [el])
    # The refused swap left the unit's own relation in place.
    assert model.p.pump.power_electrical_relation.active
    assert model.p.pump.current_surrogate_block() is None


@pytest.mark.component
@pytest.mark.parametrize(
    ("inp", "out", "match"),
    [
        ("flow_outt", "power_electrical", "input_variables names 'flow_outt'"),
        ("flow_out", "power_electrcal", "output_variables names 'power_electrcal'"),
        ("flow_in", "power_electrical", "names input 'flow_in', but"),
        ("flow_out", "flow_in", "names output 'flow_in', but"),
    ],
    ids=["input-typo", "output-typo", "wrong-input", "wrong-output"],
)
def test_a_constant_intensity_naming_a_missing_or_wrong_variable_is_refused(
    inp, out, match
):
    """The recorded names must be the unit's intensity basis and power draw."""
    spec = ConstantIntensitySurrogate.make_spec(
        energy_intensity=0.9,
        input_variables={inp: "m^3/hr"},
        output_variables={out: "kW"},
    )
    with pytest.raises(FlexConfigError, match=match):
        build(element("p.pump", "power_electrical_relation", spec))
    model = build()
    before = pyo.value(model.p.pump.energy_intensity)
    with pytest.raises(FlexConfigError, match=match):
        apply_relation_spec(model.p.pump, spec)
    assert pyo.value(model.p.pump.energy_intensity) == before


@pytest.mark.component
@pytest.mark.parametrize(
    "names",
    [("flow_out", "power_electrical"), ("p.pump.flow_out", "p.pump.power_electrical")],
    ids=["unit-local", "model-alias"],
)
def test_a_constant_intensity_accepts_local_names_and_model_aliases(names, tmp_path):
    """apply_to_model records model aliases; hand-written specs use local names."""
    inp, out = names
    spec = ConstantIntensitySurrogate.make_spec(
        energy_intensity=0.9,
        input_variables={inp: "m^3/hr"},
        output_variables={out: "kW"},
    )
    model = build(element("p.pump", "power_electrical_relation", spec))
    assert pyo.value(model.p.pump.energy_intensity) == pytest.approx(0.9)
    assert_round_trips(model, tmp_path)
