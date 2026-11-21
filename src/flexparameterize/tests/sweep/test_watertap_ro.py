"""Sweep a public WaterTAP RO flowsheet, fit its power, and embed it in flex-pse."""

import pandas as pd
import pyomo.environ as pyo
import pytest

from flexops.testing import dummy_time_block
from flexops.unit_models import ReverseOsmosis
from flexparameterize import ParameterizeConfig, TagMap, apply_to_model
from flexparameterize.sweep import fit_sweep, run_sweep

pytest.importorskip("parameter_sweep")
pytest.importorskip("sklearn")
ro = pytest.importorskip(
    "watertap.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery"
)


def hold_volumetric_feed(m) -> None:
    """Specify the feed by volumetric flow and salt mass fraction."""
    props = m.fs.feed.properties[0]
    props.flow_mass_phase_comp["Liq", "H2O"].unfix()
    props.flow_mass_phase_comp["Liq", "NaCl"].unfix()
    props.flow_vol_phase["Liq"].fix()
    props.mass_frac_phase_comp["Liq", "NaCl"].fix()


def build_model():
    """Build the RO flowsheet at its default operating point."""
    m = ro.build()
    ro.set_operating_conditions(m)
    hold_volumetric_feed(m)
    return m


def initialize(m) -> None:
    """Re-specify the feed and pump pressure for the point, then initialize."""
    props = m.fs.feed.properties[0]
    flow = props.flow_vol_phase["Liq"].value
    recovery = m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].value
    props.flow_vol_phase["Liq"].unfix()
    props.mass_frac_phase_comp["Liq", "NaCl"].unfix()
    ro.set_operating_conditions(m, water_recovery=recovery, flow_vol=flow)
    ro.initialize_system(m)
    hold_volumetric_feed(m)


def optimize(m, solver):
    """Solve the flowsheet and return the solver results."""
    return solver.solve(m)


CONFIG = ParameterizeConfig(
    inputs={
        "feed": {
            "path": "fs.feed.properties[0].flow_vol_phase[Liq]",
            "units": "m^3/s",
            "range": (5e-4, 1.2e-3),
        },
        "recovery": {
            "path": "fs.RO.recovery_mass_phase_comp[0,Liq,H2O]",
            "units": "dimensionless",
            "range": (0.40, 0.50),
        },
    },
    sampler={"type": "lhs", "n": 30, "seed": 20261118},
    solver={"name": "ipopt-watertap"},
    outputs={
        "power": {
            "path": "fs.costing.aggregate_flow_electricity",
            "units": "kW",
            "unit": "unit",
            "output_units": "kW",
            "input_names": {"feed": "feed", "recovery": "recovery"},
            "terms": ["feed", "recovery", "feed*recovery"],
        }
    },
    max_failed_fraction=0.05,
    holdout={"k": 5, "r": 0},
)


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    """One sweep of the RO flowsheet, shared by the tests below."""
    return run_sweep(CONFIG, build_model, initialize, optimize)


@pytest.mark.integration
@pytest.mark.needs_ipopt
def test_ro_sweep_is_reproducible(sweep):
    """A second sweep with the same config gives an identical frame."""
    assert sweep.solve_successful.all()
    pd.testing.assert_frame_equal(
        sweep, run_sweep(CONFIG, build_model, initialize, optimize)
    )


@pytest.mark.integration
@pytest.mark.needs_ipopt
def test_ro_fit_embeds_and_matches_watertap(sweep):
    """The fitted spec on a flex-pse RO unit predicts WaterTAP's power at holdout."""
    specs = fit_sweep(CONFIG, sweep)
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)

    apply_to_model(m, pd.DataFrame(), TagMap({}), surrogates=specs)

    for row in sweep.iloc[::5].itertuples():
        m.unit.feed[0].set_value(row.feed * 3600.0)
        m.unit.recovery[0].set_value(row.recovery)
        m.unit.power_electrical[0].set_value(0.0)
        predicted = -pyo.value(m.unit.surrogate_power_electrical.fitted[0].body)
        assert predicted == pytest.approx(row.power, rel=0.1)
