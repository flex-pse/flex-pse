"""Round-trip corpus: code-built models that must survive emit -> JSON -> build."""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexcore.config.schema import UnitCommitmentConfig
from flexops import logic
from flexops.costing import currency_units
from flexops.logic import (
    add_conditional,
    add_startup_shutdown,
    add_status,
    register_parallel_group,
)
from flexops.tests.sample_extensions import minimize_power, terminal_level

# --- api_freeze_like ---


def api_freeze_like() -> pyo.ConcreteModel:
    """The frozen API script's model, priced with energy_prices instead of a tariff."""
    m = pyo.ConcreteModel(name="waterfacility")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-30", time_step=15 * pyunits.min
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.12 * currency_units("USD") / pyunits.kWh},
    )
    m.waterfacility = fo.PlantBlock(time_block=m.time_block)
    m.waterfacility.tank = fo.Tank(property_package=m.properties)
    m.waterfacility.plant = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.waterfacility.tank_to_plant = Arc(
        source=m.waterfacility.tank.outlet, destination=m.waterfacility.plant.inlet
    )
    m.waterfacility.battery = fo.BatteryModel(
        capacity=1 * pyunits.kWh, costing_package=m.costing
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- battery_status ---


def battery_status() -> pyo.ConcreteModel:
    """A battery that builds its own status; the spec carries no status element."""
    m = pyo.ConcreteModel(name="battery_status")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.battery = fo.BatteryModel(
        capacity=10 * pyunits.kWh,
        power_charge_max=5 * pyunits.kW,
        power_discharge_max=5 * pyunits.kW,
        unit_commitment=UnitCommitmentConfig(status=True),
        costing_package=m.costing,
    )
    logic.add_startup_shutdown(m.site.battery, m.site.battery.status)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- datacenter ---


def datacenter() -> pyo.ConcreteModel:
    """Two servers, one with its own status, plus costed space cooling and chiller."""
    m = pyo.ConcreteModel(name="datacenter")
    m.time_block = fo.TimeBlock(
        start_date="2025-07-01", end_date="2025-07-01T01:00", time_step=15 * pyunits.min
    )
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.r1 = fo.Server(
        utilization=[0.4, 0.6, 0.8, 0.5],
        utilization_exponent=1.4,
        idle_power=4 * pyunits.kW,
        peak_power=12 * pyunits.kW,
        fan_rated_power=0.3 * pyunits.kW,
        rated_airflow=1.0 * pyunits.kg / pyunits.s,
        airflow=[0.9, 0.9, 1.0, 0.9],
        costing_package=m.costing,
    )
    m.site.r2 = fo.Server(
        utilization=[0.7, 0.7, 0.7, 0.7],
        idle_power=5 * pyunits.kW,
        peak_power=15 * pyunits.kW,
        liquid_fraction=0.5,
        pdu_loss_fraction=0.03,
        unit_commitment=UnitCommitmentConfig(status=True),
        rated_power=20 * pyunits.kW,
        costing_package=m.costing,
    )
    m.site.crah = fo.SpaceCooling(
        cop=4.0, ancillary_intensity=0.08, costing_package=m.costing
    )
    m.site.chiller = fo.Chiller(cop=5.5, costing_package=m.costing)
    m.site.crah.cooling_load.fix(25.0)
    m.site.chiller.cooling_load.fix(32.0)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- desalination ---

RATED_FEED = 337.67
"""Feed per RO skid, m^3/hr; a skid is off or at this flow."""


INTAKE_FLOW = 1063.5
"""Fixed seawater intake, m^3/hr."""


RECUP_STEPS = 3
"""Steps post-treatment stays out after the RO system restarts."""


DESALINATION_PRICES = [0.1 + 0.2 * ((t % 24) in range(16, 21)) for t in range(48)]
"""Electricity price per hour, USD/kWh."""


@fo.extension
def post_treatment_recuperation(model, *, steps: int):
    """Run post-treatment only once ro[0] has been on for ``steps`` steps."""
    plant = model.plant

    @plant.Constraint(model.time_block.time_index)
    def post_treatment_recuperation(b, t):
        started = sum(
            plant.ro[0].startup[s] for s in range(max(t - steps + 1, 1), t + 1)
        )
        return plant.post_treatment.status[t] <= plant.ro[0].status[t] - started


@fo.extension
def set_demand(model, *, volume: float):
    """Set the horizon's product-water obligation, in m^3."""
    model.plant.potable.delivery_min.set_value(volume)


def desalination() -> pyo.ConcreteModel:
    """Desalination gate: a trimmed copy of the flex-pse-examples desalination model."""
    m3_hr = pyunits.m**3 / pyunits.hr
    kwh_m3 = pyunits.kWh / pyunits.m**3
    m = pyo.ConcreteModel(name="desalination")
    m.time_block = fo.TimeBlock(
        start_date="2026-07-01", end_date="2026-07-03", time_step=1 * pyunits.hr
    )
    tb = m.time_block
    m.properties = fo.SimpleAqueousFlow()
    usd_per_kwh = currency_units("USD") / pyunits.kWh
    m.costing = fo.FlexCosting(
        time_block=tb,
        energy_prices={"electrical": [p * usd_per_kwh for p in DESALINATION_PRICES]},
    )
    m.plant = fo.PlantBlock(time_block=tb)
    plant = m.plant
    plant.trains = pyo.Set(initialize=range(3), ordered=True)
    costed = {"property_package": m.properties, "costing_package": m.costing}

    plant.intake_pump = fo.Pump(
        power_relation="constant_intensity", energy_intensity=0.157 * kwh_m3, **costed
    )
    plant.pretreatment = fo.ConstantEnergyIntensityModel(
        plant.trains, energy_intensity=0.01 * kwh_m3, **costed
    )
    plant.ro = fo.ReverseOsmosis(
        plant.trains,
        recovery=0.465,
        recovery_min=0.4,
        recovery_max=0.5,
        energy_intensity=3.34 * kwh_m3,
        **costed,
    )
    for i in plant.trains:
        plant.ro[i].recovery.unfix()
    plant.post_treatment = fo.ConstantEnergyIntensityModel(
        energy_intensity=0.11 * kwh_m3, **costed
    )
    plant.product_pump = fo.Pump(
        power_relation="constant_intensity", energy_intensity=0.3 * kwh_m3, **costed
    )
    plant.feed_header = fo.Splitter(
        property_package=m.properties,
        outlet_names=tuple(f"t{i}" for i in plant.trains),
    )
    plant.train_split = fo.Splitter(
        plant.trains, property_package=m.properties, outlet_names=("ro", "bypass")
    )
    plant.permeate_split = fo.Splitter(
        plant.trains, property_package=m.properties, outlet_names=("header", "offspec")
    )
    plant.permeate_header = fo.Mixer(
        property_package=m.properties,
        inlet_names=tuple(f"t{i}" for i in plant.trains),
    )
    plant.outfall = fo.Product(
        property_package=m.properties,
        inlet_names=tuple(
            f"{stream}_{i}"
            for i in plant.trains
            for stream in ("brine", "bypass", "offspec")
        ),
        resource_name="outfall",
    )
    plant.potable = fo.Product(
        property_package=m.properties,
        resource_name="potable_water",
        min_demand=1.0 * pyunits.m**3,
        demand_basis="horizon",
    )
    plant.seawater = fo.Feed(property_package=m.properties, resource_name="seawater")

    plant.seawater_to_intake = Arc(
        source=plant.seawater.outlet_a,
        destination=plant.intake_pump.inlet,
        doc="Raw seawater to the intake pump.",
    )
    plant.intake_to_header = Arc(
        source=plant.intake_pump.outlet, destination=plant.feed_header.inlet
    )

    @plant.Arc(plant.trains, doc="Feed header to each train's pretreatment.")
    def header_to_pretreatment(b, i):
        return (
            plant.feed_header.find_component(f"outlet_t{i}"),
            plant.pretreatment[i].inlet,
        )

    @plant.Arc(plant.trains)
    def pretreatment_to_split(b, i):
        return (plant.pretreatment[i].outlet, plant.train_split[i].inlet)

    @plant.Arc(plant.trains)
    def split_to_ro(b, i):
        return (plant.train_split[i].outlet_ro, plant.ro[i].inlet)

    @plant.Arc(plant.trains)
    def ro_to_permeate_split(b, i):
        return (plant.ro[i].outlet_a, plant.permeate_split[i].inlet)

    @plant.Arc(plant.trains)
    def permeate_split_to_header(b, i):
        return (
            plant.permeate_split[i].outlet_header,
            plant.permeate_header.find_component(f"inlet_t{i}"),
        )

    plant.header_to_post = Arc(
        source=plant.permeate_header.outlet, destination=plant.post_treatment.inlet
    )
    plant.post_to_product = Arc(
        source=plant.post_treatment.outlet, destination=plant.product_pump.inlet
    )
    plant.product_to_delivery = Arc(
        source=plant.product_pump.outlet, destination=plant.potable.inlet_a
    )

    @plant.Arc(plant.trains)
    def brine_to_outfall(b, i):
        return (plant.ro[i].outlet_b, plant.outfall.find_component(f"inlet_brine_{i}"))

    @plant.Arc(plant.trains)
    def bypass_to_outfall(b, i):
        return (
            plant.train_split[i].outlet_bypass,
            plant.outfall.find_component(f"inlet_bypass_{i}"),
        )

    @plant.Arc(plant.trains)
    def offspec_to_outfall(b, i):
        return (
            plant.permeate_split[i].outlet_offspec,
            plant.outfall.find_component(f"inlet_offspec_{i}"),
        )

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    always_on = [plant.intake_pump, *plant.pretreatment.values()]
    for unit in always_on:
        add_status(unit, unit.flow_in, 0 * m3_hr, INTAKE_FLOW * m3_hr)
        for t in tb.time_index:
            unit.status[t].fix(1)
    for t in tb.time_index:
        plant.seawater.withdrawal[t].fix(INTAKE_FLOW)
    for i in plant.trains:
        add_status(
            plant.ro[i], plant.ro[i].feed, RATED_FEED * m3_hr, RATED_FEED * m3_hr
        )
        add_startup_shutdown(
            plant.ro[i], plant.ro[i].status, min_uptime=RECUP_STEPS, min_downtime=1
        )
    add_status(
        plant.post_treatment,
        plant.post_treatment.flow_in,
        0 * m3_hr,
        3 * RATED_FEED * 0.5 * m3_hr,
    )
    post_treatment_recuperation(m, steps=RECUP_STEPS)
    register_parallel_group([plant.ro[i] for i in plant.trains])
    add_conditional(plant.ro[0], plant.ro[1], then="on")

    set_demand(m, volume=20000.0)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- extension_constraint_objective ---


def extension_constraint_objective() -> pyo.ConcreteModel:
    """Extensions adding a tank's terminal level and the model's objective."""
    m = pyo.ConcreteModel(name="extensions")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.tank = fo.Tank(property_package=m.properties)
    m.site.pump = fo.Pump(property_package=m.properties, costing_package=m.costing)
    m.site.tank_to_pump = Arc(source=m.site.tank.outlet, destination=m.site.pump.inlet)
    m.costing.cost_process()
    terminal_level(m, tank="site.tank", fraction=0.5)
    minimize_power(m, units=["site.pump"])
    return m


# --- indexed_trains ---


def indexed_trains() -> pyo.ConcreteModel:
    """Indexed trains joined by directed, undirected and fan-in indexed arcs."""
    m = pyo.ConcreteModel(name="trains")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.p = fo.PlantBlock(time_block=m.time_block)
    p = m.p
    p.trains = pyo.Set(initialize=[0, 1, 2], ordered=True)
    p.train = fo.ConstantEnergyIntensityModel(
        p.trains,
        property_package=m.properties,
        energy_intensity=0.3 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    p.split = fo.Splitter(
        p.trains, property_package=m.properties, outlet_names=("a", "b")
    )
    p.out = fo.Product(p.trains, property_package=m.properties, inlet_names=("a",))
    p.mix = fo.Mixer(property_package=m.properties, inlet_names=("t0", "t1", "t2"))
    p.train_to_split = Arc(
        p.trains,
        rule=lambda b, i: {
            "source": b.train[i].outlet,
            "destination": b.split[i].inlet,
        },
    )
    p.split_to_out = Arc(
        p.trains, rule=lambda b, i: (b.split[i].outlet_a, b.out[i].inlet_a)
    )
    p.split_to_mix = Arc(
        p.trains,
        rule=lambda b, i: {
            "source": b.split[i].outlet_b,
            "destination": b.mix.find_component(f"inlet_t{i}"),
        },
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- logic_cross_unit ---


def logic_cross_unit() -> pyo.ConcreteModel:
    """Three units in a parallel group, plus a conditional that turns one off."""
    m = pyo.ConcreteModel(name="cross_unit")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    units = []
    for name in ("a", "b", "c"):
        unit = fo.ConstantEnergyIntensityModel(
            property_package=m.properties,
            energy_intensity=0.4 * pyunits.kWh / pyunits.m**3,
            costing_package=m.costing,
        )
        m.site.add_component(name, unit)
        unit = m.site.find_component(name)
        logic.add_status(unit, unit.power_electrical, 0 * pyunits.kW, 50 * pyunits.kW)
        units.append(unit)
    logic.register_parallel_group(units)
    logic.add_conditional(units[0], units[2], then="off")
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- logic_indexed ---


def logic_indexed() -> pyo.ConcreteModel:
    """Indexed RO trains, each member with the same status and startup/shutdown."""
    m = pyo.ConcreteModel(name="indexed_logic")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.p = fo.PlantBlock(time_block=m.time_block)
    m.p.trains = pyo.Set(initialize=[0, 1, 2], ordered=True)
    m.p.ro = fo.ConstantEnergyIntensityModel(
        m.p.trains,
        property_package=m.properties,
        energy_intensity=0.3 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    for train in m.p.ro.values():
        status = logic.add_status(
            train, train.power_electrical, 0 * pyunits.kW, 40 * pyunits.kW
        )
        logic.add_startup_shutdown(train, status, min_uptime=2)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- logic_status_startup ---


def logic_status_startup() -> pyo.ConcreteModel:
    """A pump with a status, startup/shutdown with min up/down times, and a dwell."""
    m = pyo.ConcreteModel(name="status_startup")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T08:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    pump = m.site.pump = fo.Pump(
        property_package=m.properties, costing_package=m.costing
    )
    status = logic.add_status(
        pump, pump.power_electrical, 1 * pyunits.kW, 20 * pyunits.kW
    )
    logic.add_startup_shutdown(pump, status, min_uptime=3, min_downtime=2)
    logic.add_dwell(pump.flow_in, 2)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- long_prices ---

N_POINTS = 3000


def long_prices() -> pyo.ConcreteModel:
    """One electricity price per time point over 3,000 points."""
    m = pyo.ConcreteModel(name="long_prices")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-21T20:00", time_step=10 * pyunits.min
    )
    price_units = currency_units("USD") / pyunits.kWh
    prices = [(0.08 + 0.04 * (t % 144 >= 96)) * price_units for t in range(N_POINTS)]
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block, energy_prices={"electrical": prices}
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.plant = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- network_two_plants ---


def network_two_plants() -> pyo.ConcreteModel:
    """A network of two plants joined by one inter-plant arc."""
    m = pyo.ConcreteModel(name="network")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.net = fo.NetworkBlock(time_block=m.time_block)
    m.net.a = fo.PlantBlock(time_block=m.time_block)
    m.net.b = fo.PlantBlock(time_block=m.time_block)
    m.net.a.tank = fo.Tank(property_package=m.properties)
    m.net.a.pump = fo.ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=0.4 * pyunits.kWh / pyunits.m**3,
        costing_package=m.costing,
    )
    m.net.b.tank = fo.Tank(property_package=m.properties)
    m.net.a.tank_to_pump = Arc(
        source=m.net.a.tank.outlet, destination=m.net.a.pump.inlet
    )
    m.net.a_to_b = Arc(
        source=m.net.a.pump.outlet,
        destination=m.net.b.tank.inlet,
        doc="Transfer main from plant a to plant b.",
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- pump_scheduling ---

BAR = 1e5
"""Pascals per bar."""


DEMAND = [40.0 + 20.0 * ((t % 24) in range(7, 20)) for t in range(24)]
"""Product demand per hour, m^3/hr."""


PUMP_SCHEDULING_PRICES = [0.08 + 0.22 * ((t % 24) in range(16, 21)) for t in range(24)]
"""Electricity price per hour, USD/kWh."""


@fo.extension
def feed_pump_status(model, *, rated_flow: float, min_fraction: float):
    """Let the feed pump be off, or run at min_fraction to 1 of rated_flow (m^3/hr)."""
    pump = model.feed_pump
    pump.flow_in = pyo.Reference(pump.inlet_state.flow_vol_phase[:, "Liq"])
    flow = pyunits.m**3 / pyunits.hr
    logic.add_status(
        pump, pump.flow_in, min_fraction * rated_flow * flow, rated_flow * flow
    )


@fo.extension
def terminal_tank_volume(model, *, volume: float):
    """End the horizon with at least ``volume`` m^3 in the tank."""
    last = model.time_block.time_index.last()
    model.tank_terminal = pyo.Constraint(expr=model.tank.volume[last] >= volume)


@fo.extension
def terminal_battery_charge(model):
    """End the horizon at least as charged as the battery started."""
    last = model.time_block.time_index.last()
    model.battery_terminal = pyo.Constraint(
        expr=model.battery.charge[last] >= model.battery.charge_init
    )


def pump_scheduling() -> pyo.ConcreteModel:
    """A trimmed copy of the flex-pse-examples pump_scheduling model."""
    m = pyo.ConcreteModel(name="pump_scheduling")
    m.time_block = fo.TimeBlock(
        start_date="2026-07-01", end_date="2026-07-02", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow(
        has_pressure=True, density=1000 * pyunits.kg / pyunits.m**3
    )
    usd_per_kwh = currency_units("USD") / pyunits.kWh
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": [p * usd_per_kwh for p in PUMP_SCHEDULING_PRICES]},
    )
    m.feed_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.75,
        costing_package=m.costing,
    )
    m.tank = fo.Tank(
        property_package=m.properties,
        max_volume=600 * pyunits.m**3,
        initial_volume=300 * pyunits.m**3,
        level_min=0.1,
        level_max=0.95,
    )
    m.product_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.7,
        costing_package=m.costing,
    )
    m.feed_to_tank = Arc(source=m.feed_pump.outlet, destination=m.tank.inlet)
    m.tank_to_product = Arc(source=m.tank.outlet, destination=m.product_pump.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    for t in m.time_block.time_index:
        m.feed_pump.delta_pressure[t].fix(2.0 * BAR)
        m.product_pump.delta_pressure[t].fix(3.0 * BAR)
        m.feed_pump.inlet_state.pressure[t].fix(1.0 * BAR)
        m.product_pump.outlet_state.flow_vol_phase[t, "Liq"].fix(DEMAND[t])

    feed_pump_status(m, rated_flow=120.0, min_fraction=0.4)
    # t=0 is the initial-state snapshot; water pumped there never reaches the tank.
    m.feed_pump.flow_in[0].fix(0.0)

    rate = 30 * pyunits.kW
    m.battery = fo.BatteryModel(
        capacity=rate * 4 * pyunits.hr,
        power_charge_max=rate,
        power_discharge_max=rate,
        eta_charge=0.95,
        eta_discharge=0.95,
        soc_min=0.1,
        soc_max=0.9,
        initial_soc=0.5,
        costing_package=m.costing,
    )
    terminal_battery_charge(m)
    terminal_tank_volume(m, volume=300.0)

    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- ramp_and_bypass ---


def ramp_and_bypass() -> pyo.ConcreteModel:
    """A two-step ramp limit with units, and a bypass around a pump."""
    m = pyo.ConcreteModel(name="ramp_bypass")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    pump = m.site.pump = fo.Pump(
        property_package=m.properties, costing_package=m.costing
    )
    logic.add_ramp_rate(
        pump.power_electrical,
        ramp_up=5.0,
        ramp_down=3.0,
        ramp_rate_units=pyunits.kW / pyunits.hr,
        window=2,
    )
    logic.add_bypass(pump, pump.flow_in, 2 * pyunits.m**3 / pyunits.hr)
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- surrogate_and_dispatch ---

SURROGATE = {
    "kind": "surrogate",
    "unit": "site.pump",
    "relation": "power_electrical_relation",
    "surrogate_type": "multilinear",
    "data": {
        "input_variables": {"flow_out": "m^3/hr"},
        "output_variables": {"power_electrical": "kW"},
        "coefficients": {"flow_out": 0.8, "intercept": 1.5},
    },
    "provenance": {"source": "datasheet"},
}


def surrogate_and_dispatch() -> pyo.ConcreteModel:
    """A Pump with a multilinear surrogate and an externally dispatched Battery."""
    m = pyo.ConcreteModel(name="surrogate_and_dispatch")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T06:00", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.pump = fo.Pump(property_package=m.properties, costing_package=m.costing)
    m.site.battery = fo.BatteryModel(
        capacity=10 * pyunits.kWh, costing_package=m.costing
    )
    fo.apply_spec(m, [SURROGATE])
    charge = [0.0, 1.0, 2.0, 2.0, 0.5, 0.0]
    m.site.battery.set_external_dispatch(
        m.site.battery.power_charge, dict(enumerate(charge))
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- two_property_packages ---


def two_property_packages() -> pyo.ConcreteModel:
    """An aqueous Pump and a gas Combustor, each on its own property package."""
    m = pyo.ConcreteModel(name="two_packages")
    m.time_block = fo.TimeBlock(
        start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
    )
    m.water = fo.SimpleAqueousFlow(has_pressure=True)
    m.gas = fo.SimpleGasFlow()
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
    )
    m.site = fo.PlantBlock(time_block=m.time_block)
    m.site.pump = fo.Pump(property_package=m.water, costing_package=m.costing)
    m.site.combustor = fo.Combustor(property_package=m.gas, inlet_names=("fuel",))
    m.costing.cost_process()
    m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
    return m


# --- units_on_model ---


def units_on_model() -> pyo.ConcreteModel:
    """Units directly on the model with two arcs, the pump-scheduling layout."""
    m = pyo.ConcreteModel(name="pump_scheduling")
    m.time_block = fo.TimeBlock(
        start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
    )
    m.properties = fo.SimpleAqueousFlow(
        has_pressure=True, density=998.0 * pyunits.kg / pyunits.m**3
    )
    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        energy_prices={"electrical": 0.2 * currency_units("USD") / pyunits.kWh},
    )
    m.feed_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.7,
        costing_package=m.costing,
    )
    m.tank = fo.Tank(
        property_package=m.properties,
        max_volume=500.0 * pyunits.m**3,
        initial_volume=250.0 * pyunits.m**3,
        level_min=0.1,
        level_max=0.95,
    )
    m.product_pump = fo.Pump(
        property_package=m.properties,
        power_relation="hydraulic",
        efficiency=0.75,
        costing_package=m.costing,
    )
    m.feed_to_tank = Arc(source=m.feed_pump.outlet, destination=m.tank.inlet)
    m.tank_to_product = Arc(source=m.tank.outlet, destination=m.product_pump.inlet)
    m.battery = fo.BatteryModel(
        capacity=40.0 * pyunits.kWh,
        power_charge_max=10.0 * pyunits.kW,
        costing_package=m.costing,
    )
    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )
    return m


CASES = {
    "api_freeze_like": (api_freeze_like, True, "unit"),
    "battery_status": (battery_status, False, "unit"),
    "datacenter": (datacenter, False, "unit"),
    "desalination": (desalination, True, "component"),
    "extension_constraint_objective": (extension_constraint_objective, True, "unit"),
    "indexed_trains": (indexed_trains, True, "unit"),
    "logic_cross_unit": (logic_cross_unit, False, "unit"),
    "logic_indexed": (logic_indexed, False, "unit"),
    "logic_status_startup": (logic_status_startup, False, "unit"),
    "long_prices": (long_prices, False, "unit"),
    "network_two_plants": (network_two_plants, False, "unit"),
    "pump_scheduling": (pump_scheduling, True, "unit"),
    "ramp_and_bypass": (ramp_and_bypass, False, "unit"),
    "surrogate_and_dispatch": (surrogate_and_dispatch, False, "unit"),
    "two_property_packages": (two_property_packages, False, "unit"),
    "units_on_model": (units_on_model, True, "unit"),
}
"""Case name -> (build function, whether to expand arcs before emitting, test tier)."""
