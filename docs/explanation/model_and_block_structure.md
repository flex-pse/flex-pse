# Model and block structure

This page explains how a flex-pse model gets assembled from Pyomo/IDAES
blocks, and how variables on those blocks get indexed. Two questions come up
again and again when you read or write a unit model, and they have
different answers.

- **Time indexing.** Which variables vary over the horizon, and against what
  set?
- **Property (state) indexing.** Where does a stream's `flow_vol_phase`,
  `pressure`, and `temperature` live, and how do they carry a time
  dimension?

## The block hierarchy

A flex-pse model is a plain Pyomo `ConcreteModel` (or a `dynamic=False` IDAES
flowsheet, **never** a `dynamic=True`/Pyomo.DAE flowsheet, see
{doc}`time_and_dynamics` for why). On it sit a small, fixed set of block
kinds.

```
ConcreteModel
├── TimeBlock                      the discrete-time substrate (exactly one)
├── <PropertyPackage>              a parameter block (SimpleAqueousFlow, SimpleGasFlow, …)
└── <unit>  : OpsBlock             one per process unit
    ├── power_electrical[t]        base power Var(s), kW, indexed by time
    ├── power_thermal[t]
    ├── <state blocks>             built from the property package, one scalar block per stream
    ├── <balance constraints>      1 to 3, hand-written (no ControlVolumes)
    └── _io_registry               what this unit exposes (IORegistry)
```

### TimeBlock, the one ordered set everything indexes against

{py:class}`~flexops.core.time_block.TimeBlock` builds the ordered integer set
`time_index` equal to `0..N-1`, the elapsed time Param `time` (`i*dt`), and
the step `dt`, which carries units. It's the **only** set flex-pse variables
index over for the time dimension. Time is discrete on purpose, for unit
commitment index arithmetic like dwell times, startup delays, and rolling
horizon windows. See {doc}`time_and_dynamics` for the full rationale. There
is exactly one TimeBlock per model. A unit finds it by searching the model
(the interim `_find_time_block` helper), a stopgap until the `flowsheet()`
chain arrives.

### OpsBlock, the base of every unit model

{py:class}`~flexops.core.ops_block.OpsBlockData` inherits IDAES
`UnitModelBlockData` for its ConfigBlock, Port, and costing registration
machinery, but it uses **no ControlVolumes**. Each subclass hand writes its
one to three balance constraints as difference equations against
`time_block.dt`. The base provides the following.

- The power Vars, via {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`.
- The registration API
  ({meth}`~flexops.core.ops_block.OpsBlockData.register_io_variable`,
  {meth}`~flexops.core.ops_block.OpsBlockData.register_process_parameter`,
  {meth}`~flexops.core.ops_block.OpsBlockData.register_power`).
- The in place update hook
  {meth}`~flexops.core.ops_block.OpsBlockData.update_parameters`. flex-pse
  **never deletes** a built component. Anything else holding a reference to
  it (an aggregated power constraint, an expanded arc) would otherwise keep
  the stale one silently. Mutate Params in place, or `deactivate()`
  constraints instead.
- The stream port helper
  {meth}`~flexops.core.ops_block.OpsBlockData.add_stream_ports`. It builds
  one `{port}_state` block per requested port from the configured
  `property_package` (default one `inlet` and one `outlet`), registers the
  caller chosen `io_vars` on each (the actual state block `Var`, not a
  `Reference`) as input or output IO variables (the default is
  `flow_vol_phase`, since which variables count as meaningful IO depends on
  the property package), and exposes the ports through the inherited IDAES
  `add_inlet_port`/`add_outlet_port` helpers.

### Property packages, a parameter block plus state blocks

A property package is an IDAES `PhysicalParameterBlock` (the *parameter
block*: phases, components, supported property metadata, default units)
paired with a `StateBlockData` subclass (the *state block*: the actual state
Vars). flex-pse ships two, both modeled structurally on WaterTAP's zero
order package.

- {py:class}`~flexops.properties.simple_aqueous.SimpleAqueousFlow` is
  flow only by default. It carries `flow_vol_phase` (indexed by the single
  `Liq` phase and time), with **opt-in** `pressure`/`temperature`.
- {py:class}`~flexops.properties.simple_gas.SimpleGasFlow` is the gas
  counterpart. It **always** carries all three state variables
  (`flow_vol_phase` over the single `Vap` phase and time, plus
  `pressure`/`temperature`). A gas stream's conditions always matter. It
  imposes no equation of state. A unit adds any relation it
  needs as its own constraint. Its first consumer is
  {py:class}`~flexops.unit_models.powergeneration.combustor.Combustor`, N
  fuel sources (inlet ports and/or utility fuel names) burned into one flue
  gas outlet, exporting power. So the extensive/intensive table below
  describes something real, not a hypothetical.

Ports built from state blocks carry a stream between units over standard
IDAES/Pyomo `Arc`s, honoring the extensive/intensive split described below.

## How variable indexing works

Variables fall into two families, indexed two different ways. Getting the
family right is the single most important structural decision when you
write a unit.

### Operational variables get indexed over the time set

Everything that represents an operating decision or quantity over the
horizon is a Var indexed directly over `time_block.time_index`. This
includes both base provided and unit declared variables.

- **Power draw.** {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`
  creates `power_electrical[t]` (or `power_thermal[t]`) as
  `Var(time_block.time_index, units=kW)`. Both are powers in kW. See
  {doc}`energy_nomenclature`.
- **Process IO variables.** These are a unit's controllable inputs and
  reported outputs (flows, setpoints, on/off status). Most are indexed by
  time, but that's not required. A fixed geometry or a design decision that
  doesn't vary over the horizon is a scalar Var instead. The
  {py:class}`~flexops.core.registration.IOVariableRecord` records this in
  its `time_indexed` flag, so FlexParameterize and the docs generator know
  which axis a registered variable spans.

`time_index` is a plain integer `Set`, so difference equations get written
with ordinary index arithmetic, `V[t+1] == V[t] + dt*(inflow[t] - outflow[t])`.
That's exactly why the time set is discrete.

### Property state variables carry the time index themselves

The state variables `flow_vol_phase`, `pressure`, and `temperature` get
declared inside `StateBlockData.build`, **indexed over the time set
directly**. The extensive, per phase `flow_vol_phase` is
`Var(time, phase_list)` (time first, phase second), while the intensive
stream properties `pressure` and `temperature` drop the phase index and
become `Var(time)` (assumed equal across phases in a stream). So a single
state block describes a stream over the whole horizon.

The time set reaches the state block through its `time_index` config
option. A property package's `build_state_block(time_index=<time Set>)`
returns a single **scalar** block whose variables span that set. A unit's
{meth}`~flexops.core.ops_block.OpsBlockData.add_stream_ports` does exactly
this for its inlet and outlet streams, then wires the flows and ports.

```python
# inside a unit's build():
self.add_stream_ports()             # builds scalar inlet_state, outlet_state + ports
# then, per time point t:
self.inlet_state.flow_vol_phase[t, "Liq"]   # volumetric flow at time t
self.inlet_state.pressure[t]                # pressure at time t (when enabled)
```

So indexing stays **single layer**. There's one state block per stream, and
time is the leading index on the state variables inside it (`flow_vol_phase`
also carries a trailing phase index). That's why the tests build a three
point stream with `build_state_block(time_index=m.time)` and then read
`state.flow_vol_phase[0, "Liq"]`, `state.flow_vol_phase[1, "Liq"]`, and so
on.

### Extensive vs. intensive, how state variables cross an arc

Within a state block, the state variables split by how they combine at a
connection.

| State variable | Kind | Across an arc | Port rule |
|---|---|---|---|
| `flow_vol_phase` | **extensive** | conserved (sum of flows balances at a node) | `Port.Extensive` |
| `pressure`, `temperature` | **intensive** | equal on both sides | `Port.Equality` |

This is why volumetric flow is the quantity a mixer sums and a splitter
divides, while pressure and temperature simply get equated across a
connection. That's exactly the split
{py:class}`~flexops.unit_models.mixer.Mixer` and
{py:class}`~flexops.unit_models.splitter.Splitter` build around. Note that
the `Port rule` column describes the modeling *intent*. The state blocks
build every port member as `Port.Equality`, so one port carries one arc. A
junction gives each stream its own port instead of splitting one port
across several arcs.

## Where each thing is declared, a summary

| Quantity | Block it lives on | Indexed over | Declared by |
|---|---|---|---|
| `time_index`, `time`, `dt` | `TimeBlock` | none / `time_index` | {py:class}`~flexops.core.time_block.TimeBlock` |
| `power_electrical[t]`, `power_thermal[t]` | the unit (`OpsBlock`) | `time_index` | {meth}`~flexops.core.ops_block.OpsBlockData.declare_power` |
| process IO variables | the unit (`OpsBlock`) | `time_index` (usually) or scalar | the unit's `build`, then `register_io_variable` |
| design/regression parameters | the unit (`OpsBlock`) | usually scalar | the unit's `build`, then `register_process_parameter` |
| `flow_vol_phase`, `pressure`, `temperature` | a scalar state block (one per stream) | time on the variable, and `flow_vol_phase` also indexed by phase | the property package's `StateBlockData.build` |

Discovery ties it together. Every unit exposes what it declared through its
{py:class}`~flexops.core.registration.IORegistry`, and
{func}`~flexops.core.registration.iter_io_registry` walks the whole model to
find every block that registered something. That gives FlexParameterize and
the docs generator a model wide view.
