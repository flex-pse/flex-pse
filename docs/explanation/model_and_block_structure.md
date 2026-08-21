# Model and block structure (decisions R1, R2)

This page defines, explicitly, how a flex-pse model is assembled from Pyomo/IDAES
blocks and how variables on those blocks are indexed. Two questions recur when
reading or writing a unit model, and they have different answers:

- **Time indexing** — which variables vary over the horizon, and against what set.
- **Property (state) indexing** — where a stream's `flow_vol_phase`, `pressure`,
  and `temperature` live, and how they carry a time dimension.

## The block hierarchy

A flex-pse model is a plain Pyomo `ConcreteModel` (or a `dynamic=False` IDAES
flowsheet — **never** a `dynamic=True`/Pyomo.DAE flowsheet, decision R2). Onto it
are placed a small, fixed set of block kinds:

```
ConcreteModel
├── TimeBlock                      the discrete-time substrate (exactly one)
├── <PropertyPackage>              a parameter block (SimpleAqueousFlow, SimpleGasFlow, …)
└── <unit>  : OpsBlock             one per process unit
    ├── power_electrical[t]        base power Var(s), kW, time-indexed
    ├── power_thermal[t]
    ├── <state blocks>             built from the property package, one scalar block per stream
    ├── <balance constraints>      1–3, hand-written (no ControlVolumes)
    └── _io_registry               what this unit exposes (IORegistry)
```

### TimeBlock — the one ordered set everything indexes against

{py:class}`~flexops.core.time_block.TimeBlock` builds the ordered integer set
`time_index` = `0..N-1`, the elapsed-time Param `time` (`i*dt`), and the
unit-carrying step `dt`. It is the **only** set flex-pse variables index over
for the time dimension. Time is discrete on purpose (unit-commitment index
arithmetic — dwell times, startup delays, rolling-horizon windows); see
{doc}`time_and_dynamics` for the full rationale. There is exactly one TimeBlock
per model; a unit finds it by searching the model (the interim
`_find_time_block` helper) until the `flowsheet()` chain arrives in a later
milestone.

### OpsBlock — the base of every unit model

{py:class}`~flexops.core.ops_block.OpsBlockData` inherits IDAES
`UnitModelBlockData` for its ConfigBlock, Port, and costing-registration
machinery, but uses **no ControlVolumes** (decision R1): each subclass
hand-writes its one-to-three balance constraints as difference equations against
`time_block.dt`. The base provides:

- the power Vars via {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`;
- the registration API
  ({meth}`~flexops.core.ops_block.OpsBlockData.register_io_variable`,
  {meth}`~flexops.core.ops_block.OpsBlockData.register_process_parameter`,
  {meth}`~flexops.core.ops_block.OpsBlockData.register_power`);
- the in-place update hook
  {meth}`~flexops.core.ops_block.OpsBlockData.update_parameters` — flex-pse
  **never deletes** a built component, because anything else holding a reference
  to it (an aggregated-power constraint, an expanded arc) would silently keep the
  stale one. Mutate Params in place or `deactivate()` constraints instead.
- the stream-port helper
  {meth}`~flexops.core.ops_block.OpsBlockData.add_stream_ports` — builds one
  `{port}_state` block per requested port from the configured `property_package`
  (default one `inlet` and one `outlet`), registers the caller-chosen `io_vars`
  on each (the actual state-block `Var`, not a `Reference`) as input/output IO
  variables — the default `flow_vol_phase`, since which variables are the
  meaningful IO is property-package dependent — and exposes the ports via the
  inherited IDAES `add_inlet_port`/`add_outlet_port` helpers.

### Property packages — parameter block plus state blocks

A property package is an IDAES `PhysicalParameterBlock` (the *parameter block*:
phases, components, supported-property metadata, default units) paired with a
`StateBlockData` subclass (the *state block*: the actual state Vars). flex-pse
ships two, both structurally modeled on WaterTAP's zero-order package:

- {py:class}`~flexops.properties.simple_aqueous.SimpleAqueousFlow` — flow-only by
  default: `flow_vol_phase` (indexed by the single `Liq` phase and time), with
  **opt-in** `pressure`/`temperature`.
- {py:class}`~flexops.properties.simple_gas.SimpleGasFlow` — the gas counterpart,
  which **always** carries all three state variables (`flow_vol_phase` over the
  single `Vap` phase and time, plus `pressure`/`temperature`) because a gas
  stream's conditions always matter (no equation of state is
  imposed; a unit adds any relation it needs as its own constraint). Its first
  consumer is {py:class}`~flexops.unit_models.powergeneration.combustor.Combustor`, so the
  extensive/intensive table below is live rather than hypothetical.

Ports built from state blocks carry a stream between units via standard
IDAES/Pyomo `Arc`s, honoring the extensive/intensive split described below.

## How variable indexing works

Variables fall into two families, indexed two different ways. Getting the family
right is the single most important structural decision when writing a unit.

### Operational variables are indexed over the time set

Everything that represents an operating decision or quantity over the horizon is
a Var indexed directly over `time_block.time_index`. This includes both
base-provided and unit-declared variables:

- **Power draw.** {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`
  creates `power_electrical[t]` (or `power_thermal[t]`) as
  `Var(time_block.time_index, units=kW)`. Both are powers in kW; see
  {doc}`energy_nomenclature`.
- **Process IO variables.** A unit's controllable inputs and reported outputs
  (flows, setpoints, on/off status). These are usually time-indexed, but not
  required to be — a fixed geometry or a design decision that does not vary over
  the horizon is a scalar Var. The
  {py:class}`~flexops.core.registration.IOVariableRecord` records this in its
  `time_indexed` flag so FlexParameterize and the docs generator know which axis
  a registered variable spans.

Because `time_index` is a plain integer `Set`, difference equations are written
with ordinary index arithmetic — `V[t+1] == V[t] + dt*(inflow[t] - outflow[t])`
— which is exactly why the time set is discrete.

### Property state variables carry the time index themselves

The state variables `flow_vol_phase`, `pressure`, and `temperature`
are declared inside `StateBlockData.build`, **indexed over the time set
directly**: the extensive, per-phase `flow_vol_phase` is `Var(time, phase_list)`
(time first, phase second), while the intensive stream properties
`pressure` and `temperature` drop the phase index and are `Var(time)` (assumed
equal across phases in a stream). A single state block therefore describes a
stream over the whole horizon.

The time set is delivered to the state block through its `time_index` config
option: a property package's `build_state_block(time_index=<time Set>)` returns a
single **scalar** block whose variables span that set. A unit's
{meth}`~flexops.core.ops_block.OpsBlockData.add_stream_ports` does exactly this
for its inlet/outlet streams, then wires the flows and ports:

```python
# inside a unit's build():
self.add_stream_ports()             # builds scalar inlet_state, outlet_state + ports
# then, per time point t:
self.inlet_state.flow_vol_phase[t, "Liq"]   # volumetric flow at time t
self.inlet_state.pressure[t]                # pressure at time t (when enabled)
```

So indexing is **single-layer**: there is one state block per stream, and time
is the leading index on the state variables inside it (`flow_vol_phase`
additionally carries a trailing phase index). This is why the tests build a
three-point stream with `build_state_block(time_index=m.time)` and then read
`state.flow_vol_phase[0, "Liq"]`, `state.flow_vol_phase[1, "Liq"]`, and so on.

### Extensive vs. intensive: how state variables cross an arc

Within a state block the state variables split by how they combine at a
connection (relevant when ports/arcs are built in a later milestone):

| State variable | Kind | Across an arc | Port rule |
|---|---|---|---|
| `flow_vol_phase` | **extensive** | conserved (sum of flows balances at a node) | `Port.Extensive` |
| `pressure`, `temperature` | **intensive** | equal on both sides | `Port.Equality` |

This is why volumetric flow is the quantity a mixer sums and a splitter divides,
while pressure/temperature are simply equated across a connection — exactly the
split {py:class}`~flexops.unit_models.mixer.Mixer` and
{py:class}`~flexops.unit_models.splitter.Splitter` are built around. Note the
`Port rule` column is the modeling *intent*: the state blocks build every port
member as `Port.Equality`, so one port carries one arc. A junction therefore
gives each stream its own port rather than apportioning one port across several
arcs.

## Where each thing is declared — a summary

| Quantity | Block it lives on | Indexed over | Declared by |
|---|---|---|---|
| `time_index`, `time`, `dt` | `TimeBlock` | — / `time_index` | {py:class}`~flexops.core.time_block.TimeBlock` |
| `power_electrical[t]`, `power_thermal[t]` | the unit (`OpsBlock`) | `time_index` | {meth}`~flexops.core.ops_block.OpsBlockData.declare_power` |
| process IO variables | the unit (`OpsBlock`) | `time_index` (usually) or scalar | the unit's `build`, then `register_io_variable` |
| design/regression parameters | the unit (`OpsBlock`) | usually scalar | the unit's `build`, then `register_process_parameter` |
| `flow_vol_phase`, `pressure`, `temperature` | a scalar state block (one per stream) | time (on the variable); `flow_vol_phase` also phase-indexed | the property package's `StateBlockData.build` |

Discovery ties it together: every unit exposes what it declared through its
{py:class}`~flexops.core.registration.IORegistry`, and
{func}`~flexops.core.registration.iter_io_registry` walks the whole model to find
every block that registered something — the model-wide view FlexParameterize and
the docs generator consume.
