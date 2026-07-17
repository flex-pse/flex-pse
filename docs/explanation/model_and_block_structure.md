# Model and block structure (decisions R1, R2)

This page defines, explicitly, how a flex-pse model is assembled from Pyomo/IDAES
blocks and how variables on those blocks are indexed. Two questions recur when
reading or writing a unit model, and they have different answers:

- **Time indexing** — which variables vary over the horizon, and against what set.
- **Property (state) indexing** — where a stream's `flow_vol`, `dens_mass`,
  `pressure`, and `temperature` live, and how they pick up a time dimension.

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
    ├── <state blocks>             built from the property package, indexed over time
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

### Property packages — parameter block plus state blocks

A property package is an IDAES `PhysicalParameterBlock` (the *parameter block*:
phases, components, supported-property metadata, default units) paired with a
`StateBlockData` subclass (the *state block*: the actual state Vars). flex-pse
ships two, both structurally modeled on WaterTAP's zero-order package:

- {py:class}`~flexops.properties.simple_aqueous.SimpleAqueousFlow` — flow-only by
  default: `flow_vol` and `dens_mass` (density fixed at the configured value
  unless `fixed_density=False`), with **opt-in** `pressure`/`temperature`.
- {py:class}`~flexops.properties.simple_gas.SimpleGasFlow` — the gas counterpart,
  which **always** carries all four state variables because gas density varies
  with pressure and temperature (no equation of state is imposed; a unit adds any
  relation it needs as its own constraint).

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

### Property state variables live on state blocks, indexed over time by the unit

The state variables `flow_vol`, `dens_mass`, `pressure`, and `temperature` are
declared as **scalar** Vars inside `StateBlockData.build`. They are *not*
time-indexed at the point of declaration. A state block describes the fluid at a
**single** point in state-space.

Time variation comes from **indexing the state block itself** over the time set.
A property package's `build_state_block(index_set)` returns an *indexed* block —
one state-block member per index — so a unit builds its inlet/outlet states as:

```python
# inside a unit's build(), with tb = self._find_time_block()
self.properties_in = self.config.property_package.build_state_block(tb.time_index)
# then, per time point t:
self.properties_in[t].flow_vol      # volumetric flow at time t
self.properties_in[t].dens_mass     # density at time t
```

So indexing is **two-layer**: the *outer* index is time (carried by the indexed
state block), and the *inner* object is the scalar-per-point state Var. This is
why the tests build a three-point stream with `build_state_block([0, 1, 2])` and
then read `state[0].flow_vol`, `state[1].flow_vol`, and so on.

### Extensive vs. intensive: how state variables cross an arc

Within a state block the state variables split by how they combine at a
connection (relevant when ports/arcs are built in a later milestone):

| State variable | Kind | Across an arc | Port rule |
|---|---|---|---|
| `flow_vol` | **extensive** | conserved (sum of flows balances at a node) | `Port.Extensive` |
| `dens_mass`, `pressure`, `temperature` | **intensive** | equal on both sides | `Port.Equality` |

This is why volumetric flow is the quantity a mixer sums and a splitter divides,
while density/pressure/temperature are simply equated across a connection.

## Where each thing is declared — a summary

| Quantity | Block it lives on | Indexed over | Declared by |
|---|---|---|---|
| `time_index`, `time`, `dt` | `TimeBlock` | — / `time_index` | {py:class}`~flexops.core.time_block.TimeBlock` |
| `power_electrical[t]`, `power_thermal[t]` | the unit (`OpsBlock`) | `time_index` | {meth}`~flexops.core.ops_block.OpsBlockData.declare_power` |
| process IO variables | the unit (`OpsBlock`) | `time_index` (usually) or scalar | the unit's `build`, then `register_io_variable` |
| design/regression parameters | the unit (`OpsBlock`) | usually scalar | the unit's `build`, then `register_process_parameter` |
| `flow_vol`, `dens_mass`, `pressure`, `temperature` | an indexed state block | time (via the state block); scalar per member | the property package's `StateBlockData.build` |

Discovery ties it together: every unit exposes what it declared through its
{py:class}`~flexops.core.registration.IORegistry`, and
{func}`~flexops.core.registration.iter_io_registry` walks the whole model to find
every block that registered something — the model-wide view FlexParameterize and
the docs generator consume.
