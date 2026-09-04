flexops.unit_models
====================

Unit models are organized by inlet/outlet topology first (the ``base``
sub-package), then specialized physically (architecture §3.4, R6). The topology
base owns port construction and the per-stream mass balance; a physical subclass
adds the flow-to-energy relationship and any bounds. ``Pump`` and ``Tank``
subclass the single-inlet/single-outlet ``SISOBlock``; a ``Tank`` additionally
disables the on/off logic layer, since a tank has no unit-commitment status.

Every unit defaults to a **constant energy intensity** and builds it as the
Constraint ``power_electrical_relation`` (``power_thermal_relation`` for a heat
duty). That name is the swap contract: FlexParameterize upgrades a unit's
relationship by deactivating exactly that Constraint and attaching a fitted
replacement, reusing the same registered IO variables — so there is no separate
regression unit class (R11).

**Picking a base when adding a new unit model** (architecture §3.4): ask
whether every port shares one ``property_package``.

- If so, subclass the topology base matching the port count
  (``SISOBlock``/``SIDOBlock``/``DIDOBlock``). It already owns port
  construction and the per-stream mass balance; the new class only renames the
  topology's generic roles into its own nomenclature and adds the
  flow-to-energy relationship in ``build()``. To rename, pass a complete
  ``naming_dict`` up to the base ``build()`` — spread the base's
  ``_component_names`` and override the subset being renamed, as
  :class:`~flexops.unit_models.reverseosmosis.ReverseOsmosis` does — and the
  base registers it before building anything named. Ports are never renamed.
- If the unit needs more than one property package (e.g. a fuel stream and an
  air stream on different flow bases), subclass
  :class:`~flexops.core.ops_block.OpsBlockData` directly instead — declare one
  named property-package config slot per stream family and hand-write the
  ports and balance across them, rather than half-fitting a topology base
  built around a single shared package.
- If the unit's **port count is itself a config option** (e.g. an arbitrary
  number of gas inlets), no fixed-arity topology base fits either — subclass
  :class:`~flexops.core.ops_block.OpsBlockData` directly and hand-write the
  ports and balance, as :class:`~flexops.unit_models.powergeneration.combustor.Combustor`,
  :class:`~flexops.unit_models.mixer.Mixer`, and
  :class:`~flexops.unit_models.splitter.Splitter` do. Give each stream its own
  port: port members are equalities, so one port carries one arc — N streams
  means N ports, never one port fed by N arcs.

Topology bases
--------------

.. currentmodule:: flexops.unit_models.base.siso

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   SISOBlock

.. currentmodule:: flexops.unit_models.base.sido

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   SIDOBlock

.. currentmodule:: flexops.unit_models.base.dido

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   DIDOBlock

Physical units
--------------

.. currentmodule:: flexops.unit_models.pump

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Pump

.. currentmodule:: flexops.unit_models.storage.tank

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Tank

.. currentmodule:: flexops.unit_models.storage.battery

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   BatteryModel

.. currentmodule:: flexops.unit_models.exchanger

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Exchanger

.. currentmodule:: flexops.unit_models.reverseosmosis

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   ReverseOsmosis

.. currentmodule:: flexops.unit_models.powergeneration.combustor

An arbitrary number of fuel sources mixed into one flue-gas outlet, exporting
electrical power (``power_electrical`` upper-bounded at 0, the
battery-discharge sign convention) under a heating-value or a
constant-intensity relation, resolved automatically from whether every fuel
source was given a heating value. Fuel sources may be connected via inlet
ports (``inlet_names``), pulled from utilities (``utility_fuel_source``),
or both. The relation determines ``power_generated`` — the
non-negative generation magnitude, and the unit's registered IO output — which
the separate, deliberately unregistered ``power_electrical_sign`` constraint
negates into the draw, so the export convention holds under any surrogate
swap.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Combustor

.. currentmodule:: flexops.unit_models.mixer

An arbitrary number of named inlet streams joined into one outlet at constant
density, so volume is conserved directly and no energy is involved. Pressure
is held equal across the inlets and passed through to the outlet; the outlet
temperature is either equal to the inlets' (the linear default) or their
volume-weighted blend, per ``temperature_mixing``.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Mixer

.. currentmodule:: flexops.unit_models.splitter

One inlet stream divided among an arbitrary number of named outlets at
constant density. Conservation is its only flow constraint, so the routing
stays a decision variable — :math:`(N-1)` degrees of freedom per time point
that the enclosing model's objective resolves. Use ``SIDOBlock`` instead when
the split is meant to be prescribed rather than optimized.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Splitter

.. currentmodule:: flexops.unit_models.wastewater.digestor

An anaerobic digester accepting an arbitrary number of feed streams, each
carrying its own property package, and converting them into one biogas outlet
and an optional treated-sludge outlet. Biogas volume is driven by a
fixed-fraction correlation of the total inlet volume, replaceable by a linear
surrogate.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Digestor

Boundary blocks
---------------

.. currentmodule:: flexops.unit_models.feed

A source: zero inlets, an arbitrary number of named outlets. It owns a boundary
stream — the reference outlet's non-flow states are the boundary conditions,
and every other outlet is held at them — meters the total ``withdrawal[t]``
crossing the boundary, optionally bounds it with mutable limit Params, and
optionally prices it into opex. Its ``resource_name`` is the key the enclosing
plant aggregates it under in ``total_feed[resource, t]``, independent of the
block's own Pyomo name.

``withdrawal_basis`` (and ``demand_basis`` on a ``Product``) chooses what the
configured limits mean, because a rate ceiling and a horizon allowance are
different constraints rather than two spellings of one. On the default
``"period"`` basis the limit is a **rate** holding in every period
(``m**3/hr``), built as a Param over the time index. On the ``"horizon"`` basis
it is a **quantity** over the whole horizon (``m**3``) — a monthly abstraction
permit, a take-or-pay contract — bounding a scalar ``withdrawal_total`` defined
by ``eq_withdrawal_total`` as :math:`\sum_t withdrawal[t] \cdot dt`, which
leaves the optimizer free to shape the profile that reaches it. The metered
``withdrawal[t]`` itself stays time-indexed either way, so costing, plant
aggregation and ``set_external_dispatch`` are unaffected. A limit whose units
contradict the declared basis raises ``FlexConfigError`` naming the config key.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Feed

.. currentmodule:: flexops.unit_models.product

The mirror sink: an arbitrary number of named inlets, zero outlets, aggregating
what arrives into ``delivery[t]`` and aggregating into
``total_product[resource, t]``. It deliberately does **not** blend — each
inlet's intensive states arrive from its own arc and are left independent, so
put a ``Mixer`` upstream when a single blended stream is wanted — and its price
follows the ``register_scalar_cost`` sign convention: positive is a cost (brine
disposal), negative is revenue (potable water sold).

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Product

Generic surrogate
-----------------

.. currentmodule:: flexops.unit_models.constant_intensity

The default building block for anything without a bespoke physical topology —
a whole plant modeled as one surrogate, as in the frozen API script.

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   ConstantEnergyIntensityModel
