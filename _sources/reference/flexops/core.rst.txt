flexops.core
============

.. currentmodule:: flexops.core.time_block

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeBlock
   TimeBlockData
   TimeWindow

OpsBlock
--------

.. currentmodule:: flexops.core.ops_block

The base class of every flex-pse unit model (see
:doc:`../../explanation/relaxation_policies` for the relaxation config slot).
flex-pse never deletes model components; a built model is updated in place via
:meth:`OpsBlockData.update_parameters`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OpsBlock
   OpsBlockData
   RelaxationPolicy

A unit's own relationships are swapped in place from a
:class:`~flexops.surrogates.base.Surrogate` via :meth:`OpsBlockData.swap_relation`
— not only its energy draw: any relationship the unit declared swappable with
:meth:`OpsBlockData.register_relation` (an RO skid's recovery, a tank's
level-to-volume geometry). Conservation constraints are never registered, so
they can never be swapped. The predefined surrogate classes themselves —
which one a :class:`~flexcore.config.schema.SurrogateSpec` names and how each
validates and builds its own data — are documented on
:doc:`surrogates`.

Composition: PlantBlock and NetworkBlock
----------------------------------------

Two levels of composition (architecture §3.3, R7): a ``PlantBlock`` is a
collection of **unit** blocks — one facility, holding the arcs between its
units — and a ``NetworkBlock`` is a composition of **plant** blocks. Both are
thin ``dynamic=False`` flowsheets whose time domain is the ``TimeBlock``'s
ordered integer set (see :doc:`../../explanation/time_and_dynamics`). Their
aggregation ``Expression``\ s (``total_electrical_power``,
``total_thermal_power``, ``total_product``, ``total_fuel_usage``,
``total_feed``) are deferred, because units are normally added after the plant exists;
``FlexCosting.cost_process()`` builds them for every plant and network on the
model.

.. currentmodule:: flexops.core.plant_block

.. autosummary::
   :toctree: generated
   :nosignatures:

   PlantBlock
   PlantBlockData

.. currentmodule:: flexops.core.network_block

.. autosummary::
   :toctree: generated
   :nosignatures:

   NetworkBlock
   NetworkBlockData

Config-driven build
-------------------

.. currentmodule:: flexops.core.build

``build_model`` is the single config-driven entry point: one validated
:class:`~flexcore.config.schema.ModelConfig` yields the whole Pyomo model
(architecture §2.3, R3). See :doc:`../../how_to/build_a_plant` for the
imperative and config-driven paths side by side.

.. autofunction:: build_model

.. autofunction:: parse_quantity

.. autofunction:: parse_units

Registration
------------

.. currentmodule:: flexops.core.registration

The registry records FlexParameterize and the docs generator consume.

.. autosummary::
   :toctree: generated
   :nosignatures:

   IORegistry
   IOVariableRecord
   ParameterRecord
   PowerRecord
   FuelUsageRecord
   RelationRecord
   BoundaryRecord
   BoundaryKind

.. autofunction:: iter_io_registry

.. autofunction:: iter_swapped_relations

A boundary flow is registered with
:meth:`~flexops.core.ops_block.OpsBlockData.register_boundary_flow`, and the
enclosing plant discovers it from this registry rather than by importing the
:class:`~flexops.unit_models.feed.Feed`/
:class:`~flexops.unit_models.product.Product` classes — ``flexops.unit_models``
imports ``flexops.core``, so the reverse would be circular.
