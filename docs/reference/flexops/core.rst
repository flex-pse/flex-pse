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
flex-pse never deletes model components. A built model gets updated in place via
:meth:`OpsBlockData.update_parameters`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OpsBlock
   OpsBlockData
   RelaxationPolicy

A unit's own relationships are swapped in place from a
:class:`~flexops.surrogates.base.Surrogate` via :meth:`OpsBlockData.swap_relation`.
Its energy draw is just one example. Any relationship the unit declared
swappable with :meth:`OpsBlockData.register_relation` qualifies (an RO skid's
recovery, a tank's geometry mapping level to volume). Conservation constraints are
never registered, so
they can never be swapped. The predefined surrogate classes themselves,
which one a :class:`~flexcore.config.schema.SurrogateSpec` names and how each
validates and builds its own data, are documented on
:doc:`surrogates`.

Composition: PlantBlock and NetworkBlock
----------------------------------------

There are two levels of composition. A ``PlantBlock`` is a
collection of **unit** blocks (one facility, holding the arcs between its
units), and a ``NetworkBlock`` is a composition of **plant** blocks. Both are
thin ``dynamic=False`` flowsheets whose time domain is the ``TimeBlock``'s
ordered integer set (see :doc:`../../explanation/time_and_dynamics`). Their
aggregation ``Expression``\ s (``total_electrical_power``,
``total_thermal_power``, ``total_product``, ``total_fuel_usage``,
``total_feed``) are deferred, since units are normally added after the plant
exists. ``FlexCosting.cost_process()`` builds them for every plant and network
on the model.

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

``build_model`` is the single config driven entry point. One validated
:class:`~flexcore.config.spec.FlowsheetSpec` yields the whole Pyomo model. See
:doc:`../../how_to/build_a_plant` for the imperative and config driven paths
side by side.

.. autofunction:: build_model

.. autofunction:: parse_quantity

.. autofunction:: parse_units

Emitting a spec
---------------

.. currentmodule:: flexops.core.emit

``emit_model`` is ``build_model`` in reverse: it reads the flat spec a live
model was, or could have been, built from (see
:doc:`../../explanation/config_schema`).

.. autofunction:: emit_model

.. autofunction:: unit_model_class_name

.. currentmodule:: flexops.core.serialize

.. autofunction:: to_jsonable

.. autofunction:: units_to_str

.. currentmodule:: flexops.core.compare

.. autofunction:: model_differences

.. autofunction:: assert_models_equivalent

Extensions
----------

.. currentmodule:: flexops.core.extensions

An extension is a user function that adds constraints or an objective. Decorate
it with :func:`extension` (exported as ``flexops.extension``) and calling it
records an ``extension`` element, so ``emit_model`` writes it back. The spec only
names the function, so it builds only where the defining package is installed.

.. autofunction:: extension

.. autofunction:: resolve_extension

Build stages
------------

.. currentmodule:: flexops.core.stages

``build_model`` runs the stages in :data:`STAGES` in order. Costing is last
because ``cost_process()`` aggregates only the terms registered when it runs.
The logic stage builds its kinds in :data:`LOGIC_ORDER`, status first and
cross-unit logic last. :func:`apply_spec` applies the mutable spec elements
(surrogates, dispatches, fixes and unfixes) to a model that already exists.

.. autodata:: STAGES
   :no-value:

.. autodata:: LOGIC_ORDER
   :no-value:

.. autoclass:: BuildContext

.. autofunction:: apply_spec

.. autofunction:: elements_of

.. autofunction:: apply_relation_spec

Fixing variables
----------------

.. currentmodule:: flexops.core.fixing

The state stage applies ``fix``, ``unfix`` and ``dispatch`` elements as one
batch: it plans every point, checks conflicts, bounds and domain, and only then
changes the model (see :doc:`../../explanation/config_schema`).

.. autofunction:: apply_state

.. autoclass:: FixReport

.. autoclass:: FixViolation

.. autofunction:: element_points

.. autofunction:: variable_points

.. autofunction:: time_of

.. autofunction:: element_label

Registration
------------

.. currentmodule:: flexops.core.registration

This registry holds the records that FlexParameterize and the docs generator consume.

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
   ResourceRecord
   ResourceTerm

.. autofunction:: iter_io_registry

.. autofunction:: iter_swapped_relations

.. autofunction:: iter_resource_terms

A boundary flow is registered with
:meth:`~flexops.core.ops_block.OpsBlockData.register_boundary_flow`, and the
enclosing plant discovers it from this registry rather than by importing the
:class:`~flexops.unit_models.feed.Feed`/
:class:`~flexops.unit_models.product.Product` classes. ``flexops.unit_models``
imports ``flexops.core``, so the reverse would be circular.

Resource balances
-----------------

.. currentmodule:: flexops.core.resources

:func:`add_resource_balance` adds a balance on one named resource to a plant or
network, using the terms :func:`~flexops.core.registration.iter_resource_terms`
collects. A ``resource`` element calls it during the build. See
:doc:`/explanation/resources`.

.. autofunction:: add_resource_balance
