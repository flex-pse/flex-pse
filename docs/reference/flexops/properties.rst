flexops.properties
===================

.. currentmodule:: flexops.properties.simple_aqueous

The minimal property packages that carry flow. Ports built from their
state blocks carry volumetric flow between units via standard IDAES/Pyomo arcs.

``SimpleAqueousFlow`` state blocks carry ``flow_vol``, and you can opt into
additional pressure and temperature state variables.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleAqueousFlow
   SimpleAqueousFlowData
   SimpleAqueousStateBlockData

.. currentmodule:: flexops.properties.simple_gas

``SimpleGasFlow`` is the counterpart for the gas phase. A gas stream's
conditions always matter, so its state blocks always carry all three state
variables (``flow_vol``, ``pressure``, ``temperature``).

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleGasFlow
   SimpleGasFlowData
   SimpleGasStateBlockData

.. currentmodule:: flexops.properties.simple_utility

``SimpleUtility`` names a utility resource, like electricity or a fuel, and the units
it balances in. A :class:`~flexops.core.resources.Resource` built on one collects what
units register under that name. See :doc:`/explanation/resources`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleUtility
   SimpleUtilityData
