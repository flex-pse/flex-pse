flexops.properties
===================

.. currentmodule:: flexops.properties.simple_aqueous

The minimal flow-carrying property packages (§3.7). Ports built from their
state blocks carry volumetric flow between units via standard IDAES/Pyomo arcs.

``SimpleAqueousFlow`` state blocks carry ``flow_vol``, with opt-in
pressure/temperature state variables.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleAqueousFlow
   SimpleAqueousFlowData
   SimpleAqueousStateBlockData

.. currentmodule:: flexops.properties.simple_gas

``SimpleGasFlow`` is the gas-phase counterpart: because a gas stream's
conditions always matter, its state blocks always carry all three state
variables (``flow_vol``, ``pressure``, ``temperature``).

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleGasFlow
   SimpleGasFlowData
   SimpleGasStateBlockData
