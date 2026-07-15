flexops.properties
===================

.. currentmodule:: flexops.properties.simple_aqueous

The minimal flow-carrying property packages (§3.7). Ports built from their
state blocks carry volumetric flow between units via standard IDAES/Pyomo arcs.

``SimpleAqueousFlow`` state blocks carry ``flow_vol`` and ``dens_mass`` (fixed
at the configured density by default), with opt-in pressure/temperature state
variables.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleAqueousFlow
   SimpleAqueousFlowData
   SimpleAqueousStateBlockData

.. currentmodule:: flexops.properties.simple_gas

``SimpleGasFlow`` is the gas-phase counterpart: because gas density varies with
pressure and temperature, its state blocks always carry all four state
variables (``flow_vol``, ``dens_mass``, ``pressure``, ``temperature``).

.. autosummary::
   :toctree: generated
   :nosignatures:

   SimpleGasFlow
   SimpleGasFlowData
   SimpleGasStateBlockData
