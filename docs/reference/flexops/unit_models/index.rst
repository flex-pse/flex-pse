flexops.unit_models
===================

Unit models are organized by IO topology first (the base classes under
``flexops.unit_models.base`` own port construction, the per-stream mass
balance, and the energy-registration wiring), then specialized physically.
``Pump`` and ``StorageTank`` subclass the ``SISOBlock`` topology base and add
only their flow↔energy relationship or holdup dynamics.

.. currentmodule:: flexops.unit_models

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: unit_model.rst

   base.SISOBlock
   Pump
   StorageTank

The ``*Data`` classes carry each model's formulation — the governing
equations, config options, and build logic:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: unit_model.rst

   base.SISOBlockData
   PumpData
   StorageTankData
