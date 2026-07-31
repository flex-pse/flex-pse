flexops.unit_models
====================

Unit models are organized by inlet/outlet topology first (the ``base``
sub-package), then specialized physically. ``Pump`` and ``Tank`` both
subclass the single-inlet/single-outlet ``SISOBlock`` topology base
(architecture §3.4); a ``Tank`` additionally disables the on/off logic
layer, since a tank has no unit-commitment status (R6).

.. currentmodule:: flexops.unit_models.base.siso

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   SISOBlock

.. currentmodule:: flexops.unit_models.pump

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Pump

.. currentmodule:: flexops.unit_models.storage_tank

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   Tank

.. currentmodule:: flexops.unit_models.battery

.. autosummary::
   :toctree: generated
   :template: unit_model.rst
   :nosignatures:

   BatteryModel
