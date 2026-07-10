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

The base class of every flex-pse unit model and the hierarchy-agnostic
block-replacement helper (see :doc:`../../explanation/relaxation_policies` for
the relaxation config slot).

.. autosummary::
   :toctree: generated
   :nosignatures:

   OpsBlock
   OpsBlockData
   RelaxationPolicy

.. autofunction:: replace_unit

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
   EnergyRecord

.. autofunction:: iter_io_registry
