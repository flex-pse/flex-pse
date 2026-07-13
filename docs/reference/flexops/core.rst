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

.. autofunction:: iter_io_registry
