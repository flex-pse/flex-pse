flexops.surrogates
===================

Predefined surrogate-structure classes (architecture §3.4/§5). A
:class:`~flexcore.config.schema.SurrogateSpec` names one by
:class:`~flexcore.config.schema.SurrogateType` and carries an opaque ``data``
mapping for it; the class validates that mapping in ``__init__`` and builds
the Pyomo relationship in :meth:`~flexops.surrogates.base.Surrogate.build`.
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the only caller
of ``build``, whether the surrogate came from a config at unit construction
time or was supplied at runtime through
:func:`~flexparameterize.apply.apply_to_model`.

This is a leaf subpackage of ``flexops`` (it imports only ``flexcore`` and
Pyomo), which is what lets ``flexops.core.ops_block`` realize a config's
surrogate at construction time with no ``flexparameterize`` import anywhere in
``flexops``.

.. currentmodule:: flexops.surrogates.surrogates

.. autosummary::
   :toctree: generated
   :nosignatures:

   SURROGATES

.. autofunction:: surrogate_from_spec

Base class
----------

.. currentmodule:: flexops.surrogates.base

.. autosummary::
   :toctree: generated
   :nosignatures:

   Surrogate

Every subclass declares ``input_variables``/``output_variables`` as
``{name: units}`` mappings — the units the relationship's data was fitted or
declared in, not necessarily the model's own. ``build`` converts each factor
from the unit's actual variable into the surrogate's declared units before
using it, and :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`
converts the whole body from the surrogate's declared output units into the
registered target's own units. Both conversions double as validation: a
declared unit dimensionally incompatible with the model's variable raises a
:class:`~flexcore.exceptions.FlexConfigError` rather than silently rescaling.

Multilinear
-----------

.. currentmodule:: flexops.surrogates.multilinear

The only implemented class today. A constant plus a sum of
``coefficient * (product of distinct declared inputs)`` — the expanded form
that covers what a previous milestone called ``linear`` (no cross terms) and
``bilinear`` (one cross term). A coefficient key is a ``*``-separated product
of names from ``input_variables``, each appearing at most once (no ``^``
exponent, no repeated factor); the reserved key ``"intercept"`` is the
constant term, read in the declared output units.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MultilinearSurrogate

Not yet implemented
--------------------

Each of these is registered in
:data:`~flexops.surrogates.surrogates.SURROGATES` and
raises ``NotImplementedError`` at construction, naming
:class:`~flexops.surrogates.multilinear.MultilinearSurrogate` as the
implemented alternative.

.. currentmodule:: flexops.surrogates

.. autosummary::
   :toctree: generated
   :nosignatures:

   QuadraticSurrogate
   ExponentialSurrogate
   ArimaSurrogate
   NeuralNetworkSurrogate

``SurrogateType.CONSTANT_INTENSITY`` has no class here at all: it fixes a
unit's ``energy_intensity`` process parameter rather than swapping a
Constraint, so :func:`~flexparameterize.apply.apply_to_model` handles it
directly rather than through this registry.
