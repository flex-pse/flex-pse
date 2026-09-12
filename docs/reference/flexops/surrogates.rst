flexops.surrogates
===================

Predefined classes for surrogate structures. A
:class:`~flexcore.config.schema.SurrogateSpec` names one by
:class:`~flexcore.config.schema.SurrogateType` and carries an opaque ``data``
mapping for it. The class validates that mapping in ``__init__`` and builds
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
``{name: units}`` mappings. These are the units the relationship's data was
fitted or declared in, not necessarily the model's own. ``build`` converts
each factor from the unit's actual variable into the surrogate's declared
units before using it, and :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`
converts the whole body from the surrogate's declared output units into the
registered target's own units. Both conversions double as validation. A
declared unit dimensionally incompatible with the model's variable raises a
:class:`~flexcore.exceptions.FlexConfigError` rather than silently rescaling.

Multilinear
-----------

.. currentmodule:: flexops.surrogates.multilinear

The only implemented class today. A constant plus a sum of
``coefficient * (product of distinct declared inputs)``. This is the expanded
form that covers what a previous milestone called ``linear`` (no cross terms)
and ``bilinear`` (one cross term). A coefficient key is a product of names
from ``input_variables``, separated by ``*``, each appearing at most once (no
``^`` exponent, no repeated factor). The reserved key ``"intercept"`` is the
constant term, read in the declared output units.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MultilinearSurrogate

External model (grey box)
-------------------------

.. currentmodule:: flexops.surrogates.grey_box

Some relationships have no closed form a modeler can write as a Pyomo
expression, but *do* have external code that can report a numeric value and
its derivatives at a point -- a fitted neural network, a linearized CFD or
process simulator, a vendor's own sensitivity-aware solver, anything with
that shape. Rather than deriving a closed-form expression,
:class:`ExternalModelSurrogate` wraps that external code opaquely -- named by
a dotted ``model_path`` -- as a unit's relation via a PyNumero
``ExternalGreyBoxBlock``: only its numeric output and derivatives (evaluated
through a pluggable :class:`ExternalModelDriver`) are used. Building this
surrogate requires solving with ``SolverFactory("cyipopt")`` -- ``get_solver``
raises a clear error rather than silently misrouting the model to an ASL
solver.

Registers no coefficients: an external model's internal weights are not
something FlexParameterize can regress, so
:func:`~flexparameterize.regression.get_regressor` raises a permanent
:class:`~flexcore.exceptions.FlexConfigError` for
``SurrogateType.EXTERNAL_MODEL`` rather than a "not implemented yet" stub.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ExternalModelSurrogate

Which tool actually evaluates the wrapped model is a declared ``framework``
field, resolved to a **driver** -- a small object evaluating one model and
its first two derivatives at a point. PyTorch (wrapping a fitted ``nn.Module``
or a plain callable, via ``torch.autograd``) is the only driver implemented
today, but the abstraction exists for future non-neural-network cases too --
e.g. a CFD model exposing its own adjoint/sensitivity computation would be a
new driver here, not a new kind of surrogate. A driver's own framework is
imported only when :func:`get_driver` resolves it, so importing
``flexops.surrogates`` never imports ``torch``.

The PyTorch driver runs the model's forward/backward pass on whatever device
its own parameters already live on -- call ``model.to("cuda")`` yourself on
the object ``model_path`` names to run it on GPU. This needs no config: every
value handed back to Pyomo/CyIpopt always crosses to CPU first, so the
optimization itself stays CPU-only regardless. The model is re-evaluated
independently at every time index, every solver iteration, with no batching
or caching -- GPU is a net win mainly for a large model; a small one mostly
just pays the host/device transfer overhead.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ExternalFramework
   ExternalModelDriver

.. autofunction:: get_driver

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

``SurrogateType.CONSTANT_INTENSITY`` has no class here at all. It fixes a
unit's ``energy_intensity`` process parameter rather than swapping a
Constraint, so :func:`~flexparameterize.apply.apply_to_model` handles it
directly rather than through this registry.
