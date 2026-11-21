flexops.surrogates
==================

Predefined classes for surrogate structures. A
:class:`~flexcore.config.schema.SurrogateSpec` names one by
:class:`~flexcore.config.schema.SurrogateType` and carries its ``config`` and
``parameters``. Each class owns its contract: it defines what ``config`` holds,
which parameters (names and shapes) that config implies, and how to build the
Pyomo relationship. It validates a spec in ``__init__`` without building any
Pyomo objects, builds in :meth:`~flexops.surrogates.base.Surrogate.build`, and
reads its live spec back with :meth:`~flexops.surrogates.base.Surrogate.read_spec`.
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the only caller
of ``build``, whether the surrogate came from a spec at build time or was
supplied at runtime through :func:`~flexparameterize.apply.apply_to_model`.

This is a leaf subpackage of ``flexops`` (it imports only ``flexcore``,
pydantic and Pyomo), which is what lets ``flexops.core.ops_block`` realize a
config's surrogate at construction time with no ``flexparameterize`` import
anywhere in ``flexops``. flexparameterize builds specs only through each
class's ``make_spec``.

.. currentmodule:: flexops.surrogates.surrogates

.. autosummary::
   :toctree: generated
   :nosignatures:

   SURROGATES

.. autofunction:: surrogate_from_spec

.. autofunction:: surrogate_class

.. autofunction:: is_registered

Base class
----------

.. currentmodule:: flexops.surrogates.base

.. autosummary::
   :toctree: generated
   :nosignatures:

   Surrogate

Every subclass declares ``input_variables``/``output_variables`` as
``{name: units}`` mappings in its config. These are the units the relationship
was fitted or declared in, not necessarily the model's own. ``build`` converts
each factor from the unit's actual variable into the surrogate's declared
units before using it, and :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`
converts the whole body from the surrogate's declared output units into the
registered target's own units. Both conversions double as validation. A
declared unit dimensionally incompatible with the model's variable raises a
:class:`~flexcore.exceptions.FlexConfigError` rather than silently rescaling.

Developing a surrogate starts with its contract: subclass
:class:`~flexops.surrogates.base.Surrogate`, define ``config_model``,
``parameter_shapes`` and a typed ``make_spec``, implement ``_build``
registering every parameter, and add the class to
:data:`~flexops.surrogates.surrogates.SURROGATES`. The shared contract tests
then cover it. Regression tooling comes after, producing specs only through
``make_spec``. A change to a type's config format bumps its ``config_version``
and adds a ``migrate_config`` step.

Multilinear
-----------

.. currentmodule:: flexops.surrogates.multilinear

A constant plus a sum of ``coefficient * (product of distinct declared
inputs)``. This is the expanded form that covers what a previous milestone
called ``linear`` (no cross terms) and ``bilinear`` (one cross term).
``config.terms`` lists the products, each a product of names from
``input_variables``, separated by ``*``, each appearing at most once (no ``^``
exponent, no repeated factor). The parameters are ``intercept``, the constant
term read in the declared output units, plus one per term.

.. autosummary::
   :toctree: generated
   :nosignatures:

   MultilinearSurrogate
   MultilinearConfig

ARIMA
-----

.. currentmodule:: flexops.surrogates.arima

A non-seasonal ``ARIMA(p, d, q)`` with exogenous inputs. ``config`` holds the
order, the deterministic term (``intercept`` for ``d=0``, ``drift`` for
``d=1``, or none), an optional ``max_ar_coeff`` and the pre-horizon
``history``. The parameters are the deterministic term, ``ar.L1`` to ``ar.Lp``,
``ma.L1`` to ``ma.Lq`` and ``exog.<input>`` per input.
``unit.update_surrogate_state()`` rewrites ``history`` from a solved horizon.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ArimaSurrogate
   ArimaConfig
   ArimaHistory
   arima_parameter_names

Constant intensity
------------------

.. currentmodule:: flexops.surrogates.constant_intensity

Sets a unit's ``energy_intensity`` (its one parameter) rather than swapping a
relation; the unit already has the relationship
``power_electrical == energy_intensity * flow``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConstantIntensitySurrogate
   ConstantIntensityConfig

Not yet implemented
-------------------

.. currentmodule:: flexops.surrogates

.. autosummary::
   :toctree: generated
   :nosignatures:

   QuadraticSurrogate
   ExponentialSurrogate
   NeuralNetworkSurrogate
