flexparameterize
================

Parameterizes a FlexOps model from tabular plant data. The pipeline is
**tabular data → tag aliasing → sufficiency validation → regression →
{apply to model | emit config}**. See :doc:`../../how_to/parameterize_from_data`.

Any tabular source works, whether that's a historian export, a spreadsheet, a CSV, or a database query.
Nothing here requires a historian connection. The only requirement is that a
``TagMap`` maps the data's columns onto the right model aliases.

Tag aliasing
------------

.. currentmodule:: flexparameterize.tags

.. autosummary::
   :toctree: generated
   :nosignatures:

   TagMap
   TagReport

.. autofunction:: model_alias

Sufficiency validation
----------------------

.. currentmodule:: flexparameterize.validate

Does the data determine every registered regression problem, satisfying the
:term:`zero-degree-of-freedom regression` condition? The check reports. It never
raises.

.. autosummary::
   :toctree: generated
   :nosignatures:

   SufficiencyReport
   IOPairStatus

.. autofunction:: check_sufficiency

Regression
----------

This is the seam for pluggable regressors. Every regressor conforms to
:class:`~flexparameterize.regression.base.Regressor`, a ``runtime_checkable``
Protocol, and reduces its fit to a shared
:class:`~flexparameterize.regression.base.FitResult`.

.. currentmodule:: flexparameterize.regression.base

.. autosummary::
   :toctree: generated
   :nosignatures:

   Regressor
   FitResult

.. currentmodule:: flexparameterize.regression.constant

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConstantIntensityRegressor

.. autodata:: COEFFICIENT_NAME

.. currentmodule:: flexparameterize.regression.linear

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinearRegressor

.. currentmodule:: flexparameterize.regression

.. autofunction:: get_regressor

Applying a fit to a live model
------------------------------

.. currentmodule:: flexparameterize.apply

This is the FlexParameterize → FlexOps direction of the coupling that runs both ways. It fixes
regressed parameters in place and, where the relationship is richer than the
unit's default constant intensity, realizes the
:class:`~flexcore.config.schema.SurrogateSpec` as a
:class:`~flexops.surrogates.base.Surrogate` (:func:`~flexops.surrogates.surrogates.surrogate_from_spec`)
and swaps it in for the unit's ``power_electrical_relation`` Constraint (or
any other relation the unit registered, see ``surrogates=``), keeping the same unit
object and the same ports and arcs.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ApplyReport

.. autofunction:: apply_to_model

Emitting a config
-----------------

.. currentmodule:: flexparameterize.emit

This is the serializable direction. The same fit becomes a
:class:`~flexcore.config.schema.ModelConfig` that
:func:`flexops.core.build.build_model` rebuilds the parameterized model from.

.. autofunction:: emit_model_config

Native-constraint NLP estimation (not yet implemented)
--------------------------------------------------------

.. currentmodule:: flexparameterize.estimation

A sibling capability to the regression pipeline above, not a replacement for
it. Instead of fitting a regressor against bare tabular columns, this fits
parameters directly against a unit's own already built constraint that's
nonlinear in its parameters (a pump's hydraulic power law, for example), using
``pyomo.contrib.parmest``. The shape below is reserved. Every entry point
raises ``NotImplementedError`` until it lands.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EstimationResult
   UnitExperiment

.. autofunction:: estimate_parameters

.. autofunction:: commit_estimate
