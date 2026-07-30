flexcore
========

Shared substrate for the flex-pse tools: the exception hierarchy, the solver
facade, and (later) the versioned config schema.

Solvers
-------

.. currentmodule:: flexcore.solvers

The solver layer classifies a model, detects which solvers are installed, and
selects one — erroring loudly rather than transforming the model (decision R5;
see :doc:`../../explanation/relaxation_policies`).

.. autofunction:: get_solver

.. autoclass:: SolverFacade
   :members:

.. autofunction:: classify

.. autoclass:: ProblemClass
   :members:

.. autofunction:: available_solvers

Capability matrix
~~~~~~~~~~~~~~~~~~

:func:`available_solvers` returns the subset of this matrix whose solver is
installed. The matrix is a plain module constant (``flexcore.solvers.registry.CAPABILITIES``)
and is extensible — a user may register an additional solver entry before
calling :func:`get_solver`.

.. list-table::
   :header-rows: 1

   * - Solver
     - Supported problem classes
   * - ``highs``
     - LP, MILP
   * - ``cbc``
     - LP, MILP
   * - ``ipopt``
     - NLP (built from idaes with HSL ``ma27`` when available, else stock IPOPT)
   * - ``scip``
     - MILP, MINLP (preferred over HiGHS for MILP; the default open-source MINLP solver)
   * - ``gurobi``
     - LP, QP, MILP

Config schema
-------------

.. currentmodule:: flexcore.config.schema

The versioned, JSON-canonical config the whole model+run is built from
(decision R3; see :doc:`../../explanation/config_schema`). Pydantic is the
schema authority; :data:`CURRENT_SCHEMA_VERSION` tags what this build writes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ModelConfig
   NetworkConfig
   PlantConfig
   UnitConfig
   IOVariableSpec
   SurrogateSpec
   ExternalDispatchSpec
   UnitCommitmentConfig
   ArcSpec
   TimeConfig
   CostingConfig
   PriceSpec
   DRConfig

.. autodata:: CURRENT_SCHEMA_VERSION

Config I/O
~~~~~~~~~~

.. currentmodule:: flexcore.config.io

.. autofunction:: load_model_config

.. autofunction:: dump_model_config

.. autofunction:: export_json_schemas

Nomenclature
------------

.. currentmodule:: flexcore.nomenclature

The canonical energy-variable names (see
:doc:`../../explanation/energy_nomenclature`).

.. autodata:: POWER_ELECTRICAL

.. autodata:: POWER_THERMAL

.. autodata:: FUEL_USAGE

.. autoclass:: PowerKind
   :members:

Exceptions
----------

.. currentmodule:: flexcore.exceptions

.. autoexception:: FlexError

.. autoexception:: FlexConfigError

.. autoexception:: FlexSolverError

.. autoexception:: FlexDataError
