"""The public, shipped unit-model test harness (testing doc §2).

Every flex-pse unit model — including user-written ones — gets a ~30-line test
class subclassing :class:`UnitModelTestHarness`: a ``configure()`` method plus
the expected-DoF and expected-solution data. Pytest collects the subclass (name
it ``Test*``); the base class itself does not match pytest's collection pattern
and is never collected.

:func:`dummy_time_block` builds the small model scaffold the harness examples
and the M14 docs generator construct units on.
"""

import datetime

import pyomo.environ as pyo
import pytest

# Direct idaes/pyomo imports per decision R12 (no compat layer; the milestone
# text's `flexcore.compat.idaes` predates R12).
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from flexcore.exceptions import FlexSolverError
from flexcore.nomenclature import ELECTRICAL_POWER, THERMAL_POWER, PowerKind
from flexops.properties.simple_aqueous import SimpleAqueousFlow

_FORBIDDEN_BARE_NAMES = ("power", "energy", "work")


def dummy_time_block(n: int = 3) -> pyo.ConcreteModel:
    """Build the standard small model scaffold for unit-model tests and docs.

    Returns a ``ConcreteModel`` carrying ``m.time_block`` — a
    :class:`~flexops.core.time_block.TimeBlock` spanning exactly ``n`` points at
    ``time_step=15 * pyunits.min`` starting ``"2025-01-01"`` — and
    ``m.properties``, a flow-only
    :class:`~flexops.properties.simple_aqueous.SimpleAqueousFlow`
    (``fixed_density=True``). The M14 docs generator imports this helper with
    exactly this signature.

    Args:
        n: Number of time points.

    Returns:
        The model, ready for a unit to be attached (e.g.
        ``m.unit = Pump(property_package=m.properties)``).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> m = dummy_time_block(3)
        >>> m.time_block.n_points
        3
    """
    from flexops.core.time_block import TimeBlock

    start = datetime.datetime(2025, 1, 1)
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date=start,
        end_date=start + n * datetime.timedelta(minutes=15),
        time_step=15 * pyunits.min,
    )
    m.properties = SimpleAqueousFlow(fixed_density=True)
    return m


class UnitModelTestHarness:
    """Subclass per unit model; pytest collects the subclass.

    Override :meth:`configure` to build and return ``(model, unit)`` and set
    ``expected_dof``/``expected_solution``. The provided ``test_*`` stages are
    inherited and collected on the subclass; build/units/registration/naming/
    DoF stages are ``unit`` tier, solve/solution are ``component`` tier (and
    ``needs_highs`` — v0 models are LP).

    Every stage calls :meth:`configure` for itself and works on that fresh
    model; nothing is cached across test methods, so no stage can leak mutated
    state (fixed variables, loaded solutions) into another.

    Attributes:
        expected_dof: Degrees of freedom after fixing every registered
            ``role="input"`` IO variable.
        expected_solution: Component name (resolved dotted against the unit,
            e.g. ``"electrical_power[0]"``) to expected solved value.
        solver_tolerance: Relative tolerance for ``expected_solution`` checks.

    Example:
        >>> from flexops.testing import UnitModelTestHarness, dummy_time_block
        >>> from flexops.unit_models import Pump
        >>> class TestPump(UnitModelTestHarness):
        ...     expected_dof = 0
        ...     def configure(self):
        ...         m = dummy_time_block(3)
        ...         m.unit = Pump(property_package=m.properties)
        ...         return m, m.unit
    """

    expected_dof: int = 0
    expected_solution: dict[str, float] = {}  # component name -> value
    solver_tolerance: float = 1e-6

    def configure(self):
        """Build and return (model, unit). Override this."""
        raise NotImplementedError

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _fix_inputs(unit) -> None:
        """Fix every registered ``role="input"`` IO variable at its value."""
        for record in unit._io_registry.io_variables:
            if record.role == "input":
                record.var.fix()

    @staticmethod
    def _component_by_name(unit, name: str):
        """Resolve ``name`` (e.g. ``"electrical_power[0]"``) against the unit."""
        component = unit.find_component(name)
        if component is None:
            raise AssertionError(
                f"expected_solution names {name!r}, but {unit.name!r} has no "
                "such component."
            )
        return component

    def _solve(self):
        """Configure, fix inputs, and solve; skip cleanly when M05 is absent."""
        model, unit = self.configure()
        self._fix_inputs(unit)
        try:
            from flexcore.solvers import get_solver

            solver = get_solver(model=model)
        except (ImportError, FlexSolverError):
            pytest.skip(
                "flexcore.solvers.get_solver not available "
                "(M05 may land in parallel)"
            )
        results = solver.solve(model)
        return model, unit, results

    # -- provided test stages ------------------------------------------------

    @pytest.mark.unit
    def test_build(self):
        """``configure()`` returns (model, unit); declared ports are built."""
        model, unit = self.configure()
        assert model is not None
        assert unit is not None
        for port in unit.component_objects(Port, descend_into=True):
            assert port.is_constructed()

    @pytest.mark.unit
    def test_units_consistent(self):
        """Every constraint/expression on the unit is dimensionally consistent."""
        _, unit = self.configure()
        assert_units_consistent(unit)

    @pytest.mark.unit
    def test_io_registration(self):
        """Registered IO variables exist, carry units, are time-indexed, docked.

        The non-empty ``doc=`` requirement is user-facing documentation: the
        M14 flexdoc generator renders these strings into the reference tables.
        """
        _, unit = self.configure()
        time_index = set(unit._find_time_block().time_index)
        for record in unit._io_registry.io_variables:
            component = unit.find_component(record.name)
            assert component is not None, f"{record.name!r} not on {unit.name!r}"
            assert record.units, f"{record.name!r} carries no units"
            assert (
                set(component.index_set()) == time_index
            ), f"{record.name!r} is not indexed by t"
            assert (component.doc or "").strip(), f"{record.name!r} has no doc= string"

    @pytest.mark.unit
    def test_energy_naming(self):
        """Power Vars exist iff registered; no bare power/energy/work names."""
        _, unit = self.configure()
        registered_kinds = {record.kind for record in unit._io_registry.power}
        assert hasattr(unit, ELECTRICAL_POWER) == (
            PowerKind.ELECTRICAL.value in registered_kinds
        )
        assert hasattr(unit, THERMAL_POWER) == (
            PowerKind.THERMAL.value in registered_kinds
        )
        for component in unit.component_objects(descend_into=True):
            assert (
                component.local_name not in _FORBIDDEN_BARE_NAMES
            ), f"component named bare {component.local_name!r} on {unit.name!r}"

    @pytest.mark.unit
    def test_dof(self):
        """With every registered input fixed, DoF equals ``expected_dof``."""
        model, unit = self.configure()
        self._fix_inputs(unit)
        assert degrees_of_freedom(model) == self.expected_dof

    @pytest.mark.component
    @pytest.mark.needs_highs
    def test_solve(self):
        """With inputs fixed, the model solves to optimal termination."""
        _, _, results = self._solve()
        assert pyo.check_optimal_termination(results)

    @pytest.mark.component
    @pytest.mark.needs_highs
    def test_solution(self):
        """Solved values match ``expected_solution`` within tolerance."""
        if not self.expected_solution:
            pytest.skip("no expected_solution declared")
        _, unit, results = self._solve()
        assert pyo.check_optimal_termination(results)
        for name, expected in self.expected_solution.items():
            actual = pyo.value(self._component_by_name(unit, name))
            assert actual == pytest.approx(
                expected, rel=self.solver_tolerance
            ), f"{name}: expected {expected}, solved {actual}"
