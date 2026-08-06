"""UnitModelTestHarness: the public, shipped unit-model test harness (§2).

Every unit-model milestone subclasses :class:`UnitModelTestHarness` instead of
hand-writing the build/units/registration/DoF/solve checks: a concrete test
file is ``configure()`` plus two expected-value dicts (~30 lines). Users
writing custom unit models get the same harness.

Each stage calls :meth:`UnitModelTestHarness.configure` fresh and does not
share state across stages -- every test method builds and tears down its own
``(model, unit)`` pair.
"""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Port
from pyomo.opt import assert_optimal_termination
from pyomo.util.check_units import assert_units_consistent

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexSolverError
from flexcore.solvers import get_solver
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow

_FORBIDDEN_POWER_NAMES = ("power", "energy", "work")


def _dummy_time_block(n: int) -> pyo.ConcreteModel:
    """Build a bare model carrying only an ``n``-point TimeBlock.

    Args:
        n: Number of 15-minute time points to span.

    Returns:
        A ``pyo.ConcreteModel`` with ``m.time_block`` and no property package.
    """
    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01",
        end_date=f"2025-01-01T{(n * 15) // 60:02d}:{(n * 15) % 60:02d}",
        time_step=15 * pyunits.min,
    )
    return m


def dummy_time_block(n: int = 3) -> pyo.ConcreteModel:
    """Build a throwaway model with an ``n``-point TimeBlock + properties.

    Reused by the M14 docs generator to construct a unit model for rendering
    its Variables/Constraints/Degrees-of-Freedom tables.

    Args:
        n: Number of 15-minute time points to span.

    Returns:
        A ``pyo.ConcreteModel`` with ``m.time_block`` (an ``n``-point,
        15-minute-resolution ``TimeBlock`` starting 2025-01-01) and
        ``m.properties`` (a ``SimpleAqueousFlow()``).
    """
    m = _dummy_time_block(n)
    m.properties = SimpleAqueousFlow()
    return m


def dummy_gas_time_block(n: int = 3) -> pyo.ConcreteModel:
    """Build a throwaway model with an ``n``-point TimeBlock + gas properties.

    The gas twin of :func:`dummy_time_block`, for unit models built on
    :class:`~flexops.properties.simple_gas.SimpleGasFlow`. Its state blocks
    always carry three state variables (flow, pressure, temperature),
    where the aqueous package's default carries one, so a gas unit's
    degree-of-freedom accounting differs and needs this fixture rather than
    the aqueous one.

    Args:
        n: Number of 15-minute time points to span.

    Returns:
        A ``pyo.ConcreteModel`` with ``m.time_block`` (an ``n``-point,
        15-minute-resolution ``TimeBlock`` starting 2025-01-01) and
        ``m.properties`` (a ``SimpleGasFlow()``).
    """
    m = _dummy_time_block(n)
    m.properties = SimpleGasFlow()
    return m


def _fix_registered_inputs(unit) -> None:
    """Fix every registered ``role="input"`` IO variable at its current value.

    Args:
        unit: The unit whose ``_io_registry`` to walk.
    """
    for record in unit._io_registry.io_variables:
        if record.role != "input":
            continue
        var = record.var
        if var.is_indexed():
            for data in var.values():
                data.fix()
        else:
            var.fix()


def _solve_with_inputs_fixed(model, unit) -> None:
    """Fix every registered input, solve ``model``, and assert optimality.

    The solver is selected from the built model, so each subclass gets whatever
    its own problem class needs -- HiGHS for an LP ``Pump``, IPOPT for a ``Tank``
    whose ``capacity`` is left free (making ``level_definition`` bilinear). When
    no installed solver covers that class, ``get_solver`` raises
    :class:`~flexcore.exceptions.FlexSolverError` and the test skips carrying
    that message, which names the class and how to install a capable solver. A
    solver that is installed but *fails* still fails the test.

    Args:
        model: The model to solve.
        unit: The unit whose registered ``role="input"`` IO variables to fix.
    """
    _fix_registered_inputs(unit)
    try:
        solver = get_solver(model=model)
    except FlexSolverError as exc:
        pytest.skip(str(exc))
    assert_optimal_termination(solver.solve(model))


class UnitModelTestHarness:
    """Subclass per unit model; pytest collects the subclass.

    Attributes:
        expected_dof: Degrees of freedom expected once every registered
            ``role="input"`` IO variable is fixed.
        expected_solution: Mapping of dotted component name (e.g.
            ``"power_electrical[0]"``) to its expected solved value; doubles
            as a solution-regression baseline. Empty skips ``test_solution``.
        solution_rtol: Relative tolerance for the ``expected_solution`` check.
    """

    expected_dof: int = 0
    expected_solution: dict[str, float] = {}
    solution_rtol: float = 1e-6

    def configure(self):
        """Build and return ``(model, unit)``. Override this.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    @pytest.mark.unit
    def test_build(self):
        """``configure()`` builds without exception; declared ports are populated."""
        model, unit = self.configure()
        assert model is not None
        assert unit is not None
        for port in unit.component_objects(Port, descend_into=False):
            assert len(list(port.values())) > 0

    @pytest.mark.unit
    def test_units_consistent(self):
        """The unit's variables and constraints are dimensionally consistent."""
        _, unit = self.configure()
        assert_units_consistent(unit)

    @pytest.mark.unit
    def test_io_registration(self):
        """Every registered IO variable exists, carries units, and has a doc=."""
        _, unit = self.configure()
        for record in unit._io_registry.io_variables:
            var = record.var
            assert var.parent_block() is not None
            assert record.units
            assert record.time_indexed
            assert var.doc, f"registered IO variable {record.name!r} has no doc="

    @pytest.mark.unit
    def test_energy_naming(self):
        """power_electrical/power_thermal exist iff registered; no bare names."""
        _, unit = self.configure()
        registered_kinds = {record.kind for record in unit._io_registry.power}
        assert hasattr(unit, nm.POWER_ELECTRICAL) == (
            nm.PowerKind.ELECTRICAL in registered_kinds
        )
        assert hasattr(unit, nm.POWER_THERMAL) == (
            nm.PowerKind.THERMAL in registered_kinds
        )
        for bad_name in _FORBIDDEN_POWER_NAMES:
            assert not hasattr(unit, bad_name)

    @pytest.mark.unit
    def test_dof(self):
        """Fixing every registered input drives DoF to ``expected_dof``."""
        from idaes.core.util.model_statistics import degrees_of_freedom

        model, unit = self.configure()
        _fix_registered_inputs(unit)
        assert degrees_of_freedom(model) == self.expected_dof

    @pytest.mark.component
    def test_solve(self):
        """Solving with every registered input fixed terminates optimally."""
        model, unit = self.configure()
        _solve_with_inputs_fixed(model, unit)

    @pytest.mark.component
    def test_solution(self):
        """Solved values match ``expected_solution`` within ``solution_rtol``."""
        if not self.expected_solution:
            pytest.skip("expected_solution is empty")
        model, unit = self.configure()
        _solve_with_inputs_fixed(model, unit)
        for name, expected in self.expected_solution.items():
            component = unit.find_component(name)
            assert component is not None, f"no component named {name!r} on unit"
            assert pyo.value(component) == pytest.approx(
                expected, rel=self.solution_rtol
            )
