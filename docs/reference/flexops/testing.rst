flexops.testing
===============

.. currentmodule:: flexops.testing

The public, shipped testing utilities: every flex-pse unit model — including
user-written ones — is tested by subclassing :class:`UnitModelTestHarness`,
and :func:`dummy_time_block` provides the standard small model scaffold (the
M14 docs generator builds units on it too).

.. autoclass:: UnitModelTestHarness
   :members:

.. autofunction:: dummy_time_block

Testing your own unit model
---------------------------

A concrete unit's test class is ~30 lines: a ``configure()`` method plus the
expected-DoF and expected-solution data. Pytest collects the subclass (name it
``Test*``) and runs every provided stage against it::

    from flexops.testing import UnitModelTestHarness, dummy_time_block
    from flexops.unit_models import Pump


    class TestPump(UnitModelTestHarness):
        """100 m3/hr at the default 0.5 kWh/m3 draws 50 kW."""

        expected_dof = 0
        expected_solution = {
            "electrical_power[0]": 50.0,
            "electrical_power[1]": 50.0,
            "electrical_power[2]": 50.0,
        }

        def configure(self):
            m = dummy_time_block(3)
            m.unit = Pump(property_package=m.properties)
            for t in m.time_block.time_index:
                m.unit.flow_vol[t].set_value(100.0)
            return m, m.unit

The build/units/registration/naming/DoF stages run in the ``unit`` tier (no
solver); ``test_solve``/``test_solution`` run in the ``component`` tier and
skip cleanly when no capable solver is installed.
