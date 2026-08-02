flexops.testing
================

.. currentmodule:: flexops.testing

The public, shipped unit-model test harness (testing plan §2). Every
unit-model milestone -- and any user writing a custom unit model -- subclasses
:class:`UnitModelTestHarness` instead of hand-writing the
build/units-consistency/registration/DoF/solve checks.

This module imports ``pytest``, so it requires the ``testing`` extra:
``pip install "flex-pse[testing]"`` (already covered by the ``dev`` extra).

.. autosummary::
   :toctree: generated
   :nosignatures:

   UnitModelTestHarness

.. autofunction:: dummy_time_block

Testing your own unit model
----------------------------

A concrete unit's test file is ~30 lines: ``configure()`` plus the two
expected-value dicts.

.. code-block:: python

    from flexops.testing import UnitModelTestHarness, dummy_time_block
    from flexops.unit_models import Pump


    class TestPump(UnitModelTestHarness):
        """Fixed inlet flow determines power_electrical via energy_intensity."""

        expected_dof = 0
        expected_solution = {
            "power_electrical[0]": 50.0,
            "power_electrical[1]": 50.0,
            "power_electrical[2]": 50.0,
        }

        def configure(self):
            m = dummy_time_block(3)
            m.unit = Pump(property_package=m.properties)
            for t in m.time_block.time_index:
                m.unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(100.0)
            return m, m.unit
