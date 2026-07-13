"""Public testing utilities for flex-pse unit models (shipped, not test-only).

Users writing custom unit models subclass :class:`UnitModelTestHarness` and
build on :func:`dummy_time_block`, exactly as the in-tree unit-model tests do.
"""

from flexops.testing.harness import UnitModelTestHarness, dummy_time_block

__all__ = ["UnitModelTestHarness", "dummy_time_block"]
