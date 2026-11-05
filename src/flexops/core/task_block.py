"""Task: units grouped behind their resource connections, to treat and swap as one."""

from idaes.core import ProcessBlockData, declare_process_block_class

from flexcore.exceptions import FlexConfigError
from flexops.core.plant_block import PlantBlockData


@declare_process_block_class("Task")
class TaskData(ProcessBlockData):
    """A group of connected units that act as one task in a resource-task network."""

    CONFIG = ProcessBlockData.CONFIG()

    def build(self) -> None:
        """Build the empty task on its plant.

        Raises:
            FlexConfigError: If the parent isn't a plant, whose totals would miss
                the task's units on any other block.
        """
        super().build()
        parent = self.parent_block()
        if not isinstance(parent, PlantBlockData):
            raise FlexConfigError(
                f"Task {self.name!r} must be built directly on a PlantBlock.",
                field="parent",
                value=None if parent is None else parent.name,
            )


def task_of(block):
    """Return the task a block belongs to.

    Args:
        block: A unit or any block inside one.

    Returns:
        The enclosing :class:`Task`, or ``block`` itself if there is none.
    """
    candidate = block
    while candidate is not None:
        if isinstance(candidate, TaskData):
            return candidate
        candidate = candidate.parent_block()
    return block
