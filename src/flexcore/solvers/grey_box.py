"""Find, suspend and range-check the external grey-box blocks on a Pyomo model."""

import contextlib
import dataclasses

import pyomo.environ as pyo
from pyomo.common.errors import InfeasibleConstraintException

from flexcore.logger import get_logger

_log = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class TrainingRangeViolation:
    """One grey-box input outside the range its model was trained over.

    Attributes:
        block: Full name of the grey-box block.
        input_name: Name of the input.
        value: The input's value.
        lower: Lower end of the training range.
        upper: Upper end of the training range.
    """

    block: str
    input_name: str
    value: float
    lower: float
    upper: float


def _walk_grey_boxes(model: pyo.ConcreteModel, active: bool | None) -> list:
    """Return grey-box data blocks, walking through grey-box blocks as well as Blocks.

    Args:
        model: The Pyomo model to walk.
        active: ``True`` to skip inactive blocks and their children, ``None`` for all.

    Returns:
        The ``ExternalGreyBoxBlockData`` objects found.
    """
    # Local import so a model with no grey box never loads PyNumero's interface.
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlock,
        ExternalGreyBoxBlockData,
    )

    return [
        block
        for block in model.block_data_objects(
            active=active, descend_into=(pyo.Block, ExternalGreyBoxBlock)
        )
        if isinstance(block, ExternalGreyBoxBlockData)
    ]


def active_grey_boxes(model: pyo.ConcreteModel) -> list:
    """Return the grey-box blocks that are active, along with all their parents.

    Args:
        model: The Pyomo model to walk.

    Returns:
        The live ``ExternalGreyBoxBlockData`` objects.
    """
    return _walk_grey_boxes(model, active=True)


@contextlib.contextmanager
def suspend_orphaned_grey_boxes(model: pyo.ConcreteModel):
    """Switch off active grey boxes under an inactive parent for the ``with`` body.

    Args:
        model: The Pyomo model about to be solved.

    Yields:
        ``None``, with each such grey box deactivated until the body exits.
    """
    live = {id(block) for block in active_grey_boxes(model)}
    orphans = [
        block
        for block in _walk_grey_boxes(model, active=None)
        if block.active and id(block) not in live
    ]
    for block in orphans:
        block.deactivate()
    try:
        yield
    finally:
        for block in orphans:
            block.activate()


@contextlib.contextmanager
def suspend_trivial_constraints(model: pyo.ConcreteModel):
    """Switch off constraints with no free variables for the ``with`` body.

    Args:
        model: The Pyomo model about to be solved.

    Yields:
        ``True`` if one of those constraints is violated, so no solve can succeed.
    """
    transformation = pyo.TransformationFactory("contrib.deactivate_trivial_constraints")
    violated = False
    try:
        try:
            transformation.apply_to(model, tmp=True, tolerance=1e-6)
        except InfeasibleConstraintException:
            violated = True
        yield violated
    finally:
        # Also after a violation, which stops the transformation partway through.
        transformation.revert(model)


def grey_box_var_values(model: pyo.ConcreteModel) -> list[tuple[pyo.Var, float | None]]:
    """Return every Var on the model with its value, including those on grey boxes.

    Args:
        model: The Pyomo model to walk.

    Returns:
        ``(var, value)`` pairs.
    """
    from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

    return [
        (var, var.value)
        for var in model.component_data_objects(
            pyo.Var, descend_into=(pyo.Block, ExternalGreyBoxBlock)
        )
    ]


def check_training_range(model: pyo.ConcreteModel) -> list[TrainingRangeViolation]:
    """Warn about live grey-box inputs outside their model's ``training_range``.

    Args:
        model: The solved Pyomo model.

    Returns:
        One entry per input outside its range; models with no range are skipped.
    """
    violations = []
    for block in active_grey_boxes(model):
        ranges = getattr(block.get_external_model(), "training_range", {})
        for name, (lower, upper) in ranges.items():
            value = block.inputs[name].value
            if value is None or lower <= value <= upper:
                continue
            violations.append(
                TrainingRangeViolation(block.name, name, value, lower, upper)
            )
            _log.warning(
                "%s input %r is %g, outside its training range [%g, %g].",
                block.name,
                name,
                value,
                lower,
                upper,
            )
    return violations
