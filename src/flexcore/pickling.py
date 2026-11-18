"""Pickle built flex-pse models."""

import functools
import io
import pickle
import types

import pyomo.environ as pyo
from idaes.core.base.process_block import ProcessBlock
from pyomo.common.config import UninitializedMixin

from flexcore.exceptions import FlexError


def dumps(model: pyo.Block) -> bytes:
    """Pickle a built model to bytes that plain ``pickle.loads`` reads back.

    Args:
        model: Any Pyomo block, typically the output of ``build_model``.

    Returns:
        The pickled bytes.

    Warning:
        Construction rules defined as local functions are replaced by a
        placeholder that raises ``FlexError`` if called. Pyomo does not call
        rules after construction.
    """
    buffer = io.BytesIO()
    _ModelPickler(buffer, pickle.HIGHEST_PROTOCOL).dump(model)
    return buffer.getvalue()


# TODO: deprecate dumps once IDAES and Pyomo fix pickling upstream (issue #180).
class _ModelPickler(pickle.Pickler):
    """Pickler that handles IDAES process blocks, lazy Pyomo configs and local rules."""

    def reducer_override(self, obj):
        """Return a reduce tuple for an unpicklable case, else NotImplemented."""
        # TODO(upstream): IDAES process block classes can't be found by name.
        if isinstance(obj, ProcessBlock) and hasattr(type(obj), "__process_block__"):
            _, _, *state = obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
            args = (obj.process_block_class(), obj.__process_block__ == "indexed")
            return (_rebuild_process_block, args, *state)
        # TODO(upstream): Pyomo UninitializedMixin swaps __class__ mid-pickle.
        if isinstance(obj, UninitializedMixin):
            _ = obj._data  # resolve the lazy value so its class is final
            return NotImplemented
        if isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return (functools.partial, (_dropped_rule, obj.__qualname__))
        return NotImplemented


def _rebuild_process_block(cls: type, indexed: bool) -> ProcessBlock:
    """Create an empty scalar or indexed instance of an IDAES process block class."""
    return ProcessBlock.__new__(cls, *((None,) if indexed else ()))


def _dropped_rule(qualname: str, *args, **kwargs):
    """Raise for a local rule that was dropped when its model was pickled."""
    raise FlexError(f"{qualname} was a local function and was not pickled.")
