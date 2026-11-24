"""The shared base of surrogate types that are declared but not implemented."""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from flexops.surrogates.base import Surrogate


class _AnyConfig(BaseModel):
    """Accepts any config; a stub never reads it."""

    model_config = ConfigDict(extra="allow", frozen=True)


class StubSurrogate(Surrogate):
    """A surrogate type that raises ``NotImplementedError`` when constructed.

    Subclasses set ``not_implemented`` to the message naming the alternative.
    """

    config_model: ClassVar[type[BaseModel]] = _AnyConfig
    not_implemented: ClassVar[str]

    @classmethod
    def parameter_shapes(cls, config: BaseModel) -> dict[str, tuple]:
        """Raise: this surrogate type is not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        del config
        raise NotImplementedError(cls.not_implemented)

    @property
    def input_variables(self) -> dict[str, str]:
        """Never reached -- construction always raises first."""
        raise NotImplementedError(self.not_implemented)

    @property
    def output_variables(self) -> dict[str, str]:
        """Never reached -- construction always raises first."""
        raise NotImplementedError(self.not_implemented)

    def _build(self, unit, target) -> tuple[None, Any]:
        """Never reached -- construction always raises first."""
        raise NotImplementedError(self.not_implemented)
