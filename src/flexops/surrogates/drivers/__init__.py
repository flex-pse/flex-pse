"""Framework-specific :class:`~flexops.surrogates.external.ExternalModelDriver`
implementations.

Each module here imports its own framework at its own top level (e.g.
``torch_driver`` imports ``torch``); :func:`~flexops.surrogates.external.get_driver`
imports the right module lazily, so importing this package itself pulls in no
framework.
"""
