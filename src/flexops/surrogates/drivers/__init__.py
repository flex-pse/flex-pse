"""Framework-specific :class:`~flexops.surrogates.grey_box.ExternalModelDriver`
implementations.

Each module here imports its own framework at its own top level (e.g.
``torch_driver`` imports ``torch``); :func:`~flexops.surrogates.grey_box.get_driver`
imports the right module lazily, so importing this package itself pulls in no
framework.
"""
