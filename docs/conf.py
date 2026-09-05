"""Sphinx configuration for flex-pse."""

import sys
from importlib.metadata import version as _dist_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "flex-pse"
copyright = "2025, flex-pse contributors"
author = "flex-pse contributors"
release = _dist_version("flex-pse")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "flexdoc",
]

templates_path = ["_templates"]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable/", None),
    "idaes": ("https://idaes-pse.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpicky = True
nitpick_ignore = [
    # Artifacts of the IDAES-generated ProcessBlock wrapper docstring.
    ("py:class", "function"),
    ("py:class", "class"),
    # Sphinx's default_role cross-reference resolves a bare exception name
    # (e.g. "FlexConfigError" in a Raises: block) against the current module,
    # not the class's own page, so it needs the fully-qualified name to
    # resolve -- most Raises: entries were written with the short name.
    ("py:exc", "FlexConfigError"),
    ("py:exc", "FlexSolverError"),
    # pydantic documents its validation error under pydantic_core, so the name
    # config-facing Raises: entries use has no intersphinx target.
    ("py:exc", "pydantic.ValidationError"),
]

# autodoc renders the subscripted generics in a pydantic model's constructor
# signature (e.g. ``dict[str, float]``) as a single xref target and splits it at
# the comma, producing an unresolvable ``dict[str`` fragment. The real types
# resolve fine; ignore only these signature-parsing fragments.
nitpick_ignore_regex = [
    ("py:class", r"dict\[.*"),
]

html_theme = "furo"


def _skip_deprecated_or_todo(app, what, name, obj, skip, options):
    """Drop members marked deprecated or TODO from the generated reference.

    A member is skipped when its docstring carries a Sphinx ``.. deprecated::``
    directive or begins with a ``TODO``/``DEPRECATED`` marker, so provisional
    or retired API never renders into the public docs. Returning ``None`` for
    everything else defers to autodoc's normal decision.
    """
    doc = (getattr(obj, "__doc__", None) or "").lstrip()
    if ".. deprecated::" in doc or doc.startswith(("TODO", "DEPRECATED")):
        return True
    return None


def setup(app):
    """Register the deprecated/TODO member filter."""
    app.connect("autodoc-skip-member", _skip_deprecated_or_todo)
