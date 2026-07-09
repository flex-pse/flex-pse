"""Sphinx configuration for flex-pse.

See ``plan/03_documentation.md`` §1 for the layout and tooling this mirrors.
"""

import os

project = "flex-pse"
copyright = "2025, flex-pse contributors"
author = "flex-pse contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
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
}

# Fast local iteration skips notebook execution; the docs CI gate (M14) runs
# with execution on. See plan/03_documentation.md §4/§6.
nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "cache")

nitpicky = True
nitpick_ignore = [
    # Artifacts of the IDAES-generated ProcessBlock wrapper docstring.
    ("py:class", "function"),
    ("py:class", "class"),
    # flexcore.exceptions has no reference page yet (added in a later
    # milestone); TimeBlock's and get_solver's Raises: entries point at it.
    ("py:exc", "FlexConfigError"),
    ("py:exc", "FlexSolverError"),
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
