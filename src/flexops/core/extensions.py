"""Spec extensions: user functions a spec names to add constraints or an objective.

The spec only names an extension, so a spec that uses one builds only where the
package defining it is installed.
"""

import functools
import importlib
import importlib.metadata

from flexcore.exceptions import FlexConfigError
from flexops.core.serialize import to_jsonable
from flexops.logic._records import suppress_logic_records


def extension(fn):
    """Mark ``fn(model, **options)`` as a spec extension.

    Calling the decorated function records ``{"entry_point", "options", "name"}``
    on ``model._flex_extensions`` (so ``emit_model`` writes it back), then runs
    it. Logic helpers it calls record nothing, since rebuilding re-runs it. The
    name is ``fn.__name__``, numbered ``_2``, ``_3`` ... on repeated calls; pass
    ``_name=`` to pin it.

    Args:
        fn: A function taking the model and keyword options.

    Returns:
        The recording wrapper.

    Raises:
        FlexConfigError: When called, if the options aren't plain JSON data.
    """
    entry_point = f"{fn.__module__}:{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(model, **options):
        name = options.pop("_name", fn.__name__)
        if to_jsonable(options, where=f"{entry_point} options") != options:
            raise FlexConfigError(
                f"{entry_point} options must be plain JSON data (numbers, strings, "
                "lists and dicts) so a spec can hold them",
                field="options",
            )
        records = model.__dict__.setdefault("_flex_extensions", [])
        taken = {r["name"] for r in records}
        unique, count = name, 1
        while unique in taken:
            count += 1
            unique = f"{name}_{count}"
        records.append({"entry_point": entry_point, "options": options, "name": unique})
        # Run the extension with logic recording suppressed, so it doesn't record
        # anything that will be re-run when the spec is rebuilt.
        with suppress_logic_records():
            return fn(model, **options)

    wrapper.flex_extension = True
    return wrapper


def resolve_extension(entry_point: str):
    """Import the function an extension element names.

    Args:
        entry_point: ``'package.module:function'`` (a dotted qualname is
            allowed), or the name of an entry point in the ``flexpse.extensions``
            group.

    Returns:
        The function.

    Raises:
        FlexConfigError: If it can't be found.
    """
    missing = FlexConfigError(
        f"Extension {entry_point!r} was not found; is the package that defines it "
        "installed?",
        field="entry_point",
        value=entry_point,
    )
    if ":" not in entry_point:
        found = importlib.metadata.entry_points(
            group="flexpse.extensions", name=entry_point
        )
        if not found:
            raise missing
        return next(iter(found)).load()
    module, _, qualname = entry_point.partition(":")
    try:
        target = importlib.import_module(module)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as exc:
        raise missing from exc
    return target
