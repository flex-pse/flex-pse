"""Sphinx directives that generate unit-model reference tables from built models.

Two directives:

``.. flexops-unit-tables:: <dotted class path>``
    Builds the named unit-model class on a throwaway model and renders three
    tables (Variables, Constraints, Degrees of Freedom) straight from its
    registered IO and built Pyomo components, so a reference page can never
    drift from the code it describes.

``.. flexops-config-table:: <dotted pydantic model path>``
    Renders a pydantic model's fields (name, type, default, description) as
    one table.

:func:`collect_unit_tables` and :func:`collect_config_table` are plain
functions with no Sphinx dependency, so the unit test in
``src/flexops/tests/docs/test_flexdoc_tables.py`` calls them directly.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import pyomo.environ as pyo
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from pyomo.environ import units as pyunits

from flexcore.config.schema import UnitCommitmentConfig
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow
from flexops.testing import dummy_gas_time_block, dummy_time_block
from flexops.unit_models.base.dido import DIDOBlock
from flexops.unit_models.base.sido import SIDOBlock
from flexops.unit_models.base.siso import SISOBlock
from flexops.unit_models.constant_intensity import ConstantEnergyIntensityModel
from flexops.unit_models.electrolyzer import (
    CO2Electrolyzer,
    Electrolyzer,
    WaterElectrolyzer,
)
from flexops.unit_models.exchanger import Exchanger
from flexops.unit_models.feed import Feed
from flexops.unit_models.mixer import Mixer
from flexops.unit_models.powergeneration.combustor import Combustor
from flexops.unit_models.powergeneration.generic_renewables import GenericRenewables
from flexops.unit_models.product import Product
from flexops.unit_models.pump import Pump
from flexops.unit_models.reverseosmosis import ReverseOsmosis
from flexops.unit_models.splitter import Splitter
from flexops.unit_models.storage.battery import BatteryModel
from flexops.unit_models.storage.tank import Tank
from flexops.unit_models.wastewater.digestor import Digestor


def _build_siso():
    m = dummy_time_block(3)
    m.unit = SISOBlock(property_package=m.properties)
    return m, m.unit


def _build_sido():
    m = dummy_time_block(3)
    m.unit = SIDOBlock(property_package=m.properties)
    return m, m.unit


def _build_dido():
    m = dummy_time_block(3)
    m.unit = DIDOBlock(property_package=m.properties)
    return m, m.unit


def _build_pump():
    m = dummy_time_block(3)
    m.unit = Pump(property_package=m.properties)
    return m, m.unit


def _build_tank():
    m = dummy_time_block(3)
    m.unit = Tank(property_package=m.properties)
    return m, m.unit


def _build_battery():
    m = dummy_time_block(3)
    m.unit = BatteryModel(
        capacity=10 * pyunits.kWh,
        unit_commitment=UnitCommitmentConfig(status=False),
    )
    return m, m.unit


def _build_exchanger():
    m = dummy_time_block(3)
    m.unit = Exchanger(property_package=m.properties)
    return m, m.unit


def _build_reverseosmosis():
    m = dummy_time_block(3)
    m.unit = ReverseOsmosis(property_package=m.properties)
    return m, m.unit


def _build_constant_energy_intensity():
    m = dummy_time_block(3)
    m.unit = ConstantEnergyIntensityModel(property_package=m.properties)
    return m, m.unit


def _build_feed():
    m = dummy_time_block(3)
    m.unit = Feed(property_package=m.properties, outlet_names=("a",))
    return m, m.unit


def _build_product():
    m = dummy_time_block(3)
    m.unit = Product(property_package=m.properties, inlet_names=("a",))
    return m, m.unit


def _build_mixer():
    m = dummy_time_block(3)
    m.unit = Mixer(property_package=m.properties, inlet_names=("a", "b"))
    return m, m.unit


def _build_splitter():
    m = dummy_time_block(3)
    m.unit = Splitter(property_package=m.properties, outlet_names=("a", "b"))
    return m, m.unit


def _build_combustor():
    m = dummy_gas_time_block(3)
    m.unit = Combustor(property_package=m.properties, inlet_names=("fuel",))
    return m, m.unit


def _build_generic_renewables():
    m = dummy_time_block(3)
    m.unit = GenericRenewables(
        capacity=10 * pyunits.kW, capacity_factor=[0.2, 0.5, 0.8]
    )
    return m, m.unit


def _build_digestor():
    m = dummy_time_block(3)
    m._biogas_pkg = SimpleGasFlow()
    m._sludge_pkg = SimpleAqueousFlow()
    m.unit = Digestor(
        inlet_packages={"feed": m.properties},
        biogas_property_package=m._biogas_pkg,
        sludge_property_package=m._sludge_pkg,
    )
    return m, m.unit


def _build_electrolyzer(cls):
    m = dummy_time_block(3)
    m._gas_pkg = SimpleGasFlow()
    m.unit = cls(liquid_property_package=m.properties, gas_property_package=m._gas_pkg)
    return m, m.unit


_BUILDERS: dict[type, Callable[[], tuple[Any, Any]]] = {
    SISOBlock: _build_siso,
    SIDOBlock: _build_sido,
    DIDOBlock: _build_dido,
    Pump: _build_pump,
    Tank: _build_tank,
    BatteryModel: _build_battery,
    Exchanger: _build_exchanger,
    ReverseOsmosis: _build_reverseosmosis,
    ConstantEnergyIntensityModel: _build_constant_energy_intensity,
    Feed: _build_feed,
    Product: _build_product,
    Mixer: _build_mixer,
    Splitter: _build_splitter,
    Combustor: _build_combustor,
    GenericRenewables: _build_generic_renewables,
    Digestor: _build_digestor,
    Electrolyzer: lambda: _build_electrolyzer(Electrolyzer),
    WaterElectrolyzer: lambda: _build_electrolyzer(WaterElectrolyzer),
    CO2Electrolyzer: lambda: _build_electrolyzer(CO2Electrolyzer),
}


def _import_object(dotted_path: str) -> Any:
    """Import an object from its dotted ``module.attribute`` path.

    Args:
        dotted_path: e.g. ``"flexops.unit_models.pump.Pump"``.

    Returns:
        The imported object.

    Raises:
        ImportError: If the module or the attribute does not exist.
    """
    module_path, _, name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"{dotted_path!r} is not a dotted module.attribute path")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"{module_path!r} has no attribute {name!r}") from exc


def collect_unit_tables(cls: type) -> dict[str, list[list[str]]]:
    """Build ``cls`` and read back its Variables/Constraints/DoF tables.

    Constructs ``cls`` via its entry in the module-level builder registry (a
    throwaway model on :func:`flexops.testing.dummy_time_block`, mirroring
    what a unit's own ``UnitModelTestHarness.configure`` builds), then reads
    the tables straight from the built unit's ``_io_registry`` and its Pyomo
    ``Constraint`` components -- the same live objects the docs directive
    renders.

    Args:
        cls: The unit-model class, e.g.
            :class:`~flexops.unit_models.pump.Pump`.

    Returns:
        A dict with keys ``"variables"``, ``"constraints"``, ``"dof"``, each a
        list of rows (a header row, then one row per component).

    Raises:
        KeyError: If ``cls`` has no registered builder.
        ValueError: If a registered IO variable has no ``doc=`` string, or a
            table would otherwise render empty.
    """
    builder = _BUILDERS.get(cls)
    if builder is None:
        raise KeyError(
            f"no flexdoc builder registered for "
            f"{cls.__module__}.{cls.__qualname__}; add one to "
            "docs/_ext/flexdoc.py's _BUILDERS"
        )
    _model, unit = builder()

    variable_rows = [["Name", "Index", "Units", "Role", "Description"]]
    dof_rows = [["Name", "Units", "Description"]]
    for record in unit._io_registry.io_variables:
        if not record.var.doc:
            raise ValueError(
                f"registered IO variable {record.name!r} on "
                f"{cls.__module__}.{cls.__qualname__} has no doc= string"
            )
        index = "time" if record.time_indexed else "scalar"
        variable_rows.append(
            [record.name, index, record.units, record.role, record.var.doc]
        )
        if record.role == "input":
            dof_rows.append([record.name, record.units, record.var.doc])

    constraint_rows = [["Name", "Description"]]
    for constraint in unit.component_objects(pyo.Constraint, descend_into=True):
        constraint_rows.append([constraint.local_name, constraint.doc or ""])

    if len(variable_rows) == 1 or len(constraint_rows) == 1:
        raise ValueError(
            f"{cls.__module__}.{cls.__qualname__} rendered an empty "
            "Variables or Constraints table -- a unit with no registered IO "
            "or no built constraints is a construction bug, not real output"
        )

    return {
        "variables": variable_rows,
        "constraints": constraint_rows,
        "dof": dof_rows,
    }


def collect_config_table(cls: type) -> list[list[str]]:
    """Render a pydantic model's fields as one table.

    Args:
        cls: A pydantic ``BaseModel`` subclass, e.g.
            :class:`~flexcore.config.schema.ModelConfig`.

    Returns:
        A list of rows (a header row, then one row per field): name, type,
        default, description.
    """
    rows = [["Field", "Type", "Default", "Description"]]
    for name, info in cls.model_fields.items():
        default = "required" if info.is_required() else repr(info.default)
        annotation = getattr(info.annotation, "__name__", str(info.annotation))
        rows.append([name, annotation, default, info.description or ""])
    return rows


def _rows_to_list_table(title: str, rows: list[list[str]]) -> list[str]:
    """Render ``rows`` (a header row + data rows) as RST ``list-table`` lines."""
    lines = [
        title,
        "" if not title else "",
    ]
    if len(rows) == 1:
        # A header with no data rows is legitimate (e.g. a unit with no inputs).
        return [*lines, "None.", ""]
    lines += [
        ".. list-table::",
        "   :header-rows: 1",
        "",
    ]
    for row in rows:
        lines.append(f"   * - {row[0]}")
        for cell in row[1:]:
            lines.append(f"     - {cell}")
    lines.append("")
    return lines


class FlexopsUnitTables(Directive):
    """``.. flexops-unit-tables:: <dotted class path>``."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        cls = _import_object(self.arguments[0])
        tables = collect_unit_tables(cls)

        lines: list[str] = []
        lines += _rows_to_list_table("Variables", tables["variables"])
        lines += _rows_to_list_table("Constraints", tables["constraints"])
        lines += _rows_to_list_table("Degrees of Freedom", tables["dof"])

        node = nodes.section()
        node.document = self.state.document
        view = StringList(lines, source=self.arguments[0])
        self.state.nested_parse(view, self.content_offset, node)
        return node.children


class FlexopsConfigTable(Directive):
    """``.. flexops-config-table:: <dotted pydantic model path>``."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        cls = _import_object(self.arguments[0])
        rows = collect_config_table(cls)

        lines = _rows_to_list_table("", rows)
        node = nodes.section()
        node.document = self.state.document
        view = StringList(lines, source=self.arguments[0])
        self.state.nested_parse(view, self.content_offset, node)
        return node.children


def setup(app) -> dict[str, bool]:
    """Register the ``flexops-unit-tables``/``flexops-config-table`` directives."""
    app.add_directive("flexops-unit-tables", FlexopsUnitTables)
    app.add_directive("flexops-config-table", FlexopsConfigTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
