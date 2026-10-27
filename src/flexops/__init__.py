"""FlexOps: the flex-pse unit-model and plant-composition library."""

from importlib.metadata import version as _dist_version

from flexcore.nomenclature import PowerKind
from flexops.core.build import build_model
from flexops.core.emit import emit_model
from flexops.core.extensions import extension
from flexops.core.fixing import FixReport, FixViolation
from flexops.core.network_block import NetworkBlock
from flexops.core.plant_block import PlantBlock
from flexops.core.registration import BoundaryKind
from flexops.core.stages import apply_spec
from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow
from flexops.unit_models import (
    BatteryModel,
    Chiller,
    CO2Electrolyzer,
    Combustor,
    ConstantEnergyIntensityModel,
    DIDOBlock,
    Digestor,
    Electrolyzer,
    Exchanger,
    Feed,
    GenericRenewables,
    GridConnection,
    Mixer,
    Product,
    Pump,
    ReverseOsmosis,
    Server,
    SIDOBlock,
    SISOBlock,
    SpaceCooling,
    Splitter,
    Tank,
    WaterElectrolyzer,
)
from flexops.unit_models.base.mixer import MixerTemperatureRule
from flexops.unit_models.powergeneration.combustor import CombustorPowerRelation
from flexops.unit_models.watertreatment.electrolyzer import (
    ElectrolyzerTechnology,
    ProductPhase,
)

__version__ = _dist_version("flex-pse")

__all__ = [
    "BatteryModel",
    "BoundaryKind",
    "CO2Electrolyzer",
    "Chiller",
    "Combustor",
    "CombustorPowerRelation",
    "ConstantEnergyIntensityModel",
    "DIDOBlock",
    "Digestor",
    "Electrolyzer",
    "ElectrolyzerTechnology",
    "Exchanger",
    "Feed",
    "FixReport",
    "FixViolation",
    "FlexCosting",
    "GenericRenewables",
    "GridConnection",
    "Mixer",
    "MixerTemperatureRule",
    "NetworkBlock",
    "PlantBlock",
    "PowerKind",
    "Product",
    "ProductPhase",
    "Pump",
    "ReverseOsmosis",
    "SIDOBlock",
    "SISOBlock",
    "Server",
    "SpaceCooling",
    "SimpleAqueousFlow",
    "SimpleGasFlow",
    "Splitter",
    "Tank",
    "TimeBlock",
    "WaterElectrolyzer",
    "__version__",
    "apply_spec",
    "build_model",
    "emit_model",
    "extension",
]
