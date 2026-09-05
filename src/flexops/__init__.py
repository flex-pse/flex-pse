"""FlexOps: the flex-pse unit-model and plant-composition library."""

from importlib.metadata import version as _dist_version

from flexcore.nomenclature import PowerKind
from flexops.core.build import build_model
from flexops.core.network_block import NetworkBlock
from flexops.core.plant_block import PlantBlock
from flexops.core.registration import BoundaryKind
from flexops.core.time_block import TimeBlock
from flexops.costing import FlexCosting
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow
from flexops.unit_models import (
    BatteryModel,
    Combustor,
    ConstantEnergyIntensityModel,
    DIDOBlock,
    Digestor,
    Exchanger,
    Feed,
    GenericRenewables,
    Mixer,
    Product,
    Pump,
    ReverseOsmosis,
    SIDOBlock,
    SISOBlock,
    Splitter,
    Tank,
)
from flexops.unit_models.mixer import MixerTemperatureRule
from flexops.unit_models.powergeneration.combustor import CombustorPowerRelation

__version__ = _dist_version("flex-pse")

__all__ = [
    "BatteryModel",
    "BoundaryKind",
    "Combustor",
    "CombustorPowerRelation",
    "ConstantEnergyIntensityModel",
    "DIDOBlock",
    "Digestor",
    "Exchanger",
    "Feed",
    "FlexCosting",
    "GenericRenewables",
    "Mixer",
    "MixerTemperatureRule",
    "NetworkBlock",
    "PlantBlock",
    "PowerKind",
    "Product",
    "Pump",
    "ReverseOsmosis",
    "SIDOBlock",
    "SISOBlock",
    "SimpleAqueousFlow",
    "SimpleGasFlow",
    "Splitter",
    "Tank",
    "TimeBlock",
    "__version__",
    "build_model",
]
