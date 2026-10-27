"""Physical unit-model library.

``__all__`` is the unit-model registry ``UnitConfig.unit_model_class`` resolves
against, so it lists **only** constructible unit models; the enum-typed config
vocabularies live on their own modules and on the top-level ``flexops``.
"""

from flexops.unit_models.base import DIDOBlock, SIDOBlock, SISOBlock
from flexops.unit_models.base.constant_intensity import ConstantEnergyIntensityModel
from flexops.unit_models.base.exchanger import Exchanger
from flexops.unit_models.base.feed import Feed
from flexops.unit_models.base.mixer import Mixer
from flexops.unit_models.base.product import Product
from flexops.unit_models.base.splitter import Splitter
from flexops.unit_models.cooling import Chiller, SpaceCooling
from flexops.unit_models.datacenter import Server
from flexops.unit_models.general.battery import BatteryModel
from flexops.unit_models.general.pump import Pump
from flexops.unit_models.general.tank import Tank
from flexops.unit_models.powergeneration.combustor import Combustor
from flexops.unit_models.powergeneration.generic_renewables import GenericRenewables
from flexops.unit_models.watertreatment.digestor import Digestor
from flexops.unit_models.watertreatment.electrolyzer import (
    CO2Electrolyzer,
    Electrolyzer,
    WaterElectrolyzer,
)
from flexops.unit_models.watertreatment.grid_connection import GridConnection
from flexops.unit_models.watertreatment.reverseosmosis import ReverseOsmosis

__all__ = [
    "BatteryModel",
    "CO2Electrolyzer",
    "Chiller",
    "Combustor",
    "ConstantEnergyIntensityModel",
    "DIDOBlock",
    "Digestor",
    "Electrolyzer",
    "Exchanger",
    "Feed",
    "GenericRenewables",
    "GridConnection",
    "Mixer",
    "Product",
    "Pump",
    "ReverseOsmosis",
    "SIDOBlock",
    "SISOBlock",
    "Server",
    "SpaceCooling",
    "Splitter",
    "Tank",
    "WaterElectrolyzer",
]
