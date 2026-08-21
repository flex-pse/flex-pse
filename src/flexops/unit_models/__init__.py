"""Physical unit-model library (architecture §3.4).

``__all__`` is the unit-model registry ``UnitConfig.unit_model_class`` resolves
against, so it lists **only** constructible unit models; the enum-typed config
vocabularies live on their own modules and on the top-level ``flexops``.
"""

from flexops.unit_models.base import DIDOBlock, SIDOBlock, SISOBlock
from flexops.unit_models.constant_intensity import ConstantEnergyIntensityModel
from flexops.unit_models.exchanger import Exchanger
from flexops.unit_models.mixer import Mixer
from flexops.unit_models.powergeneration.combustor import Combustor
from flexops.unit_models.powergeneration.generic_renewables import GenericRenewables
from flexops.unit_models.pump import Pump
from flexops.unit_models.reverseosmosis import ReverseOsmosis
from flexops.unit_models.splitter import Splitter
from flexops.unit_models.storage.battery import BatteryModel
from flexops.unit_models.storage.tank import Tank

__all__ = [
    "BatteryModel",
    "Combustor",
    "ConstantEnergyIntensityModel",
    "DIDOBlock",
    "Exchanger",
    "GenericRenewables",
    "Mixer",
    "Pump",
    "ReverseOsmosis",
    "SIDOBlock",
    "SISOBlock",
    "Splitter",
    "Tank",
]
