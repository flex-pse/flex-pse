"""SimpleUtility: the property block of a utility resource, like electricity or fuel."""

from idaes.core import PhysicalParameterBlock, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError


@declare_process_block_class("SimpleUtility")
class SimpleUtilityData(PhysicalParameterBlock):
    """Parameter block naming a utility resource and the units it balances in."""

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "resource",
        ConfigValue(
            default=None,
            domain=str,
            description="Name units register the utility under, e.g. 'electricity'.",
        ),
    )
    CONFIG.declare(
        "units",
        ConfigValue(
            default=None,
            domain=str,
            description="Rate units of the utility, e.g. 'kW' or 'm^3/hr'.",
        ),
    )

    def build(self) -> None:
        """Check the utility's resource and units are set.

        Raises:
            FlexConfigError: If the resource or the units are missing.
        """
        super().build()
        if not self.config.resource or not self.config.units:
            raise FlexConfigError(
                f"SimpleUtility {self.name!r} needs a resource and units, got "
                f"{self.config.resource!r} and {self.config.units!r}.",
                field="resource",
                value=self.config.resource,
            )

    @classmethod
    def define_metadata(cls, obj) -> None:
        """Declare the five default units IDAES requires; a utility has no states."""
        obj.add_default_units(
            {
                "time": pyunits.hr,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )
