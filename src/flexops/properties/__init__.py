"""Property packages, and the registry a config names them from."""

from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.properties.simple_gas import SimpleGasFlow

PROPERTY_PACKAGES = {
    "SimpleAqueousFlow": SimpleAqueousFlow,
    "SimpleGasFlow": SimpleGasFlow,
}
__all__ = ["PROPERTY_PACKAGES", "SimpleAqueousFlow", "SimpleGasFlow"]
