"""Chiller(SpaceCooling): a chiller plant at a constant COP."""

from idaes.core import declare_process_block_class

from flexops.unit_models.cooling.space_cooling import SpaceCoolingData


@declare_process_block_class("Chiller")
class ChillerData(SpaceCoolingData):
    """A chiller plant whose draw is its load over a COP plus an ancillary share.

    Config:
        Inherits the SpaceCooling config, with a default ``cop`` of 5.
    """

    CONFIG = SpaceCoolingData.CONFIG()
    CONFIG.get("cop").set_default_value(5.0)
