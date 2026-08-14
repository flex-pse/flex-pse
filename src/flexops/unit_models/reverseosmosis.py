r"""ReverseOsmosis(SIDOBlock): feed -> permeate + brine (§3.4).

A direct physical subclass of
:class:`~flexops.unit_models.base.sido.SIDOBlockData` that renames the split
into RO vocabulary (``recovery``, ``feed``, ``permeate``, ``brine``), bounds
the recovery, and adds a constant electrical intensity.
"""

from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.unit_models.base.sido import SIDOBlockData


@declare_process_block_class("ReverseOsmosis")
class ReverseOsmosisData(SIDOBlockData):
    r"""A reverse-osmosis skid: feed split into permeate and brine.

    ``feed`` is the SIDO inlet, ``permeate`` the outlet_a stream, and
    ``brine`` the outlet_b (concentrate) stream -- both the config option and
    the Var carrying the split are named ``recovery`` (not ``split_fraction``,
    which does not exist on this unit):

    .. math::

        \dot{V}_{perm}[t] &= \text{recovery} \cdot \dot{V}_{feed}[t] \\
        P_{elec}[t] &= \text{energy\_intensity} \cdot \dot{V}_{perm}[t]

    The draw is metered on **permeate**, not feed: ``energy_intensity`` is the
    skid's specific energy consumption, the kWh per m^3 of product the
    desalination industry quotes. At a recovery below one the same number
    therefore means a smaller absolute draw than a feed-based reading would.

    ``recovery_min``/``recovery_max`` are the recovery Var's bounds, defaulted
    to the seawater-RO window. They bind once the Var is unfixed — by a design
    mode or a regression — where they keep the fitted recovery inside what the
    membrane train can actually deliver; a brackish or high-recovery train
    raises ``recovery_max``. An inverted window, or one that cannot hold the
    configured ``recovery``, raises ``FlexConfigError`` at build time rather
    than producing a Var whose bounds contradict its value.

    The RO names are registered by passing ``naming_dict`` up to the SIDO
    ``build()``; the ports (``inlet``/``outlet_a``/``outlet_b``) keep their
    topology names.

    Config:
        Inherits the SIDO/OpsBlock config with ``split_fraction`` renamed to
        ``recovery`` (default 0.45); adds ``recovery_min`` (0.3),
        ``recovery_max`` (0.6), and ``energy_intensity`` (default
        3.0 kWh/m^3 of permeate).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import ReverseOsmosis
        >>> m = dummy_time_block(3)
        >>> m.ro = ReverseOsmosis(property_package=m.properties)  # doctest: +SKIP
    """

    CONFIG = SIDOBlockData.CONFIG()
    del CONFIG["split_fraction"]  # renamed to "recovery", declared just below
    CONFIG.declare(
        "recovery",
        ConfigValue(
            default=0.45,
            domain=float,
            description="Water recovery: the permeate fraction of the feed "
            "(a fixed, regressable Var once built); the remainder leaves as "
            "brine.",
        ),
    )
    CONFIG.declare(
        "recovery_min",
        ConfigValue(
            default=0.3,
            domain=float,
            description="Lower bound on the recovery Var: the lowest recovery "
            "the membrane train can be operated or fitted at.",
        ),
    )
    CONFIG.declare(
        "recovery_max",
        ConfigValue(
            default=0.6,
            domain=float,
            description="Upper bound on the recovery Var: the highest recovery "
            "the membrane train can be operated or fitted at.",
        ),
    )
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=3.0 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume of permeate "
            "produced -- the skid's specific energy consumption, the basis the "
            "desalination industry quotes (a fixed, regressable Var once "
            "built), kWh/m^3.",
        ),
    )

    def build(self) -> None:
        """Build the SIDO base in RO vocabulary, then the electrical relation."""
        super().build(
            naming_dict={
                **SIDOBlockData._component_names,
                "flow_in": "feed",
                "flow_out_a": "permeate",
                "flow_out_b": "brine",
                "split_fraction": "recovery",
            }
        )
        self.add_constant_intensity_relation(
            self.find_component(self._named("flow_out_a")),
            kind=nm.PowerKind.ELECTRICAL,
            intensity=self.config.energy_intensity,
        )

    def _split_parameter_value(self) -> float:
        """Return the configured recovery — RO's name for the split parameter.

        Returns:
            The configured water recovery.
        """
        return self.config.recovery

    def _split_parameter_bounds(self) -> tuple[float, float]:
        """Return the configured recovery window, rejecting an unusable one.

        Returns:
            The ``(recovery_min, recovery_max)`` bounds of the recovery Var.

        Raises:
            FlexConfigError: If the window is inverted, or if the configured
                ``recovery`` falls outside it.
        """
        low, high = self.config.recovery_min, self.config.recovery_max
        if low > high:
            raise FlexConfigError(
                f"recovery_min ({low}) exceeds recovery_max ({high}); set "
                "recovery_min <= recovery_max.",
                field="recovery_min",
                value=low,
            )
        if not low <= self.config.recovery <= high:
            raise FlexConfigError(
                f"recovery ({self.config.recovery}) lies outside the window "
                f"[{low}, {high}]; move recovery inside the window or widen "
                "recovery_min/recovery_max.",
                field="recovery",
                value=self.config.recovery,
            )
        return low, high
