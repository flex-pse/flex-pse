"""Draw the battery example in docs/explanation/time_alignment.md.

Run from the repository root to regenerate both theme variants:

    python docs/figures/time_alignment_battery.py

It writes ``docs/_static/figures/time_alignment_battery_{light,dark}.svg``.
The data are synthetic and seeded, so the output is reproducible.

The figure compares three views of one afternoon of a 100 kW / 400 kWh battery
under a 16:00-21:00 peak tariff:

* flex-pse's schedule, where the point stamped ``t`` holds a setpoint over
  ``[t, t + dt)`` and the charge at ``t`` is the level at ``t + dt``;
* a sensor sampled every minute, with a short response lag and noise;
* a historian that stores hourly averages of that power stamped at the
  interval's end, and instantaneous charge readings stamped at the instant.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "_static" / "figures"

THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f1efe8",
        "flex": "#2a78d6",
        "historian": "#eb6834",
        "sensor": "#4a4945",
    },
    "dark": {
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#24241f",
        "flex": "#3987e5",
        "historian": "#d95926",
        "sensor": "#b8b6ae",
    },
}

DAY = "2025-07-08"
DT = pd.Timedelta(hours=1)
STEPS = pd.date_range(f"{DAY} 13:00", periods=10, freq=DT)  # 13:00 ... 22:00
PEAK = (pd.Timestamp(f"{DAY} 16:00"), pd.Timestamp(f"{DAY} 21:00"))
OFF_PEAK_PRICE, PEAK_PRICE = 0.09, 0.18

# flex-pse schedule: battery power per step, charging positive (kW).
# Charge 100 kW for three hours before the peak, discharge 60 kW through it.
POWER = np.array([100, 100, 100, -60, -60, -60, -60, -60, 0, 0], dtype=float)
CHARGE_INIT = 100.0  # kWh at 13:00, the schedule's start (lossless for clarity)
CHARGE = CHARGE_INIT + np.cumsum(POWER) * 1.0  # level at the END of each step


def _price(stamps: pd.DatetimeIndex) -> np.ndarray:
    hours = stamps.hour
    return np.where((hours >= 16) & (hours < 21), PEAK_PRICE, OFF_PEAK_PRICE)


def _sensor(rng: np.random.Generator):
    """Minute samples of the real battery: lagged setpoint tracking plus noise."""
    minutes = pd.date_range(STEPS[0], STEPS[-1] + DT, freq="1min", inclusive="left")
    setpoint = np.repeat(POWER, 60)
    actual = np.empty_like(setpoint)
    level = 0.0
    tau = 3.0  # minutes; the inverter ramps toward each new setpoint
    for i, target in enumerate(setpoint):
        level += (target - level) / tau
        actual[i] = level
    power = actual + rng.normal(0.0, 4.0, size=actual.size)
    charge_true = CHARGE_INIT + np.concatenate([[0.0], np.cumsum(actual) / 60.0])
    charge_stamps = minutes.append(
        pd.DatetimeIndex([minutes[-1] + pd.Timedelta("1min")])
    )
    charge = charge_true + rng.normal(0.0, 1.5, size=charge_true.size)
    return (
        pd.Series(power, index=minutes),
        pd.Series(charge, index=charge_stamps),
    )


def _style(ax, theme):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvspan(*PEAK, color=theme["band"], zorder=0, linewidth=0)


def _title(ax, theme, text, sub):
    ax.set_title(text, loc="left", fontsize=11, color=theme["ink"], pad=18)
    ax.text(
        0,
        1.02,
        sub,
        transform=ax.transAxes,
        fontsize=9,
        color=theme["ink2"],
        va="bottom",
    )


def draw(name: str, theme: dict) -> Path:
    rng = np.random.default_rng(20250708)
    sensor_power, sensor_charge = _sensor(rng)

    # The historian: hourly averages stamped at the END of the hour they cover,
    # and instantaneous charge readings on the hour.
    hour_ending = sensor_power.resample("1h", label="right", closed="left").mean()
    charge_readings = sensor_charge[sensor_charge.index.minute == 0]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2], "hspace": 0.55},
    )
    fig.patch.set_alpha(0)
    edges = STEPS.append(pd.DatetimeIndex([STEPS[-1] + DT]))

    # --- price -----------------------------------------------------------
    ax = axes[0]
    _style(ax, theme)
    price = _price(STEPS)
    ax.stairs(price, edges, color=theme["ink2"], linewidth=2, baseline=None)
    ax.set_ylim(0, 0.22)
    ax.set_yticks([0, 0.09, 0.18])
    ax.set_yticklabels(["$0", "$0.09", "$0.18"])
    _title(
        ax,
        theme,
        "Energy price, $/kWh",
        "The peak is billed for the steps stamped 16:00 to 20:00.",
    )
    ax.text(
        PEAK[0] + (PEAK[1] - PEAK[0]) / 2,
        0.195,
        "peak 16:00 to 21:00",
        ha="center",
        fontsize=9,
        color=theme["ink2"],
    )

    # --- power -----------------------------------------------------------
    ax = axes[1]
    _style(ax, theme)
    ax.axhline(0, color=theme["axis"], linewidth=1)
    ax.plot(
        sensor_power.index,
        sensor_power.values,
        color=theme["sensor"],
        linewidth=1.0,
        label="sensor, every minute",
    )
    ax.stairs(
        POWER,
        edges,
        color=theme["flex"],
        linewidth=2,
        baseline=None,
        label="flex-pse power[t], held over the step",
    )
    ax.plot(
        STEPS, POWER, "o", color=theme["flex"], markersize=7, markeredgecolor="none"
    )
    # Each historian average sits at its stamp, the end of the hour it averaged.
    ax.plot(
        hour_ending.index,
        hour_ending.values,
        "D",
        color=theme["historian"],
        markersize=7,
        markeredgecolor="none",
        label="historian hourly average, stamped at the hour's end",
    )
    # Read as if each stamp started a step, the series runs one hour late.
    naive = hour_ending[hour_ending.index < edges[-1]]
    ax.stairs(
        naive.values,
        naive.index.append(pd.DatetimeIndex([edges[-1]])),
        color=theme["historian"],
        linewidth=1.5,
        linestyle=(0, (4, 2.5)),
        baseline=None,
        label="historian used unshifted: one hour late",
    )
    ax.set_ylim(-130, 140)
    ax.set_yticks([-60, 0, 50, 100])
    ax.set_yticklabels(["−60", "0", "50", "100"])
    _title(
        ax,
        theme,
        "Battery power, kW (charging +, discharging −)",
        "The historian's 16:00 value is the 15:00 to 16:00 average, "
        "flex-pse's 15:00 step.",
    )
    ax.annotate(
        "historian at 16:00: still charging,\nbecause it averaged 15:00 to 16:00",
        xy=(PEAK[0], hour_ending[PEAK[0]]),
        xytext=(STEPS[0] + pd.Timedelta(minutes=15), 40),
        fontsize=9,
        color=theme["ink2"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8},
    )
    ax.annotate(
        "flex-pse 16:00 step: discharging\n16:00 to 17:00",
        xy=(PEAK[0] + pd.Timedelta(minutes=30), POWER[3]),
        xytext=(PEAK[0] + pd.Timedelta(minutes=20), -105),
        fontsize=9,
        color=theme["ink2"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8},
    )
    legend = ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=False,
        labelcolor=theme["ink2"],
        handlelength=2.6,
    )
    legend.set_zorder(5)

    # --- charge ----------------------------------------------------------
    ax = axes[2]
    _style(ax, theme)
    ax.plot(
        sensor_charge.index,
        sensor_charge.values,
        color=theme["sensor"],
        linewidth=1.0,
        label="sensor, every minute",
    )
    # charge[t] is one value per step, drawn over [t, t + dt) like power. It is
    # the level at the step's end, so each step's right end lands on the
    # historian's instantaneous reading at t + dt.
    ax.stairs(
        CHARGE,
        edges,
        color=theme["flex"],
        linewidth=2,
        baseline=None,
        label="flex-pse charge[t], one value per step, the level at its end",
    )
    ax.plot(
        STEPS, CHARGE, "o", color=theme["flex"], markersize=7, markeredgecolor="none"
    )
    ax.plot(
        [STEPS[0]],
        [CHARGE_INIT],
        "o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=theme["flex"],
        markeredgewidth=1.5,
        label="charge_init, the level when the schedule starts",
    )
    ax.plot(
        charge_readings.index,
        charge_readings.values,
        "D",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=theme["historian"],
        markeredgewidth=1.5,
        label="historian reading at the instant stamped",
    )
    ax.set_ylim(0, 560)
    ax.set_yticks([0, 100, 200, 300, 400])
    _title(
        ax,
        theme,
        "Battery charge, kWh",
        "Each step ends on the historian's reading at the step's end, t + dt.",
    )
    i15 = list(STEPS).index(pd.Timestamp(f"{DAY} 15:00"))
    ax.annotate(
        "the step stamped 15:00 holds 400 kWh\nand ends on the 16:00 reading",
        xy=(STEPS[i15] + DT, CHARGE[i15]),
        xytext=(STEPS[0] + pd.Timedelta(minutes=10), 40),
        fontsize=9,
        color=theme["ink2"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8},
    )
    ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=False,
        labelcolor=theme["ink2"],
        handlelength=2.2,
    )

    axes[-1].xaxis.set_major_locator(mdates.HourLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlim(STEPS[0] - pd.Timedelta(minutes=10), edges[-1])
    axes[-1].set_xlabel(f"Clock time, {DAY}", color=theme["muted"], fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"time_alignment_battery_{name}.svg"
    plt.rcParams["svg.hashsalt"] = "flex-pse"
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return path


if __name__ == "__main__":
    for theme_name, palette in THEMES.items():
        print(draw(theme_name, palette))
