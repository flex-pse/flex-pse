"""Shared config and synthetic data for the sweep tests."""

import numpy as np
import pandas as pd

from flexparameterize.config import ParameterizeConfig


def toy_config(**overrides) -> ParameterizeConfig:
    """A config sweeping ``a`` and ``b`` for ``z = 1 + 2a + 3b + 4ab`` in W."""
    raw = {
        "inputs": {
            "a": {"path": "a", "units": "m^3/s", "range": (1e-3, 5e-3)},
            "b": {"path": "b", "units": "Pa", "range": (1e5, 5e5)},
        },
        "sampler": {"type": "lhs", "n": 20, "seed": 20261118},
        "solver": {"name": "ipopt", "options": {"tol": 1e-10}},
        "outputs": {
            "power_electrical": {
                "path": "z",
                "units": "W",
                "unit": "facility.plant",
                "relation": "power_electrical_relation",
                "output_units": "kW",
                "input_names": {"a": "flow_out", "b": "outlet_state.pressure"},
                "terms": ["a", "b", "a*b"],
            }
        },
        "max_failed_fraction": 0.1,
        "holdout": {"k": 5, "r": 0},
        "acceptance": {"r2_min": 0.999, "max_rel_error": 1e-6},
    }
    return ParameterizeConfig.model_validate(raw | overrides)


def toy_output(a, b):
    """The toy model's output ``z`` in W."""
    return 1.0 + 2.0 * a + 3.0 * b + 4.0 * a * b


def toy_frame(n: int = 6) -> pd.DataFrame:
    """Noise-free sweep-shaped rows on an ``n`` x ``n`` grid of ``a`` and ``b``."""
    a, b = np.meshgrid(np.linspace(1e-3, 5e-3, n), np.linspace(1e5, 5e5, n))
    a, b = a.ravel(), b.ravel()
    return pd.DataFrame(
        {
            "index": np.arange(a.size),
            "a": a,
            "b": b,
            "power_electrical": toy_output(a, b),
            "solve_successful": True,
        }
    )
