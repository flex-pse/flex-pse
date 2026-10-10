"""Write a fingerprint of each legacy nested config's built model.

Run once on the code that built nested configs before the flat spec existed:
``python make_snapshots.py``. The snapshots then pin what the new assembler must
still build for those configs.
"""

import json
import os
import warnings
from pathlib import Path

from flexcore.config.io import load_model_config
from flexops import build_model
from flexops.testing import model_fingerprint

FIXTURES = Path(__file__).parent.parent
DEMO = json.loads((FIXTURES / "plant_config_demo.json").read_text())
SURROGATE_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"flow_out": 0.8, "intercept": 0.0},
}


def _demo_variant(**unit_overrides) -> dict:
    """Return the demo config with fields of its units overridden."""
    config = json.loads(json.dumps(DEMO))
    for unit, overrides in unit_overrides.items():
        config["plant"]["units"][unit].update(overrides)
    return config


def _demo_network() -> dict:
    """Return the demo config with its plant moved under a network."""
    config = json.loads(json.dumps(DEMO))
    config["network"] = {"name": "net", "plants": {"demo": config.pop("plant")}}
    return config


CASES = {
    "api_freeze": (
        FIXTURES / "api_freeze" / "api_freeze_config.json",
        "api_freeze/data",
    ),
    "demo": (DEMO, "."),
    "demo_network": (_demo_network(), "."),
    "demo_multilinear_surrogate": (
        _demo_variant(
            surrogate={
                "surrogate": {
                    "surrogate_type": "multilinear",
                    "data": SURROGATE_DATA,
                }
            }
        ),
        ".",
    ),
    "demo_no_package_no_costing": (
        _demo_variant(
            battery={"property_package": None},
            surrogate={"costing": False},
        ),
        ".",
    ),
}
"""Case name -> (nested config path or dict, data directory under fixtures)."""


def build_case(name: str):
    """Build one case's model from its nested config."""
    config, data_dir = CASES[name]
    previous = Path.cwd()
    os.chdir(FIXTURES / data_dir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return build_model(load_model_config(config))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    for case in CASES:
        fingerprint = model_fingerprint(build_case(case))
        text = json.dumps(fingerprint, indent=1, sort_keys=True)
        (Path(__file__).parent / f"{case}.json").write_text(text)
        print(case, len(fingerprint["components"]), "components")
