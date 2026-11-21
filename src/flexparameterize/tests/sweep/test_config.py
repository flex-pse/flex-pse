"""Tests for ParameterizeConfig validation and hashing."""

import pydantic
import pytest

from flexparameterize.config import ParameterizeConfig, SamplerType, config_hash
from flexparameterize.tests.sweep.helpers import toy_config


def _output(**changes) -> dict:
    """The toy output's raw dict with some fields replaced."""
    raw = toy_config().outputs["power_electrical"].model_dump(mode="json")
    return {"power_electrical": raw | changes}


@pytest.mark.unit
def test_toy_config_is_valid():
    """The shared toy config validates and parses its enums."""
    config = toy_config()

    assert isinstance(config, ParameterizeConfig)
    assert config.sampler.type is SamplerType.LHS
    assert config.inputs["a"].range == (1e-3, 5e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"inputs": {"a": {"path": "a", "units": "m^3/s", "range": (2.0, 1.0)}}},
            "range",
        ),
        ({"inputs": {"a": {"path": "a", "units": "m^3.5", "range": (0, 1)}}}, "m^3.5"),
        ({"outputs": _output(terms=["a", "c"])}, "'c'"),
        ({"outputs": _output(terms=["a*a"])}, "repeat"),
        ({"outputs": _output(terms=["a^2"])}, r"\^"),
        ({"outputs": _output(input_names={"a": "flow_out"})}, "input_names"),
        (
            {"outputs": _output(input_names={"a": "flow_out", "b": "flow_out"})},
            "same",
        ),
        ({"outputs": _output(surrogate="arima")}, "multilinear"),
        ({"outputs": _output(output_units="m^3.5")}, "m^3.5"),
        ({"holdout": {"k": 5, "r": 5}}, "holdout"),
        ({"max_failed_fraction": 1.0}, "max_failed_fraction"),
        ({"unexpected": 1}, "unexpected"),
    ],
    ids=[
        "inverted_range",
        "unparseable_input_units",
        "unknown_term_factor",
        "repeated_term_factor",
        "exponent_in_term",
        "factor_missing_from_input_names",
        "two_inputs_share_a_variable",
        "non_multilinear_surrogate",
        "unparseable_output_units",
        "holdout_remainder_out_of_range",
        "failed_fraction_of_one",
        "extra_key",
    ],
)
def test_malformed_config_raises(overrides, message):
    """Each malformed config fails validation with a message naming the problem."""
    with pytest.raises(pydantic.ValidationError, match=message):
        toy_config(**overrides)


@pytest.mark.unit
def test_two_outputs_on_the_same_relation_raise():
    """Two outputs may not target the same unit and relation."""
    outputs = _output() | {"other": _output()["power_electrical"] | {"path": "y"}}

    with pytest.raises(pydantic.ValidationError, match="facility.plant"):
        toy_config(outputs=outputs)


@pytest.mark.unit
def test_config_hash_is_stable_and_tracks_changes():
    """Equal configs hash equal; changing the seed changes the hash."""
    assert config_hash(toy_config()) == config_hash(toy_config())
    reseeded = toy_config(sampler={"type": "lhs", "n": 20, "seed": 1})
    assert config_hash(reseeded) != config_hash(toy_config())
