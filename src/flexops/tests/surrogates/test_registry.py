"""Tests for the flexops.surrogates registry: SURROGATES and surrogate_from_spec."""

import pyomo.environ as pyo
import pytest

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import CoefficientRegistry
from flexops.surrogates import (
    SURROGATES,
    ConstantIntensitySurrogate,
    MultilinearSurrogate,
    is_registered,
    surrogate_from_spec,
)

_MULTILINEAR_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"intercept": 1.0, "flow_out": 0.5},
}


@pytest.mark.unit
def test_every_surrogate_type_has_a_registered_class():
    """Each SurrogateType resolves to the class implementing its contract."""
    assert set(SURROGATES) == set(SurrogateType)
    for surrogate_type, cls in SURROGATES.items():
        assert cls.surrogate_type is surrogate_type


@pytest.mark.unit
def test_surrogate_from_spec_builds_a_multilinear_surrogate():
    """The class is resolved and the spec validated in one step."""
    spec = MultilinearSurrogate.make_spec(**_MULTILINEAR_DATA)

    surrogate = surrogate_from_spec(spec)

    assert isinstance(surrogate, MultilinearSurrogate)
    assert surrogate.spec == spec
    assert is_registered(surrogate)


@pytest.mark.unit
def test_surrogate_from_spec_builds_a_constant_intensity_that_swaps_nothing():
    """constant_intensity has a class, but it sets a parameter, not a relation."""
    spec = ConstantIntensitySurrogate.make_spec(energy_intensity=0.5)
    surrogate = surrogate_from_spec(spec)
    assert isinstance(surrogate, ConstantIntensitySurrogate)
    assert surrogate.replaces_relation is False


@pytest.mark.unit
def test_a_subclass_of_a_registered_class_is_not_registered():
    """Only the registered class can be rebuilt from a spec, so emitted."""

    class Custom(MultilinearSurrogate):
        pass

    assert not is_registered(
        Custom(MultilinearSurrogate.make_spec(**_MULTILINEAR_DATA))
    )


@pytest.mark.unit
def test_a_surrogate_refuses_a_spec_for_another_type():
    """Constructing a class from another type's spec says which to use."""
    spec = ConstantIntensitySurrogate.make_spec(energy_intensity=0.5)
    with pytest.raises(FlexConfigError, match="surrogate_from_spec"):
        MultilinearSurrogate(spec)


@pytest.mark.unit
def test_a_surrogate_refuses_the_old_data_layout():
    """A pre-0.2.0 data dict points to the migration helper."""
    with pytest.raises(FlexConfigError, match="migrate_surrogate_spec"):
        MultilinearSurrogate({"data": _MULTILINEAR_DATA})


@pytest.mark.unit
def test_a_newer_config_version_is_refused():
    """A spec from a newer build's config format asks for an upgrade."""
    spec = MultilinearSurrogate.make_spec(**_MULTILINEAR_DATA).model_copy(
        update={"config_version": 2}
    )
    with pytest.raises(FlexConfigError, match="Upgrade flex-pse"):
        surrogate_from_spec(spec)


@pytest.mark.unit
def test_an_older_config_version_is_migrated_on_construction(monkeypatch):
    """A type that bumps config_version upgrades older configs it is given."""

    def migrate(cls, version, config):
        assert version == 1
        return {**config, "terms": [term.upper() for term in config["terms"]]}

    monkeypatch.setattr(MultilinearSurrogate, "config_version", 2)
    monkeypatch.setattr(MultilinearSurrogate, "migrate_config", classmethod(migrate))
    old = SurrogateSpec(
        surrogate_type="multilinear",
        config_version=1,
        config={
            "input_variables": {"FLOW_OUT": "m^3/hr"},
            "output_variables": {"power_electrical": "kW"},
            "terms": ["flow_out"],
        },
        parameters={"intercept": {"value": 0.0}, "FLOW_OUT": {"value": 1.0}},
    )
    surrogate = surrogate_from_spec(old)
    assert surrogate.spec.config_version == 2
    assert surrogate.config.terms == ["FLOW_OUT"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "surrogate_type",
    [
        SurrogateType.QUADRATIC,
        SurrogateType.EXPONENTIAL,
        SurrogateType.NEURAL_NETWORK,
    ],
)
def test_surrogate_from_spec_stubs_raise_not_implemented(surrogate_type):
    """Reserved types without a full implementation raise NotImplementedError."""
    spec = SurrogateSpec(surrogate_type=surrogate_type, config={})
    with pytest.raises(NotImplementedError, match="MultilinearSurrogate"):
        surrogate_from_spec(spec)


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_success():
    """A valid name/Var pair is stored and retrievable."""
    registry = CoefficientRegistry()
    var = pyo.Var(initialize=0.0)
    registry.register_coefficient("intercept", var)
    assert registry["intercept"] is var
    assert "intercept" in registry
    assert len(registry) == 1
    assert list(registry.items()) == [("intercept", var)]


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_duplicate_raises():
    """Registering the same name twice raises FlexConfigError."""
    registry = CoefficientRegistry()
    registry.register_coefficient("a", pyo.Var(initialize=0.0))
    with pytest.raises(FlexConfigError, match="already registered"):
        registry.register_coefficient("a", pyo.Var(initialize=1.0))


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_non_var_raises():
    """Registering a non-Var value raises FlexConfigError."""
    registry = CoefficientRegistry()
    with pytest.raises(FlexConfigError, match="must be a pyo.Var"):
        registry.register_coefficient("bad", 42)


@pytest.mark.unit
def test_coefficient_registry_registers_an_indexed_var_as_one_array():
    """A whole indexed Var is one array parameter, read and set as a list."""
    m = pyo.ConcreteModel()
    m.coefs = pyo.Var(["a", "b"], initialize={"a": 1.0, "b": 2.0})

    registry = CoefficientRegistry()
    registry.register_coefficient("coefs", m.coefs)
    assert registry.names() == ["coefs"]
    assert registry.shape("coefs") == (2,)
    assert registry.value("coefs") == [1.0, 2.0]

    registry.set_value("coefs", [3.0, 4.0])
    assert m.coefs["b"].value == 4.0
    with pytest.raises(FlexConfigError, match=r"shape \(2,\)"):
        registry.set_value("coefs", [[3.0, 4.0]])


@pytest.mark.unit
def test_coefficient_registry_array_shape_folds_nested_lists_row_major():
    """A (2, 3) array reads and writes nested lists in its index order."""
    m = pyo.ConcreteModel()
    m.w = pyo.Var(range(2), range(3), initialize=0.0)
    registry = CoefficientRegistry()
    registry.register_coefficient("w", m.w, shape=(2, 3))
    registry.set_value("w", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert m.w[1, 0].value == 4.0
    assert registry.value("w") == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(FlexConfigError, match="does not fill shape"):
        registry.register_coefficient("w2", m.w, shape=(4, 2))
    with pytest.raises(FlexConfigError, match="scalar Var"):
        registry.register_coefficient("s", pyo.Var(initialize=0.0), shape=(1,))


@pytest.mark.unit
def test_coefficient_registry_fixes_one_name_or_all():
    """fix/unfix take a name; an array's flag covers every entry."""
    m = pyo.ConcreteModel()
    m.w = pyo.Var(range(2), initialize=0.0)
    m.c = pyo.Var(initialize=1.0)
    registry = CoefficientRegistry()
    registry.register_coefficient("w", m.w)
    registry.register_coefficient("c", m.c)

    registry.fix("w")
    assert registry.is_fixed("w") and not registry.is_fixed("c")
    m.w[0].unfix()
    assert not registry.is_fixed("w") and registry.is_partially_fixed("w")
    registry.fix()
    assert registry.is_fixed("w") and registry.is_fixed("c")


@pytest.mark.unit
def test_coefficient_registry_register_coefficients_bulk():
    """Bulk registration adds every valid entry."""
    registry = CoefficientRegistry()
    a = pyo.Var(initialize=0.0)
    b = pyo.Var(initialize=1.0)
    registry.register_coefficients({"a": a, "b": b})
    assert len(registry) == 2
    assert registry["a"] is a
    assert registry["b"] is b


@pytest.mark.unit
def test_coefficient_registry_register_coefficients_bulk_raises_on_bad_entry():
    """Bulk registration stops at the first invalid entry."""
    registry = CoefficientRegistry()
    registry.register_coefficients({"a": pyo.Var(initialize=0.0)})
    with pytest.raises(FlexConfigError, match="already registered"):
        registry.register_coefficients(
            {"a": pyo.Var(initialize=1.0), "b": pyo.Var(initialize=2.0)}
        )
    assert len(registry) == 1


@pytest.mark.unit
def test_coefficient_registry_fix_unfix():
    """fix() locks every Var; unfix() releases them."""
    m = pyo.ConcreteModel()
    registry = CoefficientRegistry()
    a = pyo.Var(initialize=0.0)
    b = pyo.Var(initialize=1.0)
    m.add_component("a", a)
    m.add_component("b", b)
    registry.register_coefficients({"a": a, "b": b})

    a.fix()
    registry.fix()
    assert a.is_fixed() and b.is_fixed()

    b.unfix()
    registry.unfix()
    assert not a.is_fixed() and not b.is_fixed()


@pytest.mark.unit
def test_coefficient_registry_register_indexed_var():
    """An indexed Var is registered; individual entries are surfaced by key."""
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=["intercept", "flow_out"])
    m.coefs = pyo.Var(m.idx, initialize=1.0)
    m.idx.construct()
    m.coefs.construct()

    registry = CoefficientRegistry()
    registry.register_coefficients(m.coefs)

    assert len(registry) == 2
    assert "intercept" in registry
    assert "flow_out" in registry
    assert registry["intercept"] is m.coefs["intercept"]
    assert registry["flow_out"] is m.coefs["flow_out"]
    assert set(registry) == {"intercept", "flow_out"}
    assert dict(registry.items()) == {
        "intercept": m.coefs["intercept"],
        "flow_out": m.coefs["flow_out"],
    }


@pytest.mark.unit
def test_coefficient_registry_indexed_var_getitem_missing_raises():
    """Looking up a key absent from the indexed Var raises KeyError."""
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=["a"])
    m.coefs = pyo.Var(m.idx, initialize=1.0)
    m.idx.construct()
    m.coefs.construct()

    registry = CoefficientRegistry()
    registry.register_coefficients(m.coefs)

    with pytest.raises(KeyError, match="b"):
        registry["b"]


@pytest.mark.unit
def test_coefficient_registry_bulk_indexed_var_getitem():
    """__getitem__ resolves an entry registered from an indexed Var."""
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=["a"])
    m.coefs = pyo.Var(m.idx, initialize=1.0)
    m.idx.construct()
    m.coefs.construct()

    registry = CoefficientRegistry()
    registry.register_coefficients(m.coefs)

    assert registry["a"] is m.coefs["a"]


@pytest.mark.unit
def test_coefficient_registry_bulk_indexed_var_iter():
    """__iter__ yields the index keys of a bulk-registered indexed Var."""
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=["a", "b"])
    m.coefs = pyo.Var(m.idx, initialize=1.0)
    m.idx.construct()
    m.coefs.construct()

    registry = CoefficientRegistry()
    registry.register_coefficients(m.coefs)

    assert set(registry) == {"a", "b"}


@pytest.mark.unit
def test_coefficient_registry_fix_unfix_indexed():
    """fix/unfix traverse every indexed Var entry."""
    m = pyo.ConcreteModel()
    m.idx = pyo.Set(initialize=["a", "b"])
    m.coefs = pyo.Var(m.idx, initialize=1.0)
    m.idx.construct()
    m.coefs.construct()

    registry = CoefficientRegistry()
    registry.register_coefficients(m.coefs)

    m.coefs["a"].fix()
    registry.fix()
    assert m.coefs["a"].is_fixed() and m.coefs["b"].is_fixed()

    m.coefs["b"].unfix()
    registry.unfix()
    assert not m.coefs["a"].is_fixed() and not m.coefs["b"].is_fixed()
