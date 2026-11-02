"""Unit tests for :mod:`flexcore.solvers.options`."""

import logging

import pytest

from flexcore.logger import DedupHandler
from flexcore.solvers import SolverOption, SolverSettings
from flexcore.solvers.options import OPTION_NAMES, solver_options


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def options_log():
    logger = logging.getLogger("flexcore.solvers.options")
    for handler in logger.handlers:
        if isinstance(handler, DedupHandler):
            handler.dedup_enabled[logging.WARNING] = False
    capture = _LogCapture()
    logger.addHandler(capture)
    yield capture
    logger.removeHandler(capture)


@pytest.mark.unit
def test_generic_options_translate_to_gurobi_spellings():
    native = solver_options(
        "gurobi", SolverSettings.DEFAULT, mip_gap=0.01, time_limit=60, options=None
    )
    assert native == {"MIPGap": 0.01, "TimeLimit": 60}


@pytest.mark.unit
def test_generic_options_translate_to_highs_spellings():
    native = solver_options(
        "highs", SolverSettings.DEFAULT, mip_gap=0.01, time_limit=60, options=None
    )
    assert native == {"mip_rel_gap": 0.01, "time_limit": 60}


@pytest.mark.unit
def test_appsi_highs_shares_highs_spellings():
    assert OPTION_NAMES["appsi_highs"] == OPTION_NAMES["highs"]


@pytest.mark.unit
def test_unset_generic_options_are_omitted():
    native = solver_options(
        "gurobi", SolverSettings.DEFAULT, mip_gap=None, time_limit=None, options=None
    )
    assert native == {}


@pytest.mark.unit
def test_unit_commitment_profile_is_valid_for_gurobi_and_highs():
    gurobi = solver_options(
        "gurobi",
        SolverSettings.UNIT_COMMITMENT,
        mip_gap=0.01,
        time_limit=None,
        options=None,
    )
    highs = solver_options(
        "highs",
        SolverSettings.UNIT_COMMITMENT,
        mip_gap=0.01,
        time_limit=None,
        options=None,
    )
    assert gurobi == {"MIPFocus": 3, "Symmetry": 2, "Cuts": 2, "MIPGap": 0.01}
    assert highs == {"mip_rel_gap": 0.01}


@pytest.mark.unit
def test_non_overlapping_sources_are_combined():
    native = solver_options(
        "gurobi",
        SolverSettings.UNIT_COMMITMENT,
        mip_gap=0.01,
        time_limit=None,
        options={"NonConvex": 2},
    )
    assert native == {
        "MIPFocus": 3,
        "Symmetry": 2,
        "Cuts": 2,
        "MIPGap": 0.01,
        "NonConvex": 2,
    }


@pytest.mark.unit
def test_generic_option_and_native_option_for_same_setting_raise():
    with pytest.raises(ValueError, match="MIPGap.*mip_gap.*options"):
        solver_options(
            "gurobi",
            SolverSettings.DEFAULT,
            mip_gap=0.01,
            time_limit=None,
            options={"MIPGap": 0.05},
        )


@pytest.mark.unit
def test_native_option_that_profile_sets_raises():
    with pytest.raises(ValueError, match="MIPFocus.*settings.*options"):
        solver_options(
            "gurobi",
            SolverSettings.UNIT_COMMITMENT,
            mip_gap=None,
            time_limit=None,
            options={"MIPFocus": 1},
        )


@pytest.mark.unit
def test_profile_conflict_ignored_for_solver_profile_does_not_tune():
    native = solver_options(
        "highs",
        SolverSettings.UNIT_COMMITMENT,
        mip_gap=None,
        time_limit=None,
        options={"MIPFocus": 1},
    )
    assert native == {"MIPFocus": 1}


@pytest.mark.unit
def test_untranslatable_generic_option_warns_and_is_dropped(options_log):
    native = solver_options(
        "ipopt", SolverSettings.DEFAULT, mip_gap=0.01, time_limit=10, options=None
    )
    assert native == {"max_cpu_time": 10}
    messages = [record.getMessage() for record in options_log.records]
    assert any("mip_gap" in message and "ipopt" in message for message in messages)


@pytest.mark.unit
def test_unknown_solver_drops_generic_options_with_warning(options_log):
    native = solver_options(
        "mystery", SolverSettings.DEFAULT, mip_gap=0.01, time_limit=None, options=None
    )
    assert native == {}
    assert options_log.records


@pytest.mark.unit
def test_solver_option_names_are_the_solve_keywords():
    assert {option.value for option in SolverOption} == {"mip_gap", "time_limit"}
