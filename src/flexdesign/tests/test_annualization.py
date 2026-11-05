"""present_worth_factors: discounting modeled years over the years they represent."""

import pytest
from flexcore.config.design import PlanningYear

from flexdesign import present_worth_factors


def _years() -> list[PlanningYear]:
    """Return two modeled years, standing in for 2 and 3 calendar years."""
    return [
        PlanningYear(year=2026, represents=2),
        PlanningYear(year=2028, represents=3),
    ]


@pytest.mark.unit
def test_zero_rate_factor_equals_years_represented():
    """With no discounting, each factor counts the calendar years represented."""
    assert present_worth_factors(_years(), 0.0) == {2026: 2.0, 2028: 3.0}


@pytest.mark.unit
def test_factors_match_hand_values_at_seven_percent():
    """Each year's factor discounts from the end of each year it represents."""
    factors = present_worth_factors(_years(), 0.07)
    assert factors[2026] == pytest.approx(1 / 1.07 + 1 / 1.07**2)
    assert factors[2028] == pytest.approx(1 / 1.07**3 + 1 / 1.07**4 + 1 / 1.07**5)


@pytest.mark.unit
def test_factors_sum_to_the_annuity_factor():
    """Factors over consecutive years add up to the study's annuity factor."""
    annuity = (1 - 1.07**-5) / 0.07
    assert sum(present_worth_factors(_years(), 0.07).values()) == pytest.approx(annuity)
