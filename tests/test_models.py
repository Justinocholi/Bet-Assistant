from datetime import date

import pytest

from bet_assistant.data.schema import Match, Sport, TeamForm
from bet_assistant.models.poisson import DixonColesModel
from bet_assistant.models.elo import EloModel
from bet_assistant.models.glicko import GlickoModel
from bet_assistant.models.logistic import LogisticRegression
from bet_assistant.models.base import InsufficientModelData
from bet_assistant.models.mathutils import poisson_pmf, sigmoid, wilson_half_width


def _football_match(lam_home=1.6, lam_away=1.1):
    hf = TeamForm("H", 30, goals_for_home=lam_home, goals_against_home=1.0,
                  goals_for_away=1.2, goals_against_away=1.4)
    af = TeamForm("A", 30, goals_for_home=1.5, goals_against_home=1.1,
                  goals_for_away=lam_away, goals_against_away=1.3)
    return Match(Sport.FOOTBALL, "H", "A", date(2025, 1, 1),
                 home_form=hf, away_form=af)


def test_poisson_pmf_sums_to_one():
    total = sum(poisson_pmf(k, 2.0) for k in range(0, 30))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_1x2_probabilities_valid():
    m = DixonColesModel().market_1x2(_football_match())
    probs = m.probabilities
    assert set(probs) == {"home", "draw", "away"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= p <= 1 for p in probs.values())
    # Stronger home side should be favoured.
    assert probs["home"] > probs["away"]


def test_dixon_coles_over_under_and_btts_consistent():
    m = _football_match()
    ou = DixonColesModel().market_over_under(m, line=2.5)
    assert sum(ou.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    btts = DixonColesModel().market_btts(m)
    assert sum(btts.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_requires_form():
    bare = Match(Sport.FOOTBALL, "H", "A", date(2025, 1, 1))
    with pytest.raises(InsufficientModelData):
        DixonColesModel().market_1x2(bare)


def test_elo_favours_higher_rated_and_home():
    elo = EloModel()
    # Simulate A beating B repeatedly.
    for _ in range(20):
        elo.update("A", "B", home_won=True)
    out = elo.market_moneyline("A", "B")
    assert out.probabilities["home"] > 0.5
    assert sum(out.probabilities.values()) == pytest.approx(1.0)


def test_elo_unrated_raises():
    with pytest.raises(InsufficientModelData):
        EloModel().market_moneyline("X", "Y")


def test_glicko_updates_and_widens_band_for_uncertain_players():
    g = GlickoModel()
    for _ in range(10):
        g.update("P", "Q")
    out = g.market_match_winner("P", "Q")
    assert out.probabilities["home"] > 0.5
    assert 0 < out.confidence_half_width <= 0.5


def test_logistic_learns_separable_data():
    # y = 1 when feature > 0.
    X = [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]]
    y = [0, 0, 0, 1, 1, 1]
    lr = LogisticRegression(epochs=2000, learning_rate=0.5).fit(X, y)
    assert lr.predict_proba([2.0]) > 0.6
    assert lr.predict_proba([-2.0]) < 0.4


def test_mathutils_bounds():
    assert 0 < sigmoid(0) < 1
    assert sigmoid(0) == pytest.approx(0.5)
    # More data => tighter band.
    assert wilson_half_width(0.5, 10) > wilson_half_width(0.5, 1000)
