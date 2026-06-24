import os
from datetime import date

import pytest

from bet_assistant.config import BankrollConfig
from bet_assistant.bankroll import BankrollManager, ResponsibleGambling


def test_save_load_round_trip(tmp_path):
    cfg = BankrollConfig(starting_bankroll=500.0, stop_loss_fraction=0.25)
    mgr = BankrollManager(cfg)
    rec = mgr.place_bet("1x2", "home", 10.0, 2.0, 0.6, today=date(2025, 1, 1))
    mgr.settle_bet(rec, won=True)
    mgr.place_bet("1x2", "away", 5.0, 3.0, 0.4, today=date(2025, 1, 2))  # unsettled

    path = os.path.join(tmp_path, "ledger.json")
    mgr.save(path)
    assert os.path.exists(path)

    loaded = BankrollManager.load(path, cfg)
    assert loaded.bankroll == pytest.approx(mgr.bankroll)
    assert loaded.starting == pytest.approx(500.0)
    assert len(loaded.bets) == 2
    assert loaded.bets[0].settled and loaded.bets[0].won is True
    assert loaded.bets[1].settled is False
    # Reporting still works after a reload.
    assert "ROI" in loaded.report()


def test_persisted_exclusion_survives_reload(tmp_path):
    cfg = BankrollConfig(starting_bankroll=500.0)
    rg = ResponsibleGambling()
    rg.self_exclude(30, today=date(2025, 1, 1))
    mgr = BankrollManager(cfg, responsible=rg)

    path = os.path.join(tmp_path, "ledger.json")
    mgr.save(path)
    loaded = BankrollManager.load(path, cfg)
    # Still excluded after reload.
    allowed, reason = loaded.can_stake(today=date(2025, 1, 5))
    assert not allowed and "Paused" in reason


def test_persisted_stop_loss_state_survives(tmp_path):
    cfg = BankrollConfig(starting_bankroll=100.0, stop_loss_fraction=0.25)
    mgr = BankrollManager(cfg)
    rec = mgr.place_bet("1x2", "home", 30.0, 2.0, 0.5, today=date(2025, 1, 1))
    mgr.settle_bet(rec, won=False)  # bankroll 70 < 75 floor -> halted
    assert mgr.stop_loss_triggered

    path = os.path.join(tmp_path, "ledger.json")
    mgr.save(path)
    loaded = BankrollManager.load(path, cfg)
    assert loaded.stop_loss_triggered


def test_load_rejects_unknown_schema(tmp_path):
    path = os.path.join(tmp_path, "bad.json")
    with open(path, "w") as fh:
        fh.write('{"schema_version": 999, "bets": []}')
    with pytest.raises(ValueError):
        BankrollManager.load(path, BankrollConfig())
