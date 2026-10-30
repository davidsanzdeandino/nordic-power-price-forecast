"""(f) Training protocol: no early stopping on data that is also in the fit,
quantile models stop on the disjoint validation quarter, and the frozen
hyperparameters load for every zone.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.model import train as train_mod
from src.model.config import BEST_PARAMS_PATH, LGB_COMMON, select_feature_cols


@pytest.fixture
def toy(rng):
    idx = pd.date_range("2025-06-01", "2026-02-28 23:00", freq="h", tz="UTC")
    n = len(idx)
    X = pd.DataFrame({f"feat_x{i}": rng.normal(0, 1, n) for i in range(4)}, index=idx)
    y = pd.Series(50 + 10 * X["feat_x0"] - 5 * X["feat_x1"] + rng.normal(0, 3, n), index=idx)
    splits = train_mod._split_masks(idx)
    return X, y, splits


@pytest.fixture
def fit_calls(monkeypatch):
    """Record (rows fitted, eval rows) of every LGBMRegressor.fit call."""
    calls = []
    original = lgb.LGBMRegressor.fit

    def spy(self, X, y, *args, eval_set=None, **kwargs):
        calls.append({"index": X.index, "eval": None if eval_set is None
                      else [e[0].index for e in eval_set]})
        return original(self, X, y, *args, eval_set=eval_set, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", spy)
    return calls


SMALL = {"learning_rate": 0.1, "num_leaves": 15, "max_depth": 4, "min_child_samples": 20}


def test_final_lightgbm_has_fixed_trees_and_no_eval_set(toy, fit_calls, quiet_logger):
    X, y, splits = toy
    model = train_mod.train_lgbm_tuned(X, y, splits, SMALL, quiet_logger, n_estimators=60)
    assert model.booster_.num_trees() == 60
    assert model.evals_result_ == {}
    (call,) = fit_calls
    assert call["eval"] is None
    trainval = X.index[splits["train"] | splits["val"]]
    assert call["index"].equals(trainval)
    # Changing the validation targets cannot change the number of trees.
    y2 = y.copy()
    y2[splits["val"]] += 100.0
    model2 = train_mod.train_lgbm_tuned(X, y2, splits, SMALL, quiet_logger, n_estimators=60)
    assert model2.booster_.num_trees() == 60


def test_shared_lightgbm_settings_are_applied(toy):
    X, y, splits = toy
    m = train_mod.fit_lgbm_fixed(X[splits["train"]], y[splits["train"]], SMALL, 10)
    params = m.get_params()
    for k, v in LGB_COMMON.items():
        assert params[k] == v
    assert params["n_estimators"] == 10


def test_quantile_models_fit_on_train_and_stop_on_validation(toy, fit_calls, quiet_logger,
                                                             monkeypatch):
    X, y, splits = toy
    monkeypatch.setattr(train_mod, "QUANTILE_MAX_ESTIMATORS", 300)
    models, stats = train_mod.train_quantile_models(X, y, splits, SMALL, quiet_logger)
    assert set(models) == {"P10", "P25", "P50", "P75", "P90"}
    assert len(fit_calls) == 5
    train_idx, val_idx = X.index[splits["train"]], X.index[splits["val"]]
    for call in fit_calls:
        assert call["index"].equals(train_idx)
        assert call["eval"][0].equals(val_idx)
        assert call["index"].intersection(call["eval"][0]).empty
    assert all(1 <= n <= 300 for n in stats["n_trees"].values())
    assert 0.0 <= stats["coverage_80_val_raw"] <= 1.0


def test_ensemble_stops_on_disjoint_validation(toy, fit_calls):
    from src.model.diverse_ensemble import DiverseObjectiveEnsemble
    X, y, splits = toy
    ens = DiverseObjectiveEnsemble(base_params={"n_estimators": 200, **SMALL})
    ens.fit(X[splits["train"]], y[splits["train"]], X[splits["val"]], y[splits["val"]])
    assert len(fit_calls) == 3
    for call in fit_calls:
        assert call["index"].intersection(call["eval"][0]).empty
    for m in ens.models.values():
        assert m.get_params()["deterministic"] is True


def test_frozen_hyperparameters_cover_every_zone():
    data = json.load(open(BEST_PARAMS_PATH, encoding="utf-8"))
    for zone in ("DK1", "DK2", "FI"):
        params = train_mod.load_best_params(zone)
        assert params == data["zones"][zone]["params"]
        assert "n_estimators" not in params
        assert {"learning_rate", "num_leaves", "max_depth"} <= set(params)


def test_splits_are_disjoint_and_ordered(toy):
    X, _y, splits = toy
    assert not (splits["train"] & splits["val"]).any()
    assert not (splits["val"] & splits["test"]).any()
    assert X.index[splits["train"]].max() < X.index[splits["val"]].min()
    assert X.index[splits["val"]].max() < X.index[splits["test"]].min()


def test_dk_rule_drops_only_the_finnish_coupling_block():
    cols = ["price_DK1", "feat_DK1_price_lag24h", "feat_FI_price_lag24h", "feat_SE3_price_lag24h",
            "feat_SE1_price_lag24h", "feat_SE4_price_lag24h", "feat_EE_price_lag24h",
            "feat_sched_net_FI_EE", "feat_post_ol3", "feat_sched_net_DK1_NO2",
            "feat_spread_SE3_SE4", "is_weekend", "is_holiday_dk", "is_congested_DK1_DK2"]
    dk = select_feature_cols(cols, "DK1")
    assert dk == ["feat_DK1_price_lag24h", "feat_sched_net_DK1_NO2", "feat_spread_SE3_SE4",
                  "is_weekend", "is_holiday_dk"]
    fi = select_feature_cols(cols, "FI")
    assert fi == [c for c in cols if c.startswith("feat_")] + ["is_weekend", "is_holiday_dk"]
