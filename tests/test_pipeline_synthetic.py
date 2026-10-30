"""(g) End to end on synthetic data: train_zone -> PricePredictor.

train_zone must decide everything on the validation quarter, write a
training summary without any test-window metric, store the served model,
the AR decision and the training medians in the model metadata, and the
predictor must serve exactly that (raw served model + day-ahead AR offsets).
Tree counts are reduced so the test runs in seconds.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.model import predict as predict_mod
from src.model import train as train_mod
from tests.test_features_leakage import _make_base, _run_features, _to_master


@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory):
    import logging
    from src.model import features as feat_mod

    logger = logging.getLogger("tests.synthetic")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("synthetic")
    base = _make_base(np.random.default_rng(11))
    feats = _run_features(_to_master(base, logger), tmp, "master", mp, logger)
    mp.undo()
    path = tmp / "dataset_features.parquet"
    feats.to_parquet(path)
    assert feat_mod.FEATURES_PATH != path
    return path, feats


def test_train_zone_decides_on_validation_and_predictor_serves_it(
        synthetic_features, tmp_path, monkeypatch, quiet_logger):
    path, feats = synthetic_features
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_mod, "FEATURES_PATH", path)
    monkeypatch.setattr(train_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(train_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_mod, "get_logger", lambda _n: quiet_logger)
    monkeypatch.setattr(train_mod, "save_run_metadata", lambda **_kw: None)
    monkeypatch.setattr(train_mod, "FINAL_N_ESTIMATORS", 80)
    monkeypatch.setattr(train_mod, "WF_N_ESTIMATORS", 30)
    monkeypatch.setattr(train_mod, "QUANTILE_MAX_ESTIMATORS", 80)
    n_dk = len(train_mod.select_feature_cols(feats.columns, "DK1"))
    monkeypatch.setitem(train_mod.EXPECTED_N_FEATURES, "DK1", n_dk)

    summary = train_mod.train_zone("DK1", run_optuna=False)

    # Nothing from the test window is scored in training.
    text = json.dumps(json.load(open(results_dir / "training_DK1.json", encoding="utf-8")))
    assert "MAE_test" not in text and "test_MAE" not in text
    assert summary["served"]["model"] in ("ensemble", "lightgbm")
    assert summary["selection_val"]["n_val_hours"] == int(summary["rows"]["val"])
    assert [f["score_month"] for f in summary["ar_val"]["folds"]] == ["2025-11", "2025-12"]

    meta = pd.read_pickle(models_dir / "model_meta_DK1.pkl")
    for key in ("served_model", "use_ar", "fill_values", "feature_rule", "selection"):
        assert key in meta
    assert meta["n_estimators_final"] == 80
    assert pd.read_pickle(models_dir / "lightgbm_DK1.pkl").booster_.num_trees() == 80
    for name in ("lightgbm_train", "ensemble", "quantile_models", "conformal",
                 "ar_corrector_served"):
        assert (models_dir / f"{name}_DK1.pkl").exists()
    assert not (models_dir / "ar_corrector_DK1.pkl").exists()
    assert not (models_dir / "ar_corrector_ensemble_DK1.pkl").exists()

    # The predictor serves the model chosen on validation, with the AR offsets.
    monkeypatch.setattr(predict_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predict_mod, "get_logger", lambda _n: quiet_logger)
    for use_ar in (False, True):
        meta["use_ar"] = use_ar
        pd.to_pickle(meta, models_dir / "model_meta_DK1.pkl")
        pred = predict_mod.PricePredictor("DK1", feature_data=feats)
        assert pred._served_model == meta["served_model"]
        out = predict_mod.served_predictions("DK1", "2026-01-01", "2026-01-31 23:00",
                                             predictor=pred, include_pre_conformal=True)
        raw = pred._served_predict(pred._prepare_X(out.index))
        offsets = out["price_pred"].to_numpy() - raw
        if use_ar:
            assert np.any(offsets != 0)
            # Bands are shifted by the same offset before the conformal step.
            q10 = pred._quantile_models["P10"].predict(pred._prepare_X(out.index))
            np.testing.assert_allclose(out["P10_preconf"].to_numpy() - q10, offsets, atol=1e-9)
        else:
            np.testing.assert_allclose(offsets, 0.0, atol=1e-12)
        assert (out["P90"] >= out["P90_preconf"]).all()     # conformal widens
        # Served quantiles are sorted per hour after the conformal step.
        qs = ["P10", "P25", "P50", "P75", "P90"]
        served_q = out[qs].to_numpy()
        assert np.all(np.diff(served_q, axis=1) >= 0)
        np.testing.assert_array_equal(
            served_q, np.sort(out[[f"{q}_presort" for q in qs]].to_numpy(), axis=1))
