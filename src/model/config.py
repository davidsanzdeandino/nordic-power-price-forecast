"""
config.py
=========
Shared constants of the model code: the temporal splits, the delivery time
zone, the LightGBM settings that every model shares and the per-zone feature
rule. train.py, evaluate.py, features.py, predict.py and diverse_ensemble.py
import them from here so that the dates and the thread count live in one place.

Splits (timestamps are UTC, as in the feature table):
  train       2023-01-01 -> 2025-09-30 23:00   fits every served model
  validation  2025-10-01 -> 2025-12-31 23:00   early stopping, model choice,
                                               AR term and interval calibration
  test        2026-01-01 -> 2026-05-31 23:00   scored only, never used to fit,
                                               select or calibrate anything

Threads: LGBM_N_JOBS in the environment caps LightGBM (default -1, all cores).
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ── Temporal splits ──────────────────────────────────────────────────────────
TRAIN_END  = "2025-09-30 23:00"
VAL_START  = "2025-10-01"
VAL_END    = "2025-12-31 23:00"
TEST_START = "2026-01-01"
TEST_END   = "2026-05-31 23:00"

# Delivery days of the day-ahead auction are local days in this time zone
# (CET/CEST). The auction for day D closes at 12:00 local time on D-1.
DELIVERY_TZ = "Europe/Copenhagen"

# ── LightGBM settings shared by every model ──────────────────────────────────
N_JOBS = int(os.getenv("LGBM_N_JOBS", "-1"))
LGB_COMMON = {
    "random_state":  42,
    "n_jobs":        N_JOBS,
    "deterministic": True,
    "verbose":       -1,
}

# Number of trees of the point LightGBM models. It is a declared
# hyperparameter: there is no early stopping on a set that is also in the
# training data. The diverse ensemble and the quantile models are trained on
# train only and stop early on the (disjoint) validation quarter.
FINAL_N_ESTIMATORS = 2000

# Frozen Optuna hyperparameters (one set per zone), so the models can be
# retrained without running Optuna again.
BEST_PARAMS_PATH = ROOT / "src" / "model" / "best_params.json"

# Published, code-generated result files.
RESULTS_DIR = ROOT / "results"

# ── Per-zone feature rule ────────────────────────────────────────────────────
# The Danish models do not use the Finnish coupling block (Finnish, Swedish
# SE1/SE3/SE4 and Estonian features, the Finland-Estonia schedule and the
# Olkiluoto 3 regime flag). Finland uses every feature.
DK_EXCLUDED_FEATURES = re.compile(r"^feat_(FI_|SE[134]_|EE_|sched_net_FI_|post_ol3)")
SAFE_IS_FLAGS = ("is_weekend", "is_holiday")
FEATURE_RULE = {
    "DK1": "feat_* columns not matching " + DK_EXCLUDED_FEATURES.pattern
           + ", plus the calendar flags is_weekend and is_holiday*",
    "DK2": "feat_* columns not matching " + DK_EXCLUDED_FEATURES.pattern
           + ", plus the calendar flags is_weekend and is_holiday*",
    "FI":  "all feat_* columns, plus the calendar flags is_weekend and is_holiday*",
}
# Size of the feature set on the published feature table (checked before training).
EXPECTED_N_FEATURES = {"DK1": 166, "DK2": 166, "FI": 224}

# Columns renamed when a lag changed (old -> new): the commodity lag went from
# 24 h to 48 h, and the reservoir lag from one week to 13 days
# (features.RESERVOIR_LAG).
RENAMED_FEATURES = {
    "feat_ttf_lag24h":              "feat_ttf_lag48h",
    "feat_gas_marginal_lag24h":     "feat_gas_marginal_lag48h",
    "feat_thermal_marginal_lag24h": "feat_thermal_marginal_lag48h",
    "feat_co2_lag24h":              "feat_co2_lag48h",
    "feat_reservoir_FI_lag1w":           "feat_reservoir_FI_lag13d",
    "feat_reservoir_NO_2_lag1w":         "feat_reservoir_NO_2_lag13d",
    "feat_reservoir_SE_3_lag1w":         "feat_reservoir_SE_3_lag13d",
    "feat_reservoir_SE_4_lag1w":         "feat_reservoir_SE_4_lag13d",
    "feat_reservoir_nordic_total_lag1w": "feat_reservoir_nordic_total_lag13d",
}


def select_feature_cols(columns, zone: str) -> list:
    """Model input columns for a zone, in table order (see FEATURE_RULE)."""
    cols = list(columns)
    feats = [c for c in cols if c.startswith("feat_")]
    if zone in ("DK1", "DK2"):
        feats = [c for c in feats if not DK_EXCLUDED_FEATURES.match(c)]
    flags = [c for c in cols if c.startswith("is_") and c.startswith(SAFE_IS_FLAGS)]
    return feats + flags
