# Details

Supporting material for the [README](../README.md): where each published number comes from, the full list of pipeline steps, the Battery Lab results and the history of the Danish price source.

## Where each number comes from

Every file below is written by code in this repository, except `src/model/best_params.json`, copied unchanged from the June 2026 Optuna runs (their run files are in the first public commit), and the top-level notes (`description`, `units`, `method`, `run`) of `results/battery_lab_fi.json`, written by hand; the two Battery Lab checks only replace their own blocks.

| Numbers | File | Written by |
|---|---|---|
| Served model on the test window: MAE, RMSE, directional accuracy, R², mean bias, MAE as % of the mean price, rMAE and changes against persistence and LEAR | `data/metadata/served_metrics_<zone>.json` | `scripts/gen_served_metrics.py` (steps 27–29) |
| Baselines (persistence, weekly seasonal mean, LEAR-style) | `results/baselines_test2026.json` | `src/model/baselines.py` (steps 15–17) |
| Model choice, AR decision, quantile coverage and conformal calibration on the validation quarter; walk-forward summary | `results/training_<zone>.json` | `src/model/train.py` (steps 18–20) |
| Test-window evaluation: served row and raw comparators, Diebold-Mariano (descriptive), interval coverage (split-conformal and ACI), hours with crossed quantiles before and after sorting, pinball and CRPS, battery backtest with the served and the persistence forecast | `results/evaluation_test2026_<zone>.json` | `src/model/evaluate.py` (steps 21–23) |
| Summary of the three zones (test) and of the validation decisions; battery gap between the served and the persistence forecast; mean price of train, validation and test (from the step-14 feature table) | `results/evaluation_test2026.json`, `results/model_selection_val2025.json` | `scripts/build_results.py`, which fails if the served MAE differs from the evaluation's served row |
| Sensitivity to inputs whose publication time is not stored | `results/leakage_sensitivity_test2026.json` | `scripts/leakage_sensitivity.py` |
| Finland by regime (nuclear outage, negative prices, cold hours) | `results/fi_regimes_test2026.json` | `scripts/evaluate_fi_regimes.py` (step 30) |
| Walk-forward folds | `data/models/walkforward_<zone>.csv` | `src/model/train.py` |
| Frozen hyperparameters (Optuna, 100 trials on validation) | `src/model/best_params.json` | copied from the June 2026 Optuna runs |
| Hourly master table size | `data/metadata/integration_20260605_144247.json` | `src/transform/integration.py` |
| Battery Lab | `results/battery_lab_fi.json` | `checks/test_revenue_stacking.py --write` (yearly ceiling) and `checks/test_battery_lab_data.py --write` (test window) |
| README bar chart | `docs/img/mae_by_zone_light.png`, `_dark.png` | values from the served metrics and the baselines; `scripts/make_readme_figures.py` redraws it from `results/evaluation_test2026.json` |

## Pipeline steps

| Steps | What they do |
|---|---|
| 1–7 | Extraction: ENTSO-E prices, physical data, unavailability, reservoirs and scheduled exchanges; Open-Meteo; Energi Data Service; Yahoo Finance |
| 8–9 | Cleaning and integration into the hourly master table |
| 10–13 | Additions to the master table: Finnish coupling, Fingrid forecasts, Fingrid and Energinet reserve prices |
| 14 | Feature engineering |
| 15–17 | Baselines for DK1, DK2 and FI, written to `results/baselines_test2026.json` |
| 18–20 | Training for DK1, DK2 and FI with the frozen hyperparameters: candidates fitted on train, model choice, AR decision, quantile bands and conformal calibration on validation, comparator LightGBM on train plus validation, 24-fold walk-forward |
| 21–23 | Evaluation on the test window for DK1, DK2 and FI |
| 24–26 | Day-ahead prediction check for DK1, DK2 and FI |
| 27–29 | Served-model metrics, written to `data/metadata/served_metrics_<zone>.json`; skipped if the zone's model is missing |
| 30 | Finland by regime, written to `results/fi_regimes_test2026.json` |

Outside the orchestrator: `scripts/leakage_sensitivity.py <zone>` (downloads the Open-Meteo Previous Runs data on its first run), `scripts/build_results.py`, `checks/test_revenue_stacking.py --write` and `checks/test_battery_lab_data.py --write` (Battery Lab), `scripts/make_readme_figures.py` (README bar chart), and the other manual checks in `checks/`. The published files were produced with `LGBM_N_JOBS=4` and `OMP_NUM_THREADS=4`.

## Validation decisions by zone

Validation quarter: 1 October to 31 December 2025, 2,208 hours. Diebold-Mariano with HAC variance (h = 24) and the Harvey-Leybourne-Newbold correction; DM < 0 means the ensemble is more accurate.

| Zone | Validation MAE, ensemble | Validation MAE, LightGBM (train) | DM | p | AR term: MAE raw → with AR (November and December, out of sample) | Served |
|---|---:|---:|---:|---:|---|---|
| DK1 | 12.08 | 15.77 | −3.67 | 0.0003 | 11.18 → 11.41, rejected | Diverse ensemble |
| DK2 | 13.20 | 15.30 | −2.31 | 0.02 | 13.29 → 13.39, rejected | Diverse ensemble |
| FI | 16.94 | 24.00 | −6.01 | < 0.0001 | 15.41 → 16.70, rejected | Diverse ensemble |

Quantile bands on validation cover 74.7% (DK1), 69.2% (DK2) and 77.5% (FI) of hours before calibration; the split-conformal calibrator brings them to 80.1% in sample.

## Interval bands on the test window

Since 30 October 2026 the served P10, P25, P50, P75 and P90 bands are sorted in each hour after the conformal step (`sort_quantile_bands` in `src/model/conformal.py`); the point forecast does not change. Steps 21 to 30 were run again without retraining, and only the interval figures moved (`results/evaluation_test2026_<zone>.json`, block `conformal`):

| DK1 / DK2 / FI | Before sorting | Served (sorted) |
|---|---|---|
| Hours with a crossed quantile (of 3,624) | 475 / 442 / 755 | 0 / 0 / 0 |
| P10–P90 coverage, split-conformal | 62.5 / 60.5 / 80.5% | 62.7 / 60.8 / 81.5% |
| P25–P75 coverage, split-conformal | 35.7 / 34.4 / 47.2% | 35.9 / 34.4 / 47.4% |
| Mean P10–P90 width (EUR/MWh) | 37.7 / 45.7 / 70.7 | 37.8 / 45.9 / 71.1 |
| Pinball loss (EUR/MWh) | 5.47 / 7.10 / 8.10 | 5.46 / 7.09 / 8.07 |
| CRPS, 5 quantiles (EUR/MWh) | 10.94 / 14.20 / 16.20 | 10.92 / 14.17 / 16.14 |

Before the conformal step the raw quantiles crossed in 631, 857 and 888 hours. The daily ACI (79.3 / 79.3 / 79.6%) uses the P10 and P90 before the conformal step, which the sorting does not touch, so it does not change. The CRPS approximated with 5 quantiles is twice the mean pinball loss, so its value before sorting is 2 × `pinball_loss_before_sorting`.

## Finland by regime

Served forecast against the LEAR-style baseline on the test window (`results/fi_regimes_test2026.json`): overall 21.45 vs 28.70 EUR/MWh; hours in the top quartile of total unavailable capacity in Finland (all REMIT notices, above 9,780 MW, 834 hours) 13.00 vs 28.32; negative-price hours (27) 3.32 vs 5.47; hours with a mean Helsinki–Vaasa temperature below 0 °C (1,381) 29.12 vs 32.59.

## Battery Lab: reserves plus arbitrage (Finland)

`pages/5_Battery_Lab.py` co-optimises day-ahead arbitrage with the Nordic frequency-containment reserves (FCR-N, FCR-D up and down) as a linear program. In every hour, charge plus discharge stays within the power rating, and the round-trip efficiency is applied once, split between charge and discharge. Results are settled against the realised reserve prices published by Fingrid. The commercial systems use manufacturer datasheets; the generic 1 MW / 2 MWh battery is parametric, with CAPEX from published Finnish contract values and efficiency, cycle life, auxiliary load and availability estimated (`src/model/battery_specs.py`).

- Perfect-foresight revenue ceiling in Finland for a generic 1 MW / 2 MWh battery, gross: 461.1 (2023), 408.8 (2024), 223.1 (2025) and 96.4 (2026, January to May, annualised) thousand EUR per MW and year. Reserve revenue (FCR-N plus FCR-D) fell from 443 in 2023 to 53 in 2026, at the same time as more batteries joined the Finnish reserve markets; the cause is not measured here.
- In the 2026 test window the day-ahead schedule follows the served price forecast. Reserve bids placed without knowing the reserve prices in advance (the same hour of the previous day times a bid margin, accepted only if the bid is at or below the realised clearing price) earn 70.7 thousand EUR per MW and year, gross (annualised). That is 87% of the 81.4 the same battery earns when reserve prices are known in advance (that figure also uses the forecast for its day-ahead schedule). Day-ahead arbitrage alone gives 39.7. For a 2-hour commercial system (Sungrow PowerTitan 2.0, 2.5 MW / 5 MWh) the three figures are 73.8, 83.5 (88%) and 41.9.
- aFRR, mFRR and intraday markets are not modelled; a battery that also trades in them could earn more, but that is not measured here.

## Danish prices and the retired Energi Data Service dataset

Energinet stopped publishing the Energi Data Service dataset `Elspotprices` on 30 September 2025, when the day-ahead market moved to 15-minute periods. Pipeline step 4 (`src/extract/04_energinet_dk.py`) still requests it for the whole period and gets rows only up to that date.

In the June 2026 run that built the master table, step 4 and the cleaning step removed duplicate rows by timestamp alone, which kept one zone per hour, DK2 (24,094 hourly rows). So `price_DK1` comes from the ENTSO-E day-ahead prices of step 1 (aggregated to hourly) for the whole period, and `price_DK2` comes from Energi Data Service up to 30 September 2025 and from ENTSO-E after that date, filled in by the integration step. Over those 24,094 hours the two sources agree on DK2 to within 0.0001 EUR/MWh. The October 2026 re-measurement reuses that master table.

Duplicates are now removed by timestamp and zone (DT-23 in `docs/decisions.md`), so a new run of steps 1 to 13 also takes DK1 from Energi Data Service up to 30 September 2025. If the request ever returns an error, the error is logged, the dataset is skipped and ENTSO-E becomes the only source for the Danish prices. In every case the Danish prices after 30 September 2025 come from ENTSO-E alone, without the cross-check against a second source. The pipeline does not read any newer Energi Data Service price dataset.
