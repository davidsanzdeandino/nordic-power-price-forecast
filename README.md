# Nord Pool electricity price forecasting and battery (BESS) simulator

[![CI](https://github.com/davidsanzdeandino/nordic-power-price-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/davidsanzdeandino/nordic-power-price-forecast/actions/workflows/ci.yml)

Forecasts the 24 hourly day-ahead electricity prices of three Nordic bidding zones (Denmark West DK1, Denmark East DK2 and Finland FI), and measures what the forecast is worth in a battery-dispatch backtest. Data comes from five sources: ENTSO-E, Energi Data Service (Energinet), Fingrid, Open-Meteo and Yahoo Finance (gas and carbon prices).

## Key results (test: January–May 2026, 3,624 hours per zone)

- MAE of 15.9 (DK1), 18.9 (DK2) and 21.5 (FI) EUR/MWh: 38–40% lower than 24-hour persistence and 25–29% lower than a linear ARX (LEAR-style) baseline.
- A 1 MW / 2 MWh battery dispatched on the forecast earns 94–95% of the perfect-foresight profit in Denmark (78% in Finland), against 82–84% (58%) when it is dispatched on 24-hour persistence.
- P10–P90 intervals with adaptive conformal inference, updated once a day, reach 79–80% coverage; the split-conformal calibration fitted on validation reaches 62.7% (DK1), 60.8% (DK2) and 81.5% (FI) on the test window.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/mae_by_zone_dark.png"><img alt="Bar chart of the test-window MAE by zone in EUR/MWh. Served model: 15.9 in DK1, 18.9 in DK2 and 21.5 in FI. Linear ARX (LEAR-style): 22.5, 26.3 and 28.7. 24-hour persistence: 26.7, 30.4 and 34.4." src="docs/img/mae_by_zone_light.png" width="860"></picture>

The model, whether to add the AR correction and the interval calibration are decided on the October–December 2025 validation quarter. That protocol was set in October 2026, after a first version had been scored on the test window; the first version and what changed are under [Limitations](#limitations).

The full pipeline is here: data extraction from five sources, cleaning and integration, leakage-controlled feature engineering, LightGBM models, simple and linear baselines, statistical model comparison, calibrated prediction intervals, a linear-programming battery backtest and a seven-view Streamlit dashboard. A 30-step orchestrator (`run_pipeline.py`) runs it end to end, and a unit-test suite on synthetic data runs in CI.

Status: built between February and June 2026 and evaluated as a backtest on January to May 2026. In October 2026 the battery linear programs and the LEAR baseline were fixed (DT-23 in `docs/decisions.md`), and the evaluation protocol was rebuilt so that the test window is only scored, with the reservoir levels aligned to their legal publication deadline and the last hour of each Fingrid forecast to its stated publication time (DT-24); every current figure here comes from that re-measurement (the first-version figures under Limitations are kept for comparison). On 30 October 2026 the quantile bands were also sorted per hour, which changes only the interval figures. It is not a live forecasting service. This repository is a cleaned public release, so the commit history starts at publication.

## The problem

Day-ahead prices are set in a single auction at 12:00 CET on the day before delivery (D-1), so a battery operator or a flexible consumer has to commit before it. Wind, hydro reservoirs, nuclear availability and coupling with Germany, Sweden and Estonia move Nordic prices, and spikes and negative prices are common. Goal: forecast every hour of day D with only the information published before the D-1 auction, beat standard baselines by a margin that holds up statistically, and turn the forecast into a measurable economic result.

## Results

Test window: 1 January to 31 May 2026 (UTC), 3,624 hourly prices per zone. It is not used to train or select the served models, nor to fit their split-conformal calibration: they are fitted on data up to 30 September 2025, and the model choice, the AR decision and that calibration use the October–December 2025 validation quarter (protocol revised after a first look at the test, see [Limitations](#limitations)). The adaptive bands are the exception: as the window advances they recalibrate on realised prices up to D-1 only, as a live system would. Only the walk-forward check below, which evaluates a separate LightGBM, trains on test months. Prices in EUR/MWh.

### Point forecast of the served model

| Zone | Served model (chosen on validation) | MAE | vs persistence 24 h | vs linear ARX |
|---|---|---:|---:|---:|
| DK1 | Diverse ensemble | **15.93** | −40.3% | −29.3% |
| DK2 | Diverse ensemble | **18.92** | −37.8% | −28.0% |
| FI | Diverse ensemble | **21.45** | −37.6% | −25.2% |

| Zone | RMSE | Mean bias (forecast − actual) | Directional accuracy | R² | rMAE |
|---|---:|---:|---:|---:|---:|
| DK1 | 21.40 | −11.4 | 83.9% | 0.79 | 0.597 |
| DK2 | 28.89 | −14.2 | 84.0% | 0.69 | 0.622 |
| FI | 37.57 | −10.4 | 73.1% | 0.72 | 0.624 |

The diverse ensemble averages three LightGBM models trained with L2, median and Huber objectives. Directional accuracy is the share of hours where the forecast gets the sign of the hour-to-hour price change right; rMAE is the model MAE divided by the persistence MAE. The mean test-window price was 96.6 (DK1), 99.8 (DK2) and 75.9 (FI), with standard deviations of 46.7, 52.0 and 70.7, so the MAE is 16.5%, 19.0% and 28.3% of the mean price. Test prices sat above the training level (mean of 79.1, 77.5 and 47.8 from January 2023 to September 2025), and the forecast is on average 10–14 EUR/MWh too low.

<img alt="Detailed Forecast view of the dashboard for DK1, 19 May to 1 June 2026: actual price and forecast with P10–P90 and P25–P75 bands, followed by the hourly forecast table." src="docs/img/forecast_dk1.png" width="760">

*Detailed Forecast view, DK1, the last 14 days of the test window, with the P10–P90 and P25–P75 bands. The forecast stays below most evening peaks.*

### Baselines (same test window)

| Baseline | DK1 MAE / RMSE / DirAcc / R² | DK2 MAE / RMSE / DirAcc / R² | FI MAE / RMSE / DirAcc / R² |
|---|---|---|---|
| Persistence (same hour, 24 h earlier) | 26.69 / 38.82 / 78.2% / 0.31 | 30.42 / 45.65 / 79.6% / 0.23 | 34.38 / 52.43 / 67.5% / 0.45 |
| Weekly seasonal mean (same hour, last 4 weeks) | 29.07 / 38.88 / 80.6% / 0.31 | 33.78 / 45.25 / 81.6% / 0.24 | 52.07 / 69.70 / 68.0% / 0.03 |
| Linear ARX (LEAR-style, ridge, zone-aware) | 22.54 / 29.15 / 80.1% / 0.61 | 26.27 / 35.02 / 80.2% / 0.55 | 28.70 / 42.12 / 70.6% / 0.64 |

The LEAR-style baseline is a ridge regression (alpha 0.1) on price lags of 24, 48 and 168 to 174 hours, cyclical hour, weekday and month terms, a weekend flag, and the temperature and wind forecast of its own zone. It does not include the daily LASSO recalibration of the original LEAR. Its exogenous inputs are deliberately minimal, so part of the gap to the LightGBM models comes from the extra inputs, not only from the model.

### Model choice and robustness

- **Choice on validation.** Rule written before scoring: the lower validation MAE wins, and LightGBM is served only if it is also significantly better. The ensemble won in all three zones (12.08 vs 15.77 in DK1, 13.20 vs 15.30 in DK2, 16.94 vs 24.00 in FI; Diebold-Mariano −3.67, −2.31 and −6.01, p = 0.0003, 0.021 and < 0.0001). The day-ahead AR correction is added only where it lowers the out-of-sample validation MAE, and it raised it in all three zones (DK1 11.18 → 11.41, DK2 13.29 → 13.39, FI 15.41 → 16.70), so no served model uses it. Per-zone detail in [docs/details.md](docs/details.md#validation-decisions-by-zone).
- **Diebold-Mariano on the test window (descriptive).** HAC variance (Newey-West, h = 24) with the Harvey-Leybourne-Newbold correction; DM < 0 means the served model is better. Against LEAR: −9.36 (DK1), −9.25 (DK2) and −8.18 (FI), all p < 0.0001. Against the 2,000-tree LightGBM fitted on train plus validation: +2.98 in DK1 (p = 0.003) and +3.45 in DK2 (p = 0.0006), where that LightGBM is better (MAE 15.23 and 17.96), and −3.18 in FI (p = 0.0015), where the served model is better.
- **Walk-forward validation.** A single LightGBM (500 trees, not the served model), expanding window, retrained every month, 24 monthly folds from June 2024 to May 2026. Mean monthly MAE: 14.60 (DK1), 15.50 (DK2) and 22.45 (FI), with monthly values from 8.6 to 22.7 in DK1, from 11.3 to 20.8 in DK2 and from 9.9 to 35.1 in FI, so the LightGBM's skill is not a one-season effect.
- **Prediction intervals (P10–P90, target 80%).** Quantile LightGBM models fitted on train with early stopping on validation cover 57.7% (DK1), 49.2% (DK2) and 78.2% (FI) of the test hours. The split-conformal calibrator fitted on validation raises that to 62.7%, 60.8% and 81.5%. In Denmark the calibration learnt on validation is too narrow for the test window; prices there sat higher than on validation (mean 96.6 and 99.8 against 84.4 and 85.9), but Finnish prices rose more (75.9 against 44.3) and its bands held, so the cause is not established. In Finland the raw bands already came close to the target. Adaptive conformal inference (Gibbs and Candès, 2021), with the band multiplier updated once per delivery day from errors up to D-1 and a fixed step of 0.03, reaches 79.3%, 79.3% and 79.6%, with bands of 53, 67 and 76 EUR/MWh on average.
- **Quantile order.** The five quantile models are fitted separately, and after the conformal step one quantile lay above the next in 475 (DK1), 442 (DK2) and 755 (FI) of the 3,624 test hours (for example P75 above P90). Since 30 October 2026 the served bands are sorted in each hour after the conformal step (monotone rearrangement, `sort_quantile_bands` in `src/model/conformal.py`), so none cross; the point forecast does not change. Before sorting, the split-conformal coverage was 62.5%, 60.5% and 80.5%, and the pinball loss 5.47, 7.10 and 8.10. The ACI figures use the P10 and P90 before the conformal step, which the sorting does not touch, so they do not change.
- **Proper scoring.** Pinball loss of 5.46, 7.09 and 8.07 EUR/MWh, and CRPS (approximated with 5 quantiles) of 10.92, 14.17 and 16.14, on the served (sorted) bands.

<img alt="Transparency view of the dashboard for DK1: model MAE, rMAE against the 24-hour naive forecast, naive and LEAR MAE, Diebold-Mariano tests, band coverage, monthly walk-forward MAE against the naive forecast, and error by price regime." src="docs/img/transparency.png" width="760">

*Transparency view, DK1. The page recomputes its numbers from the test predictions when it loads. It starts on 2 January (3,600 hours) because the first day is needed for the 24-hour lags, so its figures differ slightly from the tables above.*

### Economic value: battery arbitrage backtest

Setup: a 1 MW / 2 MWh battery with 90% round-trip efficiency. Each day, a linear program sets the charge and discharge schedule from a forecast, with charge plus discharge capped at the 1 MW rating in every hour, and the result is settled at the actual prices. The same program runs on the served forecast, on 24-hour persistence and on the actual prices (perfect foresight), and a fixed schedule (charge 00:00–02:00, discharge 17:00–19:00, Copenhagen time) is shown for reference. Figures are annualised from the 149 complete days of the test window (winter and spring), so they are not a full-year estimate.

| Zone | Served forecast (EUR/yr) | Share of perfect foresight | Persistence forecast (EUR/yr) | Share of perfect foresight | Perfect foresight (EUR/yr) | Fixed schedule (EUR/yr) |
|---|---:|---:|---:|---:|---:|---:|
| DK1 | 70,007 | **95%** | 61,843 | 84% | 74,033 | 8,661 |
| DK2 | 77,813 | **94%** | 67,344 | 82% | 82,349 | 16,955 |
| FI | 42,152 | **78%** | 31,395 | 58% | 54,065 | 22,017 |

The fixed schedule is a weak reference; the value of the model is the gap to the persistence-driven battery: 8,164 (DK1), 10,469 (DK2) and 10,757 (FI) EUR per year for this battery. The interactive Value Calculator page uses the same served forecast with its own default parameters, so its numbers differ slightly.

### Battery Lab: reserves plus arbitrage (Finland)

`pages/5_Battery_Lab.py` co-optimises day-ahead arbitrage with the Nordic frequency-containment reserves (FCR-N, FCR-D) as a linear program, settled against the realised reserve prices published by Fingrid, with datasheet parameters for the commercial systems and a parametric generic battery whose efficiency and cycle life are estimated. In the 2026 test window, with the day-ahead schedule from the served forecast and reserve bids placed without knowing the reserve prices, a generic 1 MW / 2 MWh battery earns 70.7 thousand EUR per MW and year (gross, annualised), 87% of what it earns with reserve prices known in advance; day-ahead arbitrage alone gives 39.7. The perfect-foresight ceiling fell from 461 thousand EUR per MW in 2023 to 96 in 2026 (January to May, annualised) as reserve prices fell, while more batteries joined the Finnish reserve markets (the cause is not measured here). Figures for a commercial system and the method are in [docs/details.md](docs/details.md#battery-lab-reserves-plus-arbitrage-finland).

<img alt="Battery Lab view for Finland with the generic 2-hour battery: its parameters (datasheet or estimated), gross revenue per MW and year from 2023 to 2026 split into FCR-N, FCR-D and day-ahead arbitrage, and the test-window result of the causal bidding strategy against the oracle ceiling." src="docs/img/battery_lab_fi.png" width="760">

Every result above is written to a JSON file by the code, except the figures inside the screenshots, which the dashboard computes when it loads; model settings quoted in the text, such as the LEAR lags or the fixed battery schedule, are in the code. The list of files is in [docs/details.md](docs/details.md#where-each-number-comes-from).

## How it works

```
ENTSO-E Transparency ─┐
Energi Data Service ──┤   src/extract/   src/transform/          src/model/
Fingrid Open Data ────┼─► 01-11 ───────► cleaning, integration ─► features ─► train ─► evaluate ─► predict
Open-Meteo ───────────┤                  (hourly master table,     │           │          │
Yahoo Finance (gas,   ┘                   Jan 2023 - May 2026)     │           │          └─► app.py + pages/
 CO2, via yfinance)                                                │           └─► choice, AR and calibration
                                                                   │               on validation; test scoring,
                                                                   │               DM test, battery LP
                                                                   └─► 166 features (DK), 224 (FI)
```

1. **Extraction** (`src/extract/`, pipeline steps 1–7 and 10–13). Day-ahead prices, load and generation (actuals and forecasts), wind and solar forecasts, cross-border flows and capacities, scheduled exchanges, unavailability notices and hydro reservoir levels come from ENTSO-E. They cover the Nordic zones plus the coupled German (DE-LU) market. The Danish TSO data comes from Energi Data Service, the Finnish TSO forecasts and reserve prices from Fingrid, the weather forecasts from Open-Meteo, and gas (TTF) and carbon (EUA) prices from Yahoo Finance.
2. **Integration** (`src/transform/`, steps 8–9). Everything is joined into one hourly table: 29,928 hours from January 2023 to May 2026. Steps 10–13 then add the Finnish coupling, Fingrid and reserve-price columns. Since 1 October 2025 the day-ahead market clears in 15-minute periods, and the pipeline averages any sub-hourly series to hourly.
3. **Features** (`src/model/features.py`, step 14). Price lags and rolling statistics, residual load (load forecast minus wind and solar forecasts) for each zone and for Germany, weather, reservoir anomalies, commodity prices, outages and calendar terms. A fixed rule in `src/model/config.py` gives 166 features in Denmark (without the Finnish coupling block) and 224 in Finland. In Denmark the most important feature is the German residual-load forecast; in Finland it is Fingrid's own residual-load forecast.
4. **Training** (`src/model/train.py`, steps 18–20). Fixed time split: train to 30 September 2025, validation October to December 2025, test January to May 2026. The hyperparameters were chosen once with Optuna (100 trials on validation) and are frozen in `src/model/best_params.json`. Two candidates are fitted on train only: a LightGBM with a fixed 2,000 trees (no early stopping) and the diverse ensemble, whose three members stop early on validation. Validation then picks the served model, decides whether to add the day-ahead AR correction (a small autoregressive model on the residuals up to 23:00 Copenhagen time on D-1 in all three zones, order up to 3 chosen by AIC) and calibrates the bands of the P10, P25, P75 and P90 quantile models. A 2,000-tree LightGBM fitted on train plus validation is kept as a comparator and for the SHAP explanations in the dashboard.
5. **Evaluation** (`src/model/baselines.py`, `src/model/evaluate.py`, steps 15–17 and 21–23). Baselines, the served forecast through the same prediction function as the dashboard, raw comparators, Diebold-Mariano tests, interval coverage, pinball loss and CRPS, and the battery backtest. Steps 24–26 check the day-ahead prediction of each zone, steps 27–29 write `data/metadata/served_metrics_<zone>.json` and step 30 the Finnish regimes.
6. **Dashboard** (`app.py`, `pages/`). Seven views: overview, market analysis, detailed forecast with SHAP explanations, value calculator, technical documentation, Battery Lab, and transparency (monthly error against the baselines).

### Preventing leakage

The model may only use what is known at the D-1 auction:

- Price lags are at least 24 hours, and rolling statistics are computed on the series already shifted by 24 hours. The German day-ahead price clears in the same auction, so it also enters with a 24-hour lag. Lags are counted in hours, so on the 25-hour day at the end of October the 24-hour lag of the last hour falls on the first hour of the same day (see Limitations).
- Load, wind and solar enter as day-ahead forecasts, never as realised values. ENTSO-E requires the load forecast before the auction, but wind and solar forecasts are due by 18:00 on D-1 (Regulation 543/2013) and the API does not store publication times, so their availability at 12:00 is an assumption (measured below). Fingrid's dataset pages give 12:00 EET on D-1 for its load, wind and solar forecasts (11:00 CET, reading EET as Finnish local time) and 10:00 for the Olkiluoto 3 forecast, which is updated from 11:00 with the capacity reserved for maintenance; each covers the Finnish day, which ends one hour after the CET day. The last CET hour of each delivery day (00:00–01:00 Finnish time on D+1) is only published on day D, so in those hours each Fingrid forecast takes the value of the hour before. Summer time is covered under Limitations.
- Weather comes from Open-Meteo's Historical Forecast API, which stitches together the first hours of successive model runs: a forecast, not reanalysis, but with a shorter lead time than at the D-1 auction.
- Gas and carbon closes are lagged by 48 hours (and interconnector utilisation too, when its columns are present), because the D-1 close and the D-1 afternoon flows arrive after the auction. Spreads and scheduled exchanges are lagged by 24 hours; the Finnish schedules are totals that include intraday trades (see Limitations). Congestion flags derived from same-hour prices are excluded.
- Hydro reservoir levels (weekly, stamped at the start of their week) are lagged by 13 days. ENTSO-E may publish a week's value as late as the third working day after that week (Regulation 543/2013, art. 16.2(d)), which is the Friday of the following week when it has two weekday holidays; with 13 days a week's value is first used for the Sunday after that Friday, whose auction closes on Saturday. A one-week lag would have used it from the Monday right after the week. The 4-week change and the anomaly against the seasonal normal use the same lagged level. If 24 December also counted as a non-working day, the Christmas weeks of 2024 and 2025 (training and validation, not the test window) would have their deadline on the following Monday, after that Sunday, so on three delivery days (29–31 December 2024 and 28–30 December 2025) the auction could close before the value was due.
- Thresholds, normalisations and fill values are computed on the training rows only: the REMIT threshold and the percentiles that normalise wind and unavailability, the seasonal normal of the reservoir levels, the fill of missing load forecasts in the residual load and the medians that fill missing features before training. During cleaning, the first 3 hours of every gap in a raw series are filled by linear interpolation towards the next observed value, and the next 3 hours repeat the last filled value; longer gaps stay missing. A filled hour can therefore depend on a later delivery day (see Limitations).
- The AR correction (tested on validation, not used by any served model) for delivery day D uses residuals up to 23:00 Copenhagen time (CET/CEST, the auction's time zone, also for Finland) on D-1, whose prices were set the day before, with steps counted from Copenhagen midnight (23- and 25-hour days included).
- No random shuffling at any point. Validation and walk-forward are strictly chronological, and the served models are only scored on the test window.

`tests/test_features_leakage.py` checks on synthetic data that changing every price and realised flow from 23 hours before a fixed hour onwards leaves the features up to that hour unchanged, that changing any flow or commodity close after 11:00 CET on D-1 leaves the features of D unchanged, and that no same-hour price, spread, flow, capacity, load or congestion column reaches the training matrix; `tests/test_day_ahead_ar.py` checks the same for the AR correction. `tests/test_features_publication.py` checks that the reservoir features of D do not change when any week whose deadline falls after the auction of D changes, that the Fingrid features of D do not change when any later Finnish day changes, and that the reservoir normal and the load-forecast fill use training rows only. `scripts/leakage_sensitivity.py` measures what cannot be fixed without new data (see [Limitations](#limitations)).

## Reproducing the results

Requirements: Python 3.11, a free ENTSO-E API token and, for Finland, a free Fingrid Open Data key. Energi Data Service, Open-Meteo and Yahoo Finance need no key.

```bash
git clone https://github.com/davidsanzdeandino/nordic-power-price-forecast.git
cd nordic-power-price-forecast
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # add ENTSOE_API_KEY and FINGRID_API_KEY
```

How to get the keys:

- ENTSO-E: register at transparency.entsoe.eu and request API access, as described in `.env.example`.
- Fingrid: register at data.fingrid.fi and subscribe to the free Open Data product.

Run the pipeline:

```bash
python run_pipeline.py                 # all 30 steps, with the frozen hyperparameters
python run_pipeline.py --optuna        # same, but searches the hyperparameters again (slow)
python run_pipeline.py --step 3        # a single step (here, the Open-Meteo extraction)
python run_pipeline.py --from 14       # from feature engineering onwards
python run_pipeline.py --zone FI       # shared steps plus the FI steps only
python scripts/leakage_sensitivity.py DK1 DK2 FI
python scripts/build_results.py
python checks/test_revenue_stacking.py --write   # Battery Lab yearly ceiling
python checks/test_battery_lab_data.py --write   # Battery Lab test window
python scripts/make_readme_figures.py            # README bar chart

streamlit run app.py
```

The step list is in [docs/details.md](docs/details.md#pipeline-steps). Without the data and models produced by the pipeline, the dashboard starts in demo mode with simulated values and a clear banner on every page. The Battery Lab and Transparency pages need the master table and do not start without it. The screenshots in this README come from a full local run.

LightGBM runs with fixed seeds (`random_state=42`) and `deterministic=True`; the environment variable `LGBM_N_JOBS` sets the number of threads. The published figures were produced with `LGBM_N_JOBS=4` and `OMP_NUM_THREADS=4`; other thread counts have not been checked to give identical results. The feature table and the trained models are not published, so the figures come from running steps 14 to 30 on downloaded data, and re-downloading can move them slightly because the sources revise past values.

### Tests

```bash
pip install pytest==9.1.1 ruff==0.17.2
ruff check .
pytest
```

`tests/` holds 106 unit tests that run on synthetic data, with no network access and no keys: feature leakage, the publication lag of the reservoir levels and the last CET hour of the Fingrid forecasts, the day-ahead AR correction by delivery day, the Diebold-Mariano test, the training protocol (what each model is fitted and stopped on), an end-to-end run from training to served prediction, the battery linear programs, conformal coverage (including the daily ACI update), the per-hour sorting of the quantile bands, the zone-aware LEAR baseline, the 15-minute to hourly aggregation and the per-zone deduplication of Energi Data Service rows. GitHub Actions runs the same lint and tests on every push and pull request (`.github/workflows/ci.yml`).

`checks/` holds manual checks that need the local data and trained models (Battery Lab data, revenue stacking, the DK2 reserve stack and the render of the Battery Lab and Transparency pages). They print their results and are not collected by pytest; with `--write`, the two Battery Lab checks also update `results/battery_lab_fi.json`.

## Repository structure

```
app.py                      dashboard entry point (sidebar navigation)
pages/                      the seven dashboard views (0_Overview.py is the landing page)
run_pipeline.py             orchestrates the 30 pipeline steps
src/extract/                one script per dataset (11 extractors over 5 sources)
src/transform/              cleaning and integration into the hourly master table
src/model/                  features, training, baselines, AR correction, ensemble, conformal
                            intervals, Diebold-Mariano, evaluation, battery backtest, revenue stacking
src/data_loader.py          data access for the dashboard (demo mode if data is missing)
src/utils/                  logging
scripts/                    served metrics, result files, leakage sensitivity, README figures,
                            and experiments (asinh transform, robust objectives, MLP member, FCR prices)
tests/                      unit tests on synthetic data (pytest, run in CI)
checks/                     manual check scripts (need local data and models)
results/                    validation decisions and test-window results (JSON, written by code)
data/metadata/              served-model metrics (JSON)
data/models/walkforward_*   walk-forward folds (CSV)
docs/details.md             sources of every number, pipeline steps, Battery Lab, Danish price source
docs/decisions.md           design log: every technical decision and its evidence
docs/data_dict.md           data dictionary
docs/roadmap.md             technical roadmap, ranked by evidence
docs/img/                   figures used in this README
.github/workflows/ci.yml    lint and tests on GitHub Actions
pyproject.toml              ruff and pytest configuration
```

`data/raw`, `data/clean`, the master table and the trained models are not included (see the next section). Code comments, most of `docs/decisions.md` and `docs/data_dict.md` are written in Spanish.

## Data sources, licences and attribution

This repository contains code and aggregated results only. No third-party data series are redistributed: the pipeline downloads everything at run time with your own keys.

| Source | Data used | Terms | In this repository |
|---|---|---|---|
| ENTSO-E Transparency Platform (transparency.entsoe.eu) | Day-ahead prices, load, generation, forecasts, flows, capacities, unavailability, reservoir levels | ENTSO-E terms and conditions. Only the datasets on ENTSO-E's list of data available for free re-use are CC BY 4.0 | Code only |
| Energi Data Service, Energinet (energidataservice.dk) | Danish day-ahead prices up to 30 September 2025, balance and settlement data, reserve and imbalance prices | Conditions for use of Danish public-sector data | Code only |
| Fingrid Open Data (data.fingrid.fi) | Finnish load, wind and solar forecasts, reserve and imbalance prices | CC BY 4.0 for datasets produced by Fingrid | Code only |
| Open-Meteo (open-meteo.com) | Historical weather forecasts | CC BY 4.0 data; the free API is for non-commercial use | Code only |
| Yahoo Finance, via yfinance | TTF gas front-month (TTF=F) and a physically backed EUA carbon ETC (CO2.L) | Yahoo terms of service; personal research use only, never redistributed | Code only |

Day-ahead prices originate from the Nordic day-ahead market (Nord Pool and the other power exchanges operating in the Nordic zones) and are obtained through the ENTSO-E Transparency Platform and Energi Data Service. No price series are redistributed here; the screenshots show two weeks of DK1 prices and aggregated figures for illustration only.

Attribution:

- Source: ENTSO-E Transparency Platform (transparency.entsoe.eu). CC BY 4.0 where applicable; data were resampled to hourly resolution and cleaned. ENTSO-E does not endorse this project.
- Contains data used pursuant to "Conditions for use of Danish public-sector data" from the Energi Data Service portal (www.energidataservice.dk). Energinet does not endorse this project.
- Source: Fingrid, data.fingrid.fi, licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Data were resampled to hourly resolution.
- Weather data by Open-Meteo.com (https://open-meteo.com), licence CC BY 4.0.
- Gas and carbon prices are downloaded at run time with yfinance from Yahoo Finance for personal research use and are not included. yfinance is not affiliated with Yahoo.

## Limitations

- **The protocol was revised after the test had been seen.** The first version chose LightGBM or the ensemble per zone with a Diebold-Mariano test on the test window, fitted the split-conformal calibrator on its first half and, in Denmark, applied an AR correction fitted on another model's in-sample residuals. It reported an MAE of 14.54 (DK1), 16.86 (DK2) and 21.05 (FI); the raw LightGBM scored 14.82, 17.14 and 25.85 then (evaluation run of 5 June 2026 on an earlier feature table) and 15.23, 17.96 and 26.49 now. Everything is now decided on validation with rules written before scoring, and the result was kept although it is worse than the first version, but the test window is no longer untouched. The revised protocol was itself run twice: the first run, before the reservoir lag, the Fingrid last-hour fix and the training-only reservoir normal and load fill, gave 15.93, 17.11 and 21.33 with the AR term in DK2 (its result files were overwritten by the second run and are not in the repository); those fixes came from a review of publication times, their effect on the test was not measured separately, and the second run is published as it came out. Some first-version choices that the current model keeps were also judged on the test window, for example adding the Fingrid forecasts to the Finnish model (DT-18). On it, the served ensemble is worse in Denmark than the LightGBM refitted on train plus validation (15.93 vs 15.23 in DK1, 18.92 vs 17.96 in DK2); in DK2 it is also worse than the LightGBM fitted on train only (17.48), which validation rejected. Details and the before-and-after table: DT-24 in `docs/decisions.md`.
- **One validation quarter does several jobs.** October to December 2025 provides early stopping for the ensemble and the quantile models, the model choice, the AR decision and the interval calibration, and the served models are not refitted on it. Optuna also tuned the hyperparameters on this quarter, which falls inside the walk-forward period, so the walk-forward is slightly optimistic for the months before it. The feature normalisations and fill values are computed on train up to September 2025, which also lies after the earlier walk-forward folds.
- **Level shift.** Test prices rose above the training level. The forecast is on average 11.4 (DK1), 14.2 (DK2) and 10.4 (FI) EUR/MWh too low and the evening peaks are under-forecast. In Denmark the split-conformal bands calibrated on validation under-cover (62.7% and 60.8% for a nominal 80%), and only the daily ACI comes close to the target (79%), with wider bands; in Finland both reach about 80%.
- **Inputs whose publication time is not stored.** Measured as the change in the served test MAE (DK1 / DK2 / FI) when each input is replaced by a more conservative version, without retraining (`results/leakage_sensitivity_test2026.json`):
  - Open-Meteo forecasts issued 24 hours before each hour (for the later hours of D, still after the auction) or 48 hours before (always before it): between −0.06 and +0.09 EUR/MWh. The Historical Forecast data is kept because the archive of earlier runs does not cover 2023.
  - REMIT unavailability frozen at its value at 11:00 CET on D-1: +0.02 / +0.02 / −0.03. The notices are aggregated by start and end without filtering by publication time, so this variant can still include notices published later.
  - Wind and solar forecasts of ENTSO-E (Germany and own zones) and Fingrid replaced by those of the previous day: +10.1 / +11.0 / +13.0. This is a pessimistic bound: it shows how much the model relies on those forecasts, not the size of a leak.
  - Fingrid forecasts in summer time: Fingrid gives the publication time of its daily load, wind and solar forecasts as 12:00 EET on D-1. As Finnish local time that is 11:00 CET or CEST; as a fixed UTC+2 it is 12:00 CEST from the last Sunday of March to the last Sunday of October, the minute the auction closes. On those days of the test window (the 1,535 summer-time hours from 29 March; the first delivery day published in summer time is 30 March, so this covers about one day more than needed), replacing them with ENTSO-E's forecasts for Finland changes the FI MAE by +0.08 EUR/MWh, and with Fingrid's forecasts of the previous day by +1.68 (pessimistic bound). The Olkiluoto 3 forecast is updated from 11:00 and the API keeps only the last version; the previous day's forecast changes the MAE by −0.01. Revisions after publication cannot be checked either.
  - Finnish scheduled exchanges (Finland with Estonia, SE1 and SE3) are ENTSO-E total schedules, which include intraday trades (the Danish ones are day-ahead schedules). For the hours of D-1 after the auction, the 24-hour lag can include trades made after it. Using the schedule of D-2 for the hours from 12:00 CET: +0.01 EUR/MWh in FI (+0.001 with D-2 in every hour).
  - These last two items and the gap-filling point below were measured separately with the same path as `scripts/leakage_sensitivity.py` (copy of the master table, variant, features, served models, no retraining) and are not in its output file.
- **Gap filling in cleaning.** When a raw series has a gap, its first 6 filled hours depend on the next observation (DT-05). In 165 cells of the master table that observation belongs to a later delivery day, so those hours use a forecast published after their own auction. All of them are ENTSO-E wind and solar forecasts for DK1 and DK2: the first six hours of days missing in the source, plus one 3-hour gap before midnight on 9 April 2025. 36 cells are in the test window (14 April and 7 May 2026). Replacing them with the last observation before the gap changes the MAE by +0.004 (DK1), +0.008 (DK2) and less than +0.001 (FI) EUR/MWh. Prices, load forecasts and weather have no such case. The remaining hours of those days stay missing, and the residual load counts them as zero wind and solar.
- **25-hour days.** Lags are counted in hours, so on the last Sunday of October the 24-hour lag of the hour from 23:00 to 24:00 CET is the price of the first hour of the same day, set in the same auction (also for spreads, scheduled exchanges and rolling statistics). It affects one hour a year: two in train, one in validation (26 October 2025) and none in the test window.
- **Short test window.** Five months, winter and spring only. The literature recommends at least a year. The 24-month walk-forward covers every season, but it evaluates a single LightGBM, not the served model.
- **Spikes dominate the error.** RMSE is well above MAE (37.6 vs 21.5 in Finland, 21.4 vs 15.9 in DK1). Extreme hours are hard to predict.
- **Hourly target and proxy inputs.** The model forecasts the hourly average, while the day-ahead market has cleared in 15-minute periods since October 2025. The carbon price is a physically backed ETC, not the ICE EUA settlement, and gas is the TTF front-month future.
- **Danish prices after 30 September 2025 come from ENTSO-E only**, without a second source to cross-check (details in [docs/details.md](docs/details.md#danish-prices-and-the-retired-energi-data-service-dataset) and DT-23).
- **Backtest, not operations.** The battery results assume a price-taking battery, perfect execution and no intraday or balancing trading. Degradation and fees are only modelled in the dashboard calculators and the Battery Lab.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The licence covers the code in this repository only. Data downloaded by the pipeline stays under each provider's terms, listed above.

## Author

David Sanz de Andino Díez, Telecommunications Engineering student at Universidad Carlos III de Madrid. [LinkedIn](https://www.linkedin.com/in/davidsanzdeandino) · [GitHub](https://github.com/davidsanzdeandino)
