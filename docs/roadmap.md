# Technical Roadmap — evidence-ranked (June 2026)

Every item cites the evidence that earned its priority. Effort: S < 1 day · M = days · L = weeks.
The guiding rule stays the project's own (DT-14): **measure before committing,
and report the measured number even when it is smaller.**

## Where the project stands

Day-ahead forecasting DK1/DK2/FI with out-of-sample test-2026 metrics (MAE 14.5 /
16.9 / 21.05 €/MWh; rMAE 0.54/0.55/0.61), conformal-ACI bands, zone-aware LEAR
baselines, DM-test model selection — plus a Battery Lab that stacks FCR-N/FCR-D
with DA arbitrage on realised TSO reserve prices and reproduces published
market benchmarks without calibration. It also covers three things that are
rarely available in the open: a Finnish BESS backtester on real Fingrid
prices, an FCR forecastability study (DT-21), and a post-2021 zonal benchmark
with a published protocol.

## Tier 1 — highest value per effort

1. **15-minute MTU upgrade path.** (M–L)
   SDAC has cleared at 15-min since delivery day 1 Oct 2025; FI imbalance is
   15-min since 19 Mar 2025. Our hourly aggregation is documented but now
   models a coarsened market. Plan: dual-resolution master (hourly history +
   15-min tail), keep the hourly model for level and add a 15-min *shape*
   model on post-Oct-2025 data; Battery Lab gains intra-hour arbitrage
   (quarter-hour spreads are where solar-driven DK1 troughs bite).

2. **Quantile forecasts of fundamentals as features.** (M)
   The highest-evidence accuracy add-on in the 2025-26 EPF literature: up to
   13-16% RMSE gains in Germany from feeding load/wind/solar *quantiles*
   instead of points (Uniejewski & Ziel 2025). Source: Open-Meteo Ensemble API
   (live) + MET Norway MEPS 2.5 km ensemble archive (as-issued forecasts
   2016→, no auth — the gold-standard leakage-free archive) + ECMWF Open Data
   ENS. Measure with the project's standard walk-forward protocol.

3. **aFRR + mFRR stacking extension in the Battery Lab.** (M)
   The realised prices are already in the master (Fingrid 52/329 series;
   Energinet AfrrReservesNordic/mFRRCapacityMarket). Post-FCR-saturation,
   this is where Finnish BESS revenue lives (Capalo AI reports €19-28k/MW/month
   for its Zeus VPP in Q2-2025, trading the full stack:
   https://capaloai.com/blog/capalo-zeus-vpp-quarterly-performance-q2-2025/ ;
   Clean Horizon expects about 2/3 of a 2h battery's long-term revenue to come
   from energy trading: https://www.cleanhorizon.com/news/finland-price-forecast-s1-2025-updated/ ).
   Model capacity markets first (simpler: reservation + price), energy
   activation as scenario analysis — and adopt an explicit realization haircut
   (Modo Energy applies an 80% calibration factor to its German benchmark:
   https://modoenergy.com/research/en/me-bess-de-1h-2h-4h-germany-battery-bess-benchmark-live-terminal-virtual-asset-simulation-may-2026 ).

4. **Hydro + outage data upgrades.** (S–M)
   NVE magasinstatistikk (weekly NO reservoir per zone since 1995, JSON, no
   auth — verified) + Energiföretagen weekly SE Excel replace/extend the
   thin ENTSO-E weekly reservoir series; NUCS UMM API (no auth, verified,
   111k messages) gives real-time outage shocks better than ENTSO-E REMIT.
   Hydro is *the* Nordic price driver; cheap wins for FI/DK2 accuracy.

5. **Probabilistic upgrade: iQRA / QRA+CP+IDR ensemble.** (M)
   The 2025-26 literature shows isotonic QRA and QRA+conformal+IDR ensembles
   beating plain conformal and DDNNs on CRPS/reliability (arXiv 2507.15079,
   2404.02270). We already log pinball/CRPS — re-fit bands, compare to the
   current ACI under the same protocol, keep the winner.

## Tier 2 — research-grade

6. **Chronos-2 covariate-mode challenger.** (S–M)
   The only TSFM with credible evidence of beating tuned ML on day-ahead
   (~5%, Belgian study, May 2026). Run as challenger under our protocol;
   document either outcome — beating it strengthens our claim, and losing to
   it is also a finding.

7. **Imbalance-exposure analysis FI (15-min).** (M)
   Post-reform FI imbalance prices hit +5,000/−10,000 €/MWh and mFRR sets
   the price 48% of the time. Frame as exposure management (passive
   balancing is only formalized in NL/BE/GB) using datasets 319 + 369 —
   a differentiated risk-analytics piece, not a trading pitch.

## Tier 3 — engineering hardening

8. **Tests + CI.** (S–M) pytest for: leakage gates (max same-timestamp
   correlation), LP invariants (causal ≤ oracle; reserve headroom respected),
   extractor schemas. GitHub Actions on push.
9. **Live-price fallback for the dashboard.** (S) Elering API (FI/Baltic,
   no auth, history to 2015 — verified) + energy-charts (DK1, CC-BY) as
   keyless fallbacks so the public dashboard never shows stale data.
10. **SE/NO reserve panel.** (M) Svenska kraftnät Mimer CSV + Statnett
    driftsdata (both verified, no auth) → a Nordic reserve-price atlas page;
    contextualizes the FI/DK2 stacking story across the whole synchronous
    area.

## Structural-break watchlist (model must flag these)

| Date | Event | Affects |
|---|---|---|
| 2022-09-07 | DK1 FCR joins German auction — local prices collapse | DK1 reserve features |
| 2023-04-16 | OL3 commercial operation (−75% FI mean price) | FI level (flag exists) |
| 2024-06-13 | SIDC intraday auctions IDA1-3 launch | ID liquidity |
| 2024-10-29/30 | Nordic flow-based DA coupling go-live | cross-zonal features, DK1↔SE4 capacity |
| 2025-02-08/09 | Baltic synchronization with Continental Europe | FI/Baltic spreads, reserve needs |
| 2025-03-04 | Nordic mFRR EAM (15-min, automated) go-live | balancing prices |
| 2025-03-18/19 | 15-min imbalance settlement + pricing reform (FI & DK) | imbalance series resolution/volatility |
| 2025-03-26/27 | Fingrid joins PICASSO | aFRR energy prices |
| 2025-10-01 | SDAC 15-min MTU first delivery | day-ahead target resolution |
| 2026-01-01 | Energinet HV power-based tariff (BESS: 59,187 DKK/MW/yr + spot-indexed loss tariff, replaces volumetric) | DK BESS economics — tailwind |
| 2026-01-01 | Fingrid fees +8%; FCR-N yearly market discontinued (hourly only) | FI FCR price regime |
| 2026-03/05 | Fingrid balance fee 1.05 → 0.65 €/MWh | FI imbalance costs |
| 2026-02/03 | Energinet abolishes first-come-first-served; connection pause (~60 GW queue) | DK demand growth path |
| 2026-07-01 | FI datacenter electricity tax II→I (+2.19 c/kWh) | FI demand trajectory |
| 2026-11 | EU co-optimization R&D conclusion (ACER 11/2024; go-live realistically post-2028) | future DA↔reserve coupling |
| 2026-Q4 | Fingrid independent aggregation in mFRR | FI mFRR liquidity |
| 2026-2027 | Skagerrak 1-2 (500 MW NO2-DK1) end-of-life, renewal politically contested; Konti-Skan excluded from SvK plan | DK1 volatility ↑, NO2-DK1 coupling ↓ |
| 2027-01 | DK offshore CfD awards (North Sea Mid, Hesselø — 2-sided capability-based) | DK RES pipeline 2032+ |
| 2027-Q1 | Nordic MARI accession (contingent target) | mFRR/imbalance prices, all zones |
| 2027-07-17 | EU two-way CfD obligation for new support schemes | SE/DK/FI subsidy design |
| 2028-01-01 | Fingrid tariff structure reform (15-min billing, power-based; decision summer 2026) | FI BESS cost stack |
| ~2028 | Statkraft-OX2 235 MW + Alpiq/Luxcara FI BESS fleet additions | FI reserve saturation deepens |
| undated | SE/NO PICASSO accession; Nordic FB intraday; mFRR CM cross-zonal reservation | aFRR/ID/balancing features |

Bidding-zone review: effectively closed for the Nordics — the TSO study (Apr 2025)
found all alternatives welfare-negative and current zones are a safe model
assumption through ≥2028.

## Explicitly deprioritized (with reasons)

- **Deep learning / TSFM as production model** — Norway 5-zone 2026 study:
  LightGBM beat all DL in every zone; zero-shot TSFMs lose to MSTL baseline.
  Keep as challengers only.
- **RL for battery dispatch** — no published 2023-26 European backtest shows
  RL beating MILP-on-forecasts; industry consensus is MILP/MPC.
- **Chasing MAE for arbitrage value** — capture is driven by price *rank*
  fidelity (Kendall τ), not MAE; our 92-93% DA capture is already in the
  τ≈0.9 band. Marginal MAE gains buy little arbitrage profit.
- **More feature pruning / VST / per-hour models** — already measured and
  rejected (DT-14/15).
