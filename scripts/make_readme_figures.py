"""
make_readme_figures.py — Gráfico «MAE by zone» del README (claro y oscuro)
==========================================================================
Lee results/evaluation_test2026.json (scripts/build_results.py) y dibuja
docs/img/mae_by_zone_light.png y docs/img/mae_by_zone_dark.png: MAE del modelo
servido, del LEAR y de la persistencia de 24 h en el test, por zona.

Uso:  python scripts/make_readme_figures.py
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "results" / "evaluation_test2026.json"
OUT_DIR = ROOT / "docs" / "img"
ZONES = ["DK1", "DK2", "FI"]

THEMES = {
    "light": {"bg": "#FCFCFB", "fg": "#141413", "muted": "#5E5D59", "grid": "#E3E2DE",
              "served": "#2B78D6", "lear": "#8A8984", "persist": "#C2C1B6"},
    "dark":  {"bg": "#1A1A19", "fg": "#F0EFEA", "muted": "#B5B3AC", "grid": "#3A3936",
              "served": "#4C93E8", "lear": "#8A8984", "persist": "#5C5B57"},
}


def draw(values: dict, n_hours: int, theme: str) -> Path:
    t = THEMES[theme]
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(12, 6), dpi=100, facecolor=t["bg"])
    ax = fig.add_axes([0.075, 0.2, 0.9, 0.58], facecolor=t["bg"])
    series = [("served", "Served model"), ("lear", "Linear ARX (LEAR-style)"),
              ("persist", "24-hour persistence")]
    width, gap = 0.28, 0.035
    ymax = max(max(v.values()) for v in values.values())
    for i, z in enumerate(ZONES):
        for j, (key, _) in enumerate(series):
            x = i * 3.0 + (j - 1) * (width + gap)
            v = values[z][key]
            ax.bar(x, v, width=width, color=t[key], zorder=3)
            ax.text(x, v + ymax * 0.025, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=11 if key == "served" else 10,
                    fontweight="bold" if key == "served" else "normal",
                    color=t["fg"] if key == "served" else t["muted"])
        chg = (values[z]["served"] - values[z]["persist"]) / values[z]["persist"] * 100
        ax.text(i * 3.0, -ymax * 0.09, z, ha="center", va="top", fontsize=12,
                fontweight="bold", color=t["fg"])
        ax.text(i * 3.0, -ymax * 0.165, f"{chg:+.1f}% vs persistence".replace("-", "−"),
                ha="center", va="top", fontsize=10.5, color=t["muted"])
    ax.set_xlim(-1.5, 7.5)
    top = 10 * (int(ymax // 10) + 1)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 10))
    ax.tick_params(axis="y", colors=t["muted"], labelsize=10, length=0)
    ax.set_xticks([])
    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.text(0.075, 0.93, "Mean absolute error by zone, test January–May 2026",
             fontsize=15, fontweight="bold", color=t["fg"])
    fig.text(0.075, 0.875, f"EUR/MWh, lower is better. {n_hours:,} hours per zone. Models "
             "chosen and calibrated on Oct–Dec 2025 validation; protocol revised after "
             "seeing the test.", fontsize=11, color=t["muted"])
    x = 0.075
    for key, label in series:
        fig.patches.append(plt.Rectangle((x, 0.818), 0.011, 0.022, color=t[key],
                                         transform=fig.transFigure, figure=fig))
        fig.text(x + 0.017, 0.82, label, fontsize=10.5, color=t["muted"])
        x += 0.13 if key == "served" else 0.2

    out = OUT_DIR / f"mae_by_zone_{theme}.png"
    fig.savefig(out, dpi=100, facecolor=t["bg"])
    plt.close(fig)
    return out


def main():
    ev = json.load(open(EVAL_PATH, encoding="utf-8"))
    values, n_hours = {}, None
    for z in ZONES:
        zz = ev["zones"][z]
        values[z] = {"served": zz["served_model"]["MAE"],
                     "lear": zz["served_model_vs_baselines"]["MAE_lear"],
                     "persist": zz["served_model_vs_baselines"]["MAE_persistence_24h"]}
        n_hours = zz["served_model"]["n_hours"]
    for theme in THEMES:
        print("OK ->", draw(values, n_hours, theme).relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
