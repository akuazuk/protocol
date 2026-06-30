#!/usr/bin/env python3
"""Демо-визуализация онкориска: evidence waterfall + Beta доверительный интервал.

Запуск:
    python3 scripts/onco_risk_demo.py

Сохраняет PNG в output/onco_risk/. Числа - советующие (decision-support), не диагноз.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from clinical_knowledge.onco_risk import OncoInputs, assess, SiteRisk  # noqa: E402

OUT_DIR = ROOT / "output" / "onco_risk"

CASE = OncoInputs(
    text="Жалобы: ректальное кровотечение, потеря веса за 3 месяца, боль в животе.",
    age=62, sex="male", symptom_duration_known=True,
    labs_text="кал на скрытую кровь положительный",
)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def plot_waterfall(site: SiteRisk, path: Path) -> None:
    labels = ["Базовый риск"] + [c.label_ru for c in site.contributors]
    values = [site.contributors[0].p_before] + [c.p_after for c in site.contributors]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = range(len(values))
    ax.plot(xs, [v * 100 for v in values], "-o", color="#2563eb", linewidth=2)
    for x, v, c in zip(xs, values, [None] + site.contributors):
        ax.annotate(_pct(v), (x, v * 100), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
        if c is not None:
            ax.annotate(f"LR x{c.lr_effective}", (x, v * 100),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=8, color="#6b7280")
    ax.axhline(3.0, color="#dc2626", linestyle="--", linewidth=1)
    ax.annotate("порог направления 3% (NICE NG12)", (0, 3.0),
                textcoords="offset points", xytext=(5, 5), color="#dc2626", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Пост-тест вероятность, %")
    ax.set_title(f"Накопление доказательств: {site.site} (советующая оценка, не диагноз)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_beta_ci(site: SiteRisk, completeness: float, path: Path) -> None:
    import numpy as np
    from scipy.stats import beta as beta_dist
    kappa = 25.0 * (0.4 + 0.6 * completeness)
    a = max(site.p * kappa, 1e-3)
    b = max((1 - site.p) * kappa, 1e-3)
    xs = np.linspace(0, min(site.ci_high * 1.6 + 0.05, 1.0), 400)
    ys = beta_dist.pdf(xs, a, b)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs * 100, ys, color="#2563eb", linewidth=2)
    ax.fill_between(xs * 100, ys, where=(xs >= site.ci_low) & (xs <= site.ci_high),
                    color="#93c5fd", alpha=0.5,
                    label=f"95% интервал {_pct(site.ci_low)}-{_pct(site.ci_high)}")
    ax.axvline(site.p * 100, color="#1d4ed8", linestyle="-", linewidth=1.5,
               label=f"оценка {_pct(site.p)}")
    ax.axvline(3.0, color="#dc2626", linestyle="--", linewidth=1, label="порог 3%")
    ax.set_xlabel("Вероятность, %")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Неопределённость оценки: {site.site} (полнота данных {completeness:.0%})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = assess(CASE)
    print(f"context={a.context} band={a.completeness.band} score={a.completeness.score}")
    print(f"triage={a.triage_level} any_cancer={_pct(a.any_cancer_p)} CI={a.any_cancer_ci}")
    if not a.sites:
        print("Нет количественной оценки для этого кейса.")
        return
    top = a.sites[0]
    wf = OUT_DIR / "waterfall_colorectal.png"
    ci = OUT_DIR / "beta_ci_colorectal.png"
    plot_waterfall(top, wf)
    plot_beta_ci(top, a.completeness.score, ci)
    print(f"saved: {wf}")
    print(f"saved: {ci}")
    print("\nB2C вопросы (нейтральные, без чисел):")
    for q in a.b2c_questions:
        print("  -", q)


if __name__ == "__main__":
    main()
