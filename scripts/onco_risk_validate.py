#!/usr/bin/env python3
"""Валидация и обзор базы знаний онкориска (data/onco_risk/*.yaml).

Запуск:
    python3 scripts/onco_risk_validate.py

Проверяет целостность (источники, LR/PPV, страты, baselines, пороги, B2C без
«страшных» слов) и печатает обзор: что готово и какие priors ещё approx (кандидаты
на рекалибровку по CI5/РНПЦ - Фаза 4). Код выхода != 0 при ошибках целостности.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_knowledge import onco_risk as orisk  # noqa: E402

FORBIDDEN = ["рак", "онколог", "злокачествен", "опухол", "метастаз", "карцином", "саркома", "меланом"]


def _features():
    d = orisk._lr_data()
    return list(d.get("features") or []) + list(d.get("lab_features") or [])


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    feats = _features()
    prior_sites = orisk._priors().get("sites") or {}

    for f in feats:
        fid = f.get("id") or "?"
        if not f.get("source"):
            errors.append(f"feature {fid}: нет source")
        if not (f.get("lr") or f.get("ppv_single")):
            errors.append(f"feature {fid}: нет lr/ppv")
        if not f.get("keywords"):
            errors.append(f"feature {fid}: нет keywords")
        site = f.get("cancer_site")
        if site and site != "unknown" and site not in prior_sites:
            errors.append(f"feature {fid}: сайт '{site}' отсутствует в priors")
        s = str((f.get("strata") or {}).get("sex") or "any")
        if s not in ("any", "male", "female"):
            errors.append(f"feature {fid}: некорректная strata.sex={s}")

    approx_sites = []
    for site, cfg in prior_sites.items():
        b = cfg.get("baseline_symptomatic")
        if b is None:
            errors.append(f"prior {site}: нет baseline_symptomatic")
            continue
        if not (0 < float(b) < 0.5):
            errors.append(f"prior {site}: baseline вне диапазона ({b})")
        src = cfg.get("baseline_source") or {}
        if "approx" in str(src).lower() or "TODO" in str(src):
            approx_sites.append(site)

    th = orisk._thresholds()
    if abs(float(th.get("referral_ppv_threshold", 0)) - 0.03) > 1e-9:
        errors.append("threshold: referral_ppv_threshold != 0.03")

    cfg = orisk._b2c()
    buckets: list[str] = []
    for k in ("by_feature", "by_context", "intake_hints_b2c"):
        for _, arr in (cfg.get(k) or {}).items():
            buckets.extend(arr or [])
    buckets.extend(cfg.get("safety_netting") or [])
    for q in buckets:
        low = q.lower()
        if any(w in low for w in FORBIDDEN) or "%" in q:
            errors.append(f"B2C: недопустимый шаблон: {q}")

    w = orisk._required().get("weights") or {}
    if abs(sum(w.values()) - 1.0) > 1e-6:
        errors.append(f"required_inputs: сумма весов != 1.0 ({sum(w.values())})")

    if approx_sites:
        warnings.append(
            "priors approx (кандидаты на рекалибровку CI5/РНПЦ, Фаза 4): "
            + ", ".join(sorted(approx_sites))
        )
    if orisk._screening().get("confirm_rb"):
        warnings.append("screening_belarus.yaml: возрасты помечены confirm_rb (подтвердить по МЗ РБ)")
    age_p = orisk._age_priors()
    if age_p.get("sites"):
        warnings.append(
            "priors_age_belarus.yaml активен: возрастные baselines для "
            + ", ".join(sorted((age_p.get("sites") or {}).keys()))
        )
    else:
        warnings.append(
            "priors_age_belarus.yaml отсутствует: используются общие baselines "
            "(запустите scripts/onco_priors_recalibrate.py с данными CI5/РНПЦ)"
        )

    print("=== Онкориск: обзор базы знаний ===")
    print(f"признаков (симптомы+анализы): {len(feats)}")
    print(f"локализаций в priors: {len(prior_sites)}")
    print(f"B2C-шаблонов проверено: {len(buckets)}")
    print()
    for wn in warnings:
        print("[warn]", wn)
    if errors:
        print()
        for e in errors:
            print("[ERROR]", e)
        print(f"\nПРОВАЛ: {len(errors)} ошибок целостности.")
        return 1
    print("\nOK: база знаний целостна.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
