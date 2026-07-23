#!/usr/bin/env python3
"""Seed возраст-специфичных правил назначения: STOPP/START v3 + Beers (AGS 2023).

Используются детектором dose/drug-disease (§6.2) и осью C (safety) для пожилых:
- STOPP: потенциально нежелательные назначения у пожилых (≥65);
- Beers: аналогично (США, AGS 2023);
- START: пропущенные показанные назначения (омиссия).

Правило: {rule_id, source, kind, atc/inn, min_age, condition, text, severity}.
kind: "avoid" (STOPP/Beers) | "consider" (START).
condition - опц. состояние/диагноз-модификатор (МКБ-глава или ключевое слово).

Выход: data/drug_safety/stopp_start_beers.json
Usage: python3 scripts/build_stopp_start_beers.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "drug_safety" / "stopp_start_beers.json"

RULES: list[dict] = [
    # --- STOPP / Beers: avoid у пожилых ---
    {"rule_id": "STOPP_benzodiazepine_elderly", "source": "STOPP v3", "kind": "avoid",
     "atc": "N05BA", "inn": ["diazepam", "lorazepam", "alprazolam", "clonazepam"],
     "min_age": 65, "condition": None, "severity": "moderate",
     "ru": ["диазепам", "феназепам", "алпразолам", "клоназепам", "лоразепам"],
     "text": "Бензодиазепины у пожилых: риск падений, спутанности, зависимости."},
    {"rule_id": "STOPP_nsaid_elderly_ckd", "source": "STOPP v3", "kind": "avoid",
     "atc": "M01A", "inn": ["diclofenac", "ibuprofen", "ketorolac", "naproxen", "meloxicam"],
     "min_age": 65, "condition": "ckd_or_htn_or_hf", "severity": "high",
     "ru": ["диклофенак", "ибупрофен", "кеторолак", "напроксен", "мелоксикам", "нпвп"],
     "text": "НПВП у пожилых с ХБП/АГ/СН: риск ОПП, задержки жидкости, кровотечения."},
    {"rule_id": "Beers_glibenclamide_elderly", "source": "Beers 2023", "kind": "avoid",
     "atc": "A10BB01", "inn": ["glibenclamide", "glyburide"],
     "min_age": 65, "condition": None, "severity": "high",
     "ru": ["глибенкламид", "манинил"],
     "text": "Глибенкламид у пожилых: длительная гипогликемия; предпочесть короткие ПСМ/др. классы."},
    {"rule_id": "Beers_anticholinergic_elderly", "source": "Beers 2023", "kind": "avoid",
     "atc": "R06A", "inn": ["diphenhydramine", "chlorphenamine", "hydroxyzine"],
     "min_age": 65, "condition": None, "severity": "moderate",
     "ru": ["димедрол", "дифенгидрамин", "хлоропирамин", "супрастин", "гидроксизин", "атаракс"],
     "text": "Антихолинергические H1 1-го поколения у пожилых: спутанность, задержка мочи, падения."},
    {"rule_id": "STOPP_ppi_longterm", "source": "STOPP v3", "kind": "avoid",
     "atc": "A02BC", "inn": ["omeprazole", "pantoprazole", "esomeprazole"],
     "min_age": 65, "condition": "duration_gt_8weeks_full_dose", "severity": "low",
     "ru": ["омепразол", "пантопразол", "эзомепразол", "нольпаза", "омез"],
     "text": "ИПП в полной дозе >8 недель без показаний: пересмотреть/снизить дозу."},
    {"rule_id": "Beers_digoxin_dose_elderly", "source": "Beers 2023", "kind": "avoid",
     "atc": "C01AA05", "inn": ["digoxin"],
     "min_age": 65, "condition": "dose_gt_0125mg", "severity": "high",
     "ru": ["дигоксин"],
     "text": "Дигоксин >0.125 мг/сут у пожилых: повышенная токсичность без прироста эффекта."},
    {"rule_id": "STOPP_tricyclic_elderly", "source": "STOPP v3", "kind": "avoid",
     "atc": "N06AA", "inn": ["amitriptyline", "imipramine", "clomipramine"],
     "min_age": 65, "condition": None, "severity": "moderate",
     "ru": ["амитриптилин", "имипрамин", "кломипрамин"],
     "text": "Трициклические антидепрессанты у пожилых: антихолинергические/кардиоэффекты."},
    {"rule_id": "STOPP_longacting_sulfonylurea", "source": "STOPP v3", "kind": "avoid",
     "atc": "A10BB", "inn": ["glimepiride", "glibenclamide"],
     "min_age": 65, "condition": None, "severity": "moderate",
     "ru": ["глимепирид", "амарил", "глибенкламид", "манинил"],
     "text": "Длительно действующие ПСМ у пожилых: риск затяжной гипогликемии."},
    {"rule_id": "Beers_nsaid_anticoag_combo", "source": "Beers 2023", "kind": "avoid",
     "atc": "M01A", "inn": ["diclofenac", "ibuprofen", "ketorolac"],
     "min_age": 65, "condition": "on_anticoagulant", "severity": "high",
     "ru": ["диклофенак", "ибупрофен", "кеторолак"],
     "text": "НПВП + антикоагулянт у пожилых: высокий риск ЖКТ-кровотечения."},
    # --- START: consider (омиссия показанного) ---
    {"rule_id": "START_statin_ascvd", "source": "START v3", "kind": "consider",
     "atc": "C10AA", "inn": ["atorvastatin", "rosuvastatin", "simvastatin"],
     "min_age": 65, "condition": "documented_ascvd", "severity": "moderate",
     "ru": ["аторвастатин", "розувастатин", "симвастатин", "статин"],
     "text": "Статин при документированном ССЗ-атеросклерозе, если нет противопоказаний."},
    {"rule_id": "START_ace_arb_hf", "source": "START v3", "kind": "consider",
     "atc": "C09", "inn": ["enalapril", "ramipril", "lisinopril", "losartan", "valsartan"],
     "min_age": 65, "condition": "heart_failure_or_post_mi", "severity": "moderate",
     "ru": ["эналаприл", "рамиприл", "лизиноприл", "лозартан", "валсартан", "иапф"],
     "text": "иАПФ/БРА при ХСН/после ИМ, если нет противопоказаний."},
    {"rule_id": "START_anticoag_af", "source": "START v3", "kind": "consider",
     "atc": "B01AF", "inn": ["apixaban", "rivaroxaban", "dabigatran", "warfarin"],
     "min_age": 65, "condition": "atrial_fibrillation", "severity": "high",
     "ru": ["апиксабан", "ривароксабан", "дабигатран", "варфарин"],
     "text": "Антикоагулянт при фибрилляции предсердий (CHA2DS2-VASc), если нет противопоказаний."},
]


def main() -> int:
    out = {
        "source": "STOPP/START v3 (2023) + AGS Beers Criteria 2023 - seed",
        "note": "Seed ключевых правил; расширяется. min_age по умолчанию 65.",
        "rules": RULES,
        "n_rules": len(RULES),
        "kinds": sorted({r["kind"] for r in RULES}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: rules={len(RULES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
