#!/usr/bin/env python3
"""Собрать курируемый список high-alert препаратов (ISMP + ONC/CredibleMeds seed).

High-alert - препараты с высоким риском тяжёлого вреда при ошибке (ISMP High-Alert
Medications in Acute Care, 2024). Для таких назначений в КЗ обязательны доза,
длительность и план мониторинга - иначе finding (ось C, risk-gate P0/P1).

Дополнительно seed CredibleMeds (QT-удлиняющие, риск TdP) для drug-disease/DDI.

Ключи - канон INN (нижний регистр, англ.), сопоставляются с назначениями через
clinical_knowledge/drug_normalizer.py. RU-синонимы даны для прямого матча по тексту КЗ.

Выход: data/drug_safety/high_alert.json
Usage: python3 scripts/build_high_alert_list.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "drug_safety" / "high_alert.json"

# category: класс ISMP; atc: якорь ATC; requires: обязательные атрибуты назначения;
# ru: синонимы для матча по свободному тексту КЗ (нижний регистр).
HIGH_ALERT: list[dict] = [
    {"inn": "insulin", "category": "insulin_subcutaneous_iv", "atc": "A10A",
     "requires": ["dose", "regimen", "glucose_monitoring"],
     "ru": ["инсулин", "инсулина", "лантус", "левемир", "хумулин", "новорапид", "апидра", "тресиба"]},
    {"inn": "heparin", "category": "anticoagulant_parenteral", "atc": "B01AB01",
     "requires": ["dose", "monitoring_aptt_antixa"],
     "ru": ["гепарин", "гепарина"]},
    {"inn": "enoxaparin", "category": "anticoagulant_lmwh", "atc": "B01AB05",
     "requires": ["dose", "renal_check"],
     "ru": ["эноксапарин", "клексан", "эниксум"]},
    {"inn": "warfarin", "category": "anticoagulant_oral_vka", "atc": "B01AA03",
     "requires": ["dose", "inr_monitoring"],
     "ru": ["варфарин", "варфарина", "варфарекс"]},
    {"inn": "rivaroxaban", "category": "anticoagulant_doac", "atc": "B01AF01",
     "requires": ["dose", "renal_check"],
     "ru": ["ривароксабан", "ксарелто"]},
    {"inn": "apixaban", "category": "anticoagulant_doac", "atc": "B01AF02",
     "requires": ["dose", "renal_check"],
     "ru": ["апиксабан", "эликвис"]},
    {"inn": "dabigatran", "category": "anticoagulant_doac", "atc": "B01AE07",
     "requires": ["dose", "renal_check"],
     "ru": ["дабигатран", "прадакса"]},
    {"inn": "methotrexate", "category": "oral_weekly_nononcology", "atc": "L04AX03",
     "requires": ["dose", "frequency_weekly", "folate_monitoring"],
     "ru": ["метотрексат", "метотрексата", "методжект"]},
    {"inn": "digoxin", "category": "cardiac_glycoside", "atc": "C01AA05",
     "requires": ["dose", "level_monitoring", "renal_check"],
     "ru": ["дигоксин", "дигоксина"]},
    {"inn": "potassium chloride", "category": "concentrated_electrolyte_iv", "atc": "B05XA01",
     "requires": ["dose", "concentration", "infusion_rate", "monitoring"],
     "ru": ["калия хлорид", "хлорид калия", "калий хлор"]},
    {"inn": "morphine", "category": "opioid", "atc": "N02AA01",
     "requires": ["dose", "frequency", "route"],
     "ru": ["морфин", "морфина"]},
    {"inn": "fentanyl", "category": "opioid", "atc": "N02AB03",
     "requires": ["dose", "route", "monitoring"],
     "ru": ["фентанил", "фентанила"]},
    {"inn": "oxycodone", "category": "opioid", "atc": "N02AA05",
     "requires": ["dose", "frequency"],
     "ru": ["оксикодон", "таргин"]},
    {"inn": "tramadol", "category": "opioid", "atc": "N02AX02",
     "requires": ["dose", "frequency", "max_daily"],
     "ru": ["трамадол", "трамадола", "трамал"]},
    {"inn": "amiodarone", "category": "antiarrhythmic_iv", "atc": "C01BD01",
     "requires": ["dose", "monitoring_qt_thyroid"],
     "ru": ["амиодарон", "кордарон"]},
    {"inn": "vancomycin", "category": "antibiotic_narrow_index", "atc": "J01XA01",
     "requires": ["dose", "level_monitoring", "renal_check"],
     "ru": ["ванкомицин", "ванкомицина"]},
    {"inn": "gentamicin", "category": "aminoglycoside", "atc": "J01GB03",
     "requires": ["dose", "level_monitoring", "renal_check"],
     "ru": ["гентамицин", "гентамицина"]},
    {"inn": "cyclophosphamide", "category": "chemotherapy", "atc": "L01AA01",
     "requires": ["dose", "protocol_ref", "monitoring"],
     "ru": ["циклофосфамид", "циклофосфан"]},
    {"inn": "magnesium sulfate", "category": "concentrated_electrolyte_iv", "atc": "B05XA05",
     "requires": ["dose", "infusion_rate", "monitoring"],
     "ru": ["магния сульфат", "сульфат магния", "магнезия"]},
    {"inn": "epinephrine", "category": "adrenergic_iv", "atc": "C01CA24",
     "requires": ["dose", "concentration", "route"],
     "ru": ["адреналин", "эпинефрин"]},
]

# CredibleMeds seed: препараты с известным риском TdP (удлинение QT). Не high-alert
# по ISMP, но требуют учёта QT/электролитов и осторожности при сочетаниях.
QT_TDP_RISK: list[dict] = [
    {"inn": "amiodarone", "ru": ["амиодарон", "кордарон"]},
    {"inn": "sotalol", "ru": ["соталол"]},
    {"inn": "haloperidol", "ru": ["галоперидол"]},
    {"inn": "azithromycin", "ru": ["азитромицин", "сумамед"]},
    {"inn": "clarithromycin", "ru": ["кларитромицин", "клацид"]},
    {"inn": "ciprofloxacin", "ru": ["ципрофлоксацин", "ципрофлоксацина"]},
    {"inn": "levofloxacin", "ru": ["левофлоксацин", "таваник"]},
    {"inn": "citalopram", "ru": ["циталопрам"]},
    {"inn": "escitalopram", "ru": ["эсциталопрам", "ципралекс"]},
    {"inn": "ondansetron", "ru": ["ондансетрон", "зофран"]},
    {"inn": "methadone", "ru": ["метадон"]},
    {"inn": "domperidone", "ru": ["домперидон", "мотилиум"]},
]


def main() -> int:
    out = {
        "source": "ISMP High-Alert Medications (Acute Care, 2024) + CredibleMeds QT seed",
        "note": "Seed-уровень: покрывает ключевые классы; расширяется по мере необходимости.",
        "high_alert": HIGH_ALERT,
        "qt_tdp_risk": QT_TDP_RISK,
        "n_high_alert": len(HIGH_ALERT),
        "n_qt": len(QT_TDP_RISK),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: high_alert={len(HIGH_ALERT)} qt_tdp={len(QT_TDP_RISK)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
