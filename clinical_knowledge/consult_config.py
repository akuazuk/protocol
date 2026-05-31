"""Загрузка конфигов оценки КЗ (веса, red flags).

YAML читается через PyYAML, если он установлен. Если PyYAML/файл недоступны —
используются встроенные дефолты (рантайм сервера не обязан иметь PyYAML).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

_DEFAULT_WEIGHTS: dict[str, Any] = {
    "weights": {
        "structural_score": 0.15,
        "patient_data_score": 0.10,
        "protocol_match_score": 0.15,
        "diagnosis_score": 0.20,
        "required_exams_score": 0.15,
        "treatment_score": 0.15,
        "safety_score": 0.05,
        "follow_up_score": 0.05,
    },
    "legacy_weights": {
        "protocol_match_score": 0.15,
        "diagnosis_score": 0.20,
        "required_exams_score": 0.20,
        "treatment_score": 0.20,
        "safety_score": 0.15,
        "documentation_quality_score": 0.10,
    },
    "status_thresholds": {
        "compliant": 90,
        "mostly_compliant": 75,
        "partially_compliant": 50,
        "non_compliant": 1,
    },
}

_DEFAULT_RED_FLAGS: dict[str, Any] = {
    "red_flags": {
        "possible_malignancy": {
            "keywords": [
                "опухолевое образование", "нельзя исключить инвазию",
                "подозрение на злокачественное", "злокачественн",
                "образование кишки", "новообразование",
            ],
            "severity": "critical",
            "expected_actions": [
                "дообследование", "консультация профильного специалиста",
                "маршрутизация", "повторная консультация",
            ],
        },
        "thrombosis": {
            "keywords": ["флеботромбоз", "тромбоз глубоких вен", "тгв", "тромбофлебит"],
            "severity": "high",
            "expected_actions": ["антикоагулянтная терапия", "контроль узи", "повторная консультация"],
        },
        "systemic_autoimmune": {
            "keywords": ["дискоидная красная волчанка", "системная красная волчанка", "ana", "anti-dna"],
            "severity": "medium",
            "expected_actions": [
                "лабораторное обследование",
                "консультация ревматолога по показаниям", "фотозащита",
            ],
        },
        "severe_infection": {
            "keywords": ["сепсис", "септическ", "высокая температура", "фебрильн", "гнойн"],
            "severity": "high",
            "expected_actions": ["антибактериальная терапия", "госпитализация по показаниям", "контроль"],
        },
        "gi_bleeding_anemia": {
            "keywords": ["железодефицит", "анемия", "кровопотер", "мелена", "кровотечени"],
            "severity": "high",
            "expected_actions": ["дообследование", "консультация профильного специалиста", "контроль анализов"],
        },
    }
}


def _load_yaml(name: str, default: dict[str, Any]) -> dict[str, Any]:
    path = CONFIG_DIR / name
    try:
        import yaml  # type: ignore
    except ImportError:
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else default
    except (OSError, ValueError):
        return default


@lru_cache(maxsize=1)
def load_compliance_weights() -> dict[str, Any]:
    return _load_yaml("compliance_weights.yaml", _DEFAULT_WEIGHTS)


@lru_cache(maxsize=1)
def load_red_flags() -> dict[str, dict[str, Any]]:
    cfg = _load_yaml("red_flags.yaml", _DEFAULT_RED_FLAGS)
    rf = cfg.get("red_flags") if isinstance(cfg, dict) else None
    return rf if isinstance(rf, dict) else _DEFAULT_RED_FLAGS["red_flags"]
