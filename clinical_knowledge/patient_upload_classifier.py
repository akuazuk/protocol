"""Определение «не тот документ» в B2C и шутливый ответ пациенту."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

UploadSlot = Literal["kz", "lab"]

# --- эвристики «похоже на КЗ» ---
_KZ_HINTS: list[tuple[int, re.Pattern[str]]] = [
    (3, re.compile(r"консультативн\w*\s+заключен", re.I)),
    (3, re.compile(r"консультаци\w*", re.I)),
    (2, re.compile(r"жалоб\w*", re.I)),
    (2, re.compile(r"\bдиагноз\b", re.I)),
    (2, re.compile(r"рекомендац\w*", re.I)),
    (2, re.compile(r"\bмкб\b|icd[\-\s]?10", re.I)),
    (2, re.compile(r"объективн\w*\s+статус", re.I)),
    (2, re.compile(r"\bанамнез\b", re.I)),
    (2, re.compile(r"врач\s*[:\-]|зав\.?\s*отделен", re.I)),
    (1, re.compile(r"заключен\w*", re.I)),
    (1, re.compile(r"контрольн\w*\s+явк|повторн\w*\s+(?:явк|визит|консультац)", re.I)),
    (1, re.compile(r"специальност\w*\s+врач", re.I)),
]

# --- типы «чужих» документов ---
_GUESS_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "recipe",
        "рецепт блюда",
        re.compile(
            r"ингредиент|приготовлен|нарезать|духовк|варить|обжарить|"
            r"столов\w*\s+ложк|чайн\w*\s+ложк|грамм\b|мл\b.*мук",
            re.I,
        ),
    ),
    (
        "menu",
        "меню или прайс кафе",
        re.compile(r"меню\b|блюдо\b|десерт\b|напиток\b|доставк\w*\s+еды|бургер|пицц", re.I),
    ),
    (
        "receipt",
        "кассовый чек",
        re.compile(r"касс\w*|чек\b|итого\b|сдача\b|унп\b|рн\s*мм|фискальн", re.I),
    ),
    (
        "passport",
        "паспорт или удостоверение",
        re.compile(
            r"паспорт\b|удостоверен\w*\s+личност|личный\s+номер|"
            r"идентификационн\w*\s+номер|выдан\b.*\d{2}\.\d{2}\.\d{4}",
            re.I,
        ),
    ),
    (
        "homework",
        "школьная тетрадь",
        re.compile(r"задач\w*\s*№|уравнен|реши\w*|контрольн\w*\s+работ|класс\b.*предмет", re.I),
    ),
    (
        "contract",
        "договор",
        re.compile(r"договор\b|сторон\w*\s+договор|подписант|юридическ\w*\s+адрес", re.I),
    ),
    (
        "resume",
        "резюме",
        re.compile(r"резюме\b|опыт\s+работ|желаем\w*\s+должност|навык\w*", re.I),
    ),
    (
        "ticket",
        "билет",
        re.compile(r"билет\b|рейс\b|посадочн\w*\s+талон|вагон\b|кинотеатр", re.I),
    ),
    (
        "social",
        "скриншот из соцсетей",
        re.compile(r"instagram|telegram|подписчик|лайк\b|репост|сторис|tiktok", re.I),
    ),
    (
        "invoice",
        "счёт на оплату",
        re.compile(r"счёт\s+на\s+оплат|счет\s*№|банковск\w*\s+реквизит|плательщик", re.I),
    ),
    (
        "parking",
        "парковочный талон",
        re.compile(r"парковк|штраф\b.*транспорт|госномер", re.I),
    ),
    (
        "pet",
        "ветеринарная выписка",
        re.compile(r"ветеринар|кошк|собак|питомец|вакцинац\w*\s+животн", re.I),
    ),
]

_JOKES: dict[str, dict[str, str]] = {
    "recipe": {
        "emoji": "🍲",
        "title": "Это похоже на кулинарный шедевр, а не на заключение",
        "body": "Борщ мы уважаем, но сверить его с протоколом Минздрава пока не научились. "
        "Загрузите консультативное заключение от врача - фото или PDF.",
    },
    "menu": {
        "emoji": "🍕",
        "title": "Вкусно, но не медицински",
        "body": "Меню отличное, только врач по нему диагноз не поставит. "
        "Нам нужно заключение после приёма - с жалобами, диагнозом и рекомендациями.",
    },
    "receipt": {
        "emoji": "🧾",
        "title": "Чек принят, сдача с протоколом - ноль",
        "body": "Кассовый чек - не консультативное заключение. "
        "Попробуйте сфотографировать лист из поликлиники с текстом врача.",
    },
    "passport": {
        "emoji": "🪪",
        "title": "Паспорт в безопасности - проверяем только КЗ",
        "body": "Документы личности лучше не светить в медицинских сервисах. "
        "Загрузите именно консультативное заключение.",
    },
    "homework": {
        "emoji": "📐",
        "title": "Двойка по медицине, пятёрка по алгебре",
        "body": "Тетрадь с задачами - не то, что мы сверяем с клиническими протоколами. "
        "Нужен лист из клиники после консультации.",
    },
    "contract": {
        "emoji": "📜",
        "title": "Договор подписан - с протоколом не подписан",
        "body": "Юридические бумаги мы не читаем. Пришлите консультативное заключение врача.",
    },
    "resume": {
        "emoji": "💼",
        "title": "Резюме сильное, КЗ не приложено",
        "body": "HR-отдел нас не интересует - только медицинское заключение после приёма.",
    },
    "ticket": {
        "emoji": "🎫",
        "title": "Приятного рейса - но не к врачу через нас",
        "body": "Билет не заменяет консультативное заключение. Загрузите выписку из клиники.",
    },
    "social": {
        "emoji": "📱",
        "title": "Лайк за креатив, но это не КЗ",
        "body": "Скрин из мессенджера или соцсети - не медицинский документ. "
        "Нужен лист с заключением врача.",
    },
    "invoice": {
        "emoji": "💳",
        "title": "Счёт оплатите в банке, КЗ - здесь",
        "body": "Счёт на оплату услуг - не то же самое, что консультативное заключение.",
    },
    "parking": {
        "emoji": "🅿️",
        "title": "Парковка оплачена, диагноз - нет",
        "body": "Талон или штраф с парковки мы сверять не будем. Нужно заключение врача.",
    },
    "pet": {
        "emoji": "🐾",
        "title": "Пушистому - к ветеринару, вам - человеческое КЗ",
        "body": "Ветеринарная выписка - отдельная история. "
        "Загрузите своё консультативное заключение.",
    },
    "lab_in_kz": {
        "emoji": "🧪",
        "title": "Это бланк анализов - он ценный, но не на этом месте",
        "body": "Похоже, вы загрузили результаты анализов в поле для заключения. "
        "КЗ - отдельно, анализы - в блок «Анализы (необязательно)» ниже.",
    },
    "kz_in_lab": {
        "emoji": "📋",
        "title": "Заключение врача - не в баночку для анализов",
        "body": "Похоже, консультативное заключение попало в поле для бланков анализов. "
        "Поменяйте файлы местами - и всё заработает.",
    },
    "protocol_pdf": {
        "emoji": "📑",
        "title": "Это клинический протокол Минздрава, а не ваше заключение",
        "body": "Вы загрузили текст протокола для врачей, а не консультативное заключение после приёма. "
        "Нужен лист из клиники с жалобами, диагнозом и рекомендациями именно по вашему визиту.",
    },
    "unknown": {
        "emoji": "🤔",
        "title": "Мы честно посмотрели - это не похоже на КЗ",
        "body": "В тексте нет типичных разделов заключения: жалобы, диагноз, рекомендации. "
        "Сфотографируйте весь лист из клиники или загрузите PDF.",
    },
    "lab_unknown": {
        "emoji": "🔬",
        "title": "В пробирке пусто - в файле тоже не анализы",
        "body": "Не нашли привычных показателей лаборатории (гемоглобин, глюкоза, СОЭ и т.п.). "
        "Загрузите бланк из лаборатории или клиники.",
    },
    "empty": {
        "emoji": "👻",
        "title": "Текста почти нет - как анализ без крови",
        "body": "Документ пустой или плохо читается. Переснимите при свете или загрузите PDF.",
    },
}


@dataclass(frozen=True)
class UploadGuess:
    slot: UploadSlot
    is_expected: bool
    kind: str
    label_ru: str
    score_kz: int
    score_lab: int


def _score_patterns(text: str, patterns: list[tuple[int, re.Pattern[str]]]) -> int:
    blob = (text or "").strip()
    if not blob:
        return 0
    return sum(pts for pts, rx in patterns if rx.search(blob))


def _guess_foreign_kind(text: str) -> tuple[str, str]:
    blob = (text or "").strip()
    for kind, label, rx in _GUESS_PATTERNS:
        if rx.search(blob):
            return kind, label
    return "unknown", "непонятный документ"


def _lab_marker_count(text: str) -> int:
    from clinical_knowledge.lab_result_parser import extract_lab_markers

    return len(extract_lab_markers(text or ""))


def _kz_score(text: str) -> int:
    return _score_patterns(text, _KZ_HINTS)


def _lab_score(text: str) -> int:
    blob = (text or "").strip()
    if not blob:
        return 0
    n = _lab_marker_count(blob)
    score = min(n * 4, 24)
    if re.search(r"биохимич\w*\s+анализ|общий\s+анализ\s+крови|анализ\s+мочи|инвитро|кравира|synlab|референс", blob, re.I):
        score += 4
    if re.search(r"ед/л|ммоль/л|г/л|результат\s+исследован", blob, re.I):
        score += 2
    return score


def _has_typical_kz_sections(text: str) -> bool:
    low = (text or "").lower()
    markers = (
        bool(re.search(r"жалоб\w*", low)),
        bool(re.search(r"\bдиагноз\b", low)),
        bool(re.search(r"рекомендац\w*", low)),
    )
    return sum(markers) >= 2


def _looks_like_minzdrav_protocol(text: str) -> bool:
    low = (text or "").lower()
    if re.search(r"консультативн\w*\s+заключен", low):
        return False
    if not re.search(r"клинический\s+протокол", low):
        return False
    return bool(
        re.search(r"министерств\w*\s+здоров", low)
        or re.search(r"утвержден\w*\s+приказ", low)
        or re.search(r"республик\w*\s+беларус", low)
        or re.search(r"общие\s+положен", low)
    )


def is_b2c_lab_filename(name: str) -> bool:
    """Имя/ case_id начинается на A/a - B2C анализы в тестовом наборе и загрузках."""
    stem = (name or "").strip()
    if not stem:
        return False
    if "/" in stem or "\\" in stem:
        stem = stem.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith((".pdf", ".txt", ".docx", ".rtf", ".odt", ".html")):
        stem = stem.rsplit(".", 1)[0]
    return stem[0].lower() == "a"


def check_consult_document(
    text: str,
    *,
    consultation_id: str = "",
    filename: str = "",
) -> UploadGuess | None:
    """None - можно гонять КЗ pipeline; иначе шутливый ответ как в B2C."""
    name = (filename or consultation_id or "").strip()
    blob = (text or "").strip()
    if is_b2c_lab_filename(name):
        return UploadGuess(
            "kz",
            False,
            "lab_in_kz",
            "бланк анализов (имя файла A/a)",
            _kz_score(blob),
            _lab_score(blob),
        )
    guess = classify_kz_upload(blob, filename=name)
    if not guess.is_expected and guess.kind in (
        "lab_in_kz",
        "recipe",
        "menu",
        "receipt",
        "passport",
        "homework",
        "contract",
        "resume",
        "ticket",
        "social",
        "invoice",
        "parking",
        "pet",
        "protocol_pdf",
        "empty",
        "unknown",
    ):
        return guess
    return None


def build_consult_upload_mismatch_response(
    guess: UploadGuess,
    *,
    consultation_id: str = "",
    review_tier: str = "L1",
) -> dict[str, Any]:
    """Ответ consult-review при загрузке не-КЗ (анализ, рецепт и т.д.)."""
    joke_report = build_upload_joke_report(guess)
    upload_joke = joke_report.get("upload_joke") or {}
    summary = str(joke_report.get("plain_summary_ru") or upload_joke.get("body_ru") or "")
    return {
        "ok": True,
        "upload_mismatch": True,
        "wrong_document_kind": guess.kind,
        "review_tier": review_tier,
        "consultation_id": consultation_id or None,
        "overall_score": None,
        "overall_status": "not_assessed",
        "confidence_score": None,
        "matched_protocols_count": 0,
        "critical_issues_count": 0,
        "llm_used": False,
        "rag_used": False,
        "criteria_source": "upload_classifier",
        "review": {
            "summary_ru": summary,
            "overall_compliance_pct": None,
            "criteria": [],
            "limitations_ru": upload_joke.get("hint_ru") or "",
            "disclaimer_ru": joke_report.get("disclaimer_ru") or "",
            "upload_joke": upload_joke,
        },
        "structured_analysis": None,
        "patient_report": joke_report,
    }


def classify_kz_upload(text: str, *, filename: str = "") -> UploadGuess:
    blob = (text or "").strip()
    if len(blob) < 40:
        return UploadGuess("kz", False, "empty", "пустой или нечитаемый файл", 0, 0)

    if is_b2c_lab_filename(filename):
        return UploadGuess(
            "kz",
            False,
            "lab_in_kz",
            "бланк анализов (имя файла A/a)",
            _kz_score(blob),
            _lab_score(blob),
        )

    kz = _kz_score(blob)
    lab = _lab_score(blob)

    if lab >= 14 and kz < 8:
        return UploadGuess("kz", False, "lab_in_kz", "бланк анализов", kz, lab)

    if _looks_like_minzdrav_protocol(blob):
        return UploadGuess("kz", False, "protocol_pdf", "клинический протокол Минздрава", kz, lab)

    if kz >= 10:
        return UploadGuess("kz", True, "kz", "консультативное заключение", kz, lab)

    kind, label = _guess_foreign_kind(blob)
    if kind != "unknown":
        return UploadGuess("kz", False, kind, label, kz, lab)

    has_sections = _has_typical_kz_sections(blob)

    if has_sections and kz >= 5 and len(blob) > 80:
        return UploadGuess("kz", True, "kz", "возможное заключение", kz, lab)

    if len(blob) < 100 and kz < 3:
        return UploadGuess("kz", False, "empty", "слишком мало текста", kz, lab)

    fn = (filename or "").lower()
    if fn and kz < 4 and not lab:
        if any(x in fn for x in ("receipt", "cheque", "menu", "recipe", "passport")):
            kind, label = _guess_foreign_kind(fn.replace("_", " "))
            if kind != "unknown":
                return UploadGuess("kz", False, kind, label, kz, lab)

    if not has_sections or kz < 4:
        return UploadGuess("kz", False, "unknown", "не похоже на КЗ", kz, lab)

    return UploadGuess("kz", True, "kz", "консультативное заключение", kz, lab)


def classify_lab_upload(text: str, *, filename: str = "") -> UploadGuess:
    blob = (text or "").strip()
    if not blob:
        return UploadGuess("lab", True, "empty", "", 0, 0)

    kz = _kz_score(blob)
    lab = _lab_score(blob)

    if kz >= 12 and lab < 8:
        return UploadGuess("lab", False, "kz_in_lab", "консультативное заключение", kz, lab)

    if lab >= 8:
        return UploadGuess("lab", True, "lab", "бланк анализов", kz, lab)

    kind, label = _guess_foreign_kind(blob)
    if kind != "unknown" and lab < 6:
        return UploadGuess("lab", False, kind, label, kz, lab)

    if lab >= 4:
        return UploadGuess("lab", True, "lab", "возможный бланк анализов", kz, lab)

    if kz >= 6 and lab < 4:
        return UploadGuess("lab", False, "kz_in_lab", "консультативное заключение", kz, lab)

    if lab < 3 and len(blob) > 60:
        return UploadGuess("lab", False, "lab_unknown", "не похоже на анализы", kz, lab)

    return UploadGuess("lab", True, "lab", "бланк анализов", kz, lab)


def _joke_for_guess(guess: UploadGuess) -> dict[str, str]:
    kind = guess.kind
    if guess.slot == "lab" and kind not in ("kz_in_lab", "lab_unknown") and kind in _JOKES:
        kind = "lab_unknown"
    if kind == "lab_in_kz":
        key = "lab_in_kz"
    elif kind == "kz_in_lab":
        key = "kz_in_lab"
    elif kind == "empty":
        key = "empty"
    elif guess.slot == "lab" and kind == "lab_unknown":
        key = "lab_unknown"
    elif kind not in _JOKES:
        key = "unknown"
    else:
        key = kind
    base = dict(_JOKES[key])
    if guess.label_ru and key not in ("empty", "unknown", "lab_unknown", "lab_in_kz", "kz_in_lab"):
        base["body"] = (
            f"Похоже, вы прислали: {guess.label_ru}. "
            + base["body"]
        )
    return base


def build_upload_joke_report(guess: UploadGuess) -> dict[str, Any]:
    joke = _joke_for_guess(guess)
    slot_label = "заключение" if guess.slot == "kz" else "анализы"
    hint = (
        "Загрузите консультативное заключение (фото или PDF всего листа)."
        if guess.slot == "kz"
        else "В блок «Анализы» - бланк из лаборатории с показателями и единицами измерения."
    )
    return {
        "report_schema_version": 2,
        "upload_mismatch": True,
        "mismatch_slot": guess.slot,
        "guessed_kind": guess.kind,
        "guessed_label_ru": guess.label_ru,
        "headline_ru": joke["title"],
        "overall_pct": None,
        "overall_label_ru": "Другой документ",
        "traffic_light": "yellow",
        "plain_summary_ru": joke["body"],
        "upload_joke": {
            "emoji": joke["emoji"],
            "title_ru": joke["title"],
            "body_ru": joke["body"],
            "guessed_what_ru": guess.label_ru,
            "slot_ru": slot_label,
            "hint_ru": hint,
        },
        "next_steps_ru": [
            hint,
            "Проверьте, что снимок без бликов и видны все страницы.",
            "После правильной загрузки нажмите «Проверить заключение» снова.",
        ],
        "questions_for_doctor": [],
        "questions_structured": [],
        "action_checklist": [],
        "blocks": [],
        "protocol_links": [],
        "protocol_citations": [],
        "matched_protocols_count": 0,
        "disclaimer_ru": (
            "Это не медицинская оценка - мы не смогли распознать нужный тип документа."
        ),
        "document_quality": {"confidence_pct": None, "level": "low", "hint_ru": hint},
    }


def check_patient_uploads(
    *,
    kz_text: str,
    lab_text: str | None = None,
    kz_filename: str = "",
    lab_filename: str = "",
) -> UploadGuess | None:
    """Вернуть первое несоответствие (КЗ важнее анализов) или None."""
    kz_guess = classify_kz_upload(kz_text, filename=kz_filename)
    if not kz_guess.is_expected:
        return kz_guess
    lab = (lab_text or "").strip()
    if lab:
        lab_guess = classify_lab_upload(lab, filename=lab_filename)
        if not lab_guess.is_expected:
            return lab_guess
    return None
