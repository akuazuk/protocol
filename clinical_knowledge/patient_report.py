"""Отчёт проверки КЗ для пациента (B2C / tier P1)."""
from __future__ import annotations

import re
from typing import Any, Literal

TrafficLight = Literal["green", "yellow", "red"]
BlockStatus = Literal["ok", "attention", "concern"]

PATIENT_DISCLAIMER_RU = (
    "Ориентировочная сверка с клиническими протоколами Минздрава РБ. "
    "Не является диагнозом, медицинским заключением или заменой очного приёма. "
    "При сомнениях обратитесь к лечащему врачу или методслужбе клиники."
)

PATIENT_BLOCK_ORDER = (
    "complaints",
    "anamnesis",
    "objective_status",
    "diagnosis",
    "exams",
    "treatment",
    "follow_up",
    "limitations",
)

BLOCK_CATEGORY_RU: dict[str, str] = {
    "complaints": "Жалобы",
    "anamnesis": "Анамнез",
    "objective_status": "Осмотр",
    "diagnosis": "Диагноз",
    "exams": "Обследования",
    "treatment": "Лечение",
    "follow_up": "Контроль",
    "limitations": "Ограничения",
    "labs": "Анализы",
    "protocol": "Протокол",
    "document": "Документ",
}

_QUESTION_PREFIXES = (
    "уточните",
    "проверьте",
    "обсудите",
    "спросите",
    "не указан",
    "не описан",
    "не отраж",
    "отсутств",
)


def _clamp_pct(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0, min(100, int(round(float(value)))))


def traffic_light_for_pct(pct: int | None) -> tuple[TrafficLight, str]:
    if pct is None:
        return "yellow", "Недостаточно данных для уверенной оценки"
    if pct >= 75:
        return "green", "В целом соответствует стандарту Минздрава"
    if pct >= 50:
        return "yellow", "Есть пробелы - задайте врачу вопросы из списка"
    return "red", "Много неучтённого по стандарту - рекомендуем обсудить с врачом"


def block_status_for_score(score_pct: int | None) -> BlockStatus:
    if score_pct is None:
        return "attention"
    if score_pct >= 75:
        return "ok"
    if score_pct >= 50:
        return "attention"
    return "concern"


def _category_for_block(block_id: str, fallback: str = "") -> str:
    bid = (block_id or "").strip().lower()
    if bid in BLOCK_CATEGORY_RU:
        return BLOCK_CATEGORY_RU[bid]
    return (fallback or "Вопрос на приёме").strip() or "Вопрос на приёме"


def _gap_to_question(gap: str, block_name: str, block_id: str = "") -> str:
    """Сформулировать вопрос так, как пациент спросил бы на приёме."""
    g = (gap or "").strip().rstrip(".")
    if not g:
        return ""
    if g.endswith("?"):
        return _polish_question(g)

    low = g.lower()
    bid = (block_id or "").strip().lower()

    if re.search(r"длительност", low) and re.search(r"терап|лечен|при[её]м", low):
        return "Доктор, на какой срок мне назначено лечение и когда его можно будет закончить?"
    if re.search(r"доз", low):
        return "Подскажите, пожалуйста, как правильно принимать препараты - дозу, время суток и сколько дней?"
    if re.search(r"узи|ультразвук", low):
        return "Мне нужно делать УЗИ? Если да - когда лучше записаться?"
    if re.search(r"\bоак\b|анализ крови", low):
        return "Нужно ли пересдать общий анализ крови, и учтены ли мои последние результаты?"
    if re.search(r"контрол|наблюден|повторн", low):
        return "Когда мне приходить на контроль и на что обратить внимание до следующего визита?"
    if re.search(r"локализац", low):
        return "Можно объяснить проще, где именно у меня проблема и что это значит?"
    if re.search(r"стади", low):
        return "На какой стадии сейчас болезнь и как это влияет на лечение?"
    if re.search(r"обязательн", low) and re.search(r"лаборатор|исследован", low):
        return "Какие анализы или обследования мне ещё нужно пройти по вашему плану лечения?"
    if re.search(r"нет\s+указан", low) or re.search(r"не\s+указан", low):
        if bid == "follow_up":
            return "Когда мне нужен следующий визит и что взять с собой на приём?"
        if bid == "treatment":
            return "В заключении не всё понятно про лечение - не могли бы вы рассказать подробнее?"

    if bid == "treatment":
        return f"По лечению я не до конца понял(а): {g}. Не могли бы вы объяснить, что это значит для меня?"
    if bid == "exams":
        return f"По обследованиям: {g} - мне это уже проходить или это было сделано раньше?"
    if bid == "diagnosis":
        return f"Можно, пожалуйста, объяснить диагноз простыми словами? Мне неясно: {g}."
    if bid == "follow_up":
        return "Когда записываться на следующий приём и что нужно принести с собой?"
    if bid == "complaints":
        return f"В жалобах я не увидел(а) упоминания про {g}. Это важно учесть в моём случае?"
    if bid == "anamnesis":
        return f"В анамнезе не указано: {g}. Нужно ли мне что-то дополнительно рассказать?"
    if bid == "objective_status":
        return f"В осмотре не описано: {g}. Можете пояснить, что это значит?"

    if block_name:
        return f"По разделу «{block_name}» хочу уточнить: {g} - не могли бы вы пояснить?"
    return f"Подскажите, пожалуйста: {g}?"


def _polish_question(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if not t.endswith("?"):
        t += "?"
    if t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _comment_to_question(comment: str, block_name: str, block_id: str) -> str:
    c = (comment or "").strip()
    if not c:
        return ""
    low = c.lower()
    if "доз" in low and ("не детализ" in low or "не указан" in low):
        return "Доктор, как именно мне принимать назначенные препараты - дозу, время приёма и на сколько дней?"
    if "мало детал" in low or "кратко" in low:
        return f"Можно подробнее рассказать про «{block_name}» - я хочу понять, всё ли учтено в моём случае?"
    if "не указан" in low and block_id == "diagnosis":
        return "Можно объяснить диагноз простыми словами и сказать, что он означает для лечения?"
    if block_id == "exams" and ("мало" in low or "не распознан" in low):
        return "Какие обследования у меня уже были и что ещё нужно пройти по вашему плану?"
    return _gap_to_question(c, block_name, block_id)


def _question_title(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Вопрос врачу"
    if "?" in t[:80]:
        return t.split("?")[0].strip()[:60] + "?"
    words = t.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def _collect_structured_questions(cards: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card.get("block_id") == "limitations":
            continue
        name = str(card.get("name_ru") or "").strip()
        bid = str(card.get("block_id") or "").strip()
        score = _clamp_pct(card.get("score_pct"))
        severity = "low"
        if score is not None and score < 50:
            severity = "high"
        elif score is not None and score < 75:
            severity = "medium"
        items: list[tuple[str, str]] = []
        if score is not None and score < 75:
            comment = str(card.get("comment_ru") or "").strip()
            if comment and len(comment) > 12:
                items.append(("comment", comment))
        for g in card.get("gaps_ru") or []:
            txt = str(g).strip()
            if txt:
                items.append(("gap", txt))
        for kind, raw in items:
            dedupe = f"{bid}:{kind}:{raw.lower()[:80]}"
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(
                {
                    "id": f"q{len(out)+1}",
                    "source_gap": raw if kind == "gap" else "",
                    "source_comment": raw if kind == "comment" else "",
                    "text": "",
                    "title": "",
                    "severity": severity,
                    "category_ru": _category_for_block(bid, name),
                    "block_id": bid,
                }
            )
            if len(out) >= limit:
                return out
    return out


def _protocol_link_dict(
    *,
    protocol_path: str | None = None,
    title: str | None = None,
    section: str | None = None,
) -> dict[str, Any] | None:
    from clinical_knowledge.protocol_links import protocol_link_payload

    payload = protocol_link_payload(protocol_path, title=title, section=section)
    return payload


def _protocol_link_from_card(card: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    return _protocol_link_dict(
        protocol_path=str(card.get("protocol_path") or ""),
        title=str(card.get("protocol_title") or "") or None,
        section=str(card.get("protocol_section") or "") or None,
    )


def _iter_protocol_path_candidates(
    cards: list[dict[str, Any]],
    l1_result: dict[str, Any],
) -> list[tuple[str, str | None]]:
    """Пары (path, title) из всех источников L1."""
    pairs: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def _add(path: str, title: str | None = None) -> None:
        p = (path or "").strip()
        if not p or p in seen:
            return
        seen.add(p)
        pairs.append((p, (title or "").strip() or None))

    align = l1_result.get("alignment") if isinstance(l1_result.get("alignment"), dict) else {}
    audit = align.get("audit_trail") if isinstance(align.get("audit_trail"), dict) else {}
    profile = align.get("protocol_profile") if isinstance(align.get("protocol_profile"), dict) else {}

    for p in audit.get("protocol_paths") or []:
        _add(str(p))
    for p in profile.get("paths") or []:
        _add(str(p))
    for m in list(audit.get("protocol_matches") or []):
        if isinstance(m, dict):
            _add(str(m.get("source_path") or m.get("local_path") or ""), str(m.get("title") or "") or None)
    sa = l1_result.get("structured_analysis") if isinstance(l1_result.get("structured_analysis"), dict) else {}
    for m in sa.get("matches") or []:
        if isinstance(m, dict):
            _add(str(m.get("source_path") or m.get("local_path") or ""), str(m.get("title") or "") or None)
    for card in cards:
        if isinstance(card, dict):
            _add(str(card.get("protocol_path") or ""), str(card.get("protocol_title") or "") or None)
    return pairs


def _collect_protocol_links(cards: list[dict[str, Any]], l1_result: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, title in _iter_protocol_path_candidates(cards, l1_result):
        link = _protocol_link_dict(protocol_path=path, title=title)
        if link and link.get("path") and link["path"] not in seen:
            seen.add(str(link["path"]))
            out.append(link)
    return out[:8]


def _enrich_protocol_links_in_report(
    report: dict[str, Any],
    *,
    protocol_context: dict[str, Any] | None,
) -> None:
    """Подставляет ссылки на PDF там, где есть только название протокола."""
    links = list(report.get("protocol_links") or [])
    seen = {str(l.get("path")) for l in links if isinstance(l, dict) and l.get("path")}

    if protocol_context and isinstance(protocol_context, dict):
        pl = protocol_context.get("protocol_link")
        if isinstance(pl, dict) and pl.get("path") and pl["path"] not in seen:
            links.insert(0, pl)
            seen.add(str(pl["path"]))
        elif not pl:
            path = str(protocol_context.get("protocol_path") or "")
            title = str(protocol_context.get("protocol_title") or "") or None
            link = _protocol_link_dict(protocol_path=path, title=title)
            if link and link.get("path") and link["path"] not in seen:
                links.insert(0, link)
                seen.add(str(link["path"]))
        protocol_context["protocol_link"] = links[0] if links else protocol_context.get("protocol_link")

    report["protocol_links"] = links[:8]
    primary = links[0] if links else None
    if not primary:
        return
    for block in report.get("blocks") or []:
        if not isinstance(block, dict) or block.get("protocol_link"):
            continue
        if block.get("protocol_excerpt") or block.get("status") in ("attention", "concern"):
            block["protocol_link"] = primary
    for cite in report.get("protocol_citations") or []:
        if isinstance(cite, dict) and not cite.get("protocol_link"):
            cite["protocol_link"] = primary


def _collect_citations(cards: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        excerpt = str(card.get("protocol_excerpt") or "").strip()
        if len(excerpt) < 24:
            continue
        key = excerpt[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        title = str(card.get("protocol_title") or card.get("name_ru") or "Клинический протокол").strip()
        section = str(card.get("protocol_section") or "").strip()
        link = _protocol_link_from_card(card)
        row: dict[str, Any] = {
            "protocol_title": title[:200],
            "section": section[:120],
            "excerpt": excerpt[:420],
        }
        if link:
            row["protocol_link"] = link
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _block_why_ru(card: dict[str, Any], status: BlockStatus, score: int | None) -> str:
    """Краткое объяснение статуса блока для пациента."""
    if status == "ok":
        return "Раздел заполнен в соответствии с типовыми требованиями протокола."
    comment = str(card.get("comment_ru") or "").strip()
    gaps = [str(g).strip() for g in card.get("gaps_ru") or [] if str(g).strip()]
    if comment and len(comment) > 12:
        return comment
    if gaps:
        return "Не хватает: " + "; ".join(gaps[:2]) + "."
    if score is not None and score < 50:
        return "По этому разделу мало информации для уверенной сверки с протоколом."
    return "Есть отдельные пробелы — уточните у врача на приёме."


def _headline_ru(light: TrafficLight, overall_label: str, conf: int | None) -> str:
    if conf is not None and conf < 55:
        return "Сначала улучшите качество фото или загрузите PDF — оценка может быть неточной"
    if light == "green":
        return "Можно спокойно идти на приём — критичных пробелов не найдено"
    if light == "yellow":
        return "Есть что обсудить с врачом — список вопросов ниже"
    return "Рекомендуем обсудить заключение с врачом — много неучтённого по стандарту"


def _patient_blocks(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(c.get("block_id")): c
        for c in cards
        if isinstance(c, dict) and c.get("block_id")
    }
    blocks: list[dict[str, Any]] = []
    for bid in PATIENT_BLOCK_ORDER:
        card = by_id.get(bid)
        if not card or bid == "limitations":
            continue
        score = _clamp_pct(card.get("score_pct"))
        status = block_status_for_score(score)
        summary = str(card.get("comment_ru") or "").strip()
        if not summary and card.get("findings_ru"):
            findings = [str(x).strip() for x in card.get("findings_ru") or [] if str(x).strip()]
            if findings:
                summary = findings[0]
        gaps = [str(g).strip() for g in card.get("gaps_ru") or [] if str(g).strip()][:3]
        excerpt = str(card.get("protocol_excerpt") or "").strip()[:420]
        plink = _protocol_link_from_card(card)
        block: dict[str, Any] = {
            "id": bid,
            "title": str(card.get("name_ru") or bid),
            "score_pct": score,
            "status": status,
            "summary_ru": summary,
            "why_ru": _block_why_ru(card, status, score),
            "protocol_excerpt": excerpt,
            "gaps": gaps,
        }
        if plink:
            block["protocol_link"] = plink
        blocks.append(block)
    return blocks


def resolve_patient_overall_pct(l1_result: dict[str, Any]) -> int | None:
    align = l1_result.get("alignment")
    if isinstance(align, dict):
        mean = _clamp_pct(align.get("alignment_mean_score"))
        if mean is not None:
            return mean
    comp = (l1_result.get("structured_analysis") or {}).get("compliance") or {}
    return _clamp_pct(comp.get("overall_score") or l1_result.get("overall_score"))


def _document_read_back(l1_result: dict[str, Any]) -> list[str]:
    doc = (l1_result.get("structured_analysis") or {}).get("document") or {}
    if not isinstance(doc, dict):
        return []
    sections = doc.get("sections") if isinstance(doc.get("sections"), dict) else {}
    lines: list[str] = []
    mapping = (
        ("complaints", "Жалобы"),
        ("diagnosis_text", "Диагноз"),
        ("recommendations_treatment", "Лечение"),
        ("follow_up_text", "Контроль"),
    )
    for key, label in mapping:
        val = str(sections.get(key) or "").strip()
        if val and len(val) > 8:
            lines.append(f"{label}: {val[:160]}{'…' if len(val) > 160 else ''}")
    diags = doc.get("diagnoses") if isinstance(doc.get("diagnoses"), list) else []
    if not any(l.startswith("Диагноз:") for l in lines) and diags:
        d0 = diags[0] if isinstance(diags[0], dict) else {}
        txt = str(d0.get("text_ru") or d0.get("icd10_code") or "").strip()
        if txt:
            lines.insert(1, f"Диагноз: {txt[:160]}")
    return lines[:5]


def _document_quality(conf: int | None, limitations: str) -> dict[str, Any]:
    hint = "Документ читается хорошо."
    level = "good"
    if conf is not None and conf < 55:
        hint = "Качество распознавания низкое - переснимите при хорошем свете или загрузите PDF."
        level = "low"
    elif conf is not None and conf < 75:
        hint = "Часть текста распознана не полностью - проверьте, что все страницы загружены."
        level = "medium"
    elif limitations:
        hint = limitations[:200]
        level = "medium"
    return {"confidence_pct": conf, "level": level, "hint_ru": hint}


def _priority_topics(
    blocks: list[dict[str, Any]],
    protocol_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    for b in blocks:
        if b.get("status") == "concern":
            topics.append(
                {
                    "topic": str(b.get("title") or b.get("id") or "Раздел"),
                    "why_ru": b.get("summary_ru") or "; ".join(b.get("gaps") or []) or "Есть замечания.",
                    "severity": "high",
                }
            )
        elif b.get("status") == "attention" and len(topics) < 5:
            why = b.get("summary_ru") or (b.get("gaps") or [""])[0]
            if why:
                topics.append({"topic": str(b.get("title") or b.get("id")), "why_ru": why, "severity": "medium"})
    if protocol_context:
        for m in protocol_context.get("missing_recommended_exams") or []:
            if not isinstance(m, dict):
                continue
            topics.insert(
                0,
                {
                    "topic": "Протокол Минздрава",
                    "why_ru": str(m.get("patient_note_ru") or m.get("exam_name") or ""),
                    "severity": str(m.get("severity") or "high"),
                },
            )
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for t in topics:
        k = t.get("topic", "") + t.get("why_ru", "")[:40]
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= 3:
            break
    return out


def _plain_summary(
    light: TrafficLight,
    blocks: list[dict[str, Any]],
    questions: list[dict[str, str]],
    protocol_context: dict[str, Any] | None,
) -> str:
    if light == "green" and not questions:
        return (
            "По основным разделам заключение в целом согласуется с клиническими протоколами Минздрава. "
            "Критичных пробелов для обсуждения с врачом не найдено."
        )
    weak = [b for b in blocks if b.get("status") in ("concern", "attention")]
    names = [str(b.get("title") or "") for b in weak[:3] if b.get("title")]
    parts: list[str] = []
    if names:
        parts.append("Обратите внимание на разделы: " + ", ".join(names) + ".")
    if questions:
        parts.append(f"Подготовлено {len(questions)} вопрос(ов) для разговора с врачом.")
    if protocol_context and protocol_context.get("missing_recommended_exams"):
        parts.append("Есть расхождения с рекомендациями протокола по обследованиям.")
    if light == "red":
        parts.insert(0, "В заключении много неучтённого по стандарту Минздрава - обсудите выписку с врачом.")
    elif light == "yellow" and not parts:
        parts.append("Есть отдельные пробелы - используйте чек-лист вопросов ниже.")
    return " ".join(parts) if parts else traffic_light_for_pct(75 if light == "green" else 60)[1]


def build_patient_report(
    l1_result: dict[str, Any],
    *,
    lab_crosscheck: dict[str, Any] | None = None,
    protocol_context: dict[str, Any] | None = None,
    exams_kz_notes: list[str] | None = None,
    question_tone: str | None = None,
) -> dict[str, Any]:
    """Преобразует результат L1 structured в отчёт для пациента."""
    from clinical_knowledge.patient_question_tone import (
        apply_tone_to_questions,
        normalize_question_tone,
        question_tones_for_api,
        questions_etiquette_ru,
        questions_panel_intro_ru,
        tone_meta,
    )

    tone = normalize_question_tone(question_tone)
    align = l1_result.get("alignment") if isinstance(l1_result.get("alignment"), dict) else {}
    cards = list(align.get("alignment_cards") or [])
    overall = resolve_patient_overall_pct(l1_result)
    light, overall_label = traffic_light_for_pct(overall)

    conf = _clamp_pct(l1_result.get("confidence_score"))
    limitations = str(
        align.get("limitations_ru")
        or (l1_result.get("review") or {}).get("limitations_ru")
        or ""
    ).strip()

    structured_questions = _collect_structured_questions(cards)
    if lab_crosscheck:
        miss_note = ""
        miss = lab_crosscheck.get("missing_in_kz_lines") or []
        if miss:
            miss_note = (
                "В моих анализах есть показатели, которых нет в заключении ("
                + "; ".join(miss[:3])
                + "). Не могли бы вы пояснить, учтены ли они в лечении?"
            )
        elif lab_crosscheck.get("summary_ru"):
            miss_note = str(lab_crosscheck.get("summary_ru") or "")
        if miss_note and not any(q.get("source_comment") == miss_note for q in structured_questions):
            structured_questions.insert(
                0,
                {
                    "id": f"q{len(structured_questions)+1}",
                    "source_comment": miss_note,
                    "intent": "labs_missing_in_kz",
                    "text": "",
                    "title": "",
                    "severity": "medium",
                    "category_ru": BLOCK_CATEGORY_RU["labs"],
                    "block_id": "labs",
                },
            )
    for note in exams_kz_notes or []:
        qtext = (
            "В заключении мало информации об обследованиях - что уже сделано и что ещё нужно пройти?"
            if "обследован" in note.lower()
            else note
        )
        if qtext and not any((q.get("source_comment") or q.get("source_gap")) == qtext for q in structured_questions):
            structured_questions.insert(
                0,
                {
                    "id": f"q{len(structured_questions)+1}",
                    "source_comment": qtext,
                    "intent": "exams_plan",
                    "text": "",
                    "title": "",
                    "severity": "medium",
                    "category_ru": BLOCK_CATEGORY_RU["exams"],
                    "block_id": "exams",
                },
            )
    if protocol_context:
        for m in protocol_context.get("missing_recommended_exams") or []:
            if not isinstance(m, dict):
                continue
            note = str(m.get("patient_note_ru") or "").strip()
            if note and not any(q.get("source_comment") == note for q in structured_questions):
                structured_questions.insert(
                    0,
                    {
                        "id": f"q{len(structured_questions)+1}",
                        "source_comment": note,
                        "intent": "exams_protocol_gap",
                        "text": "",
                        "title": "",
                        "severity": "high",
                        "category_ru": BLOCK_CATEGORY_RU["protocol"],
                        "block_id": "exams",
                    },
                )
    structured_questions = structured_questions[:8]

    if conf is not None and conf < 55 and limitations:
        warn = f"Качество распознавания документа низкое ({conf}%). {limitations}"
        structured_questions.insert(
            0,
            {
                "id": "q0",
                "source_comment": warn,
                "intent": "document_quality",
                "text": "",
                "title": "",
                "severity": "high",
                "category_ru": BLOCK_CATEGORY_RU["document"],
                "block_id": "limitations",
            },
        )
        if light == "green":
            light, overall_label = "yellow", "Качество документа низкое — переснимите или загрузите PDF"

    structured_questions = apply_tone_to_questions(structured_questions, tone)

    blocks = _patient_blocks(cards)
    if exams_kz_notes:
        for b in blocks:
            if b.get("id") == "exams":
                extra = exams_kz_notes[0]
                if extra and extra not in (b.get("summary_ru") or ""):
                    b["summary_ru"] = ((b.get("summary_ru") or "").strip() + " " + extra).strip()
                break
    action_checklist = [
        {
            "id": q["id"],
            "text": q["text"],
            "title": q["title"],
            "severity": q.get("severity", "medium"),
            "category_ru": q.get("category_ru") or BLOCK_CATEGORY_RU["document"],
            "block_id": q.get("block_id") or "",
            "tone": q.get("tone") or tone,
            "emoji": q.get("emoji") or "💬",
            "checked": False,
        }
        for q in structured_questions
    ]

    report = {
        "report_schema_version": 2,
        "headline_ru": _headline_ru(light, overall_label, conf),
        "overall_pct": overall,
        "overall_label_ru": overall_label,
        "traffic_light": light,
        "plain_summary_ru": _plain_summary(light, blocks, structured_questions, protocol_context),
        "document_read_back_ru": _document_read_back(l1_result),
        "document_quality": _document_quality(conf, limitations),
        "priority_topics": _priority_topics(blocks, protocol_context),
        "blocks": blocks,
        "questions_for_doctor": [q["text"] for q in structured_questions],
        "questions_structured": structured_questions,
        "action_checklist": action_checklist,
        "question_tone": tone,
        "question_tone_meta": tone_meta(tone),
        "question_tones_available": question_tones_for_api(),
        "questions_intro_ru": questions_panel_intro_ru(tone),
        "questions_etiquette_ru": questions_etiquette_ru(tone),
        "protocol_citations": _collect_citations(cards),
        "protocol_links": _collect_protocol_links(cards, l1_result),
        "limitations_ru": limitations,
        "confidence_score": conf,
        "disclaimer_ru": PATIENT_DISCLAIMER_RU,
        "matched_protocols_count": int(l1_result.get("matched_protocols_count") or 0),
        "next_steps_ru": [
            "Прочитайте краткий итог и вопросы ниже.",
            "Отметьте, что уже обсудили с врачом.",
            "На приёме покажите список или сохраните PDF.",
        ],
    }
    if lab_crosscheck:
        report["lab_crosscheck"] = lab_crosscheck
    if protocol_context:
        report["protocol_context"] = protocol_context
    _enrich_protocol_links_in_report(report, protocol_context=protocol_context)
    return report


def sanitize_patient_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Убирает B2B-поля (ЦИСЗ, send_gate, сырой structured)."""
    out = dict(payload)
    for key in (
        "send_gate",
        "cisz_readiness",
        "structured_analysis",
        "alignment",
        "report_html",
        "report_markdown",
        "review",
    ):
        out.pop(key, None)
    return out
