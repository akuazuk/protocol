"""Подбор карточек протоколов по фактам из КЗ."""
from __future__ import annotations

import re
from typing import Any

from .applicability import assess_card_applicability, infer_card_population
from .condition_registry import score_card_for_hint
from .diagnosis_icd import is_symptom_code, prioritize_codes
from .kp_validity import card_in_force_on, looks_omnibus
from .loader import load_protocol_cards_registry
from .patient_age import parse_iso_date
from .protocol_pick_filters import (
    clinical_relevance_multiplier,
    icd_fit_for_card,
    is_administrative_protocol,
)

# Веса match_score (ТЗ improve_kz §8.1), нормализуются к ~100
_WEIGHT_ICD = 0.40
_WEIGHT_DIAG_TEXT = 0.20
_WEIGHT_SPECIALTY = 0.15
_WEIGHT_POPULATION = 0.10
_WEIGHT_DEMO = 0.05
_WEIGHT_EXAMS = 0.05
_WEIGHT_COMPLAINTS = 0.05


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


# I84 - старый код геморроя, не вены ног. Не усиливать варикоз/ТГВ.
_VENOUS_ICD_ROOTS = frozenset({"I80", "I81", "I82", "I83", "I85", "I86", "I87", "I88", "I89"})
_NOT_VENOUS_ICD_ROOTS = frozenset({"I84"})
_VENOUS_CARD_NEEDLES = (
    "тромбоз", "тгв", "тромбоэмбол", "флеб", "вен ", "веноз", "варикоз", "тромбофлеб",
    "флеботромб", "глубоких вен", "поверхностн",
)
_HEART_FAILURE_NEEDLES = (
    "недостаточност", "сердечн", "кардиомиопат", "функциональн класс", "nyha",
)
_SPINE_ICD_ROOTS = frozenset({"M51", "M53", "M54"})
_BLADDER_PATH_NEEDLES = ("мочевого", "мочев", "пузыр", "уролог", "дизури", "цистит")
_NEOPLASM_PATH_NEEDLES = ("опухол", "новообраз", "онколог", "злокач", "metast", "metastaz", "метастаз")
_SPINE_PATH_NEEDLES = (
    "позвоноч", "радикул", "люмбо", "ишиас", "нейрохирург", "м54", "m54", "м51", "m51",
)


def _is_spine_icd(icd_list: list[str]) -> bool:
    for c in icd_list:
        root = _icd_root(c)
        if root in _SPINE_ICD_ROOTS or (len(root) >= 2 and root.startswith("M5")):
            return True
    return False


def _card_spine_bladder_relevance(card: dict[str, Any], icd_list: list[str]) -> float:
    """0.0 - чужой КП при M54* (мочевой пузырь, опухоли); 1.0 - позвоночный/нейро КП."""
    if not _is_spine_icd(icd_list):
        return 0.5
    blob = ((card.get("title") or "") + " " + (card.get("source_path") or "")).lower()
    bladder = any(n in blob for n in _BLADDER_PATH_NEEDLES)
    neoplasm = any(n in blob for n in _NEOPLASM_PATH_NEEDLES)
    spine = any(n in blob for n in _SPINE_PATH_NEEDLES)
    if (bladder or neoplasm) and not spine:
        return 0.0
    if spine and not bladder and not neoplasm:
        return 1.0
    if bladder and spine:
        return 0.4
    if neoplasm and spine:
        return 0.3
    return 0.5


def _is_venous_icd(icd_list: list[str]) -> bool:
    for c in icd_list:
        root = _icd_root(c)
        if root in _NOT_VENOUS_ICD_ROOTS:
            continue
        if root in _VENOUS_ICD_ROOTS or (len(root) >= 2 and root.startswith("I8")):
            return True
    return False


def _card_venous_relevance(card: dict[str, Any]) -> float:
    """1.0 - явно венозный КП; 0.0 - явно ЧСН без вен; 0.5 - нейтрально."""
    blob = ((card.get("title") or "") + " " + (card.get("source_path") or "")).lower()
    venous = any(n in blob for n in _VENOUS_CARD_NEEDLES)
    heart = any(n in blob for n in _HEART_FAILURE_NEEDLES)
    if venous and not heart:
        return 1.0
    if heart and not venous:
        return 0.0
    if venous and heart:
        return 0.7
    return 0.5


def _population_match(card: dict[str, Any], consult_audience: str | None) -> float:
    cp = infer_card_population(card)
    ca = (consult_audience or "").lower().strip()
    # unknown / пусто - не обнуляем детские КП (иначе suggest теряет pediatric)
    if not ca or ca in {"unknown", "any", "na", "n/a"} or cp == "any":
        return 1.0
    if cp == ca:
        return 1.0
    return 0.0


def _prefer_non_child_if_unknown(
    rows: list[dict[str, Any]],
    audience: str | None,
) -> list[dict[str, Any]]:
    """При unknown не ставить детский КП top-1, если есть взрослый/any с тем же сигналом."""
    ca = (audience or "").lower().strip()
    if ca in {"adult", "child", "newborn"} or len(rows) < 2:
        return rows
    top = rows[0]
    top_pop = infer_card_population(top)
    if top_pop != "child":
        return rows
    top_score = float(top.get("match_score") or 0)
    for idx, row in enumerate(rows[1:], start=1):
        pop = infer_card_population(row)
        if pop == "child":
            continue
        if float(row.get("match_score") or 0) >= top_score * 0.85:
            return [row] + [r for i, r in enumerate(rows) if i != idx]
    return rows


_CARD_BLOB_CACHE: dict[str, str] = {}
_LEAD_CODE_RE = re.compile(r"^\s*[A-Za-z]\d{2}(?:\.\d{1,4})?\s*[-:–—]\s*")


def _card_match_blob(
    card: dict[str, Any],
    *,
    focus_codes: set[str] | None = None,
) -> str:
    """Текст карточки для lexical Dx-match: title, path, содержание КП, RU-коды.

    Содержание берётся из protocol_content_index (summary quotes/conditions),
    не только из названия файла. Коды на карточке - русские формулировки,
    не gate по коду случая.
    """
    focus_key = ",".join(sorted(focus_codes)) if focus_codes else "*"
    cache_key = "|".join(
        [
            str(card.get("source_path") or ""),
            str(card.get("protocol_id") or ""),
            str(card.get("title") or "")[:96],
            str(card.get("condition_label") or "")[:64],
            focus_key,
        ]
    )
    cached = _CARD_BLOB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    title = str(card.get("title") or "")
    cond = str(card.get("condition_label") or "")
    path = str(card.get("source_path") or "")
    path_words = re.sub(r"[_\-/]+", " ", path.split("/")[-1] if path else "")
    path_words = re.sub(r"\.(pdf|json|html?)$", "", path_words, flags=re.IGNORECASE)
    parts = [title, cond, path, path_words]
    try:
        from clinical_knowledge.protocol_content_index import content_text_for_card

        content = content_text_for_card(card)
        if content:
            parts.append(content)
    except Exception:  # noqa: BLE001
        pass
    all_codes = [
        str(x).strip().upper()
        for x in list(card.get("icd10_primary") or []) + list(card.get("icd10_all") or [])
        if x
    ]
    if focus_codes:
        focus_roots = {_icd_root(c) for c in focus_codes}
        codes = [c for c in all_codes if c in focus_codes or _icd_root(c) in focus_roots]
    else:
        # без focus - все коды (consult path / тесты); иначе M21 за top-N теряется
        codes = all_codes
    seen: set[str] = set()
    try:
        import icd_mkb
    except Exception:  # noqa: BLE001
        icd_mkb = None  # type: ignore[assignment]
    for code in codes:
        if not code or code in seen:
            continue
        seen.add(code)
        parts.append(code)
        if icd_mkb is None:
            continue
        try:
            title_ru = icd_mkb.ru_title(code)
        except Exception:  # noqa: BLE001
            title_ru = None
        if title_ru:
            parts.append(_LEAD_CODE_RE.sub("", str(title_ru)).strip())
    blob = " ".join(parts).lower()
    if len(_CARD_BLOB_CACHE) > 8000:
        _CARD_BLOB_CACHE.clear()
    _CARD_BLOB_CACHE[cache_key] = blob
    return blob


def _diag_text_overlap(
    diag_text: str,
    card: dict[str, Any],
    *,
    focus_codes: set[str] | None = None,
) -> float:
    if not diag_text:
        return 0.0
    from clinical_knowledge.dx_query_expand import diagnosis_tokens, token_weight

    words = diagnosis_tokens(diag_text, min_len=3, limit=28)
    if not words:
        return 0.0
    blob = _card_match_blob(card, focus_codes=focus_codes)
    if not blob.strip():
        return 0.0
    hit_w = 0.0
    total_w = 0.0
    strong_hit = False
    for word in words:
        weight = token_weight(word)
        total_w += weight
        if word in blob:
            hit_w += weight
            if weight >= 1.25:
                strong_hit = True
    if total_w <= 0:
        return 0.0
    raw = hit_w / max(3.0, total_w * 0.4)
    if not strong_hit and raw < 0.55:
        # только короткие/общие токены - не считаем клиническим попаданием
        raw *= 0.45
    return min(1.0, raw)


def _text_icd_bridge_score(
    diag_text: str,
    card: dict[str, Any],
    *,
    bridge_cands: list[dict[str, Any]] | None = None,
) -> float:
    """Soft-bridge: text→ICD (справочник) ↔ коды на карточке КП.

    Не использует код МКБ из случая (mis_diagnos). Нужен для omnibus-КП,
    где title/path не содержат нозологию, а M21.* есть в icd10_all.
    """
    if bridge_cands is None:
        if not diag_text:
            return 0.0
        from clinical_knowledge.dx_query_expand import bridge_icd_candidates

        bridge_cands = bridge_icd_candidates(diag_text)
    if not bridge_cands:
        return 0.0
    card_codes = {
        str(x).strip().upper()
        for x in list(card.get("icd10_primary") or []) + list(card.get("icd10_all") or [])
        if x
    }
    if not card_codes:
        return 0.0
    card_roots = {_icd_root(c) for c in card_codes}
    exact = 0
    root_only = 0
    for item in bridge_cands:
        code = str(item.get("code") or "").strip().upper()
        if not code:
            continue
        if code in card_codes:
            exact += 1
        elif _icd_root(code) in card_roots:
            root_only += 1
    if exact:
        return min(1.0, 0.62 + 0.12 * exact)
    # root-only слишком шумный на omnibus/соседних рубриках (Q66 vs Q67)
    if root_only:
        return min(0.45, 0.22 + 0.08 * root_only)
    return 0.0

def compute_match_score(
    card: dict[str, Any],
    *,
    icd_list: list[str],
    audience: str | None,
    hints: set[str],
    specialty_slug: str | None,
    diag_text: str,
    complaints: list[str],
    performed_exams: list[str],
    use_icd: bool = True,
    bridge_cands: list[dict[str, Any]] | None = None,
) -> float:
    """Нормализованный 0-100 match score.

    use_icd=False: путь МО Suggest - только текст диагноза (+ слабые жалобы/specialty),
    без recall/гейтов по МКБ случая. text→ICD bridge - отдельный soft-сигнал.
    """
    card_icd = [str(x).upper() for x in (card.get("icd10_all") or card.get("icd10_primary") or [])]
    icd_part = 0.0
    if use_icd:
        case_codes = [c for c in icd_list if c and not is_symptom_code(c)]
        card_set = set(card_icd)
        exact_n = sum(1 for c in case_codes if c in card_set)
        icd_roots = {_icd_root(c) for c in case_codes}
        card_roots = {_icd_root(c) for c in card_icd}
        overlap = icd_roots & card_roots
        # Exact code on card must beat root-only (F41.2 vs соседний F41.0)
        if exact_n:
            icd_part = min(1.0, 0.88 + 0.04 * exact_n)
        elif overlap:
            icd_part = min(1.0, 0.6 + 0.1 * len(overlap))
        elif icd_list and card_icd:
            for c in icd_list:
                if is_symptom_code(c):
                    continue
                for cc in card_icd:
                    if c.startswith(_icd_root(cc)) or cc.startswith(_icd_root(c)):
                        icd_part = 0.5
                        break

    pop_mult = _population_match(card, audience)
    if pop_mult == 0:
        return 0.0
    pop_part = pop_mult

    spec_part = 0.0
    if specialty_slug and card.get("specialty_slug") == specialty_slug:
        spec_part = 1.0
    elif specialty_slug:
        spec_part = 0.2

    title_low = (card.get("title") or "").lower()
    blob = _card_match_blob(card)
    hint_score = 0.0
    for hint in hints:
        hint_score += score_card_for_hint(str(hint), blob, icd_list if use_icd else []) / 100.0
    hint_score = min(1.0, hint_score)

    focus_codes: set[str] | None = None
    if not use_icd and bridge_cands:
        focus_codes = {
            str(item.get("code") or "").strip().upper()
            for item in bridge_cands
            if item.get("code")
        }
    diag_part = _diag_text_overlap(diag_text, card, focus_codes=focus_codes)
    if not use_icd:
        # МО Suggest: lexical + text→ICD bridge (не код из случая)
        diag_part = max(
            diag_part,
            _text_icd_bridge_score(diag_text, card, bridge_cands=bridge_cands),
        )
    exam_blob = " ".join(performed_exams).lower()
    exam_part = 0.3 if exam_blob and any(x in exam_blob for x in title_low.split()[:3]) else 0.0
    compl_part = 0.0
    if complaints:
        cb = " ".join(complaints).lower()
        compl_part = 0.4 if any(w in blob for w in cb.split() if len(w) > 5) else 0.0

    # Suggest: вес МКБ переносится на текст диагноза.
    w_icd = _WEIGHT_ICD if use_icd else 0.0
    w_diag = _WEIGHT_DIAG_TEXT + (_WEIGHT_ICD if not use_icd else 0.0)
    w_compl = _WEIGHT_COMPLAINTS + (0.05 if not use_icd else 0.0)

    raw = (
        w_icd * icd_part
        + w_diag * max(diag_part, hint_score * 0.5)
        + _WEIGHT_SPECIALTY * spec_part
        + _WEIGHT_POPULATION * pop_part
        + _WEIGHT_DEMO * (1.0 if pop_part > 0 else 0.0)
        + _WEIGHT_EXAMS * exam_part
        + w_compl * compl_part
    )
    if (card.get("status") or "active") != "active":
        raw *= 0.7

    if use_icd and _is_venous_icd(icd_list):
        rel = _card_venous_relevance(card)
        if rel >= 0.9:
            raw = min(1.0, raw * 1.15)
        elif rel <= 0.1:
            raw *= 0.12

    if use_icd and _is_spine_icd(icd_list):
        rel = _card_spine_bladder_relevance(card, icd_list)
        if rel >= 0.9:
            raw = min(1.0, raw * 1.12)
        elif rel <= 0.05:
            raw *= 0.08

    # Свежий пост МЗ при сопоставимом ICD не должен проигрывать КП 8-12 лет назад.
    if icd_part >= 0.5:
        try:
            from clinical_knowledge.kp_sync.recency import recency_multiplier

            raw *= recency_multiplier(card)
        except Exception:  # noqa: BLE001
            pass

    # Омнибус без нозологии в содержании - даже если формально ещё в силе.
    if looks_omnibus(card) and diag_part < 0.35:
        raw *= 0.22

    return round(max(0.0, min(100.0, raw * 100)), 2)


def match_protocol_cards(
    consult_facts: dict[str, Any],
    *,
    specialty_slug: str | None = None,
    limit: int = 8,
    use_icd: bool = True,
) -> list[dict[str, Any]]:
    """Ранжированный список protocol_id.

    use_icd=True (consult/default): МКБ + популяция + нозология.
    use_icd=False (МО Suggest): текст диагноза, без поиска по МКБ.
    """
    cards = load_protocol_cards_registry()
    if specialty_slug:
        cards = [c for c in cards if c.get("specialty_slug") == specialty_slug]

    ctx = consult_facts.get("patient_context") or {}
    cons = consult_facts.get("consultation") or {}
    icd_list = (
        prioritize_codes([str(x).upper() for x in (cons.get("icd10") or []) if x])
        if use_icd
        else []
    )
    audience = ctx.get("adult_or_child")
    visit_day = parse_iso_date(ctx.get("visit_date"))
    hints = set(cons.get("conditions_hint") or [])
    diag_text = str(cons.get("diagnosis_text") or "")
    complaints = list(cons.get("complaints") or [])
    performed = list(cons.get("performed_exams") or [])
    bridge_cands: list[dict[str, Any]] = []
    if not use_icd and diag_text.strip():
        from clinical_knowledge.dx_query_expand import bridge_icd_candidates

        bridge_cands = bridge_icd_candidates(diag_text)

    scored: list[tuple[float, dict[str, Any]]] = []
    for card in cards:
        if is_administrative_protocol(card):
            continue
        if not card_in_force_on(card, visit_day):
            continue
        score = compute_match_score(
            card,
            icd_list=icd_list,
            audience=audience,
            hints=hints,
            specialty_slug=specialty_slug,
            diag_text=diag_text,
            complaints=complaints,
            performed_exams=performed,
            use_icd=use_icd,
            bridge_cands=bridge_cands,
        )
        if score <= 0:
            continue
        if use_icd:
            score = round(
                min(
                    100.0,
                    score
                    * clinical_relevance_multiplier(
                        card,
                        icd_codes=icd_list,
                        complaints=complaints,
                        ambulatory=True,
                    ),
                ),
                2,
            )
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda x: (-x[0], x[1].get("protocol_id") or ""))
    patient = ctx
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for sc, card in scored:
        if card.get("alias_of"):
            continue
        # Дедуп: один PDF в двух рубриках (разный source_path) - одна строка.
        path = str(card.get("source_path") or "").replace("\\", "/")
        fname = path.rsplit("/", 1)[-1].strip().lower()
        approval = card.get("approval") if isinstance(card.get("approval"), dict) else {}
        key = (
            str(card.get("sha256") or "").strip().lower()
            or (
                f"{approval.get('date')}|{approval.get('number')}"
                if approval.get("number")
                else ""
            )
            or fname
            or str(card.get("protocol_id") or id(card))
        )
        if key in seen_keys:
            continue
        appl, _, _ = assess_card_applicability(card, patient)
        if appl == "not_applicable":
            continue
        seen_keys.add(key)
        icd_fit = icd_fit_for_card(card, icd_list) if use_icd else []
        # icd10_* карточки оставляем и при use_icd=False: это метаданные КП для
        # lexical bridge (RU title кода на карточке), не gate по коду случая.
        out.append(
            {
                "protocol_id": card.get("protocol_id"),
                "title": card.get("title"),
                "source_path": card.get("source_path"),
                "population": card.get("population"),
                "icd10_primary": list(card.get("icd10_primary") or [])[:16],
                "icd10_all": list(card.get("icd10_all") or [])[:24],
                "match_score": round(sc, 2),
                "icd_fit": icd_fit,
                "icd_fit_label": ", ".join(
                    f"{x['code']} ({x['weight']:.2f})" for x in icd_fit[:4]
                ),
                "approval": card.get("approval"),
                "matched_condition": card.get("condition_label") or card.get("title"),
                "specialty_slug": card.get("specialty_slug"),
                "sha256": card.get("sha256"),
            }
        )
        if len(out) >= limit:
            break

    out = _prefer_non_child_if_unknown(out, audience)
    # Applicability-gate (ТЗ №2, §A): честный статус результата + безопасный re-rank,
    # чтобы неподтверждённый population-specific (особенно детский) протокол не стал
    # рекомендуемым Top-1 при взрослом/неопределённом запросе. Аддитивно, за флагом.
    try:
        from .search_applicability_gate import apply_applicability_gate, gate_enabled

        if gate_enabled():
            _pat = patient or {}
            _aud = (_pat.get("adult_or_child") or "").lower()
            pediatric_signal = _aud in ("child", "newborn")
            out = apply_applicability_gate(
                out, _pat, icd_list, pediatric_signal=pediatric_signal
            )
    except Exception:
        pass
    return out


def match_protocol_cards_by_diagnosis_text(
    consult_facts: dict[str, Any],
    *,
    specialty_slug: str | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """МО Suggest: подбор КП только по тексту диагноза (без МКБ)."""
    return match_protocol_cards(
        consult_facts,
        specialty_slug=specialty_slug,
        limit=limit,
        use_icd=False,
    )


def match_protocol_cards_for_diagnoses(
    consult_facts: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    *,
    specialty_slug: str | None = None,
    limit_per_dx: int = 3,
    limit_total: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Подбор протоколов отдельно по каждому диагнозу.

    Returns (applicable_matches, not_applicable_matches).
    """
    applicable: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    seen: set[str] = set()

    for dx in diagnoses or [{}]:
        dx_id = str(dx.get("diagnosis_id") or "")
        icd = [dx["icd10_code"]] if dx.get("icd10_code") else []
        facts = dict(consult_facts)
        cons = dict(facts.get("consultation") or {})
        cons["icd10"] = prioritize_codes(list(cons.get("icd10") or []) + icd)
        cons["diagnosis_text"] = dx.get("raw_text") or cons.get("diagnosis_text") or ""
        facts["consultation"] = cons
        if dx.get("certainty") == "suspected":
            cons["conditions_hint"] = list(set(cons.get("conditions_hint") or []) | {"suspected"})

        matches = match_protocol_cards(facts, specialty_slug=specialty_slug, limit=limit_per_dx)
        patient = consult_facts.get("patient_context") or {}
        enriched = annotate_applicability(matches, patient)
        for m in enriched:
            key = str(m.get("source_path") or m.get("protocol_id") or "")
            m["diagnosis_id"] = dx_id
            if m.get("applicability") == "not_applicable":
                if key not in seen:
                    not_applicable.append(m)
                    seen.add(key)
            elif key not in seen and len(applicable) < limit_total:
                applicable.append(m)
                seen.add(key)
    return applicable, not_applicable


def annotate_applicability(
    matches: list[dict[str, Any]],
    patient: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Аддитивно добавляет к каждому матчу applicability/match_reasons/mismatch_reasons.

    Не меняет существующие поля; исходный список не мутируется (возвращается копия).
    """
    out: list[dict[str, Any]] = []
    for m in matches:
        appl, mr, mmr = assess_card_applicability(m, patient)
        enriched = dict(m)
        enriched["applicability"] = appl
        enriched["match_reasons"] = mr
        enriched["mismatch_reasons"] = mmr
        out.append(enriched)
    return out
