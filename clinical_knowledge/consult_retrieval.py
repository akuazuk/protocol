"""Подбор релевантных PDF протоколов для анализа КЗ (только МКБ + matched cards + рубрики)."""
from __future__ import annotations

from typing import Any

from clinical_knowledge.rule_family_gates import expand_specialty_slugs_for_icd, expand_specialty_slugs_for_clinical_text
from clinical_knowledge.respiratory_path_filters import (
    filter_paths_for_respiratory_context,
    is_urti_icd_only,
    path_has_urti_markers,
    scan_registry_urti_paths,
)


def _icd_root(code: str) -> str:
    c = (code or "").upper().strip()
    return c[:3] if len(c) >= 3 else c


def _path_norm(sp: str) -> str:
    return (sp or "").replace("\\", "/").strip()


def _card_icd_roots(card: dict[str, Any]) -> set[str]:
    icd = list(card.get("icd10_all") or card.get("icd10_primary") or [])
    return {_icd_root(str(x)) for x in icd if x}


def _icd_overlap_score(card: dict[str, Any], icd_roots: set[str], icd_full: set[str]) -> float:
    if not icd_roots and not icd_full:
        return 0.0
    card_icd = [str(x).upper() for x in (card.get("icd10_all") or card.get("icd10_primary") or [])]
    card_roots = {_icd_root(c) for c in card_icd}
    overlap = icd_roots & card_roots
    score = 0.0
    if overlap:
        score += 40.0 + 8.0 * len(overlap)
    for c in icd_full:
        for cc in card_icd:
            if c.startswith(_icd_root(cc)) or cc.startswith(_icd_root(c)):
                score += 18.0
                break
    return score


_SPINE_ICD_ROOTS = frozenset({"M51", "M53", "M54"})

_SPECIALTY_SLUG_ALIASES: dict[str, str] = {
    "nevrologiya": "nevrologiya-neyrokhirurgiya",
    "neyrokhirurgiya": "nevrologiya-neyrokhirurgiya",
    "ortopediya": "travmatologiya-ortopediya",
    "ortopediya-travmatologiya": "travmatologiya-ortopediya",
    "travmatologiya": "travmatologiya-ortopediya",
    "ftiziatriya": "pulmonologiya-ftiziatriya",
    "pulmonologiya": "pulmonologiya-ftiziatriya",
    "ginekologiya": "akusherstvo-ginekologiya",
    "akusherstvo": "akusherstvo-ginekologiya",
}


def _normalize_specialty_slugs(slugs: set[str] | list[str] | None) -> set[str]:
    """Короткие slug («nevrologiya») → канон каталога."""
    out: set[str] = set()
    try:
        from clinical_knowledge.rubric_extractors import normalize_rubric_slug
    except Exception:
        normalize_rubric_slug = None  # type: ignore[assignment,misc]
    for raw in slugs or []:
        s = str(raw or "").strip()
        if not s:
            continue
        canon = _SPECIALTY_SLUG_ALIASES.get(s) or _SPECIALTY_SLUG_ALIASES.get(s.lower())
        if not canon and normalize_rubric_slug:
            canon = normalize_rubric_slug(s)
        if not canon and "-" not in s:
            for alias, target in _SPECIALTY_SLUG_ALIASES.items():
                if alias.startswith(s) or s in alias:
                    canon = target
                    break
        out.add(canon or s)
    return out


def _path_spine_domain_mismatch(sp: str, icd_roots: set[str]) -> bool:
    """M54* + путь КП мочевого пузыря или опухолей без позвоночника - чужой протокол."""
    if not icd_roots & _SPINE_ICD_ROOTS:
        return False
    low = sp.lower()
    bladder = any(n in low for n in ("мочевого", "мочев", "пузыр"))
    neoplasm = any(n in low for n in ("опухол", "новообраз", "онколог", "злокач"))
    spine = any(n in low for n in ("позвоноч", "радикул", "люмбо", "нейрохирург", "ишиас"))
    return (bladder or neoplasm) and not spine


_HIV_PATH_MARKERS = ("вич", "hiv", "антиретровир", "арт-терап", "иммунодефицит")
_HIV_CONSULT_MARKERS = (
    "вич",
    "hiv",
    "антиретровир",
    "арт ",
    " cd4",
    "viral load",
    "иммунодеф",
    "aids",
)


def _consult_audience_from_context(
    consult_text: str | None,
    facts: dict[str, Any] | None,
) -> str | None:
    """adult | child | None - из facts и текста КЗ."""
    patient = ((facts or {}).get("patient_context") or {})
    aud = str(patient.get("adult_or_child") or "").strip().lower()
    if aud in ("adult", "child"):
        return aud
    age_raw = patient.get("age_years")
    try:
        age = int(age_raw) if age_raw is not None and str(age_raw).strip() else None
    except (TypeError, ValueError):
        age = None
    if age is not None:
        if age >= 18:
            return "adult"
        if 0 < age < 18:
            return "child"
    blob = (consult_text or "").lower()
    if any(m in blob for m in ("взросл", "взр.", "взр ", "18 лет", "старше 18")):
        return "adult"
    if any(m in blob for m in ("ребен", "детск", "новорожд", "грудн", "лет ", "мес.")):
        if any(m in blob for m in ("взросл", "взр")):
            return None
        return "child"
    return None


def _consult_routing_reject(
    sp: str,
    title: str,
    *,
    consult_text: str | None,
    facts: dict[str, Any] | None,
) -> tuple[bool, str]:
    """True если протокол не подходит для RAG КЗ (до add в allowlist)."""
    from .protocol_audience import infer_protocol_audience

    low = f"{sp} {title}".lower()
    blob = (consult_text or "").lower()

    if any(m in low for m in _HIV_PATH_MARKERS):
        if not any(m in blob for m in _HIV_CONSULT_MARKERS):
            return True, "wrong_nosology_hiv_without_markers"

    consult_aud = _consult_audience_from_context(consult_text, facts)
    proto_aud = infer_protocol_audience(sp, title)
    if consult_aud == "adult" and proto_aud == "pediatric":
        return True, "population_mismatch"
    if consult_aud == "child" and proto_aud == "adult":
        return True, "population_mismatch"

    return False, ""


def filter_consult_protocol_matches(
    matches: list[dict[str, Any]],
    *,
    consult_text: str | None = None,
    consult_facts: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Routing gates для списка match_protocol_cards (до scoring/RAG)."""
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        sp = str(m.get("source_path") or "")
        title = str(m.get("title") or m.get("document_title") or "")
        reject, flag = _consult_routing_reject(
            sp, title, consult_text=consult_text, facts=consult_facts,
        )
        if reject:
            row = dict(m)
            row["applicability"] = "not_applicable"
            flags = list(row.get("pick_risk_flags") or row.get("mismatch_reasons") or [])
            if flag and flag not in flags:
                flags.append(flag)
            row["pick_risk_flags"] = flags
            row["mismatch_reasons"] = list(flags)
            rejected.append(row)
        else:
            allowed.append(m)
    return allowed, rejected


def consult_target_protocol_paths(
    *,
    merged_icd: list[str] | None,
    diag_icd: list[str] | None,
    clinical_rules: dict[str, Any] | None,
    specialty_slugs: list[str] | None,
    consult_text: str | None = None,
    consult_facts: dict[str, Any] | None = None,
    primary_specialty: str | None = None,
    max_paths: int | None = None,
    min_match_score: float = 22.0,
) -> tuple[list[str], dict[str, Any]]:
    """Список source_path PDF, по которым разрешён RAG для КЗ."""
    from .loader import load_protocol_cards_registry
    from .protocol_match import match_protocol_cards
    from .protocol_match_detail import compute_match_detail
    from .protocol_pick_filters import (
        icd_fit_for_card,
        is_administrative_protocol,
    )

    limit = max_paths
    if limit is None:
        import os

        limit = max(2, min(10, int(os.environ.get("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", "6"))))

    paths: list[str] = []
    seen: set[str] = set()
    sources: dict[str, str] = {}
    scored_entries: list[tuple[float, str, str]] = []
    protocol_matches: list[dict[str, Any]] = []
    rejected_protocols: list[dict[str, Any]] = []

    _WHY_REJECTED_RU = {
        "admin_order": "Приказ об утверждении - не клинический эталон",
        "low_score": "Низкий балл соответствия жалобам, анамнезу и МКБ",
        "population_mismatch": "Не подходит по возрасту/популяции",
        "wrong_nosology_spine": "Чужая нозология (не позвоночник/ишиас)",
        "wrong_nosology_venous": "Чужая нозология (не венозная)",
        "wrong_nosology_hiv_without_markers": "ВИЧ-протокол без маркеров ВИЧ в тексте КЗ",
        "inpatient_only": "Только стационарный уход",
        "low_icd_fit": "Слабое соответствие коду МКБ",
    }

    def _match_row_from_detail(detail: dict[str, Any], card: dict[str, Any], src: str) -> dict[str, Any]:
        flags = list(detail.get("pick_risk_flags") or [])
        why = [
            _WHY_REJECTED_RU[f]
            for f in flags
            if f in _WHY_REJECTED_RU
        ]
        if detail.get("rejected") and not why:
            why.append(_WHY_REJECTED_RU["low_score"])
        return {
            "title": card.get("title"),
            "source_path": card.get("source_path"),
            "match_score": detail.get("match_score"),
            "match_breakdown": detail.get("match_breakdown") or {},
            "specialty_slug": card.get("specialty_slug"),
            "icd_fit": detail.get("icd_fit") or [],
            "icd_fit_label": detail.get("icd_fit_label") or "",
            "pick_reason_ru": detail.get("pick_reason_ru") or "",
            "pick_risk_flags": flags,
            "why_rejected_ru": why,
            "pick_source": src,
        }

    def add(sp: str, src: str, score: float = 0.0, *, title: str = "") -> None:
        n = _path_norm(sp)
        if not n or n in seen:
            return
        reject, flag = _consult_routing_reject(
            n, title, consult_text=consult_text, facts=facts,
        )
        if reject:
            rejected_protocols.append({
                "title": title,
                "source_path": n,
                "match_score": score,
                "pick_risk_flags": [flag],
                "why_rejected_ru": [_WHY_REJECTED_RU.get(flag, flag)],
                "pick_source": src,
            })
            return
        seen.add(n)
        paths.append(n)
        sources[n] = src
        scored_entries.append((score, n, src))

    facts = consult_facts
    if not facts and isinstance(clinical_rules, dict):
        facts = clinical_rules.get("consult_facts")

    def _append_match(mp: dict[str, Any], src: str) -> None:
        sp = str(mp.get("source_path") or "")
        sc = float(mp.get("match_score") or 0)
        if not sp:
            return
        row = {
            "title": mp.get("title"),
            "source_path": sp,
            "match_score": sc,
            "match_breakdown": mp.get("match_breakdown") or {},
            "specialty_slug": mp.get("specialty_slug"),
            "icd_fit": mp.get("icd_fit") or [],
            "icd_fit_label": mp.get("icd_fit_label") or "",
            "pick_reason_ru": mp.get("pick_reason_ru") or "",
            "pick_risk_flags": mp.get("pick_risk_flags") or [],
            "why_rejected_ru": mp.get("why_rejected_ru") or [],
            "pick_source": src,
        }
        if sc < min_match_score:
            rejected_protocols.append(row)
            return
        add(sp, src, sc)
        protocol_matches.append(row)

    if isinstance(clinical_rules, dict):
        for mp in clinical_rules.get("matched_protocols") or []:
            if isinstance(mp, dict):
                _append_match(mp, "matched_protocol_card")

    diag = [str(x).upper() for x in (diag_icd or []) if x]
    merged = [str(x).upper() for x in (merged_icd or []) if x]
    primary_icd = diag or merged
    icd_roots = {_icd_root(c) for c in primary_icd}
    icd_full = set(primary_icd)
    slugs = _normalize_specialty_slugs(set(specialty_slugs or []))
    if primary_specialty:
        slugs |= _normalize_specialty_slugs([primary_specialty])
    slugs = expand_specialty_slugs_for_icd(slugs, primary_icd)
    slugs = expand_specialty_slugs_for_clinical_text(slugs, consult_text or "")

    if facts and len(paths) < limit:
        spec_try: list[str | None] = []
        if primary_specialty:
            spec_try.append(
                next(
                    iter(_normalize_specialty_slugs([primary_specialty])),
                    primary_specialty,
                )
            )
        if slugs:
            for s in sorted(slugs):
                if s not in spec_try:
                    spec_try.append(s)
        spec_try.append(None)
        for spec in spec_try:
            for m in match_protocol_cards(facts, specialty_slug=spec, limit=limit * 2):
                sp = str(m.get("source_path") or "")
                if sp in seen:
                    continue
                _append_match(m, "facts_match")
                if len(paths) >= limit:
                    break
            if len(paths) >= limit:
                break

    if primary_icd and len(paths) < limit:
        best_by_path: dict[str, float] = {}
        cons = (facts or {}).get("consultation") or {}
        patient = (facts or {}).get("patient_context") or {}
        icd_list = primary_icd
        spec_for_score = primary_specialty
        if spec_for_score:
            spec_for_score = next(
                iter(_normalize_specialty_slugs([spec_for_score])),
                spec_for_score,
            )
        if not spec_for_score and len(slugs) == 1:
            spec_for_score = next(iter(slugs))

        for card in load_protocol_cards_registry():
            sp = _path_norm(str(card.get("source_path") or ""))
            if not sp or sp in seen:
                continue
            if is_administrative_protocol(card):
                detail = compute_match_detail(
                    card,
                    icd_list=icd_list,
                    audience=patient.get("adult_or_child"),
                    hints=set(cons.get("conditions_hint") or []),
                    specialty_slug=spec_for_score,
                    diag_text=str(cons.get("diagnosis_text") or ""),
                    complaints=list(cons.get("complaints") or []),
                    performed_exams=list(cons.get("performed_exams") or []),
                )
                rejected_protocols.append(_match_row_from_detail(detail, card, "icd_registry_scan"))
                continue
            if slugs and card.get("specialty_slug") not in slugs:
                continue
            if facts:
                detail = compute_match_detail(
                    card,
                    icd_list=icd_list,
                    audience=patient.get("adult_or_child"),
                    hints=set(cons.get("conditions_hint") or []),
                    specialty_slug=spec_for_score,
                    diag_text=str(cons.get("diagnosis_text") or ""),
                    complaints=list(cons.get("complaints") or []),
                    performed_exams=list(cons.get("performed_exams") or []),
                )
                sc = float(detail.get("match_score") or 0)
            else:
                sc = _icd_overlap_score(card, icd_roots, icd_full)
                detail = None
            if sc <= 0:
                continue
            if _path_spine_domain_mismatch(sp, icd_roots):
                if detail:
                    row = _match_row_from_detail(detail, card, "icd_registry_scan")
                    row["why_rejected_ru"] = list(row.get("why_rejected_ru") or []) + [
                        "Несоответствие нозологии (позвоночник)"
                    ]
                    rejected_protocols.append(row)
                continue
            if slugs and card.get("specialty_slug") in slugs:
                sc += 8.0
            if sc >= min_match_score and sc > best_by_path.get(sp, 0.0):
                best_by_path[sp] = sc
            elif detail and detail.get("rejected"):
                rejected_protocols.append(_match_row_from_detail(detail, card, "icd_registry_scan"))

        for sp, sc in sorted(best_by_path.items(), key=lambda x: (-x[1], x[0])):
            add(sp, "icd_registry_match", sc)
            if sp not in {m.get("source_path") for m in protocol_matches}:
                card = next(
                    (
                        c for c in load_protocol_cards_registry()
                        if _path_norm(str(c.get("source_path") or "")) == sp
                    ),
                    {},
                )
                if card:
                    detail = compute_match_detail(
                        card,
                        icd_list=icd_list,
                        audience=patient.get("adult_or_child"),
                        hints=set(cons.get("conditions_hint") or []),
                        specialty_slug=spec_for_score,
                        diag_text=str(cons.get("diagnosis_text") or ""),
                        complaints=list(cons.get("complaints") or []),
                        performed_exams=list(cons.get("performed_exams") or []),
                    )
                    row = _match_row_from_detail(detail, card, "icd_registry_match")
                    row["match_score"] = sc
                    protocol_matches.append(row)
            if len(paths) >= limit:
                break

    def _ensure_icd_protocol_pick() -> None:
        """Гарантировать ≥1 КП для любого МКБ (с процентом соответствия)."""
        if paths or not primary_icd:
            return
        cons = (facts or {}).get("consultation") or {}
        patient = (facts or {}).get("patient_context") or {}
        icd_list = primary_icd
        q_parts = [
            consult_text or "",
            " ".join(cons.get("complaints") or []),
            str(cons.get("diagnosis_text") or ""),
            " ".join(icd_list),
        ]
        query = " ".join(p for p in q_parts if p).strip()
        spec_for_score = primary_specialty
        if spec_for_score:
            norm_spec = next(iter(_normalize_specialty_slugs([spec_for_score])), spec_for_score)
            spec_for_score = norm_spec

        try:
            from clinical_knowledge.protocol_icd_index import lookup_protocols_by_icd

            lk = lookup_protocols_by_icd(
                icd_codes=icd_list,
                query=query,
                population=patient.get("adult_or_child"),
                rubric_slugs=sorted(slugs) if slugs else None,
                limit=max(limit, 2),
            )
            for prot in lk.get("protocols") or []:
                if len(paths) >= limit:
                    break
                sp = _path_norm(str(prot.get("path") or ""))
                if not sp or sp in seen:
                    continue
                rel_pct = float(prot.get("icd_relevance_pct") or 0)
                match_sc = max(12.0, min(100.0, rel_pct))
                card = next(
                    (
                        c for c in load_protocol_cards_registry()
                        if _path_norm(str(c.get("source_path") or "")) == sp
                    ),
                    {"title": prot.get("title"), "source_path": sp},
                )
                detail = compute_match_detail(
                    card,
                    icd_list=icd_list,
                    audience=patient.get("adult_or_child"),
                    hints=set(cons.get("conditions_hint") or []),
                    specialty_slug=spec_for_score,
                    diag_text=str(cons.get("diagnosis_text") or ""),
                    complaints=list(cons.get("complaints") or []),
                    performed_exams=list(cons.get("performed_exams") or []),
                ) if facts else None
                row = _match_row_from_detail(detail, card, "icd_lookup_fallback") if detail else {
                    "title": prot.get("title"),
                    "source_path": sp,
                    "match_score": match_sc,
                    "match_breakdown": {"icd": round(rel_pct / 100.0, 2)},
                    "icd_fit": [{"code": c, "weight": round(rel_pct / 100.0, 2)} for c in icd_list[:2]],
                    "icd_fit_label": ", ".join(f"{c} ({rel_pct:.0f}%)" for c in icd_list[:2]),
                    "pick_reason_ru": f"Подбор по МКБ {', '.join(icd_list[:2])} - {rel_pct:.0f}%",
                    "pick_risk_flags": ["icd_lookup_fallback"],
                    "why_rejected_ru": [],
                    "pick_source": "icd_lookup_fallback",
                }
                row["match_score"] = max(float(row.get("match_score") or 0), match_sc)
                flags = list(row.get("pick_risk_flags") or [])
                if match_sc < min_match_score and "weak_icd_match" not in flags:
                    flags.append("weak_icd_match")
                row["pick_risk_flags"] = flags
                add(sp, "icd_lookup_fallback", float(row["match_score"]))
                protocol_matches.append(row)
        except Exception:
            pass

        if paths:
            return

        best: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []
        for card in load_protocol_cards_registry():
            sp = _path_norm(str(card.get("source_path") or ""))
            if not sp or sp in seen or is_administrative_protocol(card):
                continue
            if _path_spine_domain_mismatch(sp, icd_roots):
                continue
            overlap_sc = _icd_overlap_score(card, icd_roots, icd_full)
            if facts:
                detail = compute_match_detail(
                    card,
                    icd_list=icd_list,
                    audience=patient.get("adult_or_child"),
                    hints=set(cons.get("conditions_hint") or []),
                    specialty_slug=spec_for_score,
                    diag_text=str(cons.get("diagnosis_text") or ""),
                    complaints=list(cons.get("complaints") or []),
                    performed_exams=list(cons.get("performed_exams") or []),
                )
                sc = max(float(detail.get("match_score") or 0), overlap_sc)
            else:
                detail = None
                sc = overlap_sc
            if sc <= 0:
                continue
            best.append((sc, card, detail))

        best.sort(key=lambda x: (-x[0], x[1].get("source_path") or ""))
        for sc, card, detail in best[:limit]:
            sp = _path_norm(str(card.get("source_path") or ""))
            if sp in seen:
                continue
            row = (
                _match_row_from_detail(detail, card, "icd_best_effort_fallback")
                if detail
                else _match_row_from_detail(
                    {
                        "match_score": sc,
                        "match_breakdown": {"icd": round(min(1.0, sc / 100.0), 2)},
                        "icd_fit": icd_fit_for_card(card, icd_list),
                        "icd_fit_label": "",
                        "pick_reason_ru": f"Резервный подбор по МКБ, балл {sc:.0f}",
                        "pick_risk_flags": ["icd_best_effort_fallback"],
                        "rejected": sc < min_match_score,
                    },
                    card,
                    "icd_best_effort_fallback",
                )
            )
            row["match_score"] = sc
            flags = list(row.get("pick_risk_flags") or [])
            for flag in ("icd_best_effort_fallback", "weak_icd_match"):
                if sc < min_match_score and flag not in flags:
                    flags.append(flag)
            row["pick_risk_flags"] = flags
            if sc < min_match_score:
                row["pick_reason_ru"] = (
                    f"Резервный подбор по МКБ {', '.join(icd_list[:2])}: "
                    f"балл {sc:.0f} (ниже порога {min_match_score:.0f})"
                )
            add(sp, "icd_best_effort_fallback", sc)
            protocol_matches.append(row)

    _ensure_icd_protocol_pick()

    if is_urti_icd_only(primary_icd) and paths:
        titles = {str(m.get("source_path") or ""): str(m.get("title") or "") for m in protocol_matches}
        paths = filter_paths_for_respiratory_context(paths, limit=limit, titles=titles)
        if not any(path_has_urti_markers(p, titles.get(p, "")) for p in paths):
            for sp in scan_registry_urti_paths(specialty_slugs=slugs, limit=limit):
                if sp not in seen:
                    add(sp, "icd_title_urti_fallback", 22.0)
                    if len(paths) >= limit:
                        break

    if not paths and primary_icd:
        icd_roots = {_icd_root(c) for c in primary_icd}
        if icd_roots & {"J06", "J00", "J02", "J03", "J04", "J05"} or any(
            r.startswith("J06") for r in icd_roots
        ):
            urti_markers = ("орви", "респиратор", "вирусн", "орз", "орви", "инфекц")
            for card in load_protocol_cards_registry():
                if slugs and card.get("specialty_slug") not in slugs:
                    continue
                sp = _path_norm(str(card.get("source_path") or ""))
                if not sp or sp in seen:
                    continue
                blob = f"{card.get('title') or ''} {sp}".lower()
                if any(m in blob for m in urti_markers):
                    add(sp, "icd_title_urti_fallback", 20.0)

    protocol_matches.sort(key=lambda m: -(float(m.get("match_score") or 0)))
    seen_titles: set[str] = set()
    deduped_matches: list[dict[str, Any]] = []
    for m in protocol_matches:
        key = str(m.get("source_path") or m.get("title") or "")
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped_matches.append(m)

    meta: dict[str, Any] = {
        "primary_icd": primary_icd[:12],
        "specialty_slugs": sorted(slugs),
        "path_sources": sources,
        "protocol_matches": deduped_matches[:limit],
        "rejected_protocols": rejected_protocols[:20],
        "strict": bool(paths),
        "min_match_score": min_match_score,
        "icd_coverage_fallback": any(
            sources.get(p) in ("icd_lookup_fallback", "icd_best_effort_fallback")
            for p in paths
        ),
    }
    return paths[:limit], meta


def filter_retrieval_rows_by_paths(
    rows: list[dict[str, Any]],
    allowed_paths: list[str] | None,
) -> list[dict[str, Any]]:
    if not allowed_paths:
        return rows
    allow = {_path_norm(p) for p in allowed_paths}
    if not allow:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        p = _path_norm(str(row.get("path") or ""))
        if p in allow:
            out.append(row)
    return out


def filter_retrieval_by_category_slugs(
    rows: list[dict[str, Any]],
    allowed_slugs: list[str] | None,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Отбрасывает чанки вне рубрики врача КЗ (напр. акушерство при неврологии)."""
    if not allowed_slugs or not strict:
        return rows
    allow = {s.strip() for s in allowed_slugs if s and s.strip()}
    if not allow:
        return rows
    out = [r for r in rows if (r.get("category") or "").strip() in allow]
    return out if out else rows


def unify_consult_protocol_paths(
    *,
    target_paths: list[str] | None,
    rules_paths: list[str] | None,
    rag_paths: list[str] | None,
    max_paths: int | None = None,
) -> list[str]:
    """Единый список PDF: target (МКБ/cards) + rules + RAG без дублей."""
    import os

    limit = max_paths or max(2, min(10, int(os.environ.get("CONSULT_REVIEW_MAX_PROTOCOL_PATHS", "6"))))
    out: list[str] = []
    seen: set[str] = set()
    for group in (target_paths or [], rules_paths or [], rag_paths or []):
        for p in group:
            n = _path_norm(str(p))
            if n and n not in seen:
                seen.add(n)
                out.append(n)
            if len(out) >= limit:
                return out
    return out


def supplement_retrieval_from_rich_chunks(
    retrieved: list[dict[str, Any]],
    *,
    paths: list[str],
    icd_codes: list[str],
    get_chunks: Any,
    query: str = "",
) -> list[dict[str, Any]]:
    """Typed-дополнение RAG: diagnostics / treatment / dispensary из rich-чанков."""
    if not paths or not get_chunks:
        return retrieved
    from clinical_knowledge.protocol_practical_lite import _chunk_type, _pick_chunks

    seen_keys: set[str] = {
        f"{r.get('path')}|{r.get('page_from')}|{(r.get('text') or '')[:60]}"
        for r in retrieved
    }
    extra: list[dict[str, Any]] = []
    q = query or " ".join(icd_codes or [])

    type_passes: tuple[tuple[str, tuple[str, ...], int], ...] = (
        ("diagnostics", ("diagnostics", "criteria_block", "table"), 4),
        ("treatment", ("treatment", "pharmacotherapy", "drug_list"), 4),
        ("monitoring", ("dispensary", "prevention"), 2),
    )

    for path in paths[:6]:
        chunks = get_chunks(path) or []
        if not chunks:
            continue
        try:
            from clinical_knowledge.chunk_tags import chunk_usable_for_retrieval

            chunks = [c for c in chunks if chunk_usable_for_retrieval(c, ambulatory=True)]
        except Exception:
            pass
        for label, ctypes, lim in type_passes:
            picked = _pick_chunks(chunks, q, icd_codes, limit=lim, chunk_types=ctypes)
            for ch in picked:
                text = (ch.get("text") or "").strip()
                if len(text) < 40:
                    continue
                key = f"{path}|{ch.get('page_from')}|{text[:60]}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                extra.append({
                    "path": path,
                    "text": text[:1200],
                    "score": 0.72,
                    "chunk_type": _chunk_type(ch),
                    "section_title": ch.get("section_title") or label,
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to"),
                    "typed_retrieve": True,
                })

    if not extra:
        return retrieved
    return list(retrieved) + extra
