# Architecture: B2C Patient App (Protocol · «Проверь КЗ»)

> **Audience:** LLM agents, backend/frontend developers, product owners.  
> **Purpose:** Полный технический и продуктовый контекст B2C-контура для изучения, доработки и рефакторинга.  
> **Last aligned with code:** 2026-06-24 · `BUILD_VERSION` `2026-06-24-r40-sticky-logo-docs-sync`  
> **Companion docs:** [`roadmap-patient-app.md`](roadmap-patient-app.md) (волны A-D, метрики), [`patient-privacy-stub.html`](patient-privacy-stub.html)

---

## Table of contents

1. [How LLM agents should use this doc](#0-how-llm-agents-should-use-this-doc)  
2. [Product definition & JTBD](#1-product-definition--jtbd)  
3. [System layers & boundaries](#2-system-layers--boundaries)  
4. [File map](#3-file-map)  
5. [End-to-end flows](#4-end-to-end-flows)  
6. [B2B L1 dependency (critical)](#5-b2b-l1-dependency)  
7. [API reference](#6-api-reference)  
8. [patient_report schema v2 + examples](#7-patient_report-schema-v2)  
9. [Question pipeline & tones](#8-question-pipeline--tones)  
10. [Playful tone - full selection algorithm](#9-playful-tone---full-selection-algorithm)  
11. [Upload classifier & upload jokes](#10-upload-classifier--upload-jokes)  
12. [Cross-check modules](#11-cross-check-modules)  
13. [Monetization & white-label](#12-monetization--white-label)  
14. [Frontend architecture (deep)](#13-frontend-architecture-deep)  
15. [Design system](#14-design-system)  
16. [Config, env, security](#15-config-env-security)  
17. [Tests inventory](#16-tests-inventory)  
18. [Known gaps & improvement backlog](#17-known-gaps--improvement-backlog)  
19. [Improvement playbook for LLM](#18-improvement-playbook-for-llm)  
20. [Invariants & glossary](#19-invariants--glossary)

---

## 0. How LLM agents should use this doc

### 0.1 Reading order for a new task

1. Confirm task is **B2C** (not B2B consult-review / ЦИСЗ / Methodist training).
2. Read §17 (gaps) - maybe the feature already partially exists.
3. Trace data flow §4 → find module in §3.
4. If touching questions/tone → §8-9. If upload UX → §10 + §13.2.
5. After code change: run tests from §16, bump `BUILD_VERSION` + `patient-sw.js` CACHE if user-facing.

### 0.2 Hard rules

| Rule | Why |
|------|-----|
| Patient API only via `sanitize_patient_api_payload()` | B2B fields must not leak |
| Two separate «шутки» systems | Upload joke ≠ playful question tone |
| Max 8 questions per report | Product + UI limit |
| Consent server-validated | Legal / privacy |
| UI hyphen `-` not em dash | Workspace `hyphen-dash.mdc` |
| Do not edit corpus protocol quotes in `data/protocol_summaries/` for UI dashes | Source PDF text |

### 0.3 Two humor systems (do not merge)

| System | Module | Trigger | Output field |
|--------|--------|---------|--------------|
| **Upload joke** | `patient_upload_classifier.py` | Wrong file (recipe, passport, lab in KZ slot) | `upload_joke`, `upload_mismatch` |
| **Playful tone** | `patient_question_tone.py` | User selects tone «Шуточно» | `action_checklist[].text` |

---

## 1. Product definition & JTBD

### 1.1 B2B vs B2C

| Dimension | B2B | B2C |
|-----------|-----|-----|
| User | Doctor, methodist, MIS | Patient, caregiver |
| Core question | Sign & send to ЦИСЗ? | Is KZ aligned with Minzdrav protocol structure? |
| Language | gate_score, send_gate, CISZ | Traffic light, blocks, doctor questions |
| Input | FHIR, structured MIS export | Photo/PDF KZ + optional lab blanks |
| Review engine | L0-L2 + optional LLM criteria | **P1/P2** on L1 only |
| Success metric | Fewer ЦИСЗ returns | Patient arrives prepared for appointment |

### 1.2 Jobs To Be Done

| Situation | Patient need | Product delivers |
|-----------|--------------|----------------|
| Just got KZ | Understand if doctor followed standard without reading 5 pages | Traffic light + priority topics + read-back |
| Has lab blanks at home | Verify doctor referenced results | Lab crosscheck table |
| Repeat visit | Don't forget to ask | Checklist + share/print |
| Trust in clinic | Objective standard, not chat advice | Protocol PDF links + citations |

**North star outcome:** «I know what to discuss with my doctor» - **not** «I got a diagnosis score».

### 1.3 What B2C explicitly does NOT do

- Does not validate clinical correctness of diagnosis or treatment.
- Does not replace appointment or second opinion.
- Does not have legal force of MEE (МЭЭ).
- Does not store uploaded files on server after response (product promise - enforce in ops).

---

## 2. System layers & boundaries

```
┌──────────────────────────────────────────────────────────────────────────┐
│ VIEW: patient.html · patient-ui.js · patient-tokens.css · patient-sw.js  │
│       protocol-logo.svg · patient-check.html                             │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ multipart POST, SSE, GET status
┌───────────────────────────────────▼──────────────────────────────────────┐
│ API (thin): rag_server.py                                                │
│  - parse uploads (_parse_consult_review_uploads_async, _parse_patient_*) │
│  - consent, rate limit, payment gate                                     │
│  - _patient_html_response() template substitution                        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│ ORCHESTRATOR: patient_review.run_patient_review()                        │
│  early exit: upload_classifier → joke report                             │
│  else: L1 → crosschecks → build_patient_report → P2 enrich → sanitize  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│ SHARED B2B: consult_tiering.run_l1_structured_review                     │
│             lab_result_parser · protocol RAG/chunks · PDF/OCR extractors │
└──────────────────────────────────────────────────────────────────────────┘
```

**Sanitization boundary:** only output of `sanitize_patient_api_payload()` may reach `patient-ui.js`.

---

## 3. File map

### 3.1 Frontend

| Path | Lines (approx) | Responsibility |
|------|----------------|----------------|
| `patient.html` | ~1100 | DOM structure, inline layout CSS, onboarding, form, result sections, FAQ, loader overlay |
| `patient-ui.js` | ~1480 | All behavior: upload, API, render, storage, monetization, PWA install hint |
| `patient-tokens.css` | ~900 | Design tokens, report components, tone panel, tiers, a11y |
| `patient-sw.js` | ~50 | SW cache v13 (`protocol-patient-v13`), network-first for app shell |
| `patient-manifest.webmanifest` | - | PWA metadata, icons wordmark + emblem |
| `protocol-logo.svg` | - | Emblem (favicon, PWA icon) |
| `protocol-logo-wordmark.svg` | - | Full lockup in hero/footer |
| `protocol-logo-mini.svg` | - | Sticky top bar (text only, T `#1F675B`) |
| `patient-check.html` | - | SEO landing → CTA to `/patient.html` |
| `patient-app/` | - | Capacitor scaffold (App Store path) |

### 3.2 Backend modules

| Module | Entry points | Depends on |
|--------|--------------|------------|
| `patient_review.py` | `run_patient_review`, `iter_patient_review_progress` | L1, all patient_* below |
| `patient_report.py` | `build_patient_report`, `sanitize_patient_api_payload` | `patient_question_tone` |
| `patient_question_tone.py` | `apply_tone_to_questions`, `render_doctor_question` | - |
| `patient_upload_classifier.py` | `check_patient_uploads`, `build_upload_joke_report` | `lab_result_parser` (scores) |
| `patient_lab_crosscheck.py` | `crosscheck_labs_with_kz` | `lab_result_parser` |
| `patient_lab_ocr.py` | `extract_lab_text_from_bytes` | `image_ocr`, consult extractors |
| `patient_protocol_crosscheck.py` | `crosscheck_protocol_requirements` | L1 structured_analysis |
| `patient_exams_enrich.py` | `exams_block_notes_for_report` | alignment exams card |
| `patient_p2_enrich.py` | `enrich_patient_report_p2` | patient_report output |
| `patient_clinic_config.py` | `resolve_clinic`, `resolve_tier`, `TIER_CATALOG` | - |
| `patient_monetization_config.py` | `load/save`, `monetization_public_view` | `data/patient_monetization.json` |
| `patient_payment.py` | `create_payment_session`, `verify_payment_token` | tiers |
| `patient_analytics.py` | `record_patient_event` | logging only |
| `patient_account.py` | guest session (in-memory stub) | - |

### 3.3 Tests (7 files)

See §16 for full inventory.

---

## 4. End-to-end flows

### 4.1 Page load sequence (`patient-ui.js` init)

```
DOMContentLoaded / bottom of IIFE:
  refreshPatientShell()
  syncUploadFormatsFromApi()      → GET /api/patient/status
  syncQuestionTonesFromApi()      → merges question_tones catalog
  syncPatientMonetizationFromApi()→ monetization + tiers UI
  loadClinic()                    → GET /api/patient/clinic?clinic_id=
  loadQuestionTone()              → localStorage
  renderTonePicker()
  renderHistory()
  ensureGuestSession()            → POST /api/patient/account/session (stub)
  setupInstallHint()              → beforeinstallprompt
  restoreReport()                 → sessionStorage full report
```

### 4.2 Submit review

```
User: consent ✓ + kzFilesList.length > 0
  → btn-check click
  → if needsPaymentBeforeReview(): startPaymentSession()
  → else runReviewSse() (default) or runReviewFetch()
  → buildFormData(): files[], lab_files[], consent, demographics, clinic_id,
                     tier_id, payment_token, question_tone
  → POST /api/patient/review/stream
  → SSE progress events → showLoader(label_ru, pct)
  → payload.type=done → handleReviewResult → renderReport()
  → form hidden, result-card shown, scroll top
```

### 4.3 Happy path backend

```
1. check_patient_uploads(kz_text, lab_text) → None
2. run_l1_structured_review(text, demographics, skip_alignment=False)
3. crosscheck_labs_with_kz (if lab)
4. exams_block_notes_for_report(exams_card)
5. crosscheck_protocol_requirements(l1, kz, lab)
6. build_patient_report(l1, ..., question_tone)
7. enrich_patient_report_p2 if tier P2
8. sanitize_patient_api_payload
```

### 4.4 Upload mismatch path

```
check_patient_uploads → UploadGuess(is_expected=False)
  → build_upload_joke_report(guess)
  → return { upload_mismatch: true, patient_report: joke_report }
  → NO L1 call (saves CPU + avoids nonsense scores)
```

### 4.5 SSE stages

| Stage | pct | label_ru |
|-------|-----|----------|
| parse | 15 | Разбор текста заключения… |
| align | 45 | Сверка с протоколами Минздрава… |
| labs | 65 | Сверка с бланками анализов… (if lab_text) |
| report | 85 | Формирование отчёта для пациента… |
| done | 100 | full JSON in payload.result |

---

## 5. B2B L1 dependency

B2C **does not reimplement** protocol matching. It consumes `run_l1_structured_review()` from `consult_tiering`.

### 5.1 What L1 returns (internal, sanitized before patient)

Key paths used by B2C:

```python
l1 = {
  "confidence_score": int,           # OCR/parse quality 0-100
  "matched_protocols_count": int,
  "alignment": {
    "alignment_mean_score": int,       # → overall_pct
    "alignment_cards": [               # → blocks + questions
      {
        "block_id": "treatment",
        "name_ru": "Лечение",
        "score_pct": 45,
        "comment_ru": "...",
        "gaps_ru": ["Нет длительности терапии"],
        "findings_ru": [...],
        "protocol_excerpt": "...",
        "protocol_title": "...",
        "protocol_path": "...",
        "protocol_section": "...",
      },
      # ... 8 block types
    ],
    "limitations_ru": "...",
    "audit_trail": { "protocol_paths": [...], "protocol_matches": [...] },
    "protocol_profile": { "paths": [...] },
  },
  "structured_analysis": {
    "document": { "sections": {...}, "diagnoses": [...] },
    "compliance": { "exam_assessments": [...], "overall_score": ... },
    "matches": [ { "title", "source_path", ... } ],
  },
}
```

### 5.2 Mapping L1 → patient_report

| L1 source | Patient field | Transform |
|-----------|---------------|-----------|
| `alignment_mean_score` | `overall_pct` | `resolve_patient_overall_pct()` |
| `alignment_cards[]` | `blocks[]` | `_patient_blocks()` fixed order |
| `alignment_cards[]` gaps/comments | `action_checklist` | `_collect_structured_questions` + tone |
| `structured_analysis.document` | `document_read_back_ru` | `_document_read_back()` |
| `confidence_score` + limitations | `document_quality` | may cap traffic light |
| protocol paths in cards/audit | `protocol_links`, `protocol_citations` | `_collect_*` + PDF URL enrich |
| `compliance.exam_assessments` | `protocol_context` | `crosscheck_protocol_requirements` |

### 5.3 Block score → patient status

```python
score >= 75 → status "ok"       → pill «В порядке»
score >= 50 → status "attention" → pill «Стоит уточнить»
score < 50  → status "concern"   → pill «Обратите внимание»
```

### 5.4 RAG requirement

`_run_patient_review_core()` calls `_ensure_consult_rag_ready()` before L1. If RAG corpus unavailable, L1 may degrade (fewer protocol matches) - test in staging.

---

## 6. API reference

### 6.1 Routes

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/patient.html` | - | Template with `__BUILD_VERSION__`, accept attrs |
| GET | `/api/patient/status` | - | Bootstrap |
| POST | `/api/patient/review` | consent | Sync review |
| POST | `/api/patient/review/stream` | consent | SSE (preferred by UI) |
| POST | `/api/patient/review/json` | consent | Text-only dev/test |
| GET | `/api/patient/clinic` | - | White-label |
| GET | `/api/patient/tiers` | - | Tier catalog |
| POST | `/api/patient/payment/session` | - | Dev mock payment |
| POST | `/api/patient/analytics` | - | Privacy-safe events |
| POST | `/api/patient/account/session` | - | Guest stub |
| POST | `/api/patient/account/sync` | session_token | History sync stub |
| GET | `/api/patient/account/history` | session_token | |
| GET/PUT | `/api/methodist/patient-monetization` | methodist auth | Admin UI |

### 6.2 Review request (multipart)

| Field | Required | Notes |
|-------|----------|-------|
| `files` | yes | 1-5 KZ files |
| `lab_files` | no | max `PATIENT_LAB_MAX_FILES` (default 3) |
| `consent` | yes | `1/true/yes/on` |
| `age_years`, `sex` | no | L1 demographics |
| `clinic_id` | no | from URL param persisted in FormData |
| `tier_id` | no | maps to P1/P2 via `TIER_CATALOG` |
| `payment_token` | if payment required | 402 if missing |
| `question_tone` | no | default `serious` |

### 6.3 Error responses

| HTTP | When |
|------|------|
| 400 | No consent, no files, empty parse |
| 402 | Payment required, invalid token |
| 503 | `PATIENT_REVIEW_ENABLED=0` |
| 429 | Rate limit `RATE_LIMIT_PATIENT_PER_MIN` (default 5/min on review) |

---

## 7. patient_report schema v2

### 7.1 Complete field list

See original §6 in prior versions - key additions for implementers:

**`action_checklist[]` item:**

```json
{
  "id": "q1",
  "text": "На какой срок назначена терапия...?",
  "title": "На какой срок назначена терапия",
  "severity": "high",
  "category_ru": "Лечение",
  "block_id": "treatment",
  "tone": "serious",
  "emoji": "pill",
  "icon": "pill",
  "checked": false
}
```

**`blocks[]` item:**

```json
{
  "id": "treatment",
  "title": "Лечение",
  "status": "concern",
  "score_pct": 42,
  "summary_ru": "Доза не детализирована.",
  "why_ru": "Не хватает: Нет длительности терапии.",
  "gaps": ["Нет длительности терапии"],
  "protocol_excerpt": "Указывают режим и длительность...",
  "protocol_link": { "title": "...", "pdf_url": "...", "path": "..." }
}
```

**`lab_crosscheck`:**

```json
{
  "lab_count": 12,
  "panels_ru": ["Биохимия"],
  "summary_ru": "...",
  "markers_table": [
    { "marker": "Глюкоза", "value": "5.2", "unit": "ммоль/л", "flag": null, "in_kz": false }
  ],
  "missing_in_kz_lines": ["Креатинин 78 мкмоль/л"]
}
```

### 7.2 Minimal success example

```json
{
  "ok": true,
  "review_tier": "P1",
  "confidence_score": 78,
  "matched_protocols_count": 2,
  "patient_report": {
    "report_schema_version": 2,
    "traffic_light": "yellow",
    "overall_pct": 62,
    "headline_ru": "Есть что обсудить с врачом - список вопросов ниже",
    "plain_summary_ru": "Обратите внимание на разделы: Лечение. Подготовлено 3 вопрос(ов)...",
    "action_checklist": [ "..." ],
    "blocks": [ "..." ],
    "disclaimer_ru": "Ориентировочная сверка..."
  }
}
```

### 7.3 Upload joke example

```json
{
  "ok": true,
  "upload_mismatch": true,
  "guessed_kind": "recipe",
  "patient_report": {
    "upload_mismatch": true,
    "traffic_light": "yellow",
    "headline_ru": "Это похоже на кулинарный шедевр, а не на заключение",
    "upload_joke": {
      "emoji": "🍲",
      "title_ru": "...",
      "body_ru": "...",
      "hint_ru": "Загрузите консультативное заключение..."
    },
    "blocks": [],
    "action_checklist": []
  }
}
```

### 7.4 Narrative helpers

| Function | Output |
|----------|--------|
| `_plain_summary()` | Combines weak block names + question count + protocol gaps |
| `_priority_topics()` | Top 3 topics: concern blocks first, then protocol missing exams |
| `_headline_ru()` | Low confidence → photo warning; else by traffic_light |
| `_document_read_back()` | Up to 5 lines from structured document sections |

---

## 8. Question pipeline & tones

### 8.1 Three tones

| id | label_ru | Template source | Accent |
|----|----------|-----------------|--------|
| `serious` | Строго и серьёзно | `_QUESTION_BANK[intent].serious` or generic | `#1e3a5f` |
| `official` | Официально | `_QUESTION_BANK[intent].official` | `#1d4ed8` |
| `playful` | Шуточно | `_PLAYFUL_VARIANTS[intent]` or `_PLAYFUL_GENERIC` | `#b8860b` |

Default: `serious`. Aliases: `шуточно`→playful, `friendly`→serious (legacy).

### 8.2 Pipeline steps

1. **`_collect_structured_questions(cards, limit=8)`** - from alignment cards with score < 75.
2. **Inject** lab/protocol/exams/OCR questions (prepend, high priority).
3. **Truncate** to 8.
4. **`apply_tone_to_questions(raw, tone)`** - style each seed.
5. **Build `action_checklist`** - copy for UI with icons from `CATEGORY_EMOJI`.

### 8.3 Intent detection (`detect_question_intent`)

Full rule table - see §7.4 in previous revision. Fallback by `block_id`:

| block_id | default intent |
|----------|--------------|
| treatment | treatment_unclear |
| exams | exams_plan |
| diagnosis | diagnosis_gap |
| follow_up | follow_up |
| complaints | complaints_gap |
| anamnesis | anamnesis_gap |
| objective_status | objective_gap |
| limitations + качество | document_quality |

### 8.4 `render_doctor_question` decision tree

```
normalize tone
detect intent from gap/comment/block
IF intent in _QUESTION_BANK:
  playful → _pick_playful_text
  else → bank[tone] or bank[serious]
ELIF raw ends with '?':
  wrap for official/playful
ELSE:
  _generic_by_tone (block-specific serious templates)
_ensure_question (?, capitalize)
```

---

## 9. Playful tone - full selection algorithm

### 9.1 Design principles (Belarus RB)

- Context: поликлиника, талон, регистратура, бланк из лаборатории, протокол Минздрава.
- Tone: warm, self-deprecating patient voice - **never** mocking the doctor.
- Forbidden patterns (removed in r26): Hollywood, series finales, «parallel universe», pension jokes that confuse tests.
- Test guardrails: no «дурак», «идиот»; prefer keywords «поликлиник», «талон», «регистратур» in fixtures.

### 9.2 Data structures

**`_QUESTION_BANK`:** 18 intents × 2 tones (serious, official). No playful column - playful uses variants.

**`_PLAYFUL_VARIANTS`:** 18 intents × 3-4 strings each (~65 total variants).

| intent | variant count | Theme |
|--------|---------------|-------|
| treatment_duration | 3 | рецепт, талон, «пока не забуду» |
| treatment_dose | 3 | почерк, баночка, очереди по кабинетам |
| treatment_unclear | 3 | объявление мелким шрифтом, туман |
| exams_uzi | 3 | талон «на потом», график |
| exams_oak | 3 | бланк в сумке, поликлиника |
| exams_plan | 3 | список без галочек, маршрут по кабинетам |
| exams_protocol_gap | 3 | две памятки, протокол Минздрава |
| follow_up | 3 | блокнот, запись по талону |
| diagnosis_plain | 3 | латынь в карте, МКБ |
| diagnosis_gap | 3 | смс без конца, гугл ночью |
| complaints_gap | 3 | урезанные жалобы |
| anamnesis_gap | 3 | черновик на коленке |
| objective_gap | 3 | фото обрезали |
| localization | 3 | «где сидит» |
| staging | 3 | начало пути / серьёзный участок |
| labs_plan | 3 | один заход в поликлинику |
| labs_missing_in_kz | 4 | бланк со стрелочками |
| document_quality | 3 | смазанное фото из автобуса |

**`_PLAYFUL_GENERIC`:** 4 templates with `{name}` and `{gap}` placeholders for unknown intents.

### 9.3 `_pick_playful_text(intent, slot, used)` - pseudocode

```python
variants = _PLAYFUL_VARIANTS.get(intent, [])
if not variants:
    return _QUESTION_BANK.get(intent, {}).get("playful") or ""

for i in range(len(variants)):
    candidate = variants[(slot + i) % len(variants)]
    if candidate.lower() not in used:
        return candidate
return variants[slot % len(variants)]  # repeat allowed if exhausted
```

**slot:** per-intent counter in `intent_slots` inside `apply_tone_to_questions`.  
**used:** global set per report; after each playful question, add `styled.lower()`.

### 9.4 `_pick_playful_generic(gap, block_name, slot, used)`

Rotates through 4 generic templates with `.format(name=..., gap=...)`. Same dedup logic.

### 9.5 Worked example (3 questions, playful)

| # | source | intent | slot | picked variant index |
|---|--------|--------|------|----------------------|
| q1 | Нет длительности терапии | treatment_duration | 0 | 0 (рецепт/регистратура) |
| q2 | Доза не детализирована | treatment_dose | 0 | 0 (почерк) |
| q3 | Нет срока приёма | treatment_duration | 1 | 1 (таблетки/талон) |

### 9.6 Adding new playful copy

1. Choose intent (or add to `detect_question_intent` if new gap type).
2. Add 3+ variants to `_PLAYFUL_VARIANTS[intent]` - distinct wording.
3. Add serious/official to `_QUESTION_BANK` if new intent.
4. Extend `tests/test_patient_question_tone.py`.
5. Manual check: read aloud - would a Belarus polyclinic patient say this to their doctor?

### 9.7 Panel copy per tone

| tone | `questions_intro_ru` gist | `questions_etiquette_ru` gist |
|------|---------------------------|-------------------------------|
| serious | Short, respectful | One question at a time |
| official | Formal «Вы» | Keep business tone |
| playful | Warm, BY context | Joke to ease corridor wait, not to argue |

---

## 10. Upload classifier & upload jokes

### 10.1 Purpose

Prevent running L1 on recipes/passports; guide user to re-upload. Separate from question tone.

### 10.2 Scoring

**KZ hints** (`_KZ_HINTS`): weighted regex sum. Strong signals: «консультативное заключение» (+3), «жалобы», «диагноз» (+2).

**Lab score:** +4 per parsed marker (cap 24), +4 lab header keywords (Invitro, Synlab, биохимия…), +2 unit patterns.

### 10.3 KZ slot classification (summary)

```
len < 40 → empty
lab >= 14 AND kz < 8 → lab_in_kz
kz >= 10 → OK
foreign pattern match → recipe/menu/receipt/... (14 types)
kz >= 5 AND len > 120 → weak OK
kz < 4 → unknown
else → OK
```

### 10.4 Foreign document types (`_GUESS_PATTERNS`)

| kind | label_ru |
|------|----------|
| recipe | рецепт блюда |
| menu | меню или прайс кафе |
| receipt | кассовый чек |
| passport | паспорт |
| homework | школьная тетрадь |
| contract | договор |
| resume | резюме |
| ticket | билет |
| social | скриншот соцсетей |
| invoice | счёт на оплату |
| parking | парковочный талон |
| pet | ветеринарная выписка |

Plus slot errors: `lab_in_kz`, `kz_in_lab`, `empty`, `unknown`, `lab_unknown`.

### 10.5 Joke payload (`_JOKES`)

Each: `{ emoji, title, body }`. Examples:

- **recipe:** «Борщ мы уважаем, но сверить с протоколом Минздрава пока не научились»
- **lab_in_kz:** «Загрузите анализы в блок ниже»
- **empty:** «Текста почти нет - как анализ без крови»

`_joke_for_guess()` may prefix: «Похоже, вы прислали: {label_ru}.»

### 10.6 UI rendering

`renderUploadJokeCard(pr)` in `patient-ui.js`:

- Hides `#result-body`
- Shows `#upload-joke-card` with emoji, title, body, hint
- `btn-again` → «Загрузить правильный документ»

---

## 11. Cross-check modules

### 11.1 Lab crosscheck

**Input:** `kz_text`, `lab_text` (concatenated OCR).  
**Process:** `extract_lab_markers` → for each marker check presence in KZ (with aliases).  
**Output:** table + `missing_in_kz_lines` → may become question intent `labs_missing_in_kz`.

**Known weakness:** OCR on phone photos; limited lab format templates (Invitro/Synlab heuristics only).

### 11.2 Protocol crosscheck

Reads L1 `compliance.exam_assessments` where status ∈ `missing_required`, `missing_conditional`.  
Checks if exam name appears in KZ or lab text.  
Builds `patient_note_ru` for questions and `priority_topics`.

### 11.3 Exams enrich

If exams block score < 75 or has gaps → human-readable note appended to exams block summary and may trigger `exams_plan` question.

### 11.4 P2 enrich

For blocks with status != ok → `plain_narratives[]` with rule-based text (no LLM).  
Adds next_step «Прочитайте пояснения…».

---

## 12. Monetization & white-label

### 12.1 Product tiers

| tier_id | BYN | API tier | Includes |
|---------|-----|----------|----------|
| promo | 2.99 | P1 | basic report |
| basic | 4.99 | P1 | + citations |
| plus | 6.99 | P1 | + lab + protocol crosscheck |
| detailed | 9.99 | P2 | + plain_narratives |
| onco | 14.99 | P2 | + safety priority |

### 12.2 Config flow

```
data/patient_monetization.json
  ← PUT /api/methodist/patient-monetization (index.html tab)
  → GET /api/patient/status → monetization_public_view()
  → patient-ui.js renderMonetizationUi(), renderTierBar()
```

Defaults: monetization **off**, free demo.

### 12.3 Payment (dev)

`create_payment_session` → `payment_token` dev-* → redirect `/patient.html?paid=TOKEN&tier=`.  
`verify_payment_token`: accepts dev tokens when `payment_required()`.

### 12.4 White-label

URL: `/patient.html?clinic=kravira&tier=promo`

`loadClinic()` sets banner, `--g600`, brand lockup name/tagline, footer in disclaimer.

---

## 13. Frontend architecture (deep)

### 13.1 `patient.html` section map

| Section ID | Phase | Content |
|------------|-------|---------|
| `#onboard` | First visit | What we check / don't check |
| `#form-card` | Input | Upload zones, demographics, tone, tiers, consent, submit |
| `#loader` | Processing | Full-screen SSE progress |
| `#result-card` | Output | Hero, questions, blocks, labs, protocol |
| `#upload-joke-card` | Error | Wrong document |
| `#faq` | Help | Static FAQ |
| `.foot` | - | Disclaimer, clear data, version |

### 13.2 `renderReport(pr)` order (Results IA v2)

```
1. upload_mismatch? → renderUploadJokeCard → STOP
2. headline_ru, traffic pill, quality banner, score ring
3. protocol strip (PDF chips)
4. plain_summary_ru
5. next_steps_ru (ul)
6. document_read_back_ru
7. action_checklist → renderQuestionCards (MAIN CTA)
8. priority_topics
9. blocks → renderBlocksPanel (collapsible details)
10. plain_narratives (P2)
11. lab_crosscheck panel
12. protocol_context panel
13. protocol_citations
14. disclaimer + clinic footer
15. saveHistory + analytics
```

### 13.3 Key JS functions

| Function | Role |
|----------|------|
| `wireUploadZone` | Camera vs file pick, merge file lists |
| `buildFormData` | Multipart assembly |
| `runReviewSse` | SSE with fallback to fetch |
| `renderQuestionCards` | Checklist + tone styling + localStorage checkmarks |
| `renderBlocksPanel` | 8 blocks with why_ru, protocol excerpt |
| `buildShareText` | Plain text for Web Share |
| `syncPatientMonetizationFromApi` | Tier/payment UI |
| `startPaymentSession` | POST payment → redirect or token |
| `restoreReport` | sessionStorage on reload |

### 13.4 Client storage

| Key | Storage | Purpose |
|-----|---------|---------|
| `protocol_patient_history_v2` | localStorage | 5 entries: ts, pct, label |
| `protocol_patient_checklist_v1` | localStorage | question id → checked |
| `protocol_patient_last_report_v3` | sessionStorage | Full report (tab reload) |
| `protocol_patient_question_tone_v2` | localStorage | Tone preference |
| `protocol_patient_payment_token` | localStorage | Dev payment |
| `protocol_patient_onboard_done` | localStorage | Hide onboarding |
| `protocol_patient_reminder_v1` | localStorage | 48h reminder timestamp |

### 13.5 Upload UX (mobile)

Per zone (KZ, lab):

- `#kz-files-camera` / `#lab-files-camera` - `capture="environment"`, image only
- `#kz-files-pick` / `#lab-files-pick` - full accept from API (PDF, Word, images)
- In-memory `kzFilesList[]`, `labFilesList[]` with dedup
- Consent **directly above** `#btn-check`

### 13.6 PWA / install

- `setupInstallHint()` - `beforeinstallprompt` on Android/desktop
- iOS: manual «Add to Home Screen» (no native prompt)
- SW: network-first prevents stale JS after deploy

---

## 14. Design system

### 14.1 Brand Protocol

- **Product name (B2C):** «Проверь КЗ» · powered by Protocol
- **Wordmark:** `protocol-logo-wordmark.svg` (hero, footer)
- **Sticky mini:** `protocol-logo-mini.svg` in top bar (letter T `#1F675B`)
- **Emblem:** `protocol-logo.svg` - Kravira shield + cross (favicon, PWA)
- **Colors:** teal family `#063d35` → `#1a8a72` → `#3db896`
- **Premium accent:** gold `#b8860b` (playful tier)
- **Fonts:** Outfit (display) + DM Sans (body)

### 14.2 CSS architecture

| Layer | File | Contents |
|-------|------|----------|
| Tokens | `patient-tokens.css` | `:root` vars, traffic pills, questions panel, tiers |
| Layout | `patient.html` `<style>` | hero, form, drop-zones, onboarding |
| Dynamic | `patient-ui.js` | SVG score ring, question cards HTML |

### 14.3 Tone panel CSS variables

```css
.questions-panel--tone-playful {
  --q-accent: rgba(184, 134, 11, 0.34);
  --q-accent-solid: #b8860b;
  --q-bg1: #fffbeb;
  ...
}
```

### 14.4 a11y checklist (partial done)

- [x] `prefers-reduced-motion`
- [x] `:focus-visible` ring
- [x] `aria-live` on status/result
- [x] Touch min 44px token
- [ ] Full keyboard path audit
- [ ] axe-core CI
- [ ] Contrast audit on `--muted`

---

## 15. Config, env, security

### 15.1 Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `PATIENT_REVIEW_ENABLED` | 1 | Master switch |
| `PATIENT_REVIEW_MAX_FILES` | 5 | KZ files |
| `PATIENT_LAB_MAX_FILES` | 3 | Lab files |
| `PATIENT_LAB_OCR` | 1 | OCR fallback for labs |
| `PATIENT_PAYMENT_REQUIRED` | 0 | Env override for payment |
| `PATIENT_MONETIZATION_CONFIG` | data/patient_monetization.json | Config path |
| `RATE_LIMIT_PATIENT_PER_MIN` | 5 | Review endpoints |

### 15.2 Privacy model

- Server: process in memory, no persistent KZ storage in code path.
- Client: history/checklist local only.
- Analytics: allowlist events + numeric meta only (`patient_analytics.py`).
- Account stub: in-memory `_SESSIONS` - not production ready.

### 15.3 Rate limiting

Separate bucket for `/api/patient/review` and `/review/stream` vs analytics (30/min).

---

## 16. Tests inventory

```bash
pytest tests/test_patient*.py tests/test_lab_result_parser.py -q
```

| File | Covers |
|------|--------|
| `test_patient_report.py` | build_patient_report fields, blocks, questions |
| `test_patient_question_tone.py` | 3 tones differ, playful dedup, respectful |
| `test_patient_upload_classifier.py` | recipe/KZ/lab slot swaps, joke report |
| `test_patient_protocol_crosscheck.py` | missing exams detection |
| `test_patient_exams_enrich.py` | exams notes |
| `test_patient_monetization_config.py` | tiers filter, payment_required |
| `test_patient_waves.py` | schema v2, low confidence cap, P2, clinic, payment |

**Not yet:**

- OpenAPI contract validation
- Playwright UI smoke
- axe a11y
- k6 load
- «B2B fields never leak» dedicated test (recommended)

---

## 17. Known gaps & improvement backlog

Prioritized for next iterations. Status from code review June 2026.

### 17.1 P0 - High user impact

| Gap | Current state | Suggested improvement | Files |
|-----|---------------|----------------------|-------|
| No page preview before upload | Files merged blindly | Thumbnail list + remove button per chip | `patient-ui.js`, CSS |
| Weak lab OCR | pytesseract optional, generic | Lab-specific templates (Invitro, Synlab, BY clinics) | `patient_lab_ocr.py`, `lab_result_parser.py` |
| Tier gating not enforced in API | All tiers get full report | Truncate questions/blocks for promo/free preview | `patient_review.py`, config |
| No idempotency | Double-click may duplicate | `Idempotency-Key` header + short TTL cache | `rag_server.py` |

### 17.2 P1 - Product completeness

| Gap | Suggested improvement |
|-----|----------------------|
| ERIP/bePaid production payment | Replace dev mock in `patient_payment.py` |
| PDF «Лист на приём» one-pager | Server or client PDF generation from `action_checklist` |
| Protocol «what it recommends overall» | Summary panel even when no gaps |
| iOS install onboarding | Custom hint for Safari «На экран Домой» |
| Figma design system | 7 screens per roadmap B |

### 17.3 P2 - Scale & quality

| Gap | Suggested improvement |
|-----|----------------------|
| OpenAPI + JSON Schema | `docs/openapi/patient.yaml`, CI validation |
| Cloud account | Replace `patient_account` in-memory with encrypted store |
| Push reminders | Capacitor plugin, 48h after check |
| Aggregated analytics for Minzdrav | Gap topic counts, no PII |
| Document scanner UX | Crop guidance, multi-page queue |

### 17.4 Technical debt

| Item | Notes |
|------|-------|
| `patient-ui.js` monolith ~1160 lines | Split: upload, report, monetization modules |
| Duplicate CSS in `patient.html` + tokens | Consolidate |
| SSE errors silently fallback | Better user message on stream fail |
| `QUESTION_TONE_CATALOG` description_ru in code vs UI catalog in JS | Single source from API only (partially done) |
| Roadmap audit table outdated | Monetization now in code - update roadmap §34 |

---

## 18. Improvement playbook for LLM

### 18.1 «Improve question quality»

1. Collect real `gaps_ru` from probe fixtures in `data/ml/reports/`.
2. Map to intents via `detect_question_intent` - add rules if gaps misfire.
3. Edit `_QUESTION_BANK` (serious/official) and `_PLAYFUL_VARIANTS`.
4. Run `pytest tests/test_patient_question_tone.py`.
5. Manual: generate report with `question_tone=playful` on sample KZ.

### 18.2 «Improve upload UX»

1. `patient.html` upload-actions pattern (camera/pick) - extend with preview/remove.
2. `syncUploadFormatsFromApi` - accept string on pick inputs only.
3. Test iOS Safari + Android Chrome file picker behavior separately.

### 18.3 «Add report field»

1. Compute in `build_patient_report()`.
2. Add to `renderReport()` or sub-render function.
3. Bump `report_schema_version` if breaking.
4. Add test in `test_patient_report.py` or `test_patient_waves.py`.
5. Document field in this file §7.

### 18.4 «Enable monetization pilot»

1. Methodist UI → enable monetization, pick tiers.
2. Or edit `data/patient_monetization.json` directly.
3. Set `payment_required` + integrate real payment provider.
4. Test flow: tier select → pay → token → review.

### 18.5 «Improve upload jokes»

1. Add regex to `_GUESS_PATTERNS`.
2. Add entry to `_JOKES` with emoji, title, body (BY-friendly humor).
3. Test in `test_patient_upload_classifier.py`.
4. Verify `renderUploadJokeCard` displays new fields.

### 18.6 Pre-commit checklist

- [ ] `pytest tests/test_patient*.py -q`
- [ ] No em dash in UI strings touched
- [ ] `BUILD_VERSION` bumped in `rag_server.py`
- [ ] `patient-sw.js` CACHE incremented if static assets changed
- [ ] Playful: no duplicate variants in same report (test)

---

## 19. Invariants & glossary

### 19.1 Invariants

1. Patient API never returns ЦИСЗ / send_gate / raw `structured_analysis`.
2. Consent required - HTTP 400 without.
3. Upload mismatch skips L1.
4. Max 8 doctor questions.
5. Playful: dedup when variants available; no insults to doctor.
6. Files not persisted server-side (ops must enforce).
7. UI short hyphen `-`.

### 19.2 Glossary

| Term | Meaning |
|------|---------|
| **КЗ** | Консультативное заключение |
| **КП** | Клинический протокол Минздрава РБ |
| **L1** | Structured review + deterministic alignment |
| **P1/P2** | Patient product depth (P2 adds narratives) |
| **intent** | Question template key |
| **slot** | N-th occurrence of intent in one report |
| **alignment_card** | One section score from L1 |
| **upload joke** | Wrong file humor response |
| **playful tone** | Humorous doctor questions |
| **read-back** | Echo of parsed document sections for trust |

---

*Document maintained for B2C evolution. When implementing roadmap items, update §17 status and this file's «Last aligned» date.*
