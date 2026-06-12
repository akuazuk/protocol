# Feedback events (без ПДн)

JSONL-файлы: одна строка = один объект. Поле `event_type` обязательно.

## l0_screen

```json
{
  "event_type": "l0_screen",
  "ts": "2026-06-01T12:00:00Z",
  "text_hash": "sha256:…",
  "tier": "L0",
  "gate_score": 72,
  "latency_ms": 840,
  "rubric": "gastroenterologiya",
  "send_decision": "allowed_with_warnings"
}
```

## retrieval_fix

```json
{
  "event_type": "retrieval_fix",
  "ts": "2026-06-01T12:00:00Z",
  "query": "K21.9 ГЭРБ изжога",
  "rejected_path": "gastroenterologiya/old.pdf",
  "chosen_path": "gastroenterologiya/kp_gerd.pdf",
  "reviewer": "methodist"
}
```

## methodist_override

```json
{
  "event_type": "methodist_override",
  "ts": "2026-06-01T12:00:00Z",
  "rule_id": "required_exam_egds",
  "text_hash": "sha256:…",
  "system_pass": true,
  "human_pass": false,
  "note": "ЭГДС указана сокращённо"
}
```

## kz_analysis (автолог прогона в режиме методиста)

```json
{
  "event_type": "kz_analysis",
  "analysis_id": "uuid",
  "ts": "2026-06-01T12:00:00Z",
  "text_hash": "sha256:…",
  "tier": "L0",
  "gate_score": 72,
  "send_decision": "allowed_with_warnings",
  "sandbox": false
}
```

## analysis_review (оценка методиста)

```json
{
  "event_type": "analysis_review",
  "analysis_id": "uuid",
  "text_hash": "sha256:…",
  "rating": 4,
  "verdict": "mostly_correct",
  "tags": ["false_positive_rule"],
  "reviewer": "И.И.",
  "overrides": [
    {"rule_id": "required_exam_egds", "system_pass": true, "human_pass": false, "note": "…"}
  ],
  "retrieval_fix": {
    "query": "ГЭРБ",
    "rejected_path": "gastro/old.pdf",
    "chosen_path": "gastro/kp_gerd.pdf"
  }
}
```

API: `POST /api/ml/feedback` (заголовок `X-Methodist-Token`).  
Полный текст КЗ: `data/ml/secure/kz_text/{hash}.txt` (не в git).  
Снимок ответа: `data/ml/analyses/{analysis_id}.json`.

Экспорт: `python3 scripts/export_training_feedback.py`
