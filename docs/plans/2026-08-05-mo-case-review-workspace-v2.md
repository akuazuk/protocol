# МО: workspace «Разбор случая» + обучение + протоколы МЗ (v2)

Дата: 2026-08-05  
Статус: active (W0-W3 в проде: PR #20, `2026-08-05-r19-mo-case-review-workspace`)  
Преемник: заменяет `2026-08-05-mo-case-review-workspace-v1.md`  
Связанные: `2026-08-05-mo-methodist-review-pack-v1.md` (хранение пакетов - уже в проде),
`2026-08-05-mo-case-protocol-suggest-v1.md` (движок подбора КП),
`2026-08-05-mo-llm-action-queue-judge-v1.md`,
`2026-08-05-mo-expert-reviewer-portal-v1.md`.
Handoff: `docs/handoff/2026-08-06-next-agent-mo-review.md`.

---

## 1. Контекст

Нужно не только улучшить UI разбора, но и замкнуть цикл:

```text
МО → авторазбор (scorer + LLM) → методист → пакет разбора
        → gold / feedback → улучшение точности
        → подбор КП МЗ РБ → оценка подбора методистом → улучшение suggest
```

Что уже есть в проде:

| Компонент | Где | Статус |
|--|--|--|
| Пакет разбора | SQLite `crm_review_pack` на Render disk (`MO_ANALYTICS_DB`) | работает |
| CRM статус | `crm_case_state` / `crm_case_event` | работает |
| Feedback consult | `data/ml/feedback/*.jsonl` (`feedback_store.py`) | другой контур (consult), паттерн переиспользуем |
| Подбор КП | `match_protocol_cards` / план Case→Protocol Suggest | **не встроен** в разбор МО |
| Export gold nightly | план review-pack фаза C | **не сделано** |

---

## 2. Цель и метрики

### UI (из v1)

| Метрика | Было | Цель |
|--|--|--|
| Sticky | форма решения | **МО слева** закреплено / свой scroll; справа разбор |
| Поля `%` | 3 number input | **удалены** |
| Textarea | rows=6 / 4000 | ≥12 строк / **12000** |
| EN в drawer | много | **0** видимых EN (кроме МКБ/ID) |
| Таблица дня | пагинация 50, sort select | день целиком + фильтры + sort по th |

### Данные и обучение (новое в v2)

| Метрика | Было | Цель v2 |
|--|--|--|
| Куда пишется решение | `crm_review_pack` + CRM | то же + **явный export path** для обучения |
| Состав пакета | clinical + system + decision | + **protocol_suggest snapshot** + **оценки подбора КП** |
| Использование gold | флаг `training_use`, нет пайплайна | weekly/nightly JSONL → eval LLM/scorer + suggest |
| КП в разборе | нет | top-3 КП МЗ + reasons + оценка методиста |
| Согласие по подбору КП | нет | ≥60% «полезно» на первых 50 размеченных |

---

## 3. Куда сохраняются данные после разбора

### 3.1 Primary store (OLTP, уже выбран)

**Render persistent disk**, файл витрины:

```text
/var/data/medical_exams/warehouse/mo_analytics.sqlite
```

Таблицы:

1. **`crm_review_pack`** - полная версия решения (append-only, каждая «Сохранить» = новый `pack_id`).
2. **`crm_case_state`** - текущий статус очереди (быстрый индекс).
3. **`crm_case_event`** - audit trail смены статуса / назначения.

**Не пишем** решения методиста в MariaDB МИС и не делаем BigQuery OLTP.

### 3.2 Что лежит в одном пакете (расширение схемы decision/system)

Текущие поля остаются. В v2 добавляем в JSON (без ломки колонок таблицы):

```json
{
  "pack_id": "uuid",
  "case_id": "3646270",
  "visit_id": "3646270",
  "patient_id": "…",
  "visit_date": "2026-08-04",
  "clinical_snapshot": { "complaints": "…", "…": "…" },
  "system_snapshot": {
    "overall_pct": 72,
    "findings": [],
    "llm_action_judge": {},
    "rubric_mz": {},
    "protocol_suggest": {
      "engine": "case_protocol_suggest_v1",
      "generated_at": "…",
      "items": [
        {
          "protocol_id": "…",
          "title": "…",
          "source_path": "minzdrav_protocols/…",
          "score": 78,
          "match_kind": "clinical|code_only|ddx|specialty",
          "reasons": [{"code": "…", "text": "…"}]
        }
      ]
    }
  },
  "methodist_decision": {
    "status": "confirmed_issue",
    "verdict_completeness": "agree|partial|disagree|unreviewed",
    "verdict_diagnosis": "agree|partial|disagree|unreviewed",
    "verdict_recommendations": "agree|partial|disagree|unreviewed",
    "summary_ru": "…",
    "finding_decisions": { "FINDING_CODE": "confirmed|false_positive|…" },
    "protocol_ratings": [
      {
        "protocol_id": "…",
        "relevance": "relevant|partial|irrelevant|unreviewed",
        "rank_ok": true,
        "note_ru": "опционально"
      }
    ],
    "training_use": true,
    "assignee": "",
    "due_date": "",
    "tags": []
  },
  "actor": "methodist@… / expert@…",
  "created_at": "…",
  "supersedes_pack_id": null
}
```

Правила:

- `corrected_scores` **не собираем в UI** (поля `%` убраны); старые пакеты читаются.
- `training_use=false` → пакет **не** попадает в gold export.
- `patient_id` только для methodist/lead/admin/expert; в публичные API не отдавать.
- Клинический текст копируется в snapshot (не зависит от ротации CSV).

### 3.3 Secondary store (для обучения, offline)

Nightly / weekly job (после накопления ≥50 пакетов с `training_use=1`):

```text
/var/data/medical_exams/gold_review/YYYY-MM-DD/
  review_packs.jsonl          # без лишних ПДн или с хешем patient_id
  protocol_ratings.jsonl      # плоский срез оценок КП
  manifest.json               # counts, build_version, date range
```

Опционально позже: sync → GCS → BigQuery **только как аналитическая витрина**, не как запись из UI.

Паттерн записи событий (по аналогии с consult):

```text
data/ml/feedback/mo_review_*.jsonl   # на Mac/dev
или на Render disk рядом с gold_review
```

Типы событий:

| event_type | Смысл |
|--|--|
| `mo_review_pack_saved` | сохранён пакет (ссылка pack_id) |
| `mo_finding_label` | confirmed / false_positive по finding |
| `mo_llm_verdict` | agree/partial/disagree по 3 вопросам |
| `mo_protocol_rating` | оценка релевантности предложенного КП |

---

## 4. Как данные улучшают точность оценок

Три независимых контура улучшения (не смешивать в один score):

### 4.A Качество оформления / deep scorer (L1)

**Вход gold:** `finding_decisions` + `summary_ru` + clinical snapshot.

| Сигнал методиста | Использование |
|--|--|
| `false_positive` на finding | снизить вес / поправить rule / regex; eval precision findings |
| `confirmed` + P0/P1 | positive examples для regression tests |
| `needs_more_data` | не штрафовать scorer как «ошибка модели» |
| `verdict_* = disagree` при высоком LLM KPI | калибровка LLM-judge промпта / порогов escalate |
| `verdict_* = agree` | confidence, что shadow можно подтягивать к primary позже |

Пайплайн:

1. Export gold JSONL.
2. `scripts/eval_mo_review_gold.py` (новый): confusion LLM vs methodist по 3 вопросам; FP-rate findings.
3. Отчёт в `ml/experiments/mo_review_gold_YYYY-MM-DD/REPORT.md`.
4. Правки rules/scorer только после отчёта (не авто-обучение в v2).

### 4.B LLM action-judge (полнота / диагноз / план)

| Сигнал | Использование |
|--|--|
| agree/partial/disagree × 3 | калибровка промпта A/B, порог needs_human |
| длинный `summary_ru` | qualitative review + few-shot examples (вручную курируемые) |

Не fine-tune Gemini в v2 - только prompt/eval loop.

### 4.C Подбор протоколов МЗ РБ (suggest)

| Сигнал | Использование |
|--|--|
| `relevance=relevant` + rank_ok | Hit@3 positive |
| `relevance=irrelevant` | hard negative для recall/score |
| `partial` + note | подкрутка weights / audience filters |
| пустой suggest + методист добавил КП вручную (v2.1) | coverage gap корпуса |

Eval: расширить `eval/case_protocol_suggest/` обезличенными fact graphs из gold (без PHI в git - хеш case_id).

---

## 5. Протоколы МЗ РБ в «Разборе случая»

Опираемся на план `mo-case-protocol-suggest-v1`, встраиваем в workspace.

### 5.1 Когда считаем suggest

При открытии case detail (или lazy по кнопке «Подобрать протоколы»):

1. Собрать CaseFactGraph из clinical slots + ICD + specialty + age/audience.
2. Добавить gaps из LLM-judge / findings (если есть).
3. Вызвать `case_protocol_suggest` → top-3..5.
4. Показать блок в **правой колонке** (над или под LLM-разбором), явно отделённый от L1 score.

### 5.2 UI-блок «Протоколы МЗ к случаю»

```text
Протоколы МЗ РБ (подбор, не оценка оформления)
┌─────────────────────────────────────────────────────────┐
│ 1. «…название…»     [Клиника]  78                        │
│    почему: закрывает отёк колена; детская аудитория     │
│    [Открыть КП]  Релевантность: ○ да ○ частично ○ нет   │
├─────────────────────────────────────────────────────────┤
│ 2. …                                                    │
└─────────────────────────────────────────────────────────┘
[+ Добавить другой протокол из каталога]  (v2.1, если успеем)
```

Оценки релевантности:

- обязательны для save? **нет** в v1 UI-итерации; **желательны** если методист трогал блок;
- при «Сохранить пакет» пишем `protocol_ratings` + snapshot `protocol_suggest` как было на экране;
- если методист не оценивал - `unreviewed` (не считать ни hit, ни miss в eval).

### 5.3 Связь с точностью

```text
suggest (детерминированный)
   → методист relevance
   → gold_review/protocol_ratings.jsonl
   → weekly: Hit@3 / FP@3 / audience mismatch rate
   → правка весов §4.1 в case_protocol_suggest / filters
```

LLM-rerank протоколов - только фаза S3 suggest-плана, после стабильного thumbs.

---

## 6. Макет workspace (UI)

```text
┌─ Разбор случая · визит · пациент · врач · [←][→] [×] ──────────────────┐
│ ┌─ МО sticky ───────────────────┐  ┌─ Разбор scroll ─────────────────┐ │
│ │ клинические слоты             │  │ LLM: 3 вопроса (RU)             │ │
│ │ подсветка linked fields       │  │ Протоколы МЗ + оценки           │ │
│ │ (details) оси / рубрика МЗ    │  │ Findings + решения              │ │
│ │                               │  │ Решение методиста               │ │
│ │                               │  │   вердикты (без % )             │ │
│ │                               │  │   ★ Развёрнутый разбор          │ │
│ │                               │  │   [x] для обучения              │ │
│ │                               │  │ История пакетов                 │ │
│ └───────────────────────────────┘  └─────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

Scroll: два pane; убрать sticky с формы решения. Mobile: стек.

Форма решения: без Полнота%/Диагноз%/Рек.%; textarea крупная; чекбокс «Можно использовать для обучения».

---

## 7. Русификация

Как в v1: shadow→черновик, Follow-up→наблюдение, n/a→нет данных, gold→для обучения,
conf/model/engine спрятать или перевести, outcome→ок/противоречие/пробел.
Match badges: Клиника / Только код / Дифдиагноз / Специальность.

---

## 8. Полная таблица за каждый день

- Deep-link из Отчёты/Вчера: `date_from=date_to=D` → «Все случаи».
- Фильтры multi-select + поиск.
- Sort кликом по th (+ API whitelist).
- Дневной `page_size=100` или «все за день» при N≤500.
- Колонка «Пакет» / «Есть разбор» (есть ли review_pack) - полезно для методиста.

---

## 9. Шаги реализации (после «приступать»)

### Фаза W0 - UI workspace (быстро)

- [x] P0 Layout: sticky/own-scroll МО слева, разбор справа.
- [x] P0 Форма: убрать `%`, увеличить textarea; save без corrected_scores.
- [x] P0 RU drawer.
- [x] P1 Таблица дня + sort по th + page_size.
- [x] P1 Prev/Next в очереди.

### Фаза W1 - данные / обучение (контракт)

- [x] Расширить `system_snapshot` / `decision_json` полями `protocol_suggest` + `protocol_ratings` (без ALTER, JSON).
- [x] API save/load уже через review-pack - принять новые ключи, валидировать enum relevance.
- [x] Скрипт export: `scripts/export_mo_review_gold.py` → `gold_review/`.
- [x] Скрипт eval: `scripts/eval_mo_review_gold.py` (LLM vs methodist + finding FP).
- [ ] Документировать в плане метрики «было/стало» после первого export (нужны реальные packs).

### Фаза W2 - протоколы в разборе

- [x] Engine MVP (`case_protocol_suggest.py` поверх `match_protocol_cards` + gaps).
- [x] Endpoint `GET .../cases/{id}/protocol-suggest`.
- [x] UI-блок + radio relevance + запись в pack.
- [x] Feature flag `CASE_PROTOCOL_SUGGEST` (default on).
- [x] Smoke на проде: `/api/version` = r19; endpoint protocol-suggest отвечает 403 без токена (жив).

### Фаза W3 - ship

- [x] Тесты API/UI; BUILD_VERSION r19; PR #20 squash-merge; Render release live (`c6bc7046`).

---

## 10. Риски

| Риск | Митигация |
|--|--|
| Путают «балл МО» и «подбор КП» | разные блоки, подписи RU, разный engine id |
| ПДн в gold export | хеш patient_id; training_use gate; не в git |
| Suggest тормозит открытие кейса | lazy load / кнопка; timeout; кэш на case_id |
| Мало разметок для eval | не трогать primary scorer до n≥50 training packs |
| Двойной источник правды (CRM vs pack) | pack = полный снимок; CRM = индекс статуса |

---

## 11. Definition of Done

1. UI: МО слева закреплено, разбор справа скроллится; нет полей `%`; textarea крупная; RU.
2. Таблица за день с фильтрами и сортировкой.
3. При сохранении пакета на Render в `crm_review_pack` пишутся вердикты, summary, findings labels,
   snapshot suggest (если был) и `protocol_ratings`.
4. Есть CLI export gold + черновик eval-отчёта (можно пустой, если пакетов мало).
5. В разборе видны top-3 КП МЗ с возможностью отметить релевантность; оценка уходит в пакет.
6. methodist и expert ведут себя одинаково; версия задеплоена.

---

## 12. Вне скоупа v2

- Fine-tune моделей.
- Авто-правка rules без human review отчёта.
- Запись решений в МИС MariaDB.
- BigQuery как online store.
- Полный LLM на все кейсы дня (остаётся night queue).

---

## 13. Статус после ship (2026-08-05 вечер)

Задеплоено. Осталось из плана: метрики после первого gold export (нужны packs с
`training_use`), улучшения suggest (audience/DDx) - в связанном плане protocol-suggest.
LLM August night-queue продолжает крутиться на Render (не убивать без нужды;
после deploy - перезапуск supervisor).
