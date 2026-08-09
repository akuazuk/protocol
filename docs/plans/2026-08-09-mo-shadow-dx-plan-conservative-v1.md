# МО: shadow Dx/Plan (вариант B), консервативные пороги

Дата: 2026-08-09
Статус: **active**

Связанные планы:
- `2026-08-09-mo-score-ssot-llm-recompute-v3.md` - калибровка и SSOT (C ещё закрыт);
- `2026-08-09-mo-calibration-confirmatory-proxy-v1.md` - C9A winners;
- `2026-08-09-mo-calibration-llm-methodist-proxy-v1.md` - C6B/C7 proxy-gold;
- `2026-08-08-mo-action-queue-precise-signals-v2.md` - очередь (не ломать).

---

## 1. Контекст и решение владельца

Выбран **вариант B**: показать калибровочные Dx/Plan как **shadow-подсказку**,
не меняя официальные scores, №55, zones SSOT и полный recompute.

Дополнительное требование владельца: показатели **не слишком жёсткие**.
Красное / «плохо» / «критично» только там, где кейс **действительно poor или
critical**. `partial` не считается браком для внимания и очереди.

Опора на C9A (July n=100, proxy-gold, production_rollout=false):

| Endpoint | Лучший кандидат | Proxy bad share |
|--|--|--|
| Dx | blind LLM (`pass_1` / mean / adjudicated при 1 pass) | 16% |
| Plan | `ensemble.arm_d_blind_mean` = mean(concordance, blind plan) | 35% |

Текущие `zone2a` и один `clinical_concordance` плохо ловят брак и/или шумят -
в B они **не** становятся источником красного флага.

---

## 2. Что изменено в production (цель плана)

### Меняем

1. Night/GCE считает shadow Endpoint C/D для clinical_visit МО.
2. Результаты кладутся в **отдельные shadow-поля** (не primary score columns).
3. В разборе случая - блок «Калибровка (shadow)» с явной подписью.
4. Опционально позже: мягкий фильтр очереди по shadow-attention.

### Не меняем

- `overall_pct`, `overall_pct_v3`, axes, zones, rubric, `reg55_section_pct`;
- формулу warehouse SSOT и historical recompute;
- primary action-queue reason (в волне 1);
- объявление human/methodist gold.

---

## 3. Консервативная семантика внимания

### 3.1 Сырые scores (считаем всегда)

| Поле | Источник |
|--|--|
| `shadow_dx_evidence_pct` + `shadow_dx_verdict` | blind LLM Dx, 1 pass, flash-класс модель для масштаба |
| `shadow_plan_llm_pct` + `shadow_plan_verdict` | blind LLM Plan (KP-grounded или no-KP route) |
| `shadow_plan_ensemble_pct` | mean(`axis.clinical_concordance`, `shadow_plan_llm_pct`) когда оба числовые |

`blocked` / `na` → attention = none, в UI «нет оценки / недостаточно данных».

### 3.2 Attention band (то, что красим)

Только по **verdict**, не по мягкому порогу 55:

| Band | Условие | UI |
|--|--|--|
| `none` | нет payload, blocked, na, good, **partial** | без красного |
| `poor` | verdict == `poor` | «Плохо (shadow)» |
| `critical` | verdict == `critical` **или** (`poor` и `potential_harm=true`) | «Критично (shadow)» |

Дополнительный числовой предохранитель (чтобы LLM не ставил poor при высоком %):

- `poor` принимаем только если score_pct ≤ **45**;
- `critical` принимаем только если score_pct ≤ **30** **или** `potential_harm=true` при score_pct ≤ 45;
- если verdict poor/critical, но score выше порога → понижаем band до `none` и пишем `softened=true` в audit (не молча теряем сигнал: виден в debug/JSON).

Для **плана** attention берём от:

1. primary: `shadow_plan_verdict` (LLM) с теми же soften-правилами;
2. ensemble **не** повышает band выше LLM (чтобы concordance не ужесточал);
3. ensemble может только **понизить** показ severity на один шаг, если
   `shadow_plan_ensemble_pct ≥ 60` и LLM = poor без harm → band `none`
   (согласованность движка говорит «не всё плохо»).

Для **Dx** ensemble с zone2a **не используем** (C9A: zone2a слаб).

### 3.3 Почему не порог 55

На C9A blind plan при &lt;55 красил ~38% кейсов. Владелец просит не раздувать
красные. Verdict poor/critical + числовой soften режет «почти плохо» (`partial`).

Ожидаемый порядок величины attention (ориентир, не KPI-контракт):

- Dx attention: заметно ниже 16% labeled-bad (часть bad = soft/partial);
- Plan attention: заметно ниже 35%;
- цель волны 1: **precision важнее recall** (лучше пропустить серое, чем залить очередь).

---

## 4. UX

### Волна B1 (обязательная) - только разбор случая

В case review отдельный блок ниже официальных зон/№55:

- заголовок: «Клиническая калибровка (shadow) - не официальная оценка»;
- две строки: Диагноз / План;
- для каждой: band, краткий `summary_ru` (≤200 символов), score если есть;
- `partial`/`good` - нейтральный текст без красного;
- `poor`/`critical` - цвет предупреждения, без замены overall.

Не показывать engine dumps, второй LLM pass, proxy-gold labels.

### Волна B2 (после B1 smoke) - очередь

- фильтр «Только shadow: плохо/критично»;
- **не** менять primary sort/reason;
- бейдж на карточке очереди только при band poor/critical.

### Волна B3 (отдельно, не в этом плане) - soft boost

Только по явному решению владельца после 1-2 недель B1/B2.

---

## 5. Runtime и данные

1. Счёт только на **GCE** (`MO_LLM_EXECUTION_HOST=gce`), не с Mac.
2. Модель волны 1: `gemini-3.6-flash` (как confirmatory blind); Pro - opt-in escalate.
3. Resume/idempotent write по `mis_id` / `visit_id` + content hash.
4. Хранение: shadow sidecar или отдельные колонки
   `shadow_dx_*` / `shadow_plan_*` / `shadow_attention_*` - **не** перетирать
   `overall_pct` / zones / reg55.
5. PHI: clinical text не в git/PR/handoff; в UI только уже открытый case scope.
6. Каталог: запись в `docs/methodist/mo-evaluation-catalog.md` - shadow, не SSOT.

---

## 6. Метрики

| Метрика | Было | Цель волны 1 |
|--|--|--|
| Official scores / SSOT | без изменений | без изменений |
| Primary queue reason | прежний | прежний |
| Shadow coverage clinical_visit (GCE days) | 0% | ≥80% за пилотный горизонт 7 дней |
| Share cases с Dx attention poor/critical | n/a | ориентир ≤10% (не KPI-fail) |
| Share cases с Plan attention poor/critical | n/a | ориентир ≤15% |
| Softened verdict rate (verdict poor но band none) | n/a | мониторинг, не zero |
| Geo/API errors в shadow runner | - | 0 устойчивых geo |
| Подпись «не официальная оценка» в UI | нет | 100% shadow surfaces |

---

## 7. Шаги

### Волна 0 - контракт (1 PR)

- [ ] **W0.1** Зафиксировать JSON schema shadow payload + attention rules в коде/тестах.
- [ ] **W0.2** Unit tests: partial→none; poor@80→softened none; critical+harm→critical;
  plan ensemble downgrade; blocked→none.
- [ ] **W0.3** Каталог методиста: новые поля помечены shadow.

### Волна 1 - B1 compute + case UI

- [ ] **W1.1** GCE runner: shadow Dx/Plan 1-pass flash, resume, write sidecar.
- [ ] **W1.2** API case detail отдаёт shadow block (`no-store` если нужно).
- [ ] **W1.3** UI блок в разборе случая + copy про неофициальность.
- [ ] **W1.4** Smoke на GCE: 20 clinical cases, ручной просмотр ≥5 poor/critical.
- [ ] **W1.5** BUILD_VERSION / PR; production official scores не меняются.

### Волна 2 - B2 очередь (отдельный PR после W1)

- [ ] **W2.1** Фильтр очереди по shadow attention.
- [ ] **W2.2** Бейдж poor/critical; primary reason не трогать.
- [ ] **W2.3** Смоук: доля attention на живом дне в пределах ориентиров; иначе
  ужесточить soften (сдвинуть 45→40) без смены official scores.

### Явно вне плана

- [ ] Вариант C / SSOT / полный recompute.
- [ ] Сделать shadow primary queue signal.
- [ ] Human methodist gold как обязательный gate для B1 (уже waived для калибровки;
  для C - нет).

---

## 8. Риски

| Риск | Митигация |
|--|--|
| LLM ставит poor слишком часто | soften по score; partial не attention; plan ensemble downgrade |
| Люди принимают shadow за SSOT | постоянная подпись; отдельный блок; не писать в overall |
| Стоимость Gemini | flash 1-pass; только clinical_visit; resume |
| Geo с Mac | только GCE runner |
| Расхождение flash vs Pro C9A | мониторинг; выборочный Pro escalate на critical |
| Очередь раздуется в B2 | B2 после метрик B1; фильтр opt-in |

---

## 9. Safe next command

После merge этого плана (docs-only) - отдельная task-ветка на W0:

```bash
scripts/ops/git_task_start.sh mo-shadow-dx-plan-w0 --pc=pc1 \
  --branch=cursor/mo-shadow-dx-plan-w0-pc1
```

Первый код: schema + unit tests attention rules, без UI и без live LLM.
