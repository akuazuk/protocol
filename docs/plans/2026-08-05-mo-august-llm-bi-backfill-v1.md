# МО: LLM backfill с августа + continuous + починка BI (v1)

Дата: 2026-08-05  
Статус: active  
Связанные: `2026-08-05-mo-llm-action-queue-judge-v1.md`,
`2026-08-04-mo-runtime-stabilization-v1.md`, `2026-07-28-mo-daily-bi-platform-v1.md`.

---

## 1. Контекст

Нужно: прогнать августовские МО через LLM, автоматически гонять новые дни,
заполнить таблицы/графики, починить фильтры.

На Render на 2026-08-05:

| Артефакт | 08-01..04 |
|--|--|
| deep `*_cases.jsonl` | есть |
| night `*_llm_grades.jsonl` (~80/день) | есть |
| action-judge `judges.jsonl` | только 08-04 |
| `fact_llm_usage` | пусто |
| графики врачей / specialty / filial | сломаны |

### Корневая ошибка BI

`merge_sql` в `mo_publish.py` делает `INSERT OR REPLACE INTO t SELECT * FROM pub.t`.
Прод-витрина создана старым `CREATE` + `ALTER ADD` (колонки в конце), снапшот -
новым `CREATE` (другой порядок). `SELECT *` пишет **по позиции** →
`doctor_key`=`status`, `specialty`=`scorer_version`, `filial`=`schema`, и т.д.

---

## 2. Метрики

| Метрика | Было | Цель |
|--|--|--|
| `fact_mo_case.doctor_key` за август | статусы good/review | hash FIO |
| specialty/filial | v4.0.0 / 4.0 | реальные метки |
| action-judge за 08-01..вчера | только 04 | все дни |
| `llm_queue_pending` при drained grades | 80 (stale report) | 0 |
| doctor_outliers / doctor_case_mix | available=false | true при n |
| auto LLM на новый день | только night в pipeline | + action-judge post-step |

---

## 3. Шаги

- [x] P0: починить `merge_sql` (named columns) + тест на разный порядок колонок
- [x] P1: на Render rebuild `fact_mo_case`/`dim_*` за август из secure CSV+cases
- [x] P2: пересобрать report.json после night LLM (`llm_queue_pending=0`) - `recompute_mo_days` без pandas, CSV из secure_cases
- [x] P3: action-judge batch 2026-08-01..04 на Render disk
- [x] P4: continuous: `run_mo_render_llm_backfill.sh` + post-step launchd `llm-yesterday`
- [x] P5: doctor charts: `enough_data` без жёсткого R²; фильтры facets на месте
- [x] P5b: `usage_date` = visit_date; таблица покрытия night/action-judge в «Расходы AI»
- [x] P6: BUILD_VERSION r16, PR #17 merge + deploy; night LLM backfill 08-01..04 на Render (DONE 80/день)
- [x] P7: фильтр филиала - `|` + heuristic CSV; r17 (#18)
- [x] P8: grades 01-04 DONE; recompute fix `2026-08-06-r1-recompute-no-pandas`
- [ ] P9: день **2026-08-05+** на Render disk (сейчас нет secure/report за 05)
- [ ] P9b: лаунчер backfill - static `mo_llm_range_runner.sh` (nested heredoc схлопывал `$DATA`); false ALREADY_RUNNING от pgrep
- [ ] P9c: «Врачи ниже ожидаемого» - убрать жёсткий `case_mix_reliable` за день (R² < 0.30 почти всегда)
- [ ] P10: repair **июльского** warehouse (bad doctor_key ~11902/13591)
- [ ] P11: gold export после ≥50 `training_use` packs (сейчас 0)

---

## 4. Риски

| Риск | Митигация |
|--|--|
| Полный LLM на все ~1500 кейсов дорого | night queue 80 + action ≤20; full corpus не в v1 |
| Rebuild затрёт CRM | только fact_/dim_; CRM_TABLES не трогать |
| Mac без medical_exams | работать по SSH на Render disk |
| Gemini geo-block на Mac | action-judge с ключом; при fail - Render shell |

---

## 5. Definition of Done

1. Named-column merge в проде; August doctor/specialty/filial читаемы.
2. Action-judge JSONL за все августовские дни на Render disk.
3. Launchd/post-pipeline гоняет judge для нового дня.
4. Графики врачей и фильтры specialty/filial/doctor отвечают на данных августа.
