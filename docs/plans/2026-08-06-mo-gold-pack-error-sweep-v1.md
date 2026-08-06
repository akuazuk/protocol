# МО: устранение ошибок по gold review packs (v1)

Дата: 2026-08-06  
Статус: active (волна в работе / код S1-S4)  
Источник: 2 пакета в `crm_review_pack` (`3650612` уролог, `3643304` невролог) + ops gaps.  
Связанные:  
`2026-08-06-mo-icd-full-document-search-v1.md` (P0-P2 done; P3/P4 open),  
`2026-08-05-mo-case-protocol-suggest-v1.md`,  
`2026-08-06-mo-case-findings-clarity-v1.md`,  
`2026-08-05-mo-august-llm-bi-backfill-v1.md`,  
handoff: `docs/handoff/2026-08-06-afternoon-next-agent.md`.

Цель одной волны: закрыть **все подтверждённые ошибки оценки/UI/данных** из экспертных
разборов и связанный stale-state (pending LLM, старые findings), чтобы повторный
разбор тех же кейсов не повторял FP/нерелевантные КП.

---

## 1. Контекст (что сохранено)

| Pack | Кейс | Спец. | Эксперт сказал |
|--|--|--|--|
| `f236d3a5…` | **3650612** | Уролог | FP: `D_reg55_p0`, `C_nsaid_dup`. Confirmed: `B_icd_invalid`, `C_ddi`, `E_template_copy`. Все 3 КП suggest = irrelevant (челюстно-лицевые). Summary: неполный диагноз, неверный КП, ложный НПВП-dup. |
| `f75cd89f…` | **3643304** | Невролог | Summary: не учтён МКБ из «Диагноз МИС» (`M54.8` есть в `mis_diagnos`). Findings/protocols в основном **unreviewed**. Suggest слабо по теме (СМА/наследственные НМЗ). |

Клиника 3650612 (ключевой FP):  
`При болях "Ибупрофен" ("Кетопрофен", "Найз" и т.п.)` - альтернативы в скобках, не одновременный приём.

---

## 2. Метрики

| Метрика | Было (на gold) | Цель |
|--|--|--|
| `C_nsaid_dup` на 3650612 (скобки / «и т.п.») | fire P1 | **нет finding** |
| Suggest top-3 specialty mismatch (уролог → ЧЛХ) | 3/3 irrelevant | ≥2/3 релевантны или пусто с честной причиной |
| `B_icd_invalid` когда `mis_diagnos`/`M54.8` в clinical | эксперт: «не учтён» | код из MIS-слота/full-text учтён; ложный invalid нет |
| Oral НПВП + топический гель (аэртал + вольтарен-гель) | `C_nsaid_dup` | **не** dup (или severity advisory), если один путь топический |
| `D_reg55_p0` без evidence/red flags | FP P0 | не P0 / нет finding без цитаты |
| `C_ddi` evidence vs title согласованы | путаница текстов | evidence = те же препараты, что в title |
| `llm_queue_pending` при grades ok | stale 80 | **0** после recompute |
| Findings в warehouse за Aug после правил | старый L1 | пересчёт L1/deep ≥ 2026-08-01..05 |
| Packs с `training_use` и все findings unreviewed | бывает | soft-gate или warning в UI |

---

## 3. Ошибки → работы (единый backlog)

### A. Scorer / medication (P0) - gold 3650612

| ID | Ошибка | Фикс | Где |
|--|--|--|--|
| A1 | `C_nsaid_dup` на альтернативах в скобках / «или» / «и т.п.» | Парсер: сегменты в `()` после НПВП = alternatives, не co-admin; токены «или», «либо», «и т.п.», «и др.» | `medication_safety` / deep NSAID |
| A2 | Словарь: Дикловит и аналоги | Добавить в NSAID labels; сигнал только с пероральным НПВП без явного «не сочетать» | словарь НПВП |
| A3 | Oral + topical gel | Правило: системный + топический того же/другого НПВП ≠ `C_nsaid_dup` (отдельный soft finding опционально) | тот же модуль |
| A4 | `C_ddi` грязный evidence | Evidence/title из одной пары препаратов; не смешивать sertraline/NSAID | DDI formatter / deep |
| A5 | `D_reg55_p0` FP без red flags | Не поднимать P0 при пустом evidence; demote или suppress «нет признаков» как critical | reg55 / clarity |

### B. Protocol suggest (P0)

| ID | Ошибка | Фикс |
|--|--|--|
| B1 | Уролог + postop → КП ЧЛХ/стоматология | Hard filter: specialty / specialty family; gaps не перевешивают specialty mismatch |
| B2 | Слабые нейро-КП на цервикалгию | Ranking: ICD `M54*` + specialty невролог > редкие СМА/наследственные без match |
| B3 | UI: все irrelevant без «почему пусто» | Если после фильтра candidates=0 - явная RU-причина, не 3 мусорных КП |

### C. МКБ (P0/P1) - #32 частично done

| ID | Ошибка | Фикс |
|--|--|--|
| C1 | `mis_diagnos` / MIS-слот не всегда в resolve | Включить `mis_diagnos`, structured diagnosis slots в `resolve_icd_codes_from_mo` |
| C2 | Старые `cases.jsonl` без full-doc ICD | **Re-score L1/deep** Aug 01-05 (не только LLM grades) |
| C3 | ICD plan P3 | soft-fill `mkb_code_main` в warehouse/export |
| C4 | ICD plan P4 | строка в LLM judge / methodist prompts: «МКБ может быть в любом разделе» |

### D. Review pack UX (P1)

| ID | Ошибка | Фикс |
|--|--|--|
| D1 | Save с `training_use` при всех findings unreviewed | Warning или confirm; опционально не default-check training_use |
| D2 | Summary vs finding_decisions расходятся | Подсветка: «summary упоминает FP, но finding confirmed» (мягко) |
| D3 | `finding_decisions` без note | Опциональное `note_ru` на FP (почему отклонили) для gold |

### E. Ops / данные (P1) - чтобы ошибки не «возвращались» в UI

| ID | Ошибка | Фикс |
|--|--|--|
| E1 | Stale `llm_queue_pending=80` при grades ok | После LLM range - всегда recompute; при geo-block stubs - `--retry-errors` only on Render |
| E2 | Grades 01-04 перезаписаны geo-block | Запрет grade с Mac; мониторинг ok/err; не считать `_error` успешным |
| E3 | Июль bad `doctor_key` | `repair_mo_warehouse_from_secure` за июльские CSV |
| E4 | Код в проде ≠ пересчёт findings | Явный шаг: re-score + recompute в DoD каждой scorer-задачи |

---

## 4. Фазы реализации (одна волна, несколько PR)

Делать **последовательно**, но без пауз на согласование внутри волны после «приступать».

### Фаза S0 - фиксация gold (0.5 ч)

- [x] Юнит-фикстуры из текстов 3650612 (NSAID скобки) и 3643304 (`mis_diagnos: M54.8`) - `tests/test_nsaid_alternatives_topical.py`.
- [ ] Экспорт 2 packs в `gold_review/` на Render (не в git с ПДн) - ops после merge.

### Фаза S1 - medication + reg55 (1 PR)

- [x] A1, A2, A3, A4 + тесты (`medication_safety`, deep, MED_*).
- [x] A5: в `mz_2021_55` уже нет P0; stale `D_reg55_p0` уйдёт после re-score.
- [x] Прогон фикстур: 3650612-like → нет `C_nsaid_dup`; oral+gel → нет dup.

### Фаза S2 - protocol suggest

- [x] B1/B3: path-block ЧЛХ/стоматология для уролога; gaps C_nsaid/C_ddi не кормят suggest.
- [x] Тест `test_suggest_blocks_stomatology_for_urologist`.

### Фаза S3 - ICD wiring + re-score

- [x] C1: `mis_diagnos` в `mo_icd_resolve`.
- [ ] C3/C4 warehouse soft-fill + LLM prompts - residual (не блокер волны).
- [ ] На Render: `rescore_mo_deep_days.py` + `recompute_mo_days` Aug 01-05 - после deploy.

### Фаза S4 - UX pack + july repair

- [x] D1: confirm при training_use + все findings unreviewed (`mo-app.js`).
- [ ] D2/D3 notes - residual.
- [ ] E3 july repair - ops отдельным шагом после merge.
- [x] E1/E2: runner уже `--retry-errors` + Gemini only Render.

### Фаза S5 - приёмка

- [ ] Deploy + rescore/recompute на Render.
- [ ] Smoke 3650612 / 3643304.
- [ ] Обновить handoff.

---

## 5. Порядок файлов (ориентир)

| Область | Файлы |
|--|--|
| NSAID / DDI | `clinical_knowledge` medication_safety / kz_deep_eval; тесты |
| reg55 P0 | `reg55_criteria.py`, findings clarity |
| Suggest | `case_protocol_suggest.py`, `methodist_protocol_search.py`, mo-app suggest block |
| ICD | `mo_icd_resolve.py`, deep/engine, prompts |
| Re-score ops | batch L1 script + `recompute_mo_days.py` на Render |
| Pack UX | `mo_review_pack.py`, `mo-app.js` |
| Warehouse july | `repair_mo_warehouse_from_secure.py` |

---

## 6. Риски

| Риск | Митигация |
|--|--|
| Ослабление NSAID → пропуск реального dup | Тесты: два НПВП в разных предложениях без «или» → finding остаётся |
| Specialty filter съест редкие но верные КП | Fallback: если 0 candidates - broader search + label «слабое совпадение» |
| Re-score дорогой по времени | Ночью на Render; по дням; не трогать CRM packs |
| Geo-block снова сотрёт grades | Только Render LLM; `--retry-errors`; не запускать grade с Mac |
| Expert pack 3643304 неполный как gold | Не использовать unreviewed labels как hard gold; доразметить после S3 |

---

## 7. Definition of Done

1. На тексте 3650612 нет `C_nsaid_dup`; suggest не отдаёт ЧЛХ как top без specialty match.
2. `mis_diagnos` / full-text МКБ учтён; ложный invalid на 3643304-like нет.
3. Oral+topical не даёт жёсткий dup.
4. `D_reg55_p0` без evidence не critical FP.
5. Aug 01-05: L1/deep пересчитаны новым кодом; reports `llm_queue_pending=0` при grades ok.
6. UX: нельзя молча копить «training_use» с пустыми finding_decisions (warning/confirm).
7. План и handoff обновлены; версия в проде = UTC stamp последнего PR волны.

---

## 8. Вне скоупа этой волны

- Fine-tune LLM / полный LLM на все ~1500 кейсов дня.
- GCE europe-north1.
- Автозапись решений в МИС MariaDB.
- ≥50 packs gold export (запустится naturally после UX + методистов).

---

## 9. Статус

Волна запущена по команде «делай всю волну». Код S1-S4 в PR; ops rescore/recompute
на Render - сразу после merge/deploy.
