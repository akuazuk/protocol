# Оценка МО/КЗ: усиление согласованности по кейсу Смирнова (v1)

Дата: 2026-08-05  
Статус: active  
Автор: агент + владелец (разбор `Downloads/KZ/smirnova.pdf`)  
Связанные:

- `2026-08-05-mo-case-protocol-suggest-v1.md` - подбор КП (отдельный модуль);
- `2026-08-03-mo-rubric-mz-scoring-viz-v1.md` - рубрика МЗ 0/0.5/1;
- `2026-07-30-mo-analytics-bi-redesign-v1.md` - deep / оси;
- эталон: ребёнок, хромота 3 мес, отёк колена в статусе, диагноз только M60 миозит, план НПВП+массаж обоих бёдер+контроль «при ухудшении».

---

## 1. Контекст: что «пропустила» текущая оценка

По Смирновой методически важны не опечатки, а **клинико-логические разрывы**:

1. Находка в статусе (отёк колена) **не попала в диагноз**.
2. Хроническая хромота у ребёнка при **бедном анамнезе**.
3. План **слишком лёгкий** для сценария (нет обследований, контроль только при ухудшении).
4. Лечение **не латерально** (массаж обоих бёдер при односторонней жалобе).
5. Код **M60** без клиники инфекции / подтверждения - слабая опора.
6. Нужные КП МЗ (ЮА/ревмо, детская ортопедия, Пертес) не участвуют в оценке, если смотреть только на выставленный ICD.

Сейчас deep даёт ось `clinical_concordance`, но таких правил либо нет, либо они не срабатывают на этом паттерне. Цель плана - **добавить детерминированные findings** (и при необходимости критерии рубрики МЗ), не смешивая с Protocol Suggest.

---

## 2. Что менять (продуктово)

### A. Новые / усиленные findings (deep / L1-deep)

| Код finding | Ось | Severity (старт) | Условие срабатывания (черновик) | Зачем (Смирнова) |
|--|--|--|--|--|
| `finding_not_in_diagnosis` | clinical_concordance | P1 | Есть структурированная находка (отёк/выпот/ограничение ROM сустава) и ни ICD, ни текст диагноза её не покрывают | Отёк колена vs только миозит |
| `anamnesis_thin_for_duration` | documentation | P2 | Длительность жалобы ≥ 28 дней (или «месяц/месяца») и анамнез без ключевых флагов (травма/лихорадка/динамика/нагрузка - хотя бы 2 из набора, иначе fail) | 3 мес хромоты, анамнез из 1 строки |
| `underworkup_chronic_red_flag` | safety | P1 (ped P0-кандидат после калибровки) | Pediatric + хромота/отёк сустава + duration≥28д + в плане нет imaging/labs + follow-up только «при ухудшении» | Нет УЗИ/анализов, контроль условный |
| `plan_laterality_mismatch` | clinical_concordance | P3 | Жалоба/статус unilateral, в плане билатеральная процедура на парный орган без обоснования | Массаж обоих бёдер |
| `icd_weakly_supported` | clinical_concordance | P2 | ICD из группы «требует уточнения» (старт: M60*) при отсутствии supporting signs (infection/systemic/CK/imaging) в тексте | M60 «в воздухе» |
| `pediatric_limp_ddx_not_addressed` | clinical_concordance / regulatory | P2 | Pediatric limp ≥4 нед и диагноз не из whitelist DDx-корней (M08, M91, M25, S/T trauma, инфекция…) и нет явного «исключить Пертес/ЮА» в плане | Не закрыт DDx |

Правила - **детерминированные** (regex + словари органов/сторон + ICD roots). LLM только optional explanation, не gate.

### B. Связка с рубрикой МЗ (`mo_rubric_mz`)

Не дублировать всё в deep. Маппинг:

| Finding / паттерн | Критерий рубрики (ориентир) | Действие |
|--|--|--|
| thin anamnesis | полнота анамнеза | score 0 или 0.5 |
| underworkup / follow-up on worsening | follow-up / план обследования | 0, если нет срока контроля и обследований |
| finding_not_in_diagnosis | согласованность диагноза со статусом (если есть в sheet) или новый подкритерий | shadow → primary после калибровки |

Шаг: сверить формулировки sheet владельцем; добавить 1-2 критерия только если sheet это покрывает, иначе оставить в deep findings.

### C. Protocol Suggest - вход для applicability, не для балла оформления

Уже есть план `2026-08-05-mo-case-protocol-suggest-v1.md`.

Для оценки использовать suggest так:

- если top clinical KP (ревмо/ортопед детский) **не** среди matched по выставленному ICD → finding `likely_missed_protocol_family` (P2, informational / regulatory soft);
- **не** штрафовать врача автоматически за «не тот PDF», пока Hit@3 suggest не стабилен.

### D. Словари и парсер статуса (инфраструктура оценки)

Чтобы A сработало, нужен лёгкий extractor (можно в `clinical_knowledge/mo_case_signals.py`):

- сторона: прав/лев/right/left;
- сустав: колен/тазобедрен/голеностоп…;
- знаки: отёк/отек/выпот/сглаженность / болезненность / хромот;
- длительность: N мес/нед/дн;
- план: УЗИ|рентген|МРТ|ОАК|СОЭ|СРБ; «при отрицательной динамике» / «при ухудшении».

Без этого findings будут шумными.

---

## 3. Метрики

| Метрика | Было | Цель после v1 |
|--|--|--|
| Эталон Смирнова ловит ≥4 из 6 паттернов A | 0 (вручную) | ≥4 findings с expected codes |
| False positive rate на 50 «чистых» МО без разрывов | н/д | <10% по P1+ |
| Изменение среднего deep overall на месяце | baseline | сдвиг объяснить; не >3 п.п. без калибровки порогов |
| Согласие методиста на 20 кейсах очереди | н/д | ≥70% «finding справедлив» |

---

## 4. Шаги реализации

### Этап E0 - зафиксировать эталон (0.5 дня)

- [x] Обезличенный fixture: fact graph + ожидаемые finding codes (Смирнова) - `tests/test_mo_concordance_smirnova.py`.
- [ ] Ещё 5 positive / 5 negative кейсов (ортопедия/педиатрия) в `eval/mo_concordance/`.
- [x] Не класть PHI PDF в git.

### Этап E1 - signals + findings (2-4 дня)

- [x] `mo_case_signals.py` - извлечение стороны/сустава/длительности/red flags плана.
- [x] Реализовать findings из таблицы A (feature flag `MO_CONCORDANCE_FINDINGS`, default on для shadow).
- [x] Unit-тесты на fixture Смирнова.
- [x] Shadow-only: писать в `shadow_findings` deep block; `MO_CONCORDANCE_PRIMARY` default off.

### Этап E2 - калибровка (2-3 дня)

- [ ] Прогон на выборке июля/августа МО (без публикации в прод-дашборд как blocking).
- [ ] Подкрутить severity / порог duration / whitelist ICD.
- [ ] Решить: `underworkup` = P1 всегда или P1 только pediatric.

### Этап E3 - UI / очередь (1-2 дня)

- [ ] Русские title в dim_finding / finding labels.
- [ ] Попадание P1 в очередь «Вчера» / action queue как сейчас для P0/P1.
- [ ] В case detail: явная связка «находка ↔ пробел в диагнозе».

### Этап E4 - рубрика МЗ и suggest (параллельно, не блокер E1)

- [ ] Сверка с `mo_rubric_mz.yaml` + sheet.
- [ ] Мягкий `likely_missed_protocol_family` после стабилизации suggest Hit@3.

---

## 5. Что не делать в этом плане

- Не заменять deep primary на LLM-judge по каждому МО.
- Не ставить P0 за «не тот протокол» без подтверждённого suggest.
- Не менять `assess_completeness` / LLM advisory policy Phase A.
- Не включать findings в blocking data-quality gate ETL.

---

## 6. Риски

| Риск | Митигация |
|--|--|
| Ложные P1 на шаблонном «суставы без особенностей» | требовать позитивный знак (отёк/боль), не отсутствие текста |
| Русские орфоварианты (отёк/отек) | словарь нормализации |
| Взрослые случаи с хронической болью | audience gate; underworkup строже для pediatric |
| Раздувание average score вниз | shadow → калибровка → primary |
| Дубль с rubric_mz | маппинг §2B; один код - один primary слой |

---

## 7. Владение файлами

- `clinical_knowledge/mo_case_signals.py` (новый) - E1 done
- `clinical_knowledge/mo_concordance_findings.py` (новый) - E1 done
- `clinical_knowledge/kz_deep_eval.py` - hook `shadow_findings` - E1 done
- `tests/test_mo_concordance_smirnova.py` - E1 done
- `eval/mo_concordance/` - E0 remainder (5+5)
- при необходимости строки в `config/mo_rubric_mz.yaml`

Не пересекать без согласования: `publish_mo_to_render.py`, launchd, Phase A completeness.

---

## 8. Definition of Done

1. Fixture Смирнова стабильно даёт `finding_not_in_diagnosis` + `underworkup_chronic_red_flag` + `anamnesis_thin_for_duration`.
2. Shadow включён за flag; primary deep без регресса метрик до калибровки.
3. Методист на 10 кейсах подтверждает полезность ≥7.
4. Короткий handoff в `docs/reports/`.

---

## 9. Первая безопасная команда

```bash
scripts/ops/git_task_start.sh mo-concordance-smirnova --pc=pc1 \
  --branch=codex/mo-concordance-smirnova-agent1-pc1
# E0: fixture fact-graph + expected findings, без включения в primary score
```
