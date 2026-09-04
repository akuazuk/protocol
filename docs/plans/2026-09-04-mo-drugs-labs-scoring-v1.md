# МО: лекарства и анализы в оценке (v1)

Дата: 2026-09-04  
Статус: **active**  
Владелец: задача после статьи РЗ (оценка качества МО)  
Связанные (не дублировать, стыковать):

- [Лаборатория из `mis_tests`](2026-08-26-mo-lab-from-mis-tests-v1.md) - склад, shadow, `MO_LAB_IN_PRIMARY` default 0
- [Rceth ЛС](2026-08-14-rceth-drug-labels-mo-v1.md) - инструкции РБ, shadow `C_rceth_*`
- [DDI topical demote](2026-08-11-mo-ddi-topical-demote-v1.md) - completed; хвост backfill
- [Grade ladder / Rceth](2026-08-20-mo-grade-ladder-v1.md)
- [Eval quality followups](2026-08-10-mo-eval-quality-followups-v2.md) - drug-norm, DDI
- [Daily BI platform](2026-07-28-mo-daily-bi-platform-v1.md) - открыты «Дозировки/формуляр», «Mapping услуг»
- Статья РЗ: `docs/public_realition/2026-09-03-kuzavka-quality-scores-rz-print.html`  
  (аномалия «анализ сделан, но не учтён» - пока сильнее продукта)

---

## 1. Контекст

Статья для «Руководитель. Здравоохранение» обещает единую систему, где каждое МО
оценивается по протоколам, МКБ, реестру ЛС и **использованию анализов**. В пилоте
МО Аналитики картина асимметрична:

| Область | Уже в primary / живом scorer | В основном shadow / advisory | Дыра vs статья |
|--|--|--|--|
| Лекарства | DDI (`C_ddi`), дубль НПВС (`C_nsaid_dup`), high-alert без дозы, STOPP/Beers, `B_tx_offprotocol`, нормализация brand→INN | Rceth off-label / contra / age (`C_rceth_*`), `MED_dose_context_missing` почти без штрафа | Доза «по смыслу», дубли **классов** шире НПВС, формуляр РБ, препарат↔диагноз |
| Анализы | `B_exams_gap` (обязательные обследования КП **упомянуты** в тексте) | `B_lab_present_not_in_mo`, `B_lab_ordered_already_done`; B2C `patient_lab_crosscheck` | **Результат не учтён в диагнозе/плане**; anomalous value ignored |

**Главный разрыв доверия:** в статье аномалия №2 звучит как рабочая проверка;
в scorer нет primary-finding «лабораторный результат есть, а в диагнозе и плане
о нём ни слова». Сейчас проверяется упоминание обследования / наличие в складе,
а не влияние результата на решение врача.

Цель этого плана: закрыть разрыв **честно** (сначала shadow + gold, потом primary)
и усилить лекарственный контур до уровня, обещанного статьёй, без подмены
согласованности с экспертом взысканиями.

---

## 2. Что уже есть в проде (база)

Не переписывать с нуля. Опираться на:

| Слой | Файлы / коды | Заметка |
|--|--|--|
| Safety drugs | `kz_deep_eval._axis_safety`, `medication_safety`, `medication_findings` | P1 на major DDI и NSAID dup; topical demote |
| Labels RU | `mo_finding_labels_ru` (`C_ddi`, `C_nsaid_dup`, `B_exams_gap`, …) | Формулировки для UI |
| Lab warehouse | `mo_lab.sqlite` / `fact_mo_lab`, ingest `mis_tests` | Ключ patient + окно дат, не visit_id |
| Lab shadow | `mo_lab_shadow`, флаги `MO_LAB_IN_PRIMARY` | Default off |
| Rceth | `rceth_label_findings`, `MO_RCETH_LABEL_PRIMARY` | Default off |
| Risk-gate | P0→overall≤40, P1→≤60 | Не ломать SSOT №55 без отдельного решения |

---

## 3. Целевая модель оценки

Две оси, которые статья уже рисует; продукт должен уметь показать их раздельно
и брать **минимум** как итог допуска (не среднее).

```text
Клиническая оценка (протоколы + безопасность ЛС + полнота + использование анализов)
        ×
Готовность документа (отдельный контур; не смешивать в один mean)
        →
Итог допуска = min(клиника, готовность)
```

Новые findings не должны сразу резать зарплату/аттестацию: до gold κ / FP-порога
только подсказка и очередь методиста (как в статье).

### 3.1. Анализы - лестница findings (новые)

| Код | Смысл | Severity | Primary? |
|--|--|--|--|
| `B_lab_unused_in_dx` | Результат в `exam_data` / `fact_mo_lab`, в диагнозе нет mention канона | P1 | после FP-калибровки |
| `B_lab_unused_in_plan` | Результат есть, план лечения/контроля не учитывает | P1 | после FP-калибровки |
| `B_lab_abnormal_ignored` | Значение вне референса, в МО «норма» / пусто | P0/P1 | **волна 3**; нужны референсы |
| `B_lab_ordered_not_used` | Заказан ранее, результат пришёл, в текущем МО не разобран | P1 | после unused MVP |
| `B_exams_gap` (усиление) | Не substring: семантика + услуга МИС → обследование КП | P2→P1 если КП жёстко | mapping услуг |

MVP сначала: **unused mention** (без референсов). Abnormal - отдельная волна.

### 3.2. Лекарства - усиления

| Код / тема | Смысл | Severity | Primary? |
|--|--|--|--|
| Rceth dose / label | Off-label, contra, age, доза vs инструкция | P1/P2 | после калибровки (`MO_RCETH_LABEL_PRIMARY`) |
| Дубли терапевтических классов | ИПП, антигистаминные, антикоагулянты/антиагреганты, … | P1 | после словаря классов |
| Drug–disease | Препарат не согласован с Dx / группой КП | P1/P2 | DDSI + `B_tx_offprotocol` |
| Формуляр / реестр РБ | «Можно ли назначать» сверх brand→INN | P2/P3 | из daily-bi backlog |
| Dose context (дети / elderly) | Нет возраста/веса/СКФ при high-alert | жёстче P2 | связать STOPP + high-alert |
| Backfill topical DDI/NSAID | Старые ложные P1 портят динамику | - | rescore дней |

---

## 4. Метрики

| Метрика | Было (2026-09-04) | Цель волны 1 | Цель волны 2-3 |
|--|--:|--:|--:|
| Primary finding «результат не учтён в Dx/плане» | нет | shadow на ≥1 нед. потока | primary при FP ≤ 15% на gold ≥100 |
| Доля визитов с same-day / −14д lab в сверке unused | склад есть (~16-34% coverage) | unused считается на всём покрытии | то же + timeline ordered→result |
| Rceth в primary | default 0 | gold 100 кейсов размечен | `MO_RCETH_LABEL_PRIMARY=1` при FP ≤ 15% |
| Дубли классов сверх НПВС | только НПВС | словарь ≥5 классов в shadow | 2+ класса в primary |
| Отдельные KPI в BI: % unused lab, % drug-safety | нет / в общем safety | полоски на Обзоре (shadow) | drill-down врач→случай |
| Отдельный gold: lab unused | 0 | 50 методист | 100 |
| Отдельный gold: drug label/dose | частично Rceth | 50 | 100 |
| Согласованность с экспертом по этим осям | не измерена | shadow only | κ / agreement ≥ 0.7 перед взысканиями |

Числа FP и κ уточняются после первой разметки; до этого primary **запрещён**.

---

## 5. Волны работ

### Волна 1 - Unused lab MVP (2-3 недели)  ← старт здесь

**Сделано:** план; склад `mis_tests` / shadow lab (см. lab-from-mis-tests).

**В работе / дальше:**

1. [ ] Канон тестов: словарь `indicator`/`type` → канон (ОАК, глюкоза, СРБ, АЛТ/АСТ, креатинин, ТТГ, …) минимум 15 панелей.
2. [ ] Детектор mention канона в слотах диагноз + план (и `exam_data` как источник «результат есть»).
3. [ ] Findings `B_lab_unused_in_dx` / `B_lab_unused_in_plan` **только shadow**; RU-лейблы как в статье: «Готовый анализ не учтён в диагнозе / плане».
4. [ ] Окно: same-day + lookback 14д (как lab v1); не клеить чужой patient.
5. [ ] Gold-пакет 50 случаев (методист confirm/reject); журнал FP.
6. [ ] Дашборд: доля МО с unused lab (shadow-бейдж), drill до случая.
7. [ ] Порог: FP ≤ 15% на 50+ → кандидат в primary; иначе ещё итерация словаря.

**Не в волне 1:** abnormal values, референсы, взыскания, изменение SSOT №55.

### Волна 2 - Лекарства: Rceth primary + классы

1. [ ] Дожать корпус/parse Rceth по плану rceth-v1; калибровка off-label FP.
2. [ ] Gold 100: label / dose / age.
3. [ ] Включить `MO_RCETH_LABEL_PRIMARY` только после порога FP.
4. [ ] Словарь терапевтических дублей (≥5 классов) → shadow `C_*_dup` / расширение NSAID-логики.
5. [ ] Rescore / backfill дней с topical Major→Moderate (хвост ddi-topical).
6. [ ] BI: три колонки drug-safety - взаимодействия / дубли / доза-инструкция.

### Волна 3 - Abnormal labs + mapping + формуляр

1. [ ] Узкая панель референсов (ОАК, СРБ, глюкоза, АЛТ/АСТ, креатинин, ТТГ) - источник и версия в метаданных.
2. [ ] `B_lab_abnormal_ignored` shadow → primary по тому же FP-процессу.
3. [ ] Mapping услуг МИС → обследования КП (`B_exams_gap` семантика) - пункт daily-bi.
4. [ ] Формуляр / реестр РБ как отдельный soft-finding.
5. [ ] Drug–disease (DDSI) в shadow.
6. [ ] `B_lab_ordered_not_used` по timeline визитов.

### Волна 4 - Оценки и дашборды «как в статье»

1. [ ] Аномалии статьи (топ-10) = first-class коды findings с % по врачу/специальности.
2. [ ] Явный показ двух оценок + min как итог (клиника vs готовность документа).
3. [ ] Risk-adjust: сравнение врачей от ≥20 случаев + поправка на состав (хотя бы грубая).
4. [ ] Матрица 2×2 «лечение × документы» на Обзоре организации.
5. [ ] Отдельный agreement report: lab unused + drug label vs методист (κ / %).

---

## 6. Что меняется в проде (по волнам)

| Волна | В проде | Не в проде |
|--|--|--|
| 1 | Shadow findings + RU labels + shadow KPI на дашборде; словарь канонов в репо | Primary score, hard-cap, блокировка подписи |
| 2 | При выполнении порога - Rceth primary; shadow class-dup | Формуляр primary, DDSI primary |
| 3 | Abnormal shadow; mapping услуг (по готовности данных) | Автоматические взыскания |
| 4 | KPI аномалий, две оценки на UI | Публичные рейтинги врачей |

Night LLM / Gemini - только через GCE (`run_on_gce.sh`), не с Mac.

---

## 7. Шаги (чеклист владельца)

### Сейчас

- [x] Зафиксировать разрыв статья ↔ продукт в плане
- [x] Связать с lab-from-mis-tests и rceth без дублирования волн 0-2 lab
- [ ] Согласовать старт волны 1 (unused lab) vs параллель волны 2 (Rceth)
- [ ] Назначить методиста на gold 50 unused

### Волна 1 (acceptance)

- [ ] На 20 вручную отобранных «анализ есть, в Dx нет» - finding срабатывает ≥16
- [ ] На 20 «анализ учтён» - false positive ≤3
- [ ] В UI разбора случая: формулировка понятна врачу без P0/P1-жаргона
- [ ] В Обзоре: % unused lab за период (shadow), клик → список визитов

### Стоп-условия primary

- [ ] Gold ≥100 (unused) или ≥100 (Rceth) с разметкой методиста
- [ ] FP ≤ 15% (или согласованный владельцем порог)
- [ ] Agreement с экспертом ≥ 0.7 на контрольной выборке
- [ ] Явный флаг env / настройка организации; default off до решения владельца

---

## 8. Риски

| Риск | Митигация |
|--|--|
| Статья уже «продала» unused lab | В UI и PR-текстах: shadow / «в апробации»; не писать «критично блокирует» до primary |
| FP на синонимах («глюкоза» vs «сахар крови» vs лат.) | Канон + синомы; gold-итерации; не primary на сыром substring |
| Чужие анализы patient_id | Пустой блок лучше чужого; канон lab v1 |
| Нет референсов в `mis_tests` | Abnormal только волна 3; не смешивать с unused MVP |
| Rceth FP на off-label частых схемах | Калибровка; specialty allowlist; shadow дольше |
| Дубли классов ловят «или/либо» | Переиспользовать парсер альтернатив из NSAID topical |
| Ломаем SSOT №55 / overall | Новые findings сначала вне hard-cap; отдельное решение на risk-gate |
| Параллельные PR на те же файлы | Не трогать чужой worktree; стык с lab/rceth owners |

---

## 9. Вне скоупа v1

- Блокировка электронной подписи врача по unused lab
- Публичные рейтинги врачей
- Полный DailyMed / зарубежные dose-range как primary
- Замена concordances / wrong-working-dx этим планом
- Переписывание статьи под «ещё не умеем» без согласования с владельцем

---

## 10. Одна безопасная следующая команда

После merge этого плана (docs-only):

```bash
# Ветка реализации волны 1 (отдельный task-worktree от свежего origin/main):
scripts/ops/git_task_start.sh mo-lab-unused-mvp --pc=pc1 \
  --branch=cursor/mo-lab-unused-mvp-agent1-pc1
```

Первый код волны 1: словарь канонов тестов + shadow `B_lab_unused_in_dx` /
`B_lab_unused_in_plan` без включения в overall.
