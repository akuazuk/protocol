# МО из БД: ежедневная загрузка, объективная оценка, CRM/BI-дашборд и единая терминология

**Дата:** 2026-07-28
**Статус:** active
**Источник истины:** этот план заменяет разрозненные планы MIS-КЗ, deep-eval, scoring и
dashboard в части массового анализа данных из БД.
**Режим исполнения:** автономно, последовательно, без пропуска пунктов; после каждой
фазы - тесты и фиксация метрик; финал - отчёт, commit и push.
**Текущий production baseline:** `e48d40d`, `BUILD_VERSION=2026-07-27-r20-mis-kz-route`.

---

## 0. Зафиксированные определения владельца продукта

Эти определения обязательны для UI, документации и новой архитектуры:

1. **МО** - продуктовый контур массового анализа записей, полученных прямым подключением
   к БД МИС. МО включает ежедневные, месячные и произвольные выборки из БД.
   Это определение по **источнику и режиму анализа**, а не утверждение, что каждая строка
   БД является медицинским осмотром одного типа.
2. **КЗ** - пациентская вкладка: пациент загружает своё консультативное заключение и
   получает понятный разбор.
3. **Одиночный анализ файла** - отдельный нейтральный сценарий **«Анализ документа»**.
   Загруженный файл может оказаться КЗ или МО; тип определяется автоматически и
   подтверждается пользователем, если уверенность классификации низкая.
4. Внутренние legacy-идентификаторы `kz_*`, таблица `mis_protocol`, старые API и
   исторические артефакты не определяют пользовательскую терминологию.
5. В пользовательском UI не показывать внутренние технологии и поставщиков:
   `RAG`, `LLM`, `Cursor`, `Gemini`, `Render`, названия облачных/модельных провайдеров.
   Использовать «нейросетевой анализ», «машинная проверка», «поиск по клиническим
   протоколам», «сервер».
6. Названия медицинских источников сохранять в provenance и методической справке:
   пользователю показывать назначение источника, а точное название - в раскрываемом
   блоке «Источники».

### 0.1. Матрица терминов по поверхностям

| Поверхность | Основное название | Допустимые подписи |
|---|---|---|
| Пациент | КЗ / заключение после приёма | «Проверить КЗ», «Заключение врача» |
| Одиночный файл | Анализ документа | «Определено: КЗ» / «Определено: МО» |
| Методист, данные БД | МО | «Аналитика МО», «Качество МО», «Отчёт МО за вчера» |
| Внутренний код | `kz_*`, `mis_protocol`, `consult_*` | оставить до совместимой миграции |
| Интегратор | полные технические термины | только в технической документации |

### 0.2. Словарь замены технического языка

- `RAG` -> «поиск по клиническим протоколам»;
- `LLM` -> «нейросетевой анализ»;
- `Gemini/OpenAI` -> «нейросетевая модель»;
- `Render` -> «сервер» / «облачная среда»;
- `L0/L1/L2` -> «быстрая проверка» / «полная проверка» / «нейросетевой анализ»;
- `overall score` -> «итоговая оценка»;
- `risk-gate` -> «ограничение оценки при критическом риске»;
- `coverage` -> «полнота проверки»;
- `confidence` -> «надёжность результата»;
- `finding` -> «выявленное замечание»;
- `deep-eval` -> «углублённая оценка».

---

## 1. Контекст и текущее состояние

### 1.1. Что уже работает в production

- `mis-kz-quality.html` - канонический standalone-дашборд массовой оценки;
- `index.html` - сводка и переход в полный дашборд;
- API `/api/methodist/mis-kz-quality/*`;
- фильтры по месяцу, специальности, филиалу, врачу, статусу, датам и МКБ;
- обзор, специальности/врачи, главы МКБ, кейсы, динамика;
- scorer v3 с осями A/B/C/D, coverage, confidence и risk в shadow-режиме;
- trust-aware правила протоколов: неподтверждённые требования не штрафуют;
- applicability-gate поиска;
- отдельные прямые URL врача и методиста.

### 1.2. Доказанные ограничения

1. Дашборд читает месячные JSONL/CSV и не имеет полноценной ежедневной витрины.
2. Нет автономного ежедневного задания «вчера».
3. Нет гарантированного catch-up после сна/выключения компьютера.
4. Нет атомарного incremental merge - текущий экспорт чаще пересобирает месяц.
5. UI содержит две навигации, дублирующие обзоры и технические термины.
6. Часть фильтров зависит от отсутствующих deep-полей.
7. Несколько конкурирующих итоговых оценок ещё существуют параллельно.
8. Scorer v3 остаётся shadow: `PRIMARY=0`, `GATE=0`.
9. 477 протоколов и 11 259 требований, но penalty-eligible coverage = 0% без
   методистского подтверждения.
10. Нет реального gold set 800-1200 случаев с двойной разметкой.
11. Нет CRM-состояния случая: ответственный, статус разбора, SLA, комментарии,
    подтверждение/отклонение замечания, аудит изменений.
12. Нет browser/a11y/visual-regression матрицы.
13. Multi-month сейчас действует в основном в сравнении и динамике: таблица случаев,
    KPI и большинство диаграмм используют один `active month`.
14. Категориальные фильтры сейчас обычные single-select; требуемого множественного
    выбора по филиалам, специальностям, врачам, МКБ и статусам нет.
15. Facets строятся до применения фильтров и не всегда отражают доступные комбинации
    текущего среза.
16. В standalone одновременно существуют topnav, deep-tabs и legacy `<details>`:
    три уровня навигации дублируют обзор, случаи и динамику.
17. API уже покрывает базовые KPI, фильтры, drill-down и объяснение оценки, но не имеет
    saved views, exports, alerts, CRM actions и role-based scopes.
18. В соседнем проекте `sql_epam` уже работают launchd-задачи синхронизации в 06:30,
    retry в 09:00 и backfill в 00:30, но это выгрузка в SQLite/Zoho, а не МО-scoring.
19. Batch и его state привязаны к месяцу; date-range export существует, но отсутствуют
    безопасный daily-to-month merge и ежедневный report generator.

### 1.3. Технические источники истины

- БД: `kravira_mc.mis_protocol + mis_data`;
- экспорт: `scripts/export_mis_protocol_month.py`;
- парсер: `clinical_knowledge/mis_protocol_parse.py`;
- массовая обработка: `scripts/run_mis_protocol_l1_batch.py`;
- текущая витрина/API: `clinical_knowledge/mis_kz_quality.py`;
- scorer v3: `clinical_knowledge/kz_evaluation_schema.py` и
  `clinical_knowledge/kz_evaluation_engine.py`;
- UI: `mis-kz-quality.html`;
- версия: `BUILD_VERSION` в `rag_server.py`.

---

## 2. Цели и измеримые результаты

### 2.1. Главные цели

1. Каждый день автоматически получать из БД все МО за вчера.
2. Не терять дни при сне, перезагрузке, отсутствии БД или VPN-конфликте.
3. Строить детальный ежедневный отчёт и обновлять month-to-date.
4. Сделать единый современный CRM/BI-дашборд МО.
5. Сделать оценку объяснимой: итог -> оси -> критерии -> evidence -> source -> cap.
6. Сделать очередь методиста рабочим процессом, а не только таблицей.
7. Развести КЗ, МО и одиночный анализ во всём пользовательском интерфейсе.
8. Убрать технический жаргон и названия сторонних поставщиков из UI.
9. Сохранить обратную совместимость API, файлов и старых отчётов.
10. Не публиковать ПДн и сырой клинический текст за пределами разрешённого контура.

### 2.2. Метрики: было / цель

| Метрика | Было | Цель |
|---|---:|---:|
| Автоматических загрузок МО за вчера | 0 | 1 успешная в день |
| Пропущенных календарных дней без catch-up | возможно | 0 |
| Повторный запуск создаёт дубли | не гарантировано | 0 дублей |
| Паритет строк БД -> raw partition | не формализован | 100% |
| `parse_ok` | измеряется | >=99.5% или блок публикации |
| Заполненная каноническая дата | измеряется | >=99.9% |
| Заполненный врач | ~99% | >=98%, alert при снижении |
| Дата mismatch | измеряется | <0.2%, alert выше |
| Успешная оценка eligible-строк | нерегулярно | >=99.5% |
| Время появления отчёта за вчера | отсутствует | до 08:00 Europe/Minsk |
| UI-термины RAG/LLM/Gemini/Render | десятки | 0 |
| CRM-статус и ответственный | 0 | 100% кейсов очереди |
| P0/P1 с evidence/source | частично | 100% отображаемых |
| Старые маршруты после миграции | риск поломки | 100% redirect/alias |
| Critical/serious a11y | не проверено | 0 |

---

## 3. Целевая архитектура

```text
MariaDB MIS
  -> безопасное ежедневное извлечение [вчера 00:00, сегодня 00:00)
  -> raw immutable partition
  -> валидация качества и quarantine
  -> нормализация + классификация документов / исключения внутри контура МО
  -> idempotent merge по id/visit_id
  -> legacy score + evaluation v3 shadow
  -> ежедневные агрегаты + month-to-date
  -> локальная аналитическая БД
  -> обезличенная публикационная витрина
  -> API МО
  -> CRM/BI-дашборд
  -> уведомление о готовности / ошибке
```

### 3.1. Хранилище

Сырые данные и ПДн - только локально/в разрешённом контуре:

```text
data/medical_exams/
  raw/YYYY/MM/mo_YYYY-MM-DD.parquet
  raw/YYYY/MM/mo_YYYY-MM-DD.meta.json
  quarantine/YYYY/MM/DD/
  warehouse/mo_analytics.sqlite
  secure_cases/YYYY/MM/mo_cases_YYYY-MM-DD.parquet
  reports/YYYY/MM/DD/
    report.json
    report.html
    report.csv
    quality.json
    run.json
  public/
    latest.json
    daily/YYYY-MM-DD.json
    monthly/YYYY-MM.json
```

Правила:

- `raw`, `secure_cases`, SQLite, CSV с врачами/пациентами - gitignored;
- `public` содержит только обезличенные агрегаты;
- запись через temporary file + atomic rename;
- каждый run имеет `run_id`, дату, source window, row counts, hash, версии parser/scorer;
- повторный запуск той же даты заменяет partition после полной проверки;
- поздние изменения в БД учитываются повторной сверкой последних 3 дней.

### 3.2. Аналитическая модель CRM/BI

Локальная SQLite-витрина со star schema:

- `fact_mo_case`;
- `fact_mo_finding`;
- `fact_mo_score_axis`;
- `fact_mo_daily`;
- `dim_date`;
- `dim_doctor`;
- `dim_specialty`;
- `dim_branch`;
- `dim_diagnosis`;
- `dim_service`;
- `dim_document_kind`;
- `crm_case_state`;
- `crm_case_event`;
- `saved_view`.

В публичной витрине:

- без `patient_id`;
- `visit_id` заменён устойчивым HMAC-hash;
- без сырого текста жалоб/анамнеза;
- агрегаты с suppression для малых групп (`n < 5`);
- врачебные ФИО доступны только методисту в разрешённом контуре.

---

## 4. Ежедневный автономный pipeline МО

### 4.1. Точка запуска

Запускать на Mac, который имеет доступ к MariaDB MIS. GitHub Actions и обычный Render
cron не подходят: БД доступна только с этой машины при выключенном VanyaVPN.
Учитывать существующий `sql_epam` sync в 06:30 и не запускать конкурирующее тяжёлое
SQL-задание в то же время.

Новый entrypoint:

```text
scripts/run_mo_daily_pipeline.py
```

Команды:

```bash
python3 scripts/run_mo_daily_pipeline.py --date yesterday
python3 scripts/run_mo_daily_pipeline.py --date 2026-07-27
python3 scripts/run_mo_daily_pipeline.py --catch-up
python3 scripts/run_mo_daily_pipeline.py --reconcile-days 3
python3 scripts/run_mo_daily_pipeline.py --dry-run
```

### 4.2. Планировщик

Использовать macOS `launchd`, а не Cursor/IDE:

```text
~/Library/LaunchAgents/by.protocol.mo-daily.plist
```

Настройки:

- основной `StartCalendarInterval`: 07:00 Europe/Minsk, после `sql_epam` 06:30;
- отдельный retry: 10:00 Europe/Minsk, если daily status не `success`;
- `RunAtLoad=true`;
- дополнительная проверка раз в час до успешного отчёта;
- `caffeinate -dimsu` только на время выполнения;
- lock-файл через `fcntl`, второй экземпляр завершается;
- catch-up всех отсутствующих дат от последнего success до вчера;
- ограничение catch-up за один запуск, например 31 день, с продолжением следующей итерацией;
- журнал без секретов и ПДн.

Если компьютер спал:

- `RunAtLoad` запускает catch-up после пробуждения;
- hourly check догружает пропущенное;
- отдельный health-флаг показывает lateness;
- гарантированный отчёт требует, чтобы Mac был включён или автоматически просыпался.
  Настройку `pmset wakeorpoweron` выполнять только после явного подтверждения владельца,
  так как она меняет системное расписание питания.

### 4.3. VPN state machine

Каждый SQL-run:

1. получить исходный статус `~/CURSOR/bin/vanya_vpn.sh status`;
2. `ensure-off`;
3. дождаться `Disconnected`;
4. проверить TCP/SQL connection;
5. выполнить все SQL-операции;
6. закрыть engine/connections;
7. в `finally` восстановить исходное состояние:
   - был Connected -> `ensure-on`;
   - был Disconnected -> оставить выключенным;
8. проверить финальный статус;
9. при невозможности выключить VPN - SQL не запускать;
10. секреты и connection URL не писать в лог.

### 4.4. Окно данных

Для отчёта за вчера:

```text
[yesterday 00:00:00, today 00:00:00), timezone Europe/Minsk
```

Использовать server-side даты БД и каноническую дату визита. Нельзя использовать
«последние 24 часа»: это ломает календарные сравнения.

### 4.5. Извлечение

Расширить существующий экспортёр или вынести общую библиотеку:

```text
clinical_knowledge/mis_export.py
scripts/export_mis_protocol_month.py        # compatibility wrapper
scripts/export_mo_daily.py                  # daily wrapper
```

Извлекать:

- `mis_protocol.id/date/visit_id/patient_id/result`;
- автор/специальность;
- филиал;
- тип оплаты;
- дата/время визита;
- дата рождения;
- диагнозы;
- услуги и их коды;
- все клинические поля `result`;
- данные, необходимые для точного различения МО, КЗ, справки и диагностики.

SQL только параметризованный. Join должен сохранять `1 protocol row = 1 output row`.

### 4.6. Классификация записей внутри контура МО

Ввести нейтральное поле `document_kind`:

```text
medical_exam
consultation
certificate
diagnostic
non_clinical
empty
unknown
```

Legacy `kz_kind` сохраняется.

Вся выборка БД относится к продуктовому контуру МО, но не каждая строка автоматически
идёт в клинический score. `document_kind` нужен для честного отбора и аналитики, а не
для переименования самого контура.

Для определения подтипа `medical_exam` использовать:

- тип оплаты;
- service codes/names;
- специализацию;
- название бланка/текст;
- поля объективного статуса;
- подтверждённые признаки профосмотра;
- конфиг правил с тестами.

Не считать все `certificate` медицинскими осмотрами. Случаи `unknown` не штрафовать,
а направлять в очередь уточнения.

### 4.7. Data-quality gate до оценки

Обязательные проверки:

- row count source = raw output;
- уникальность `id`;
- отсутствие дублей `(id, visit_id)`;
- все даты внутри окна или явно объяснены;
- заполненность `visit_date`;
- `parse_ok`;
- процент date mismatch;
- заполненность врача, специальности, филиала, возраста, МКБ;
- join не увеличил число строк;
- валидность JSON/Parquet;
- отсутствие NaN/Inf в агрегатах;
- сравнение объёма с медианой того же дня недели за 4-8 недель;
- аномальное падение/рост >50% требует warning;
- нулевой день допустим только после успешного SQL и проверки source count.

При критическом нарушении:

- partition переносится в quarantine;
- warehouse и dashboard не обновляются;
- предыдущий успешный отчёт остаётся current;
- Telegram получает краткое сообщение без ПДн;
- retry по расписанию.

### 4.8. Idempotent merge

- primary key raw: `mis_protocol.id`;
- case key: `id`, дополнительно `visit_id`;
- upsert, не append;
- hash содержимого для определения обновлённой строки;
- повторный run не меняет количество строк без изменения source;
- последние 3 дня reconciliation;
- monthly partition пересобирается из daily partitions, а не дописывается вслепую.

### 4.9. Оценка

Для каждой eligible-записи:

1. legacy L1 - для совместимости;
2. углублённые оси A/B/C/D;
3. `evaluation_v3` shadow;
4. coverage/confidence;
5. trust/applicability;
6. регуляторные требования №55/№127;
7. лекарственные и safety-проверки;
8. findings P0-P3;
9. статус ручного разбора.

До gold-калибровки:

- v3 не заменяет production gate;
- неподтверждённые протокольные правила advisory-only;
- «ошибка диагноза/лечения» в UI формулировать как «требует проверки»;
- каждый P0/P1 обязан иметь evidence, source и trust level;
- отсутствие данных отражается в coverage, а не превращается в нулевой клинический score.

Нейросетевой анализ:

- не обязателен для ежедневного завершения;
- не блокирует deterministic/v3 отчёт;
- raw ПДн не отправлять внешнему провайдеру без разрешённого контура;
- quota/spend-cap -> статус «нейросетевой анализ недоступен», без потери отчёта;
- provider name остаётся только в adapter/log для администратора.

### 4.10. Публикация и уведомления

После успешной обработки:

- обновить daily и month-to-date агрегаты;
- сформировать HTML/JSON/CSV отчёт;
- обновить SQLite;
- сформировать обезличенный public snapshot;
- загрузить разрешённые агрегаты на persistent storage сервера;
- проверить API freshness;
- отправить Telegram:
  - дата;
  - количество строк;
  - eligible МО;
  - исключено;
  - средняя оценка;
  - на разбор;
  - P0/P1;
  - data-quality warnings;
  - ссылка на отчёт.

Текущий `scripts/render_mis_protocol_data.sh upload` отправляет месячные CSV/Parquet.
Не использовать его автоматически для нового daily-процесса, пока отдельно не
подтверждён разрешённый контур хранения ПДн. Реализовать отдельную публикацию
обезличенного `public` snapshot; secure case-detail оставлять локально либо отдавать
через утверждённый защищённый канал.

При ошибке - причина, этап, число retries, следующая попытка. ПДн не отправлять.

### 4.11. Retry и recovery

- DB connect: 5 попыток с backoff;
- transient SQL/read timeout: retry;
- scoring: resume по `id`;
- upload/API smoke: retry отдельно без повторного SQL;
- state machine этапов:
  `pending -> extracting -> validating -> scoring -> reporting -> publishing -> success`;
- heartbeat;
- stale run >2 часов -> failed и новый retry;
- ручной resume с последнего успешного этапа;
- ежедневный reconciliation последних 3 дней;
- weekly full month reconciliation;
- backfill с 2026-01-01 отдельной командой.

---

## 5. Подробный отчёт МО за вчера

Отчёт является immutable snapshot с revision. Он открывается из дашборда и хранится
в папке даты.

### 5.1. Шапка и freshness

- дата отчёта;
- окно БД;
- время загрузки;
- время окончания обработки;
- source row count;
- revision;
- parser/scorer/build versions;
- статус качества данных;
- опоздание относительно SLA 07:30.

### 5.2. Executive summary

- всего записей из БД;
- допущено к оценке в контуре МО;
- консультации/справки/диагностика/исключённые;
- оценено;
- средняя/медианная оценка;
- доля «хорошо / требует внимания / критично»;
- P0/P1;
- coverage и confidence;
- изменение ко вчера;
- изменение к среднему того же дня недели;
- month-to-date.

### 5.3. Клиническая аналитика

- оси A/B/C/D;
- причины снижения оценки;
- Pareto топ-10 нарушений;
- red flags;
- подозрительные диагнозы;
- расхождение МКБ в документе и МИС;
- неполные обследования;
- лечение, требующее проверки;
- лекарственные взаимодействия;
- отсутствие маршрутизации/наблюдения;
- регуляторные дефекты;
- распределение по confidence/coverage.

### 5.4. Организационные срезы

- филиал;
- специальность;
- врач;
- диагноз/глава МКБ;
- услуга;
- тип оплаты;
- возрастная группа;
- первичный/повторный приём;
- тип документа;
- час приёма;
- объём vs качество.

Для малых групп применять suppression.

### 5.5. Сравнения

- вчера vs позавчера;
- вчера vs среднее последних 7 дней;
- вчера vs тот же день недели за 4 недели;
- month-to-date vs предыдущий месяц на сопоставимом числе дней;
- выбранные произвольные периоды.

Не сравнивать неполный день с полным без явной пометки.

### 5.6. Очередь действий

- критические случаи;
- случаи с потенциальным вредом;
- низкая coverage/confidence;
- расхождение диагноза/МКБ;
- лечение/обследования на ручную проверку;
- data-quality anomalies;
- повторные нарушения врача/подразделения;
- просроченные CRM-задачи.

Каждая строка содержит:

- приоритет;
- причину;
- evidence;
- источник;
- оценку и оси;
- ответственного;
- SLA;
- статус;
- историю решений.

### 5.7. Выгрузки

- PDF/print-ready HTML без технического жаргона;
- XLSX/CSV текущей отфильтрованной выборки;
- JSON агрегатов;
- отдельный secure case export только для авторизованного методиста;
- имя файла включает дату, revision и filter hash.

---

## 6. Целевой CRM/BI-дизайн дашборда МО

### 6.1. Принципы

- единый app shell, без двух вложенных навигаций;
- desktop-first для методиста, responsive до 390 px;
- современная плотность данных, но не перегруженность;
- sans-serif;
- нейтральная светлая палитра, один основной accent;
- без декоративных градиентов и эмодзи в клинических статусах;
- цвет не единственный носитель смысла;
- sticky global context;
- drill-down без потери фильтров;
- один очевидный primary action;
- сохранённые представления и ссылки на текущий срез;
- на каждом числе источник, период и знаменатель.

### 6.2. Информационная архитектура

Левое меню:

```text
МО Аналитика
├── Обзор
├── Отчёт за вчера
├── Очередь разбора
├── Все случаи
├── Врачи
├── Специальности
├── Диагнозы и МКБ
├── Безопасность
├── Качество данных
├── Отчёты
└── Настройки
```

Верхняя глобальная панель:

- период: вчера / 7 дней / месяц / произвольный;
- сравнение периода;
- филиалы multi-select;
- специальности multi-select;
- врачи multi-select;
- тип документа;
- статусы;
- сохранённое представление;
- freshness;
- экспорт;
- сброс фильтров.

Все фильтры:

- поддерживают множественный выбор для периода/месяцев, филиалов, специальностей,
  врачей, типов документа, МКБ, статусов, severity и осей;
- применяются одинаково к KPI, графикам, таблице, профилям и экспорту;
- сериализуются в URL;
- восстанавливаются после reload/back;
- показываются chips;
- доступны для удаления по одному;
- имеют поиск;
- возвращают facet counts;
- пересчитывают зависимые facets после изменения среза;
- поддерживают Include/Exclude;
- сохраняются как private/team view.

Один `active month` больше не является скрытым источником данных. API получает
`date_from/date_to` или массив `periods`, а сравнение - отдельный `compare_period`.

### 6.3. Обзор

Верхний ряд KPI:

- МО за период;
- оценено;
- итоговая оценка;
- требует внимания;
- критические;
- полнота проверки;
- надёжность;
- свежесть данных.

Каждый KPI:

- значение;
- delta;
- сравниваемый период;
- sparkline;
- tooltip с формулой;
- click -> отфильтрованные случаи.

Ниже:

- тренд объёма и качества;
- funnel: БД -> валидные -> МО -> оценено -> внимание -> критично;
- Pareto причин;
- heatmap «специальность x ось»;
- scatter «объём x качество» по врачу/филиалу;
- главы МКБ;
- блок «Что требует внимания сегодня»;
- data freshness/quality.

### 6.4. Отчёт за вчера

Отдельная first-class страница:

- executive summary;
- сравнения;
- риски;
- динамика;
- лучшие/слабые срезы с минимальным N;
- очередь действий;
- data-quality;
- кнопки «Скачать отчёт», «Открыть очередь», «Поделиться ссылкой на срез».

Нельзя скрывать неполную загрузку: banner `данные неполные`, publish заблокирован.

### 6.5. Очередь разбора как CRM

Колонки:

- приоритет;
- дата;
- филиал;
- врач/специальность;
- диагноз;
- итог/оси;
- P0/P1;
- coverage/confidence;
- причина;
- ответственный;
- SLA;
- статус;
- последнее действие.

CRM-статусы:

```text
new
assigned
in_review
confirmed_issue
false_positive
needs_more_data
sent_to_doctor
resolved
closed
```

Функции:

- назначить ответственного;
- bulk assign/status;
- комментарий;
- теги;
- due date/SLA;
- подтверждение/отклонение каждой находки;
- запрос врачу;
- шаблоны действий;
- audit log;
- saved views;
- уведомления;
- export текущего среза.

### 6.6. Таблица случаев

- server-side pagination/filter/sort;
- sticky header и первые колонки;
- настройка видимых колонок;
- сохранение density/columns;
- multi-select;
- быстрые filters;
- row preview;
- keyboard navigation;
- virtualized rendering при больших страницах;
- total и агрегат пересчитываются по всему фильтру, не только текущей странице;
- экспорт всей выборки выполняется сервером job-ом.

### 6.7. Разбор случая

Открывать в правой drawer/panel, не уводя из списка:

1. Header: дата, врач, специальность, филиал, диагноз, статус CRM.
2. Итоговая оценка и версия scorer.
3. Оси A/B/C/D.
4. Coverage/confidence.
5. Ограничения оценки и причины.
6. P0-P3.
7. Клиническая цепочка:
   жалобы + анамнез + статус -> диагноз -> МКБ -> обследования -> лечение -> наблюдение.
8. Evidence из документа.
9. Evidence из протокола/НПА.
10. Применимость протокола и trust.
11. Пройденные проверки.
12. Нейросетевой анализ как дополнительный слой.
13. Решение методиста и audit log.
14. Действие «Проверить этот документ отдельно» - открыть нейтральный
    «Анализ документа» с разрешённым prefill, не переименовывая запись в КЗ.

### 6.8. Врачи, специальности, филиалы

Профиль сущности:

- объём;
- тренд;
- score/axes;
- risk mix;
- coverage/confidence;
- частые причины;
- диагнозы;
- динамика относительно собственной baseline;
- сравнение только с сопоставимой специальностью и достаточным N;
- очередь открытых случаев;
- исключить публичные рейтинги при малом N.

### 6.9. Диагнозы и безопасность

- главы/коды МКБ;
- mismatch документа и МИС;
- жалобы/анамнез не подтверждают диагноз;
- обследования не соответствуют диагностической гипотезе;
- лечение требует проверки;
- safety findings;
- протокол применим / advisory / недостаточно данных;
- динамика и drill-down.

### 6.10. Качество данных

- freshness;
- row parity;
- parse rate;
- date mismatch;
- missing doctor/specialty/age/MKB;
- join anomalies;
- duplicates;
- source volume anomaly;
- scoring failures;
- quarantined partitions;
- retry/reconcile state;
- версия данных и кода.

---

## 7. API МО и обратная совместимость

Новый namespace:

```text
GET  /api/methodist/mo/overview
GET  /api/methodist/mo/daily-report?date=YYYY-MM-DD
GET  /api/methodist/mo/trends
GET  /api/methodist/mo/facets
GET  /api/methodist/mo/cases
GET  /api/methodist/mo/cases/{case_id}
POST /api/methodist/mo/cases/bulk-action
GET  /api/methodist/mo/entities/doctors/{id}
GET  /api/methodist/mo/entities/specialties/{id}
GET  /api/methodist/mo/data-quality
GET  /api/methodist/mo/reports
POST /api/methodist/mo/exports
GET  /api/methodist/mo/saved-views
POST /api/methodist/mo/saved-views
```

UI route:

```text
/methodist/mo
/methodist/mo/yesterday
/methodist/mo/cases
```

Compatibility:

- старые `/api/methodist/mis-kz-quality/*` работают минимум два релиза;
- старый `/methodist/mis-kz-quality` делает redirect на `/methodist/mo`;
- response содержит deprecation metadata;
- старые поля `kz_kind` и `evaluation_v3` не удалять;
- добавить `document_kind` и человекочитаемый `document_kind_label`;
- compatibility tests обязательны.

---

## 8. Разделение КЗ, МО и одиночного анализа в продукте

### 8.1. Пациент

- вкладка: «Проверить КЗ»;
- загрузка одного пациентского заключения;
- понятный B2C-отчёт;
- не показывать массовую аналитику врачей/филиалов;
- не показывать технические уровни, модели и инфраструктуру.

### 8.2. Одиночный анализ

- название: «Анализ документа»;
- принимает PDF/TXT/DOCX;
- классифицирует `КЗ / МО / другой документ`;
- показывает confidence типа;
- при низкой confidence предлагает выбрать тип;
- применяет соответствующую рубрику;
- результат явно показывает, по какой рубрике оценён.

### 8.3. МО

- только методист/администратор;
- источник - БД МИС;
- массовые агрегаты, сравнения, очередь и CRM;
- ежедневный отчёт;
- drill-down до случая в разрешённом контуре;
- журнал действий.

---

## 9. Точность и объективность оценки

### 9.1. До production rollout v3

Обязательные условия:

1. gold 800-1200 МО;
2. двойная разметка;
3. арбитраж разногласий;
4. стратификация по специальностям, диагнозам, филиалам, риску и качеству данных;
5. MAE/QWK;
6. harm recall;
7. false-critical rate;
8. calibration curves;
9. inter-rater agreement;
10. shadow не менее 30 календарных дней;
11. сравнение legacy/v3;
12. письменное решение о `PRIMARY=1`;
13. отдельное решение о `GATE=1`.

### 9.2. Протоколы

- ручная валидация top-протоколов;
- trust B/A только с подтверждённой цитатой;
- применимость по возрасту, полу, беременности, setting, МКБ, версии;
- неподтверждённое не штрафует;
- versioned requirements;
- audit trail методиста.

### 9.3. Лекарственные проверки

- база дозировок;
- возраст/вес/беременность;
- дубли;
- взаимодействия;
- high-alert;
- мониторинг;
- формуляр;
- источник и версия;
- finding без достаточного контекста -> needs human, не hard error.

### 9.4. Explainability contract

Каждая оценка обязана содержать:

- score и status;
- оси;
- coverage/confidence;
- risk/cap;
- findings;
- evidence;
- source_ref;
- trust;
- applicability;
- scorer/build/data version;
- причины отсутствия проверки;
- legacy comparison до завершения миграции.

---

## 10. Очистка терминологии и брендов

### 10.1. Первая очередь

- `index.html`;
- `mis-kz-quality.html`;
- `patient.html`;
- `patient-ui.js`;
- manifest/title/aria/tooltip;
- пользовательские сообщения `rag_server.py`;
- `rule_labels_ru.py`;
- `patient_upload_classifier.py`;
- `kz_block_sources.py`;
- `search_analytics_public.py`.

### 10.2. Вторая очередь

- README;
- министерские материалы;
- презентация;
- актуальные архитектурные print HTML;
- активные планы;
- внешние отчёты.

### 10.3. Не менять механически

- extracted protocol text;
- `data/protocol_summaries/`, `corpus/`, `output/`;
- regex-классификаторы без тестов;
- `kz_*` identifiers;
- env `RAG_*`;
- старые API/JSON events;
- provider adapters;
- медицинские источники в provenance;
- исторические отчёты.

### 10.4. Автоматическая проверка

Добавить terminology lint для user-facing файлов:

- запрещённые слова;
- allowlist для технических секций;
- не считать CSS `cursor:`;
- snapshot-тесты UI-строк;
- проверка новых OpenAPI descriptions.

---

## 11. Безопасность и доступ

- methodist auth обязателен;
- роли: viewer / methodist / lead / admin;
- row-level доступ по филиалу при необходимости;
- audit log всех CRM-изменений;
- no-store для case detail;
- CSRF-защита mutating endpoints;
- export job с TTL;
- HMAC вместо открытого patient/visit id в public;
- raw text не попадает в Telegram, git, public snapshots;
- secrets только env/keychain;
- backup SQLite/raw manifests;
- retention policy;
- право на удаление/пересборку;
- suppression малых групп;
- действия модели не считаются решением врача.

---

## 12. Производительность

Цели:

- overview API p95 <500 ms;
- facets p95 <800 ms;
- cases p95 <1 s;
- case detail p95 <1.5 s без нейросети;
- first dashboard render <2.5 s;
- daily pipeline <60 минут при обычном объёме;
- memory в пределах текущего server plan.

Меры:

- precomputed daily/monthly aggregates;
- SQLite indexes;
- server-side pagination;
- bounded facets;
- response compression;
- ETag/data revision;
- lazy detail;
- async export jobs;
- кэш по filter hash;
- invalidate после нового successful report.

---

## 13. Тестирование

### 13.1. Export/pipeline

- date window/timezone;
- VPN state restoration;
- DB retry;
- empty day;
- late rows;
- duplicate id;
- idempotent rerun;
- catch-up;
- quarantine;
- atomic write;
- resume;
- no secrets in logs.

### 13.2. Classification

- МО;
- КЗ;
- справка;
- диагностика;
- non-clinical;
- empty;
- unknown;
- услуги/оплата/специализация;
- ambiguous -> manual.

### 13.3. Scoring

- v3 schema;
- caps;
- coverage/confidence;
- trust;
- applicability;
- P0/P1 evidence;
- no penalty C/D;
- no NaN/Inf;
- legacy comparison.

### 13.4. API

- auth/RBAC;
- filters;
- multi-select;
- comparison;
- facets;
- pagination/sort;
- daily report;
- exports;
- CRM actions;
- audit log;
- old-route compatibility.

### 13.5. Frontend

- URL state;
- saved views;
- no-data/loading/error/stale/partial;
- keyboard;
- focus;
- 390x844, 768x1024, 1280x720, 1440x900;
- no horizontal overflow;
- axe critical/serious = 0;
- visual regression;
- terminology lint;
- no provider brands in UI.

### 13.6. Production smoke

- `/health`, `/api/version`;
- freshness;
- yesterday report;
- global filters;
- case drill-down;
- CRM save/reload;
- export;
- old redirect;
- Telegram success;
- VPN final state.

---

## 14. Фазы реализации без пропусков

### Фаза 0. Governance и baseline

- [x] Сохранить baseline status/branch/SHA/version: `main`, `e48d40d`,
      `2026-07-27-r20-mis-kz-route`.
- [x] Обновить индекс планов, старые планы отметить archived с этим преемником.
- [x] Зафиксировать glossary.
- [x] Зафиксировать текущие API и UI snapshots в аудите/коде; browser screenshots
      выполняются повторно после frontend-фазы для сравнения.
- [x] Выполнить baseline tests: 1068 passed, 1 skipped, 3 baseline failures
      (`consult_cache`, `medication_safety pregnancy`, nested dashboard route).
- [x] Зафиксировать метрики: 477 протоколов, 11 259 требований, penalty-ready 0%;
      27.07.2026 из БД - 580 строк, 481 строка / 472 уникальных визита eligible,
      doctor 99.5%, date mismatch 0.

### Фаза 1. Терминология и маршруты

- [x] Развести КЗ / МО / Анализ документа в UI.
- [x] Убрать RAG/LLM/provider brands из user-facing текстов.
- [x] Добавить terminology lint для МО-дашборда.
- [x] Добавить новые МО routes/API aliases.
- [x] Сохранить совместимость старых маршрутов.
- [x] Обновить актуальные docs.

### Фаза 2. Daily extraction foundation

- [x] Вынести reusable export library.
- [x] Добавить daily exporter.
- [x] Добавить `scripts/validate_mis_export.py` с blocking/warning gates.
- [x] Добавить `scripts/merge_mis_protocol_export.py`: daily Parquet -> rolling month,
      upsert по `mis_protocol.id`, atomic replace, контроль дублей.
- [x] Добавить `scripts/run_mo_daily_report.py`: единый Python-orchestrator и генератор
      daily JSON/HTML/CSV.
- [x] Добавить thin launchd-wrapper без бизнес-логики и отдельный retry-wrapper.
- [x] Добавить VPN state machine.
- [x] Добавить lock/state/retry/resume.
- [x] Добавить raw partition/meta/quarantine.
- [x] Добавить data-quality gate.
- [x] Добавить idempotent merge/reconcile/catch-up.
- [x] Добавить launchd installer/status/uninstall.
- [x] Добавить Telegram status.
- [x] Согласовать запуск с `sql_epam` 06:30 и использовать его status как дополнительный
      health-сигнал, не смешивая Zoho sync и МО-scoring.

Реализовано и проверено 2026-07-28 реальным SQL/VPN-прогоном: основной launchd назначен
на 07:00, retry на 10:00, hourly catch-up; все три задания установлены и загружены.
`MO_SQL_EPAM_STATUS_FILE` читается как необязательный read-only health-сигнал и не запускает
соседний контур; в текущем окружении отдельный status-файл `sql_epam` не настроен.

### Фаза 3. Document taxonomy и warehouse

- [x] `document_kind`.
- [x] Правила МО с config.
- [x] Golden fixtures классификации.
- [x] SQLite star schema.
- [x] Миграция истории с 2026-01-01.
- [x] Проверка row parity/duplicates.
- [x] Secure/public split.

История перенесена из уже проверенных локальных выгрузок без повторной нагрузки на БД:
92 177 записей, 208 календарных дней, 2026-01-02 - 2026-07-27, дублей `mis_id` нет.
Повторяемый импорт: `scripts/backfill_mo_warehouse.py`; даты после вчера автоматически
исключаются из локальной витрины.

### Фаза 4. Daily scoring/report

- [x] Legacy + deep + v3 shadow.
- [x] Daily aggregates.
- [x] Month-to-date.
- [x] Yesterday report JSON/HTML/CSV.
- [x] Comparisons.
- [x] Data-quality appendix.
- [x] CRM action queue.
- [x] Локальный обезличенный public snapshot + freshness smoke.
- [x] Late-data revision.

Реальный отчёт за 27.07.2026: 580 строк, 472 оценённых уникальных случая, 0 ошибок,
средний балл 71.5, требуют внимания 160, критические 25 (17 по клиническим рискам,
10 по низкому баллу; категории пересекаются). Data-quality gate пройден.

Удалённая публикация не включена: до подтверждения разрешённого persistent storage
pipeline публикует только локальный обезличенный snapshot и не вызывает legacy upload
месячных CSV/Parquet.

### Фаза 5. CRM/BI backend

- [x] Новый `/api/methodist/mo/*`.
- [x] Overview/trends/facets.
- [x] Cases/detail.
- [x] Entities.
- [x] CRM state/events.
- [x] Saved views.
- [x] Export jobs с приватным скачиванием и сроком действия.
- [x] RBAC/audit/security.
- [x] Performance indexes/caches.

### Фаза 6. CRM/BI frontend

- [x] Единый app shell.
- [x] Global context bar и множественный выбор месяцев/организационных фильтров.
- [x] Обзор.
- [x] Отчёт за вчера.
- [x] Очередь.
- [x] Все случаи.
- [x] Врачи/специальности/филиалы.
- [x] Диагнозы/МКБ.
- [x] Безопасность.
- [x] Качество данных.
- [x] Reports/exports.
- [x] Case drawer с критериями, доказательствами и решением методиста.
- [x] Saved views.
- [x] Responsive/a11y и статические проверки JavaScript/keyboard invariants.
- [ ] Удалить дубли старого UI после compatibility period.

### Фаза 7. Точность v3 и методическая работа

- [x] Provenance spans для точных цитат из полей документа: field/start/end/text.
- [ ] Top-протоколы trust B/A.
- [x] Стратифицированный манифест на 1000 МО: все доступные низкие score bands,
      117 справок и 588 red-flag; raw manifest остаётся gitignored.
- [ ] Двойная человеческая разметка 1000 МО и арбитраж - нельзя подменять
      нейросетевыми proxy-метками.
- [x] Proxy calibration/evaluation: n=411, corr=0.605, precision bad=0.84,
      recall bad=0.37; human calibration остаётся обязательной до primary.
- [~] Доизвлечение лекарственных режимов: prompt/merger теперь сохраняют дозу,
      путь, частоту, длительность и мониторинг; нужен повторный серверный прогон корпуса.
- [ ] Дозировки/формуляр.
- [ ] Mapping услуг к обследованиям.
- [ ] Shadow 30 дней.
- [ ] Решение о primary.
- [ ] Отдельное решение о gate.

### Фаза 8. Acceptance и rollout

- [x] Unit/integration/API/frontend tests.
- [x] Full relevant pytest: 1118 passed, 1 skipped.
- [x] Python diagnostics, JavaScript syntax и `git diff --check`; ruff отсутствует
      в текущем окружении и не является зависимостью проекта.
- [ ] Browser matrix.
- [ ] Axe/Lighthouse.
- [ ] Visual regression.
- [x] Performance benchmark: warm overview 1.37 с, facets 1.38 с, cases 2.59 с,
      trends 2.87 с.
- [x] Security/PII review.
- [x] Backfill/reconciliation: 92 177 строк, 208 дней, 0 дублей, cutoff вчера.
- [ ] 7 дней ежедневных успешных runs.
- [x] Итоговый отчёт «до/после»:
      `docs/reports/2026-07-28-mo-daily-bi-result-v1.md`.
- [x] Обновить BUILD_VERSION: `2026-07-28-r2-mo-daily-crm-bi`.
- [ ] Commit.
- [ ] Push.
- [ ] Deploy smoke.
- [x] Проверить Telegram и VPN final state: Telegram enabled/доставка успешна,
      VanyaVPN восстановлен в `Connected`, три launchd job загружены.

---

## 15. Перенесённый backlog из старых планов

Чтобы новый план был полным, сюда перенесены незавершённые обязательства:

- реальный gold set и методистская валидация;
- provenance spans;
- top-50 протоколов;
- полная база дозировок/формуляр;
- production rollout v3;
- массовый пересчёт MIS;
- deep-фильтры axes/severity/harm/age/agreement;
- ручная проверка слабых/14 fallback-протоколов;
- mapping услуг к обследованиям;
- окончательная проверка type/specialty/УЗИ;
- precompute section overviews;
- axe/keyboard/reflow/visual regression;
- browser navigation tests;
- чипы/saved views/URL state;
- единый summary source;
- 6 baseline test failures должны быть повторно проверены и либо исправлены, либо
  формально изолированы с issue/обоснованием.

---

## 16. Риски и rollback

1. **БД недоступна** - retry/catch-up, предыдущий отчёт остаётся current.
2. **VPN не выключился** - abort до SQL.
3. **VPN не восстановился** - critical alert.
4. **Поздние записи** - reconcile 3 дней + revision.
5. **ПДн** - secure/public split, suppression, no raw upload.
6. **Ложные клинические штрафы** - trust/applicability, v3 shadow, gold до primary.
7. **Сломана совместимость** - aliases, redirects, contract tests.
8. **UI слишком сложный** - progressive disclosure, saved views, role defaults.
9. **Сравнение малых групп** - minimum N и suppression.
10. **Нейросетевая quota** - deterministic report не блокируется.
11. **Сон/выключение Mac** - RunAtLoad/hourly catch-up; SLA требует включённый Mac.
12. **SQLite corruption** - raw immutable partitions + rebuild command + backup.
13. **Новый scorer ухудшает метрики** - feature flags и мгновенный rollback на legacy.

Rollback:

- отключить новый UI feature flag;
- старые routes/API остаются;
- `KZ_EVALUATION_V3_PRIMARY=0`, `GATE=0`;
- current report pointer вернуть на предыдущую revision;
- warehouse полностью пересобирается из raw partitions.

---

## 17. Definition of Done

Задача полностью завершена только когда одновременно:

1. терминология КЗ/МО/Анализ документа соответствует §0;
2. в user-facing UI отсутствуют RAG/LLM/provider brands;
3. ежедневный pipeline автономно обрабатывает вчера;
4. пропущенные дни догружаются;
5. rerun идемпотентен;
6. VPN всегда восстанавливается;
7. data-quality gate блокирует плохую публикацию;
8. отчёт за вчера содержит все разделы §5;
9. dashboard реализует все страницы §6;
10. CRM workflow сохраняется и аудируется;
11. API совместим со старыми клиентами;
12. ПДн не покидают разрешённый контур;
13. scorer объясним и versioned;
14. v3 не включён primary/gate без gold и решения;
15. tests/a11y/performance/security пройдены;
16. backfill с начала 2026 года выполнен;
17. минимум 7 последовательных ежедневных runs успешны;
18. план и метрики обновлены;
19. создан итоговый отчёт;
20. изменения закоммичены, отправлены в origin и проверены после deploy.
