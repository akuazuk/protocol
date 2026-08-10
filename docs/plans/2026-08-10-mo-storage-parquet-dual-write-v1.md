# Миграция хранения МО: parquet + sqlite, без ломки night cron (v1)

Дата: 2026-08-10  
Статус: active  
Связанные: `2026-08-07-by-home-gcp-llm-split-v1.md` (E2 night cron),
`deploy/mac-bridge/extract-contract.md`, `deploy/gcp-app/night_mis_pipeline.sh`,
`deploy/gcp-app/score_inbound_day.sh`

---

## 1. Контекст

Сейчас на GCE:

| Слой | Факт |
|--|--|
| Night cron | 02:00/03:00 UTC → SQL → CSV inbound → score → sqlite |
| `secure_cases` | дневные `mo_*.csv` плотно с 2026-07-26 |
| `warehouse` | sqlite **2026-01-02…2026-08-09** (канон для API/BI) |
| Дубли | inbound CSV + secure CSV + jsonl cases |

CSV удобен как transit, но плохой долговременный SSOT (размер, типы, два режима хранения).

**Жёсткое ограничение:** не ломать `night_mis_pipeline.sh` / cron. Миграция только **additive** → dual-read → optional drop CSV.

---

## 2. Целевая схема (без big-bang)

```text
Marina SQL
  → night extract (как сейчас)
  → inbound/extract/
       mo_DAY.parquet     (новый канон)
       mo_DAY.csv         (пока пишем - совместимость cron/score)
       mo_DAY.meta.json   (+ checksum parquet)
  → secure_cases/YYYY/MM/
       mo_DAY.parquet     (канон day detail)
       mo_DAY.csv         (shim, пока score читает CSV)
       kz_l1_*            (без изменений на фазе 1-2)
  → warehouse/mo_analytics.sqlite  (без изменений; SSOT для API)
```

API/дашборды **не** переезжают с sqlite. CSV остаётся export/shim, не источник истины.

---

## 3. Метрики

| Метрика | Было | Цель |
|--|--|--|
| Night cron status | success (smoke 09.08) | success каждый день; dual-write не роняет job |
| Формат inbound | CSV only | parquet + CSV (фаза 1) |
| Score input | только CSV | CSV или parquet (фаза 2) |
| Disk `secure_cases` CSV >30д | растёт линейно | CSV TTL / не писать (фаза 3) |
| API `/api/methodist/mo/*` | sqlite | без регрессии |

---

## 4. Шаги (без ломки cron)

### Фаза 0 - зафиксировать контракт (0.5 дня) - дальше

- [ ] P0. Дописать в `extract-contract.md`: optional `mo_DAY.parquet`, `checksum_sha256_parquet`, `format_version`.
- [ ] P0. Явно: night job **обязан** писать CSV, пока `MO_INBOUND_REQUIRE_CSV=1` (default).

### Фаза 1 - dual-write (1-2 дня) - дальше

- [ ] P1. В `night_mis_pipeline.sh` после успешного export:
  - уже есть parquet у экспортёра → копировать в `inbound/mo_DAY.parquet`;
  - CSV как сейчас;
  - meta: оба checksum.
- [ ] P1. `score_inbound_day.sh` **не менять вход** (по-прежнему CSV). Cron путь идентичен.
- [ ] P1. Smoke: один день dual-write; сравнить `row_count` csv vs parquet.
- [ ] P1. Метрика в status json: `formats=["csv","parquet"]`.

### Фаза 2 - dual-read score (2-3 дня) - дальше

- [ ] P2. `run_mis_protocol_l1_batch` / wrapper: если есть parquet и `MO_SCORE_PREFER_PARQUET=1` - читать parquet, иначе CSV.
- [ ] P2. Default на проде **CSV**, пока 3 зелёные ночи подряд на parquet shadow.
- [ ] P2. Feature flag в `.env.gcp-staging` / night env; rollback = unset flag (cron тот же).

### Фаза 3 - ужать CSV (после стабилизации) - дальше

- [ ] P3. Default score = parquet; CSV писать только если `MO_WRITE_DAY_CSV=1`.
- [ ] P3. TTL: удалять `secure_cases/**/mo_*.csv` старше N дней (N=14), **не** трогать parquet/jsonl/sqlite.
- [ ] P3. Опциональный `scripts/export_mo_day_csv.py` для методиста «скачать день».
- [ ] P3. Обновить night cron docs; Mac path уже off.

### Вне скоупа v1

- Перенос warehouse на Postgres.
- Сжатие `kz_l1_*_cases.jsonl`.
- Backfill всех янв-июл в parquet files (не нужно для API).

---

## 5. Как не сломать night cron

1. **Не менять** расписание `0 2` / `0 3` и entrypoint `night_mis_pipeline.sh main|retry`.
2. Любой новый формат - **после** успешного CSV write; ошибка parquet = warning, status всё ещё success если CSV+score ok (фаза 1).
3. Флаги только env; rollback без redeploy кода образа web - достаточно env/script на VM.
4. Один smoke-день перед включением dual-read default.
5. Не параллелить с PR, которые трогают `score_inbound_day.sh` без очереди.

Rollback одной командой (фаза 2):

```bash
# на VM: выключить parquet-prefer, оставить CSV path
sudo sed -i '/MO_SCORE_PREFER_PARQUET/d' /opt/protocol/.env.mis || true
```

---

## 6. Риски

| Риск | Митигация |
|--|--|
| Parquet schema drift vs CSV columns | snapshot schema в meta; тест row_count + key columns |
| Score читает другой набор колонок | golden day compare CSV vs parquet scores |
| Диск забьётся обоими форматами | фаза 3 TTL CSV; следить `du -sh secure_cases` |
| Путаница inbound/secure | один helper `promote_inbound_day()` |

---

## 7. Владение файлами

План владеет: `deploy/gcp-app/night_mis_pipeline.sh`, `score_inbound_day.sh`,
`deploy/mac-bridge/extract-contract.md`, тонкий reader в `scripts/run_mis_protocol_l1_batch.py`
(только вход файла).

Не смешивать с scorer/ICD product PR.

---

## 8. Definition of Done v1

1. Контракт parquet описан; dual-write в проде ≥3 ночи.
2. Cron 02:00/03:00 без изменений поведения success.
3. Flag dual-read готов; default всё ещё CSV или явно задокументирован parquet.
4. План фазы 3 (TTL CSV) согласован, не обязательно выполнен в том же PR.

---

## 9. Статус

План создан 2026-08-10. Реализация не начата. Night cron E2 уже работает на CSV - это база для dual-write.
