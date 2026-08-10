# Night GCE: скорость score + skip unchanged + алерт fail (v1)

Дата: 2026-08-10  
Статус: active  
Связанные: `2026-08-07-by-home-gcp-llm-split-v1.md` (E2 cron),
`2026-08-10-mo-storage-parquet-dual-write-v1.md` (хранение, отдельно),
`deploy/gcp-app/night_mis_pipeline.sh`, `score_inbound_day.sh`

---

## 1. Контекст

Night на GCE уже работает (02:00 UTC main, 03:00 retry). Узкие места не в SQL join, а в:

- score часто `workers=1`;
- retry с `--force` пересчитывает день даже если extract не изменился;
- fail после 03:00 не алертится (нужно смотреть логи вручную).

Цель v1: **быстрее ночь / дешевле retry / видно падение** - без смены расписания cron и без parquet (это другой план).

---

## 2. Что изменится в проде (после реализации)

| Было | Станет |
|--|--|
| L1 score 1 worker | `MO_DAILY_WORKERS=2` (или 3) на GCE night |
| retry всегда `--force` | `--force` только если checksum inbound ≠ последнего success |
| fail тихий в json/log | Telegram (или лог+exit) после retry fail |

---

## 3. Метрики

| Метрика | Было (ориентир) | Цель |
|--|--|--|
| Wall time score ~400 clinical visits | десятки минут, 1 worker | примерно −40…60% при workers=2 |
| Retry при том же CSV | полный re-score | skip score, status success/`unchanged` |
| Обнаружение fail | ручной `tail` лога | алерт ≤5 мин после 03:00 job |

---

## 4. Шаги

### A. Workers (сделать первым)

- [x] A1. В `night_mis_pipeline.sh` / cron env: `MO_DAILY_WORKERS=2` (default на VM e2-standard-2).
- [x] A2. Проброс в `score_inbound_day.sh` → `run_mis_protocol_l1_batch.py --workers`.
- [x] A3. Smoke на одном дне (например вчера): сравнить wall time и `avg_score` / n cases vs baseline.
- [x] A4. Если CPU throttle / OOM - откат на `1` одной строкой env.

### B. Skip unchanged (без ломки retry)

- [x] B1. В status / meta писать `inbound_sha256` (CSV или parquet checksum).
- [x] B2. Retry: если предыдущий status=`success` **или** sha совпал с уже scored secure day → **не** `--force`, при полном совпадении sha со success - exit 0 early.
- [x] B3. Main: если sha == last success и cases на месте → score resume only (как сейчас todo=0), не чистить kz_l1_*.
- [x] B4. Явный `--force` / `MO_NIGHT_FORCE=1` для ручного пересчёта.

### C. Алерт fail

- [x] C1. После `retry` с status≠success: вызвать существующий `telegram_notify.py` (токены из `.env.gcp-staging`, не в git).
- [x] C2. Текст без PHI: день, mode, detail (`export_failed` / `doctor_join` / `score`), host=gce.
- [x] C3. Опционально cron 03:15 UTC: `check_gce_night_status.sh` если status-файл stale/failed.
- [x] C4. Если Telegram не настроен - писать `ALERT_NEEDED` в лог + non-zero только у checker (night success path не ломать ложным rc от notify).

---

## 5. Не в этом мини-плане

- Parquet dual-write (план storage).
- Secret Manager.
- Сдвиг слота относительно sql_epam.
- Postgres / vacuum sqlite (отдельный ops).

---

## 6. Риски

| Риск | Митигация |
|--|--|
| workers=2 искажает score | детерминизм batch; smoke compare counts/avg |
| Skip скрыл реальный re-extract | сравнивать sha; при fail gate doctor_join не skip |
| Telegram spam | только terminal fail после retry; не на каждую attempt |

---

## 7. DoD

1. Night main с workers≥2 зелёный на 1 smoke-дне.  
2. Повторный retry на том же extract не гоняет полный force-score.  
3. Искусственный fail → алерт (или явный ALERT_NEEDED в логе).  
4. Cron 02:00/03:00 entrypoint без смены имён режимов `main|retry`.

---

## 8. Статус

Реализация в коде 2026-08-10 (`night_mis_pipeline.sh`, `check_gce_night_status.sh`,
`score_inbound_day.sh` workers env, cron 03:15). Smoke на GCE - в том же PR/сессии.
