# План: массовый L1-анализ mis_protocol + дашборд методиста (v1)

- Дата: 2026-07-21
- Статус: active
- Связано с: выгрузка MIS `mis_protocol` (июль 2026, 7916 КЗ)

## 1. Контекст

На Render в `/var/data/mis_protocol/` лежит выгрузка КЗ за июль.
Нужен массовый срез качества по врачам/специальностям/филиалам **без LLM-затрат**,
документация для другого компьютера и визуализация в кабинете методиста.

## 2. Что сделано в проде / репо

| Артефакт | Назначение |
|----------|------------|
| `scripts/run_mis_protocol_l1_batch.py` | L1 batch + resume + summary по врачам |
| `/var/data/mis_protocol/kz_l1_2026-07_*.{jsonl,json,log}` | Полный прогон на Render (cases = ПДн) |
| `data/mis_protocol/kz_l1_2026-07_summary.json` | Агрегаты (без patient_id) - в git |
| `GET /api/methodist/mis-kz-quality` | Отдача summary для UI |
| `mis-kz-quality.html` + вкладка методиста | Визуализация |

## 3. Как повторить на другом компьютере

```bash
# 1) Данные уже на Render; локально (ПДн не в git):
bash scripts/render_mis_protocol_data.sh list

# 2) Запуск L1 на Render Web Shell / SSH:
cd /opt/render/project/src
PYTHONPATH=. python3 scripts/run_mis_protocol_l1_batch.py \
  --csv /var/data/mis_protocol/mis_protocol_2026-07.csv \
  --out-dir /var/data/mis_protocol \
  --month 2026-07 --resume --reset-fails --direct --workers 1

# 3) Забрать только summary (без cases.jsonl с ПДн):
scp ...:/var/data/mis_protocol/kz_l1_2026-07_summary.json data/mis_protocol/

# 4) Пересобрать summary из уже готового jsonl:
python3 scripts/run_mis_protocol_l1_batch.py \
  --csv /var/data/mis_protocol/mis_protocol_2026-07.csv \
  --out-dir /var/data/mis_protocol --month 2026-07 --rebuild-summary-only
```

**Важно:** `*_print` в MIS = флаги `on`/`off`, не текст. В КЗ-текст идут клинические столбцы.
L1 = без RAG/LLM → API cost ~$0. ETA на 7916 при ~100-150 мс **`--direct`**: **~15-25 мин**.
HTTP-режим упирается в rate-limit 60 POST/мин → используйте `--direct`.

## 4. Метрики

| Показатель | Цель | Факт (2026-07-21) |
|------------|------|-------------------|
| Покрытие L1 | все КЗ месяца | **7648** уникальных визитов; **7643** ok / **5** errors |
| Средний overall | ориентир 65-80% | **69.1%** (медиана **69.6%**) |
| API cost | ~$0 | **$0** (`llm_used=false`, `--direct`) |
| UI | вкладка + отдельная страница | вкладка «MIS · КЗ» + `mis-kz-quality.html` |

Гистограмма overall: 0-49: 227; 50-59: 800; 60-69: 2863; 70-79: 2691; 80-89: 899; 90-100: 3.
Слабые блоки L1 в среднем: exams **16.0**, treatment **20.3** (часто пусто в тексте КЗ).

## 5. Шаги

- [x] Скрипт batch + resume
- [x] Полный прогон на Render (`--direct`, workers=1)
- [x] Summary в git + API + UI + пуш
- [ ] (позже) L2 только на bottom-decile

## 6. Риски

- ПДн в `*_cases.jsonl` - не коммитить, только `/var/data`.
- Параллельность >2 на standard Render может упереться в CPU/RAM.
- Системный штраф без даты/ФИО пациента - в batch дата филиала/врача уже добавляются в текст.
