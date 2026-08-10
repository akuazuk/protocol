# Локальные выгрузки mis_protocol (КЗ из МИС)

Данные **персональные** - в git не коммитятся (`*.parquet`, `*.csv` в `.gitignore`).

В git можно класть только агрегаты: `kz_l1_YYYY-MM_summary.json` (без patient_id / полного текста КЗ).

## Пересоздать за месяц

```bash
python3 scripts/export_mis_protocol_month.py --month 2026-07
```

Нужен `KRAVIRA_DB_PASSWORD`: на GCE `/opt/protocol/.env.mis`; локально
`~/CURSOR/sql_epam/.env` (fallback).

Схема полей: `epam/scheme_mis_protocols.docx`, парсер: `clinical_knowledge/mis_protocol_parse.py`.

### Разделение КЗ и не-КЗ (`kz_kind`)

Экспортёр тянет `type` (в схеме КЗ = 9), резолвит специальность автора по `doctor_id`
протокола и классифицирует каждую строку (`classify_kz_kind`) в столбец `kz_kind`:

- `kz` - клиническое консультативное заключение (оцениваем);
- `certificate` - справка/профосмотр `pay_type=12` (оцениваем отдельной рубрикой);
- `diagnostic` - УЗИ / рентген / функц. диагностика / эндоскопия / лаборатория (**НЕ оцениваем**);
- `non_clinical` - медсестра / стоматология / логопед (НЕ оцениваем);
- `empty` - нет клинического содержания (НЕ оцениваем).

В L1/L2 идут только `kz` + `certificate`; остальное сводится в `excluded_breakdown`
summary (панель «Гигиена данных»). Счётчики `doc_type_distribution` / `kz_kind_counts`
пишутся в `*.meta.json`. См. `docs/plans/2026-07-22-kz-data-separation-viz-v1.md`.

## Массовый L1-анализ качества КЗ

Детерминированный L1 (без LLM, API cost ~$0). Полный прогон июля (~7.6k уникальных визитов) на Render: ~15-25 мин при `--direct --workers 1`.

```bash
# На Render SSH / Web Shell (рекомендуется --direct: без HTTP/429):
cd /opt/render/project/src
PYTHONPATH=. python3 scripts/run_mis_protocol_l1_batch.py \
  --csv /var/data/mis_protocol/mis_protocol_2026-07.csv \
  --out-dir /var/data/mis_protocol \
  --month 2026-07 --resume --reset-fails --direct --workers 1
```

Артефакты на диске:
- `kz_l1_2026-07_cases.jsonl` - построчно (ПДн, **не в git**)
- `kz_l1_2026-07_summary.json` - агрегаты по врачам/спец./филиалам (**можно в git**)
- `kz_l1_2026-07_state.jsonl` - resume

Дашборд: кабинет методиста → вкладка «MIS · КЗ» или страница `mis-kz-quality.html`
(включая топ-30 слабых визитов с visit_id, датой и комментарием).
API: `GET /api/methodist/mis-kz-quality` (нужен methodist token).

План: `docs/plans/2026-07-21-mis-kz-l1-batch-v1.md`.

## На Render (тестовый диск)

Файлы кладутся в `/var/data/mis_protocol/` (persistent disk), не в git.

```bash
bash scripts/render_mis_protocol_data.sh upload 2026-07
bash scripts/render_mis_protocol_data.sh list
bash scripts/render_mis_protocol_data.sh delete 2026-07
bash scripts/render_mis_protocol_data.sh delete-all
```

На самом Render (SSH):

```bash
ls -lh /var/data/mis_protocol/
rm -f /var/data/mis_protocol/mis_protocol_2026-07.*
```
