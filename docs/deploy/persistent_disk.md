# Persistent Disk на Render для RAG и MO

Без диска каждый деплой перечитывает `corpus_chunks_parts/` из образа: cold start 1-3 мин и пик RAM.

## Шаги

1. **Dashboard → protocol-rag → Disks → Add Disk**
   - Name: `rag-chunks`
   - Mount path: `/var/data`
   - Size: 10 GB (или больше)

2. **Залить чанки на диск** (один раз):
   ```bash
   # Локально: собрать rich chunks или скопировать corpus_chunks_parts
   rsync -avz --progress corpus_chunks_parts/ \
     render:/var/data/corpus_chunks_parts/
   ```
   Либо через Render Shell после деплоя с подключённым диском.

3. **Переменные окружения** (уже в `render.yaml`):
   - `RAG_CHUNKS_DIR=/var/data`
   - `RAG_MEMORY_SAVER=1`
   - `MO_DATA_ROOT=/var/data/medical_exams`

4. **Офлайн-эмбеддинги** (ускорение поиска):
   ```bash
   python3 scripts/build_chunk_embeddings.py --input /var/data/corpus_chunks_parts
   ```
   Затем `RAG_PRECOMPUTED_CHUNK_EMBED=1` (в `render.yaml`).

5. **Blueprint sync** - после создания диска в UI синхронизируйте Blueprint или проверьте `disk:` в `render.yaml`.

## Проверка

```bash
curl -s https://protocol-bimy.onrender.com/health | jq '{chunks, rag_ready, memory_saver}'
```

После заливки `chunks` должен совпадать с локальным корпусом (~55k+).

**Деплой:** корпус грузится в фоне после bind порта (`lifespan` + `RAG_STARTUP_LOAD_DELAY_SEC`). Сразу после деплоя `rag_ready` может быть `false` 1-2 минуты - это нормально; `/health` отвечает сразу.

## Данные MO Аналитики

Ежедневный pipeline работает на рабочей машине и после прогона публикует данные
в `/var/data/medical_exams`:

```bash
python3 scripts/publish_mo_to_render.py --methodist-token "$METHODIST_TOKEN"
```

На диск попадают:

- `warehouse/mo_analytics.sqlite` - факты и справочники; CRM-строки из локального
  файла не публикуются, чтобы не затереть работу методистов в проде;
- `reports/`, `state/`, `public/`;
- `secure_cases/YYYY/MM/` за последние 45 дней.

Месячные deep-оценки после исторического пересчёта:

```bash
python3 scripts/publish_mo_to_render.py \
  --legacy-first-month 2026-01 --legacy-last-month 2026-07
```

Проверка:

```bash
curl -s -H "X-Methodist-Token: $METHODIST_TOKEN" \
  https://protocol-bimy.onrender.com/api/methodist/mo/freshness | \
  jq '{status, lag_days, data_through, roots}'
```

Норма: `status = "fresh"`, `lag_days <= 1`, первый root -
`/var/data/medical_exams` с `has_reports = true` и `has_secure_cases = true`.
