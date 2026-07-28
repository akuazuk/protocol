# Persistent Disk на Render для корпуса RAG

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
