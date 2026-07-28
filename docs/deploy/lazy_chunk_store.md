# Lazy Chunk Store на Render

Manifest + disk-backed chunks вместо загрузки ~55k чанков в RAM при старте.

## Артефакты на Persistent Disk

После заливки `corpus_chunks_parts/*.jsonl` на `/var/data`:

```bash
# Path manifest (один JSONL-ряд на PDF)
python3 scripts/build_corpus_path_manifest.py \
  --corpus /var/data/corpus_chunks_parts \
  --output /var/data/corpus_path_manifest.jsonl

# Опционально: lex shards по rubric (ускорение retrieve)
python3 scripts/build_path_lex_shards.py \
  --corpus /var/data/corpus_chunks_parts \
  --output /var/data/lex_shards
```

## Env (Render)

| Переменная | Значение |
|------------|----------|
| `RAG_STARTUP_MODE` | `manifest` |
| `RAG_LAZY_CHUNK_STORE` | `1` |
| `RAG_LAZY_RETRIEVE` | `1` |
| `RAG_FORBID_FULL_CORPUS_RETRIEVE` | `1` |
| `RAG_MANIFEST_PATH` | `/var/data/corpus_path_manifest.jsonl` |
| `RAG_PATH_LEX_SHARDS` | `1` (если собраны shards) |
| `RAG_LEX_SHARDS_DIR` | `/var/data/lex_shards` |

## Проверка после деплоя

```bash
curl -s "$URL/health" | jq '{startup_mode, manifest_paths, chunks, rag_ready, chunk_store}'
```

Ожидание: `startup_mode=manifest`, `chunks=0`, `rag_ready=true`, `manifest_paths>0`.

## Откат

```bash
RAG_STARTUP_MODE=full
RAG_LAZY_CHUNK_STORE=0
RAG_LAZY_RETRIEVE=0
```

Redeploy. Поведение как до lazy store (с OOM caps consult).

## Local dev

По умолчанию `RAG_STARTUP_MODE=full` - полная загрузка корпуса. Для проверки manifest mode:

```bash
export RAG_STARTUP_MODE=manifest
export RAG_MANIFEST_PATH=data/catalog/corpus_path_manifest.jsonl
python3 scripts/build_corpus_path_manifest.py
uvicorn rag_server:app --reload
```

`RAG_STARTUP_MODE=full` оставлен для локальной разработки; на Render рекомендуется `manifest`.
