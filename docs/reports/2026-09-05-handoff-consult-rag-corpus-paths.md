# Handoff: вкладка «Анализ документа» - пустой RAG на GCE

Дата: 2026-09-05
Репозиторий: `akuazuk/protocol`
Ветка: `cursor/consult-rag-corpus-paths-agent1-pc1`
Worktree: `/private/tmp/protocol-task-consult-rag-corpus-paths-pc1`
Base: `53d61e51` (`origin/main`)
Production SHA: тот же `53d61e51` (код не выкладывали; починили данные и env)

## Причина ошибки

Сообщение «Укажите коды МКБ-10» было ложным: в `gastro_1` коды есть (K30, Q43.8, E61.1, R14).

На GCE `RAG_STARTUP_MODE=manifest`, но `RAG_MANIFEST_PATH` указывал на
`data/catalog/corpus_path_manifest.jsonl` (файла нет в образе). `manifest_paths=0`.
Даже после сборки манифеста из `output/chunks/chunks.jsonl` там было только 110 PDF.
L2 выбирал КП 2025 (пищевод/кишечник), которых в этом extract нет, и `retrieve()`
возвращал пусто.

## Что сделано на GCE (тот же image)

- Залит полный `corpus_chunks_parts` (478 PDF, 104687 чанков).
- Собран `/var/data/protocol_corpus/corpus_chunks_parts/corpus_path_manifest.jsonl`.
- Env/container: `RAG_CHUNKS_DIR` и `RAG_MANIFEST_PATH` смотрят на этот каталог.
- Smoke: `gastro_1` L2 → HTTP 200, протоколы пищевода и кишечника 2025.

## Код в этой ветке

- Деплой больше не сбрасывает RAG на пустой gitignored path.
- L2 оставляет в allowlist только PDF, которые есть в manifest.
- Если МКБ уже есть, ошибка говорит про корпус, а не про отсутствие кодов.

Параллельный PR #187 (лекарства/анализы) эти файлы не трогает. Если он
перезапустит `protocol-web` до merge этой ветки, взять RAG-пути из
`/opt/protocol/.env.gcp-staging` (уже обновлены на VM).
