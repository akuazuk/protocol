# Как использовать экспорт для улучшения системы (Cursor)

1. Скачайте latest.jsonl:
   GET /api/consult-archive/export/latest

2. Положите в репозиторий (опционально):
   cp latest.jsonl tests/fixtures/consult_replay.jsonl

3. Добавьте PDF-кейсы в clients_consult/ (если есть локально).

4. Запустите регрессию:
   python scripts/replay_consult_archive.py --fixtures tests/fixtures/consult_replay.jsonl

5. В Cursor откройте чат и опишите расхождения:
   «По replay_consult_archive DIFF на pl_1_f.pdf - исправь парсер/матчер,
    добавь регресс-тест».

Это не дообучение LLM, а накопление эталонных метрик + правка детерминированного кода.
Чем больше снимков - тем стабильнее подбор протоколов и разбор КЗ.
