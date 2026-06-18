# Папка для ваших КЗ (ПДн — не коммитить PDF)

Положите сюда PDF/TXT/DOCX консультативных заключений. Имена файлов = `case_id` в отчётах batch.

## Пилот (10 КЗ)

```bash
# Импорт из Downloads/КЗ и полный прогон L1 + ИИ-предразметка:
python3 scripts/run_kz_pilot_batch.py --import-from ~/Downloads/КЗ

# Только прогон (файлы уже в этой папке):
python3 scripts/run_kz_pilot_batch.py

# Сравнение уровней L0/L1/L2 (дольше):
python3 scripts/run_kz_pilot_batch.py --compare-tiers

# Отправить одобрения на Render (нужен METHODIST_TOKEN в .env):
python3 scripts/run_kz_pilot_batch.py --submit-render

# Экспорт для правок движка:
python3 scripts/run_kz_pilot_batch.py --export-feedback
```

Отчёт: `ml/experiments/kz_pilot_*/REVIEW_QUEUE.md` · разметка в UI: **Очередь** → **Открыть**.
