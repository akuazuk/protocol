# Пакет для конкурса Белинфонда 2026 (МЦ «Кравира»)

**Срок подачи:** до 01.08.2026 · [konkurs.belinfond.by/participants](https://konkurs.belinfond.by/participants)

## Основной формат: PDF

| Файл | Назначение |
|------|------------|
| [01_Zayavka_Kravira_Protocol.pdf](01_Zayavka_Kravira_Protocol.pdf) | Заявка |
| [02_Pasport_Kravira_Protocol.pdf](02_Pasport_Kravira_Protocol.pdf) | Паспорт инновационного проекта |
| [03_Biznes_plan_Kravira_Protocol.pdf](03_Biznes_plan_Kravira_Protocol.pdf) | **Бизнес-план** (разделы 1-16, социальная значимость, графики) |
| [04_Strategiya_Kravira_Protocol.pdf](04_Strategiya_Kravira_Protocol.pdf) | Стратегия коммерциализации |
| [06_ROI_Kravira.pdf](06_ROI_Kravira.pdf) | ROI якорного клиента |

**Пересборка PDF (рекомендуется):**

```bash
python3 scripts/build_konkurs_pdf.py
```

Требуется Google Chrome (headless). Исходники: `*-print.html` в этой же папке.

## Дополнительно

| Файл | Назначение |
|------|------------|
| `*.docx` | Legacy-формы Word (опционально: `--with-docx`) |
| [05_Biznes_plan_Prilozheniya.html](05_Biznes_plan_Prilozheniya.html) | Интерактивные приложения (браузер) |
| [docx-preprint-checklist.md](docx-preprint-checklist.md) | Чек-лист перед подачей |
| [financial_assumptions.md](financial_assumptions.md) | Единые допущения финмодели |

## Якорь рынка

| Показатель | Значение |
|------------|----------|
| КЗ/мес в Кравире | **25 000** (= **1%** частных ОЗ РБ) |
| TAM | **2,5 млн КЗ/мес** · **30 млн/год** |
| Рынок платных медуслуг РБ, 2025 | **~2,7 млрд BYN** (+24% г/г) |

## Социальная значимость Protocol

- **Пациенты** - B2C проверка КЗ по 478 КП Минздрава
- **Врачи** - L0 подсказка до ЭЦП, send_gate по правилам
- **Клиники** - 100% потока, ЦИСЗ + клиника в одном контуре
- **Государство** - качество данных в ЦИСЗ, исполнение КП

## Подача

1. Загрузка PDF на сайт конкурса.
2. Печать, подпись, печать Кравиры (при бумажной подаче).
3. Белинфонд: 220072, г. Минск, пр. Независимости, 68-2, каб. 112.

Контакт: rkip@belinfond.by, +375 17 270-84-29.
