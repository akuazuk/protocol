# Пакет для конкурса Белинфонда 2026 (МЦ «Кравира»)

**Срок подачи:** до 01.08.2026 · [konkurs.belinfund.by/participants](https://konkurs.belinfund.by/participants)

## Готовые формы (.docx)

| Файл | Назначение |
|------|------------|
| [01_Zayavka_Kravira_Protocol.docx](01_Zayavka_Kravira_Protocol.docx) | Заявка |
| [02_Pasport_Kravira_Protocol.docx](02_Pasport_Kravira_Protocol.docx) | Паспорт инновационного проекта |
| [03_Biznes_plan_Kravira_Protocol.docx](03_Biznes_plan_Kravira_Protocol.docx) | Бизнес-план (разделы 1-16 + таблицы и графики) |
| [04_Strategiya_Kravira_Protocol.docx](04_Strategiya_Kravira_Protocol.docx) | Стратегия коммерциализации |
| [05_Biznes_plan_Prilozheniya.html](05_Biznes_plan_Prilozheniya.html) | Приложения А-Е: таблицы и графики (печать в PDF) |
| [06_ROI_Kravira.html](06_ROI_Kravira.html) | Приложение Ж: ROI якорного клиента (1 стр., PDF) |

**Перед печатью:** [docx-preprint-checklist.md](docx-preprint-checklist.md) · **Расчёты:** [financial_assumptions.md](financial_assumptions.md) · **Черновики текстов:** [kravira-belinfund-2026-drafts.md](kravira-belinfund-2026-drafts.md)

Пересборка из шаблонов сайта:

```bash
python3 scripts/fill_konkurs_docx.py
```

## Якорь рынка (единое допущение)

| Показатель | Значение |
|------------|----------|
| КЗ/мес в Кравире | **25 000** |
| Доля на рынке платных КЗ частных ОЗ РБ | **1%** |
| Рынок РБ (экстраполяция) | **2 500 000 КЗ/мес** · **30 млн/год** |
| Тариф якоря L0 «Сеть» | **0,69 BYN/КЗ** → **207 000 BYN/год** |

## Три канала монетизации

| Канал | Кому | Тариф (черновик) |
|-------|------|------------------|
| **B2B** | клиники, МИС | от **0,99 BYN**/КЗ (L0), пакеты 0,79 / 0,69 |
| **B2B API** | вендоры МИС | внедрение + поддержка |
| **B2C** | **физические лица** (пациенты) | **4,99** / **9,99 BYN** за проверку своего КЗ |

## Подача

1. Регистрация и загрузка на сайте конкурса (.doc/.pdf).
2. Печать, подпись, печать Кравиры.
3. Белинфонд: 220072, г. Минск, пр. Независимости, 68-2, каб. 112.

Контакт: rkip@belinfund.by, +375 17 270-84-29.
