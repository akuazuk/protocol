# Handoff для нового чата / вкладки Cursor

Дата: 2026-08-21  
Репозиторий: `akuazuk/protocol`  
Канон: `origin/main` `22806bb` (#174)  
Грязный `/Users/pavel/CURSOR/Protocol/protocol` на `main` не использовать.

Primary: `https://protocol.kravira.by`  
Прод: **`2026-08-21-062249Z-kp-omnibus-norm`** (smoke `/health/live` ok).

---

## Сделано

- Влиты `#169` `#170` `#171` `#172` `#173` `#174`.
- КП только от диагноза/МКБ. Жалобы не ищут. Омнибус не подставляем.
- Overlap без тела PDF, целые слова / основа. Путь с `_` нормализуется.
- Миелома только при корне C90.
- Sample 300 после #174 (`kp_suggest_eval_sample_post174.json`):
  hit **40.3%**, омнибус **0**, adult+child **0**, жалобы **0**,
  ПЦД / ГСК / экстренка / нейрохирургия / миелома **0**.
- Деплой GCE с `22806bb`. `kp-eval-full` пишет
  `/var/data/medical_exams/reports/kp_suggest_eval_post174.json` (~8 ч).

## Делается

Полный CSV `kp-eval-full` (one-off, не `protocol-web`).

## Нужно

1. Дождаться `kp-eval-full` exit 0. Сверить те же чужие файлы = 0.
2. Не включать `MO_RCETH_LABEL_PRIMARY`.
3. Волна 5 плана: паспорта карт (`icd10_mentions`), когда полный CSV готов.

## Запрет

- Второй full Rceth parse, Gemini с Mac, push в `main`, PHI
- Рестарт `protocol-web`, пока `kp-eval-full` running, без нужды
