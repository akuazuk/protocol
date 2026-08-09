# МО: независимый AI-proxy до human gold (C6A/C7A)

Дата: 2026-08-09  
Статус: **completed** (exploratory C6A/C7A; formal C6-C9 не закрыты)

Связанный основной план:
`2026-08-09-mo-score-ssot-llm-recompute-v3.md`.

## Контекст

Основной план остановлен на C6: методист ещё не заполнил 22 endpoint labels для
18 disagreement cases. Владелец попросил выполнить автоматическую предварительную
проверку. Автоматическая оценка не является human gold и не может закрыть C6-C9,
поэтому этот контур регистрируется отдельно как exploratory C6A/C7A.

## Что изменено в production

- Secure C6 UI развёрнут на GCE primary.
- Production scoring, action queue, warehouse и отображаемые оценки не меняются.
- Proxy artifacts остаются в secret calibration directory на GCE.
- В git допускаются только агрегаты без clinical text и идентификаторов.

## Дизайн

1. **Proxy arm E**: один blind pass `gemini-3.1-pro-preview` по тем же 30 frozen
   cases и тем же Endpoint C/D contracts.
2. Proxy не получает deterministic scores, два исходных blind passes,
   adjudication или methodist labels.
3. Human template `methodist_labels.jsonl` не изменяется.
4. C7A сравнивает deterministic score families, исходные blind scores и простые
   pre-registered ensembles против proxy отдельно для Dx и plan.
5. Отчёт содержит MAE, Spearman, ROC-AUC, PR-AUC, threshold-55 confusion metrics
   и bootstrap 95% CI.
6. Любой вывод маркируется `proxy_not_human_gold`; production decision запрещён.

## Метрики

- Было: proxy labels `0/30`, exploratory comparison отсутствует.
- Стало C6A: 30 rows; Dx payload `30/30`, plan payload `29/30`;
  1 plan contract error, leakage/geo errors `0`.
- Scored non-abstain proxy: Dx `25` (`2 bad`), plan `21` (`8 bad`);
  abstain соответственно `5` и `8`.
- Стало C7A: Dx `10` primary + `12` control candidates, plan `7` primary
  + `13` control candidates; 2,000 bootstrap iterations; отчёт не содержит
  IDs/clinical text.
- Цель `30/30` valid по обоим endpoints не достигнута: один grounded plan
  стабильно нарушил nullability contract. Для exploratory анализа валидный Dx
  этого row сохранён, невалидный plan исключён.
- Formal C6: остаётся `0/22`, цель `22/22` human labels.
- Formal C7-C9: остаются незавершёнными до human gate.

## Шаги

- [x] A0 Зафиксировать отдельный proxy-контур без подмены human gold.
- [x] A1 Добавить PHI-safe evaluator и synthetic tests.
- [x] A2 Выполнить GCE proxy 30 × 1.
- [x] A3 Выполнить C7A aggregate comparison + bootstrap CI.
- [x] A4 Зафиксировать exploratory report и расхождения.
- [ ] A5 Передать 18 disagreement cases реальному методисту (остаётся formal C6
  основного плана).

## Риски и ограничения

- Proxy и исходный judge относятся к одному vendor family; model diversity не
  заменяет независимого клинического эксперта.
- Автоматически проверить корректность proxy без human gold невозможно.
- Small-N CI широкие; ranking нестабилен при малом числе proxy-bad cases.
- Proxy нельзя записывать через methodist UI или использовать как reviewer gold.
- C7A/C8A не разрешают rollout, изменение порогов или выбор primary queue signal.
