# Handoff: КП suggest - префильтр и пустые кластеры

- branch: `cursor/kp-suggest-eval-speed-pc1`
- worktree: `/private/tmp/protocol-task-kp-suggest-eval-speed-pc1`
- base: `origin/main` `#152` (`0a15236`)
- план: `docs/plans/2026-08-14-mo-kp-suggest-accuracy-v2.md` шаги 5 и 7

## Сделано

- Префильтр по токенам и корням МКБ, плюс text→ICD bridge.
- Коды МКБ из содержания КП на карточку (K21/K30/E03 больше не теряются).
- Омнибус без нозологии не становится clinical (B07/2008 кожа, урология 2011).
- J06, M54, A63, B07 - честное «нет»: действующего нозологического КП нет.

## Не сделано

Шаг 6: CSV-прогон 26-31.07 и августа на GCE после deploy.
Шаг 8: golden 40.

## Тесты

`pytest --noconftest` clusters / suggest / I84 / flatfoot / superseded - ок.

## Следующая команда

После merge: `deploy_to_gce.sh`, затем eval на VM без печати ДР.
