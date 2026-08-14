# Handoff: КП suggest - возраст и срок действия

- repo: `akuazuk/protocol`
- branch: `cursor/kp-suggest-age-valid-pc1`
- worktree: `/private/tmp/protocol-task-kp-suggest-age-valid-pc1`
- base: `origin/main` после `#151` (`6016231`)
- план: `docs/plans/2026-08-14-mo-kp-suggest-accuracy-v2.md` шаги 1-4

## Сделано

1. Возраст: готовые годы, иначе ДР + дата визита (не сегодня).
2. Даты силы на карточке из уже известных полей; пустое не выдумываем.
3. Hard-filter только если `valid_to` / `superseded_on` раньше визита.
   `sync_missing` без даты отмены не выкидываем.
4. Демоут омнибуса без нозологии в содержании.

Не печатать ДР и возраст в логах.

## Не сделано

Шаги 5-8 плана: пустые кластеры, CSV-прогон, скорость, golden 40.

## Тесты

`pytest --noconftest` по age/validity/suggest/matcher - ок.
Полный conftest с RAG в этой среде упирается в таймаут корпуса (baseline).

## Следующая команда

После merge: `GCE_OPS_USER=pavel SYNC_PROTOCOL_CORPUS=0 COPYFILE_DISABLE=1 bash deploy/gcp-app/deploy_to_gce.sh`
затем smoke `/health/live` и `/api/version` на `protocol.kravira.by`.
