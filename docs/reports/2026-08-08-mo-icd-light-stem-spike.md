# Spike: лёгкий stem для ICD text fit (фаза 6)

Дата: 2026-08-08  
План: `docs/plans/2026-08-08-mo-icd-dx-matching-pipeline-v3.md` §4.4 / фаза 6  
Флаг: `MO_ICD_LIGHT_STEM` (default `0`)

## Зачем

Пары вроде «боль в животе» ↔ title «… живота» давали `token_coverage=0` при exact-token
пересечении (`животе` ≠ `живота`). Fuzz частично тянул score, но coverage-ветка
оставалась нулевой.

## Решение (не pymorphy)

В `clinical_text_similarity`:

- для токенов **≥6** символов отрезать **1-2** типичных русских окончания;
- stem остаётся ≥4;
- применяется в `tokens()` → jaccard/coverage/`combined_score` (name_only);
- `mo_icd_directory_eval.title_match_score` читает те же токены.

## Локальные цифры (фикстура)

| Пара | stem off | stem on |
|--|--|--|
| coverage «боль в животе» ↔ «боли в области живота» | 0.0 | ~0.33 |
| combined той же пары | ~0.19 | ~0.39 |

## Primary

Флаг **выкл** по умолчанию. Name_only уже в primary (`MO_ICD_NAME_IN_PRIMARY=1`) -
включать stem на GCE отдельно после smoke, не смешивать с embeddings.

Embeddings spike **не** делали - stem закрыл эталонный кейс склонений.

## Smoke на GCE (после deploy)

```bash
# в .env.gcp-staging или docker -e:
MO_ICD_LIGHT_STEM=1
# затем точечный score / recompute одного дня и сравнение chip name_fit
```
