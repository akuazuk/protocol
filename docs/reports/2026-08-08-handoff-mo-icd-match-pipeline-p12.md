# Handoff: ICD match pipeline v3 phases 1-2

Дата: 2026-08-08

## Repo / Git

| | |
|--|--|
| PR | https://github.com/akuazuk/protocol/pull/53 (merged) |
| merge SHA | `310090c214b4927b76e6981e91a1ea32c1a38d04` |
| branch (done) | `cursor/mo-icd-match-pipeline-p12-pc1` |
| `BUILD_VERSION` | `2026-08-08-081015Z-icd-match-p12` |
| GCE | https://protocol.kravira.by `/api/version` = version выше |

## Сделано

Фаза 1:
- `clinical_knowledge/mo_icd_match_pipeline.py` - оркестратор Dx↔МКБ
- Чип case detail / `compute_icd_visit_status` из `pipeline.chip`
- Case detail: поле `icd_match` (summary)
- `kz_deep_eval` B3: `mkb_code_agreement` ∈ match|partial|mismatch|unknown (+ legacy 0/1)

Фаза 2:
- Compact ICD `K293` → `K29.3` только если ∈ RU JSON (`icd_mkb._canonicalize_icd_like_token`)
- `mo_icd_aliases.py` + `data/icd_reference/dx_aliases_ru.json` + seed `diagnosis_icd`
- Aliases boost query; не ставят chip ok сами
- План v3 active в `docs/plans/`; фазы 1-2 отмечены [x]

## Не сделано (следующие фазы плана)

- Фаза 3: калибровка → `MO_ICD_*_IN_PRIMARY`
- Фаза 4: LLM review серой зоны
- Фаза 5: warehouse soft-fill
- История пациента - отдельный план

## Тесты

- Локально: 42 ICD-related pytest green
- CI PR #53: lint-and-test / docker-images / manifest-mode pass

## Следующая безопасная команда

Калибровка на дне warehouse (фаза 3) или ручной smoke чипов в UI на визитах с ОРВИ / compact-кодом / `mkb_code_agreement=mismatch`.

## Не трогать параллельно

- `icd_mkb.py` canonicalize
- `mo_icd_match_pipeline.py` / `mo_icd_aliases.py`
- флаги `MO_ICD_*_IN_PRIMARY` до калибровки
