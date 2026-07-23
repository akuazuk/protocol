# Калибровка/валидация deep-оценки КЗ (2026-01, proxy-LLM)

Join: **n=411** размеченных gold-КЗ (LLM overall не пуст), LLM-bad=279 (67.9%).

- corr(deep_overall, llm_overall) = **0.605**, MAE = **15.5** п.п.

## Детекция плохих КЗ (deep review/poor/critical или overall<cutoff) vs LLM-bad

| конфиг | good | acc | flag_cut | min_axis | recall | prec | F1 | QWK | flag_rate |
|--|--|--|--|--|--|--|--|--|--|
| baseline | 80 | 60 | - | - | 0.16 | 0.88 | 0.27 | 0.258 | 0.122 |
| **калибр.** | 78 | 58 | 0 | 60 | 0.37 | 0.84 | 0.51 | 0.258 | 0.299 |

## Корреляция осей с LLM-overall (для будущих весов)

| ось | corr |
|--|--|
| documentation | 0.571 |
| clinical_concordance | 0.352 |
| safety | 0.1 |
| regulatory | 0.556 |

Конфиг записан: `config/deep_thresholds.yaml`.

