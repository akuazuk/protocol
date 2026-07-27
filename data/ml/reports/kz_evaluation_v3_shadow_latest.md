# Shadow benchmark: legacy deep vs scorer v3

- N кейсов: **10**
- Legacy score: mean 62.7 / median 63.1
- V3 score: mean 83.2 / median 89.4
- Coverage: mean 1.0
- Confidence: mean 0.6
- Cap применён: **1**
- Смен статуса: **8**
- Legacy высок, v3 низок: **0**
- C/D findings исключены из штрафа: **4**
- Протокол advisory (не penalty-eligible): **2**

## По кейсам

| id | legacy | v3 | статус v3 | cov | conf | cap | proto pen | C/D adv |
|---|---|---|---|---|---|---|---|---|
| syn_good_adult | 60.0 | 90.4 | good | 0.825 | 0.705 | False | True | 2 |
| syn_missing_diagnosis | 57.1 | 61.4 | review | 1.0 | 0.4 | False | None | 0 |
| syn_no_protocol | 71.4 | 89.4 | good | 1.0 | 0.575 | False | None | 0 |
| syn_child_protocol_adult_kz | 63.1 | 89.4 | good | 0.883 | 0.573 | False | False | 1 |
| syn_inpatient_protocol | 63.1 | 89.4 | good | 0.883 | 0.596 | False | False | 1 |
| syn_double_nsaid | 60.0 | 60.0 | review | 1.0 | 0.575 | True | None | 0 |
| syn_uncertainty_unrouted | 71.4 | 89.4 | good | 1.0 | 0.575 | False | None | 0 |
| syn_repeat_visit | 67.8 | 88.9 | good | 1.0 | 0.575 | False | None | 0 |
| syn_empty | 38.1 | None | insufficient_data | None | None | False | None | 0 |
| syn_reviewed_protocol | 75.0 | 90.4 | good | 1.0 | 0.912 | False | True | 0 |

> Shadow-режим: production score/gate не переключаются. C/D findings всегда
> исключены из штрафа (advisory), что устраняет архитектурный источник ложных
> штрафов по недоверенным правилам (ТЗ §2.2, §6).
