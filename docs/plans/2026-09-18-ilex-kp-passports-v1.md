# Ilex-паспорта КП для подбора в МО (v1)

Статус: **active**
Дата: 2026-09-18
Связанные: `2026-08-21-mo-kp-diagnosis-only-v1.md`,
`2026-08-08-mo-icd-first-kp-suggest-v1.md`.

Канон: **диагноз → нозология → КП. Пусто лучше чужого протокола.**
PDF МЗ остаётся источником текста. Ilex даёт паспорт: точное имя,
МКБ главы 1, аудитория, действует / утратил силу.

Лицензионный HTML Ilex в git не кладём. В репозитории только компактный
JSONL паспортов (названия, коды, даты, статус).

---

## Контекст

Карты `protocol_cards.jsonl` часто с обрубленным заголовком
(`КЛИНИЧЕСКИЙ ПРОТОКОЛ`) и с `icd10_all` из тела омнибуса. Suggest ищет
похожую карту по словам, а не протокол этой нозологии.

Выгрузка Ilex (вне репо, `Protocol_Private/ilex_clinical_protocols`):
242 акта, 329 именованных протоколов, HTML с `ogl-tag` / главами /
шифром МКБ в п.1.

---

## Что меняется в проде

Пока ветка не в `origin/main` и не на GCE - ничего.
После merge: overlay паспортов на карты при загрузке реестра
(`ILEX_PASSPORTS=0` выключает). `BUILD_VERSION` поднимается в том же
коммите. Деплой только координатором после merge.

---

## Метрики

Шаг 1 (паспорта, без новых PDF):

| Метрика | Было | Стало |
|--|--|--|
| Прямой Ilex-паспорт (АГ / бронхит / синусит / ФП) | нет слоя | **4/4**; ФП МКБ `I48` |
| Golden 40 | baseline | 0 регрессий |
| Suggest top-1 на 4 нозологиях | 3/4 | 3/4 (нет PDF 2026) |
| Overlay точного имени | 0 | 1273 |

Шаг 2 (PDF АГ N 38 и аритмии N 34 + recency на текстовом пути):

| Метрика | Было (шаг 1) | Стало |
|--|--|--|
| Карты в `protocol_cards.jsonl` | 5121 | **5193** (+7 PDF, 72 карты) |
| Suggest overlay on, 4 нозологии | 3/4 | **4/4**, чужих 0 |
| Suggest overlay off | 3/4 | 3/4 (АГ 2026 в топе, имя файла «АГ» без нозологии) |
| Golden 40 | 0 fail | 0 fail (АГ: путь 2026 N 38 тоже принимается) |
| Overlay точного имени | 1273 | **1373** |
| HTML Ilex в git | нет | нет |

---

## Шаги

0. [x] Сравнить Ilex HTML с картами: Ilex - паспорт, PDF - тело.
1. [x] Парсер выгрузки → `output/registry/ilex_protocol_passports.jsonl`.
2. [x] Overlay на карты: title / `condition_label` / МКБ главы 1 / статус.
3. [x] Eval golden + нозологический набор (было / стало).
4. [x] Golden зелёный; overlay включён. Suggest top-1 упёрся в отсутствующие PDF 2026.
5. [x] PDF МЗ: АГ 2026 N 38 и N 34 КП1-КП6 (ФП = КП4). Unique-PDF overlay для «АГ». Recency на текстовом пути МО.
6. [ ] Следующие волны: остальные PDF 2026, главы для plan-score, nightly sync Ilex.

---

## Риски

- Стык дата+номер постановления: в картах дата часто `YYYY-01-01`.
  Стыкуем по **году + номеру**, не по дню.
- Несколько КП в одном акте: не подменяем title одним именем, пишем все
  в `condition_label`.
- Неполный Ilex (242 vs ~478 PDF): overlay только при стыке, остальное
  как было.
- Лицензия Ilex: HTML, card.json, segments не коммитить.

---

## Инструкция: пересобрать паспорта

Выгрузка должна лежать локально (не в git):

```bash
# из task-worktree
python3 scripts/build_ilex_protocol_passports.py \
  --dump "$HOME/Protocol_Private/ilex_clinical_protocols" \
  --out output/registry/ilex_protocol_passports.jsonl
```

Проверка подбора:

```bash
PYTHONPATH=. python3 scripts/eval_ilex_kp_suggest.py
PYTHONPATH=. python3 -m pytest tests/test_ilex_protocol_passports.py \
  tests/test_mo_kp_suggest_golden.py tests/test_case_protocol_suggest.py \
  tests/test_plans_index.py --noconftest -q
```

Выключить overlay: `ILEX_PASSPORTS=0`.
Другой файл: `ILEX_PASSPORTS_PATH=/path/to.jsonl`.

---

## Следующая безопасная команда

Не деплоить. После merge PR - координатор `bash deploy/gcp-app/deploy_to_gce.sh`.
GCE не трогать с этой task-ветки.
