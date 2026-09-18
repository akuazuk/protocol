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

## Метрики (шаг 1)

| Метрика | Было | Стало (эта ветка) |
|--|--|--|
| Прямой Ilex-паспорт (АГ / бронхит / синусит / ФП) | нет слоя | **4/4** верные названия; ФП МКБ `I48` |
| Golden 40 кейсов | baseline | **0 регрессий** (off и on) |
| Suggest top-1 на тех же 4 нозологиях | 3/4 (нет ФП 2026 PDF) | 3/4, **без новых чужих** |
| Карты с overlay точного имени | 0 | 1273 (только стык год+номер **и** нозология в пути) |
| HTML Ilex в git | нет | нет |

Suggest не сдвинул АГ 2026 и ФП 2026: этих PDF нет в `protocol_cards.jsonl`.
Паспорт их уже знает. Следующая волна - добрать PDF с сайта МЗ, не ослаблять фильтр.

---

## Шаги

0. [x] Сравнить Ilex HTML с картами: Ilex - паспорт, PDF - тело.
1. [x] Парсер выгрузки → `output/registry/ilex_protocol_passports.jsonl`.
2. [x] Overlay на карты: title / `condition_label` / МКБ главы 1 / статус.
3. [x] Eval golden + нозологический набор (было / стало).
4. [x] Golden зелёный; overlay включён. Suggest top-1 упёрся в отсутствующие PDF 2026.
5. [ ] Следующие волны (не этот PR): добор PDF АГ N 38 / ФП N 34, главы для plan-score, nightly sync Ilex.

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

Не деплоить. После зелёного eval - commit/push task-ветки и PR.
GCE не трогать, пока PR не в `origin/main`.
