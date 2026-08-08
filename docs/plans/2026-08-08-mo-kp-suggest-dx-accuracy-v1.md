# МО: точный подбор КП по диагнозу (suggest v4)

Статус: **active**  
Дата: 2026-08-08

## Контекст

Сверка случая МО с клиническим протоколом МЗ требует сначала **найти правильный КП**.
Suggest v3 часто возвращал specialty-filler (огнестрельные / позвоночник) для
формулировок вроде «плосковальгусная установка стоп»: title/path omnibus-КП №109
не содержат нозологию, а M21.* лежат за пределами top-10 кодов карточки.
`audience=unknown` дополнительно обнулял детские карточки.

Продуктовое правило сохраняется: **не ранжировать КП по коду МКБ из случая**
(`mis_diagnos`). Коды на карточке и text→ICD из справочника - только lexical bridge.

## Что меняем

1. `dx_query_expand.py` - алиасы Dx, токены, text→ICD bridge + refine/фильтры шума.
2. `protocol_match.py` - soft audience unknown; blob/titles с focus на bridge-коды;
   diag_part = max(lexical, bridge); кэш.
3. `case_protocol_suggest.py` - engine **v4**, возраст → audience, clinical-only top.
4. `protocol_links.py` - boilerplate titles → имя файла.

## Метрики

| Кейс | Было | Стало | Цель |
|------|------|-------|------|
| Плосковальгус / 7 лет / ортопед | specialty огнестрельные ~44 | clinical КП детс ортопедо №109 ~90 | top-1 clinical нужный КП |
| ОРВИ (контроль шума) | инородные тела / паллиатив | ENT/J06-семейство, без C/T/W | без онко/инородных |

## Шаги

- [x] Age-aware audience; unknown не убивает pediatric
- [x] Alias + weighted tokens + full/focus ICD titles на карточке
- [x] text→ICD bridge (не код случая) + refine от шума
- [x] Clinical-only top; engine v4
- [x] Тесты flatfoot / bridge ORVI
- [x] Преемник: история-эпизод + golden → `2026-08-08-mo-kp-history-episode-suggest-v1.md`
- [ ] Проводка improved suggest в №55 (39.2/39.5/39.8) - следующий шаг
- [ ] BUILD_VERSION + PR по запросу

## Риски

- Omnibus-КП с широким icd10_all могут давать несколько clinical-хитов одной семьи.
- Педиатр без rubric slug → поиск по всему каталогу (как раньше).
- Bridge зависит от качества `icd_mkb.suggest_icd_from_russian`.
