#!/usr/bin/env python3
"""
Локальный RAG: отбор фрагментов из корпусных JSONL (corpus_chunks_parts/*.jsonl) или из chunks.json и ответ по ним.

Запуск: pip install -r requirements-rag.txt, скопировать .env.example в .env и задать ключ API.
Переменные — из .env / .env.local (python-dotenv). См. комментарии в .env.example.

Фронт (index.html) вызывает POST /api/assist; ключ к API не передаётся в браузер.
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent

from env_load import load_project_env

from icd_mkb import (
    analyze_query_for_icd,
    count_icd_code_mentions,
    describe_code,
    extract_icd_codes_raw,
    icd_tokens_for_lex,
    normalize_icd_code,
    ru_lexicon_scored_entries,
)

from retrieval_bm25 import build_bm25_index

load_project_env(ROOT)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(f"Установите: pip install -r requirements-rag.txt ({e})") from e

CHUNKS_PATH = ROOT / "chunks.json"
CORPUS_CHUNKS_PARTS_GLOB = "corpus_chunks_parts/chunks.part.*.jsonl"
PROTOCOLS_PATH = ROOT / "protocols.json"


def _chunks_data_root() -> Path:
    """Каталог с JSONL-чанками: по умолчанию рядом с rag_server.py; на Render — смонтированный диск.

    Задаётся RAG_CHUNKS_DIR (например /var/data при Persistent Disk). Глобы RAG_CHUNKS_JSONL_GLOB
    и corpus_chunks_parts/*.jsonl считаются относительно этого каталога.
    """
    raw = (os.environ.get("RAG_CHUNKS_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ROOT

_chunks: list[dict] = []
_chunks_by_path: dict[str, list[dict]] = {}
_protocols_by_path: dict[str, dict] = {}
_protocol_meta: dict[str, dict] = {}
_structured_by_path: dict[str, dict] = {}
_routing: dict = {}
_model = None
_retrieval_embed_meta: dict | None = None
_bm25_index = None
_chunks_load_done = threading.Event()
_chunks_load_error: str | None = None

PROTOCOL_META_PATH = ROOT / "protocol_meta.json"
STRUCTURED_INDEX_PATH = ROOT / "structured_index.json"

ALLOWED_SPECIALTY_SLUGS = frozenset(
    [
        "akusherstvo-ginekologiya",
        "allergologiya-immunologiya",
        "anesteziologiya-reanimatologiya",
        "bolezni-sistemy-krovoobrashcheniya",
        "dermatovenerologiya",
        "endokrinologiya-narusheniya-obmena-veshchestv",
        "gastroenterologiya",
        "gematologiya",
        "infektsionnye-zabolevaniya",
        "khirurgiya",
        "nefrologiya",
        "nevrologiya-neyrokhirurgiya",
        "novoobrazovaniya",
        "oftalmologiya",
        "otorinolaringologiya",
        "palliativnaya-pomoshch",
        "psikhiatriya-narkologiya",
        "pulmonologiya-ftiziatriya",
        "revmatologiya",
        "stomatologiya",
        "transplantatsiya-organov-i-tkaney",
        "travmatologiya-ortopediya",
        "urologiya",
        "zabolevaniya-perinatalnogo-perioda",
    ]
)

# Рубрики каталога Минздрава РБ (slug → подпись для UI и /api/specialties).
SPECIALTY_LABELS_RU: dict[str, str] = {
    "akusherstvo-ginekologiya": "Акушерство и гинекология",
    "allergologiya-immunologiya": "Аллергология и иммунология",
    "anesteziologiya-reanimatologiya": "Анестезиология и реаниматология",
    "bolezni-sistemy-krovoobrashcheniya": "Болезни системы кровообращения",
    "dermatovenerologiya": "Дерматовенерология",
    "endokrinologiya-narusheniya-obmena-veshchestv": "Эндокринология и обмен веществ",
    "gastroenterologiya": "Гастроэнтерология",
    "gematologiya": "Гематология",
    "infektsionnye-zabolevaniya": "Инфекционные заболевания",
    "khirurgiya": "Хирургия",
    "nefrologiya": "Нефрология",
    "nevrologiya-neyrokhirurgiya": "Неврология и нейрохирургия",
    "novoobrazovaniya": "Новообразования",
    "oftalmologiya": "Офтальмология",
    "otorinolaringologiya": "Оториноларингология",
    "palliativnaya-pomoshch": "Паллиативная помощь",
    "psikhiatriya-narkologiya": "Психиатрия и наркология",
    "pulmonologiya-ftiziatriya": "Пульмонология и фтизиатрия",
    "revmatologiya": "Ревматология",
    "stomatologiya": "Стоматология",
    "transplantatsiya-organov-i-tkaney": "Трансплантация органов и тканей",
    "travmatologiya-ortopediya": "Травматология и ортопедия",
    "urologiya": "Урология",
    "zabolevaniya-perinatalnogo-perioda": "Перинатальный период",
}

SYSTEM_JSON = """Ты помощник врача по клиническим протоколам Минздрава Республики Беларусь.
Фрагменты PDF ниже могут быть неполными. Не выдумывай факты вне фрагментов.
Если в запросе есть блок «=== Контекст пациента ===» (возраст, пол и т.д.) и «=== Жалобы и вопрос ===», учитывай контекст при выборе детских vs взрослых протоколов и в формулировке summary.
Если возраст явно взрослый (например «49 лет», ≥18 лет) — не включай в protocols детские КП: в списке должны остаться только path из входных фрагментов; если фрагменты только детские (маловероятно), опирайся на них осторожно и не выдавай детский протокол как основной без пометки.
Клиническая калибровка (обязательно):
- Опирайся на симптомы и формулировки из «Запрос пользователя». Не приписывай пациенту симптомов, которых там нет (в частности: насморк, боль в горле, ангина, ОРВИ, если их не указали в запросе). Не переноси симптомы из фрагментов протоколов в описание жалобы, если их не было в запросе.
- Если фрагменты явно про другую нозологию или орган (например вестибулярная патология или деформации позвоночника при жалобе на гайморит/синусы), не используй их как основу ответа и не повышай им confidence; укажи в match_reason несоответствие теме запроса или опусти такие протоколы из верхних позиций.
- ОРВИ, фарингит, тонзиллит, риносинусит и др. типичные ЛОР-причины — только если они явно следуют из запроса пользователя и/или из приведённых фрагментов; не подставляй их «по умолчанию».
- Редкие неотложные состояния (острый эпиглоттит, ретрофарингеальный абсцесс и т.п.) — только при явных красных флагах в тексте запроса (выраженная одышка, слюнотечение, невозможность глотать слюну, быстрое ухудшение) или если это прямо следует из фрагментов. Если пользователь указал нормальное дыхание без одышки — не ставь эпиглоттит первым в дифференциальный ряд и не формулируй ответ так, будто он наиболее вероятен.
- Не противоречь явным фактам из запроса (например «дыхание нормальное»).
- Summary — только краткое сопоставление запроса с отобранными протоколами. Не перечисляй в нём конкретные лекарства, дозы, схемы и перечни анализов/инструментальных исследований (их место — в развёрнутой выдержке по выбранному протоколу); не выдумывай детали вне фрагментов.
- Дифференциальный ряд (differential) в первую очередь из гипотез, согласующихся с тематикой входных фрагментов; расширяй до 3–5 пунктов только если запрос или фрагменты действительно допускают широкий дифференциал.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема полей:
{
  "summary": "…",
  "protocols": [{"path":"…","title":"…","match_reason":"…","confidence":"низкая|средняя|высокая","confidence_score":0.0}],
  "differential": ["…","…"],
  "questions_for_patient": [] или ["…","…"],
  "disclaimer": "Информация из протоколов; не замена очной консультации."
}
Не добавляй в JSON поле icd_codes — список кодов МКБ-10 формирует сервер.
ЖЁСТКИЕ ЛИМИТЫ (иначе ответ обрежется посередине):
- summary: РОВНО 2 предложения на русском, каждое заканчивается точкой. Вместе НЕ ДЛИННЕЕ 280 символов (с пробелами). Без тире в конце; последний символ — точка. Не формулируй как установленный диагноз; это краткое сопоставление запроса с протоколами.
- match_reason: не длиннее 70 символов, одно короткое предложение или фраза, законченная по смыслу.
- differential: только дифференциальные ГИПОТЕЗЫ для обсуждения с врачом; не окончательный диагноз. Не используй формулировки «диагноз:», «установлен», «подтверждён». В приоритете точность. По умолчанию 2 короткие строки (каждая 3–8 слов), порядок по убыванию вероятности. Добавь 3-й–5-й пункт только при явно широком дифференциале; не больше 5 строк.
- questions_for_patient: если хотя бы у одного протокола confidence_score равен 1.0 (полное соответствие запросу) — пустой массив []. Иначе ровно 2 коротких вопроса.
- protocols: все уникальные path из входных фрагментов; confidence_score 0.0–1.0. Каждый path и каждый протокол по названию (title) указывай только один раз — не дублируй одинаковые строки.
Если не хватает места — сожми формулировки, но НЕ обрывай слова и НЕ оставляй незаконченное предложение в summary."""

SYSTEM_JSON_RETRY = """Повтори задачу: нужен ОДИН компактный JSON (без markdown).
Не добавляй симптомы носа/горла/ОРВИ, если их не было в запросе пользователя. Эпиглоттит и др. редкие неотложные — только при красных флагах или прямо в фрагментах; при нормальном дыхании не веди с эпиглоттита.
Не дублируй один и тот же протокол в protocols (один path / один title).
Предыдущая попытка оборвалась по длине. Сделай ещё короче:
- summary: РОВНО 2 коротких предложения, ВМЕСТЕ максимум 220 символов, последний символ — точка.
- match_reason: до 55 символов на протокол.
- differential: только гипотезы, не окончательный диагноз; без формулировок «диагноз установлен». 2 коротких пункта (или до 5 только если дифференциал широкий); по убыванию вероятности; questions_for_patient: [] если есть протокол с confidence_score 1.0, иначе 2 коротких вопроса.
Сохрани все path из фрагментов. Не обрывай слова."""

ASSIST_USER_CONTEXT_GUIDE = """Как читать фрагменты выше:
- Они перечислены в порядке отбора; поля score и lexical_score отражают силу совпадения с поисковым запросом (ориентир, не клинический скоринг).
- При противоречии между фрагментами разных протоколов приоритет — согласованность с формулировкой «Запрос пользователя» и с рубрикой фрагмента; не смешивай тактику из явно нерелевантного фрагмента в summary и match_reason.
- Детальные назначения, обследования и режимы лечения не раскрывай в summary JSON — они доступны пользователю при раскрытии протокола (вторая ступень)."""

SYSTEM_EXTRACT = """Ты помощник врача. По фрагментам клинического протокола Минздрава Республики Беларусь извлеки факты, относящиеся к запросу пользователя.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания протокола по тексту (1–5 предложений)",
  "treatment_methods": ["метод или этап лечения — по тексту протокола"],
  "medications": ["группы препаратов или МНН, если названы во входном тексте — без выдуманных доз"],
  "note": "кратко: чего нет в фрагментах или что требует очной консультации"
}
Не придумывай препараты, дозы и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_FULL = """Ты помощник врача. По ПОЛНОМУ тексту фрагментов клинического протокола Минздрава Республики Беларусь извлеки структурированные сведения, релевантные запросу пользователя.
Запрос хорошо соответствует протоколу (оценка модели обычно ≥80%); это не обязательно «идеальные 100%» — дай развёрнутый практичный разбор строго по тексту протокола.
В списках investigations, medications, treatment_methods и recommendations сначала помещай пункты, прямо относящиеся к формулировке запроса пользователя (симптомы, этап, цель обращения), затем — остальные релевантные пункты протокола по возрастанию общности.

Различай четыре блока (если в тексте нет данных — пустой массив [] или пустая строка ""):
- investigations: только диагностика и обследование (анализы, инструментальные методы, осмотры, критерии до постановки диагноза). Пример: «Общий анализ крови», «УЗИ органов брюшной полости», «рентгенография» — если это в тексте.
- medications: только лекарственные группы, МНН, режимы из текста (не дублируй сюда немедикаментозное лечение).
- treatment_methods: немедикаментозное и медикаментозное лечение как этапы/тактика (операции, режим, физиотерапия, схемы терапии словами протокола). Не копируй сюда дословно длинные таблицы доз — кратко по строкам.
- monitoring_frequency: только кратность и сроки наблюдения (через сколько недель визит, диспансеризация раз в год и т.п.) — одна строка или короткие фразы через «;».

monitoring_followup — отдельно: когда срочно обращаться, реабилитация, прочие формулировки наблюдения без дублирования кратности из monitoring_frequency; если вся наблюдательная информация только про сроки визитов — оставь monitoring_followup пустой строкой "".

Если протокол содержит алгоритмы (ветвления «если/то», пошаговые действия, эскалация помощи), заполни care_algorithms как список структур:
- title: название алгоритма/сценария;
- entry_conditions: короткие условия старта;
- steps: пошаговые действия в порядке выполнения (короткие пункты).
Если алгоритмов нет — верни [].

Если в тексте есть нумерация разделов/подпунктов («п. 2.1», «раздел 3») — по возможности укажи короткую отсылку в пункте списка (только если она есть в OCR).

Верни ОДИН JSON-объект (без markdown, без текста до/после).
Схема:
{
  "diagnosis": "диагнозы, состояния, показания (2–8 предложений)",
  "investigations": ["пункты обследования — по тексту протокола"],
  "medications": ["группы препаратов, МНН, режимы — только если есть во входном тексте"],
  "treatment_methods": ["этапы и методы лечения — по тексту протокола"],
  "monitoring_frequency": "кратность наблюдения одной строкой или пустая строка",
  "recommendations": ["рекомендации и алгоритм действий для врача/пациента — по тексту"],
  "monitoring_followup": "прочие формулировки наблюдения, когда обращаться — если уместно; иначе пустая строка",
  "care_algorithms": [{"title":"название алгоритма","entry_conditions":["условие старта"],"steps":["шаг 1","шаг 2"]}],
  "contraindications": "противопоказания и ограничения — если названы во фрагментах, иначе пустая строка",
  "note": "чего нет в фрагментах; необходимость очной консультации"
}
Не придумывай дозировки, препараты и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_GAP_SCAN = """Ты помощник врача. Ниже — полный текст фрагментов клинического протокола Минздрава Республики Беларусь (и при необходимости — выдержка из индекса).
В первом проходе не были извлечены или остались пустыми разделы: {fields_ru}.
Задача: ещё раз внимательно прочитай ВЕСЬ текст ниже и найди в нём сведения, относящиеся к этим разделам.
Сопоставь с формулировкой запроса пользователя: для investigations и medications в первую очередь извлеки то, что напрямую относится к жалобе/ситуации из запроса, затем прочие пункты из протокола.
Особое внимание: таблицы (каждая осмысленная строка таблицы с обследованием, препаратом или режимом — отдельный короткий пункт списка, если ячейки читаемы), перечни с маркерами, подпункты в скобках.
Каждый пункт — не длиннее ~2 строк текста; при необходимости разбей на несколько пунктов.
Верни ОДИН JSON-объект (без markdown, без текста до/после).
Включай в ответ ТОЛЬКО те ключи из этого списка, которые были пусты: {keys_json}.
- investigations, medications, treatment_methods — массивы строк (короткие пункты);
- monitoring_frequency — одна строка или пустая строка "".
Если в тексте протокола для раздела действительно нет данных — верни [] или "".
Не придумывай препараты, дозы и процедуры, которых нет во входном тексте."""

SYSTEM_EXTRACT_NON_PROTOCOL = """Ты помощник врача. По сути запроса и названию протокола ниже для разделов, которые в тексте протокола не найдены или пусты, дай краткие обобщённые общеклинические ориентиры для врача (не цитата из КП).
Правила:
- Не выдавай это за текст протокола; формулируй осторожно и обобщённо.
- Каждая строка в списках и строка кратности наблюдения ДОЛЖНЫ начинаться с точной пометки «[не из протокола]» (с квадратными скобками).
- Не указывай конкретные дозировки и схемы; не придумывай названия препаратов, которых нет в общих клинических стандартах.
- Если раздел заполнить нельзя безопасно — верни [] или "".
Специальность (рубрика): {spec}
Название протокола: {title}
Запрос пользователя:
{query}
Заполнить только пустые поля: {fields_ru}
Верни ОДИН JSON (без markdown) с ключами ТОЛЬКО из: {keys_json}
Типы: investigations, medications, treatment_methods — массивы строк; monitoring_frequency — одна строка."""

SYSTEM_CLASSIFY = """По краткому медицинскому запросу пациента выбери до трёх рубрик клинических протоколов (slug), которым соответствует ситуация.
Верни ОДИН JSON: {"categories": ["slug1"], "note": "одно короткое предложение"}
slug ТОЛЬКО из этого списка (копируй точно):
""" + ", ".join(sorted(ALLOWED_SPECIALTY_SLUGS)) + """
Если нельзя уверенно сопоставить — верни "categories": []."""

SYSTEM_QUERY_SPELLFIX = """По фрагменту текста жалобы (русский) исправь только орфографию и очевидные опечатки, в том числе в медицинских терминах (например лишние/пропущенные буквы).
Не меняй смысл, не добавляй симптомы и диагнозы, не сокращай и не перефразируй свободно.
Верни ОДИН JSON-объект (без markdown):
{"corrected": "<тот же текст целиком, с исправлениями или без изменений>"}"""

SYSTEM_CLINICAL_QUERY_REFINE = """Ты помощник врача. Ниже — текст жалобы/клинического запроса (русский) для автоматического поиска по клиническим протоколам и справочнику МКБ-10.
Задача: привести формулировки к общепринятой клинической терминологии (как в русскоязычных названиях МКБ-10 и протоколах), сохранив смысл и объём жалобы.
Правила:
- Не добавляй симптомы, жалобы, диагнозы и обстоятельства, которых нет во входном тексте.
- Не приписывай пациенту пол, возраст и сопутствующие болезни, если их нет во входе (если в дополнительном контексте ниже указаны возраст/пол — можно использовать только для согласования формулировок «ребёнок/взрослый», без выдумок).
- Разговорные названия замени на клинические эквиваленты там, где это однозначно (например «гайморит» → можно уточнить «острый/хронический верхнечелюстной синусит» только если степень остроты явно следует из текста; иначе оставь «гайморит» или нейтрально «синусит верхнечелюстной пазухи»).
- Не ставь окончательный клинический диагноз; это подготовка текста к поиску, не заключение.
- Исправь опечатки в медицинских терминах.
- Сохрани структуру: если несколько предложений — не сливай в одно без необходимости; итог не длиннее исходного более чем на ~30% (не раздувай).
Верни ОДИН JSON-объект (без markdown):
{"refined": "<итоговый текст>", "applied": true или false, "note": "<одно короткое предложение или пустая строка: что изменилось; если изменений нет — пусто>"}
Поле applied: false, если текст уже корректен и ты вернул его без существенных правок (допустимы микроисправления — тогда applied: true и кратко в note)."""

SYSTEM_ICD_POOL_SELECT = """Ты помощник врача. По клиническому запросу (жалобы, симптомы) выбери до 5 кодов МКБ-10 ТОЛЬКО из списка allowed ниже.
Запрещено: коды вне списка, выдуманные обозначения, текст вне JSON.
Дублируй поле code ТОЧНО как в списке (латиница и цифры).
Верни ОДИН JSON-объект (без markdown):
{"codes":[{"code":"J20.9","rationale":"одно короткое предложение"}]}
Если ни один код из списка не подходит — {"codes":[]}."""

SYSTEM_CONSULTATION_TEMPLATE = """Ты помощник врача. По развёрнутой выдержке из клинического протокола Минздрава Республики Беларусь (структура JSON ниже) и по сути запроса пользователя составь текстовый ШАБЛОН консультативного заключения.
Правила:
- Опирайся только на поля выдержки и на запрос; не выдумывай диагнозы, препараты, дозы и процедуры, которых нет во входных данных.
- Если передан блок selected_facts_payload (структурированные выбранные пункты), это ПРИОРИТЕТНЫЙ источник: отрази каждый выбранный пункт в профильном разделе заключения.
- Не пропускай выбранные пункты. Если пункт нельзя включить дословно, включи клинически эквивалентную формулировку без потери смысла.
- Если в «Запрос пользователя» или в блоке «Контекст пациента» указаны возраст и пол — подставь их в разделы «Жалобы» и «Анамнез» в связном тексте (например: «Пациент 49 лет, мужского пола, предъявляет жалобы…»). Не используй плейсхолдер [ФИО, возраст] или аналог, если возраст и пол уже известны из контекста. Фамилию, имя, отчество не выдумывай: если ФИО в данных нет — формулируй без ФИО («Пациент», «Пациентка» + возраст + пол).
- Плейсхолдеры в квадратных скобках должны быть ПОНЯТНЫМИ: не повторяй без разбора общую фразу [уточнить при осмотре]. Вместо этого указывай, ЧТО именно внести, например: [перечислить сопутствующие заболевания, аллергоанамнез, перенесённые операции], [описать объективный статус: общее состояние, местный статус прямой кишки и промежности], [указать сроки и режим наблюдения после лечения], [указать ограничения по выбору метода лечения при отсутствии данных в выдержке]. Если контекст узкий — короткая подсказка в скобках допустима.
- Общую форму [уточнить при осмотре] используй только если нельзя сформулировать конкретнее.
- Акценты разделяй: строки с «ВАЖНО:» — для критичных предупреждений; строки с «Внимание:» — для напоминаний и уточнений (не столь срочных). Отдельный абзац или пункт списка, без markdown.
- Стиль: официально-деловой, медицинский, пригодный для МИС или печати.
- Структура: разделы с заголовками в одну строку с двоеточием в конце, например: Жалобы: / Анамнез: / Объективно: / Диагноз по протоколу Минздрава РБ: / Рекомендации по протоколу: / Наблюдение и контроль: / Дополнительно:
- ОБЯЗАТЕЛЬНО выведи текст ПОЛНОСТЬЮ: не обрывай на середине слова, фразы или раздела; заверши каждый раздел; не используй «…» вместо целых абзацев протокола. Если объём большой — всё равно доведи структуру до конца.
- ЗАПРЕЩЕНО оформление в markdown: не используй звёздочки *, **, подчёркивания для выделения, решётки #, обратные кавычки. Пиши обычным текстом.
- Списки: строки с дефисом и пробелом в начале (- пункт) или нумерация 1. 2.
- Не дублируй заголовок «Консультативное заключение» в тексте — с него начинать не нужно (он будет на экране отдельно).
- В конце кратко: шаблон не заменяет очный осмотр и оформление документации лечащим врачом.
Верни ТОЛЬКО текст шаблона, без вступления «вот шаблон»."""

SYSTEM_CONSULTATION_REFINE = """Ты помощник врача. Ниже — черновик консультативного заключения (часть полей пользователь уже заполнил вместо плейсхолдеров в квадратных скобках). Также даны дополнительные сведения от пользователя (если есть).
Задача: выдай ПОЛНЫЙ итоговый текст заключения, согласованный с развёрнутой выдержкой из протокола Минздрава РБ (JSON ниже) и запросом. Дополни недостающие разделы по протоколу; где данных по-прежнему нет — оставь плейсхолдер с КОНКРЕТНОЙ подсказкой в скобках (что внести), избегая безликого [уточнить при осмотре], если можно уточнить формулировку.
- Если в запросе или в «Контекст пациента» есть возраст и пол — сохрани их в «Жалобы» и «Анамнез»; не возвращай плейсхолдер [ФИО, возраст], если эти данные уже заданы. ФИО не выдумывай.
- Критичное — с префиксом «ВАЖНО:»; напоминания — с «Внимание:».
- Не сокращай и не обрывай текст посередине; заверши все разделы.
- Не выдумывай факты, которых нет в выдержке, запросе или в дополнениях пользователя.
- ЗАПРЕЩЕНО markdown (*, **, #, обратные кавычки).
- Не дублируй заголовок «Консультативное заключение» в теле текста.
Верни ТОЛЬКО полный текст заключения."""

SYSTEM_CONFIDENCE_REFINE = """Ты помощник врача. По запросу и кратким сведениям о протоколе оцени, насколько протокол соответствует сути жалобы (0.0–1.0).
Верни ОДИН JSON без markdown: {"scores":[{"path":"…","confidence_score":0.0}]}
Копируй path точно из списка ниже; не добавляй протоколы вне списка."""


def _jsonl_chunk_files() -> list[Path]:
    """Порядок: один файл из RAG_CHUNKS_JSONL, либо glob из RAG_CHUNKS_JSONL_GLOB, либо части corpus_chunks_parts."""
    base = _chunks_data_root()
    one = (os.environ.get("RAG_CHUNKS_JSONL") or "").strip()
    if one:
        p = Path(one).expanduser()
        if not p.is_file():
            raise SystemExit(f"RAG_CHUNKS_JSONL: файл не найден: {p}")
        return [p.resolve()]
    gl = (os.environ.get("RAG_CHUNKS_JSONL_GLOB") or "").strip()
    if gl:
        paths = sorted(base.glob(gl))
        if not paths:
            raise SystemExit(f"RAG_CHUNKS_JSONL_GLOB: нет файлов по шаблону {gl!r} в {base}")
        return paths
    return sorted(base.glob(CORPUS_CHUNKS_PARTS_GLOB))


def _memory_saver_enabled() -> bool:
    """По умолчанию — полный lex (embedding_ready_text при отличии от text).

    На слабом инстансе (например 512Mi) задайте RAG_MEMORY_SAVER=1 — без дубля
    embedding_ready_text, чтобы избежать OOM при старте.
    """
    v = (os.environ.get("RAG_MEMORY_SAVER") or "").strip().lower()
    return v in ("1", "true", "yes")


def _load_chunks_from_jsonl(part_paths: list[Path]) -> list[dict]:
    """Корпусный pipeline: строки JSONL → формат retrieve() / gather_protocol_text.

    Без промежуточного списка «всех сырых строк» — сразу группировка по path и только
    нужные поля (экономия RAM). lex_text хранится только если отличается от text.
    """
    memory_saver = _memory_saver_enabled()
    lex_cap = int(os.environ.get("RAG_LEXICAL_MAX_CHARS", "0") or "0")
    by_path: dict[str, list[dict]] = {}
    for pp in part_paths:
        with pp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                p = (row.get("source_path") or "").strip()
                if not p:
                    continue
                text = (row.get("text") or "").strip()
                slim: dict = {
                    "page_from": int(row.get("page_from") or 0),
                    "page_to": int(row.get("page_to") or 0),
                    "chunk_id": row.get("chunk_id"),
                    "text": text,
                    "chunk_type": (row.get("chunk_type") or "body").strip() or "body",
                }
                if not memory_saver:
                    ert = (row.get("embedding_ready_text") or "").strip()
                    if ert and ert != text:
                        if lex_cap > 0 and len(ert) > lex_cap:
                            ert = ert[:lex_cap]
                        slim["lex_text"] = ert
                elif lex_cap > 0 and len(text) > lex_cap:
                    slim["lex_text"] = text[:lex_cap]
                by_path.setdefault(p, []).append(slim)
    out: list[dict] = []
    for p in sorted(by_path.keys()):
        rows = sorted(
            by_path[p],
            key=lambda r: (
                r["page_from"],
                r["page_to"],
                str(r.get("chunk_id") or ""),
            ),
        )
        for i, row in enumerate(rows):
            text = (row.get("text") or "").strip()
            rec: dict = {
                "path": p,
                "text": text,
                "title": "",
                "category": "",
                "kind": row.get("chunk_type") or "body",
                "chunk_index": i,
                "chunk_id": row.get("chunk_id"),
            }
            if "lex_text" in row:
                rec["lex_text"] = row["lex_text"]
            out.append(rec)
    return out


def _use_jsonl_chunks() -> bool:
    """По умолчанию — JSONL-чанки (корпус), если явно не задан RAG_CHUNKS_SOURCE=json."""
    src = (os.environ.get("RAG_CHUNKS_SOURCE") or "").strip().lower()
    if src in ("json", "legacy", "chunks.json"):
        return False
    if src in ("jsonl", "corpus", "parts", "1", "true", "yes"):
        return True
    # авто: есть части corpus → jsonl; иначе chunks.json
    return bool(_jsonl_chunk_files())


def _enrich_chunks_from_index() -> None:
    """Заголовок и рубрика из protocols.json / protocol_meta для routing и retrieve."""
    for ch in _chunks:
        p = ch.get("path") or ""
        if not p:
            continue
        pr = _protocols_by_path.get(p) or {}
        pm = _protocol_meta.get(p) or {}
        if not (ch.get("title") or "").strip():
            ch["title"] = (pr.get("title") or pm.get("title") or "").strip() or Path(
                p
            ).stem
        if not (ch.get("category") or "").strip():
            ch["category"] = (pr.get("category") or pm.get("category") or "").strip()


def load_data() -> None:
    global _chunks, _chunks_by_path, _protocols_by_path, _protocol_meta, _structured_by_path, _routing
    _protocols_by_path = {}
    if PROTOCOLS_PATH.is_file():
        for row in json.loads(PROTOCOLS_PATH.read_text(encoding="utf-8")):
            _protocols_by_path[row["path"]] = row
    if PROTOCOL_META_PATH.is_file():
        _protocol_meta = json.loads(PROTOCOL_META_PATH.read_text(encoding="utf-8"))
    else:
        _protocol_meta = {}
    if STRUCTURED_INDEX_PATH.is_file():
        _structured_by_path = {
            row["path"]: row
            for row in json.loads(STRUCTURED_INDEX_PATH.read_text(encoding="utf-8"))
            if row.get("path")
        }
    else:
        _structured_by_path = {}
    rp = ROOT / "symptom_routing.json"
    if rp.is_file():
        _routing = json.loads(rp.read_text(encoding="utf-8"))
    else:
        _routing = {}

    if _use_jsonl_chunks():
        parts = _jsonl_chunk_files()
        if not parts:
            raise SystemExit(
                f"Нет JSONL-чанков ({CORPUS_CHUNKS_PARTS_GLOB} или RAG_CHUNKS_JSONL) "
                f"в { _chunks_data_root() }. Соберите корпус, задайте RAG_CHUNKS_DIR на диск с данными "
                "или RAG_CHUNKS_SOURCE=json при наличии chunks.json"
            )
        _chunks = _load_chunks_from_jsonl(parts)
    else:
        if not CHUNKS_PATH.is_file():
            raise SystemExit(
                f"Нет {CHUNKS_PATH}. Запустите: python3 build_chunks.py "
                "или положите corpus_chunks_parts/*.jsonl и уберите RAG_CHUNKS_SOURCE=json"
            )
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    _enrich_chunks_from_index()
    _chunks_by_path = {}
    for ch in _chunks:
        p = ch.get("path") or ""
        if not p:
            continue
        _chunks_by_path.setdefault(p, []).append(ch)
    for plist in _chunks_by_path.values():
        plist.sort(key=lambda x: int(x.get("chunk_index", 0)))
    gc.collect()

    global _bm25_index
    _bm25_alpha_chk = float(os.environ.get("RAG_LEX_BM25_ALPHA", "0.55"))
    _pool_merge_chk = os.environ.get("RAG_EMBED_POOL_MERGE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if _bm25_alpha_chk < 0.999 or _pool_merge_chk:
        _bm25_index = build_bm25_index(_chunks, tokenize_ru)
    else:
        _bm25_index = None


def _run_load_data_background() -> None:
    """Тяжёлый корпус грузится в фоне — uvicorn успевает открыть порт (Render health check)."""
    global _chunks_load_error
    try:
        load_data()
        _chunks_load_error = None
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            _chunks_load_error = code
        elif isinstance(code, int):
            _chunks_load_error = f"Ошибка запуска (код {code})"
        else:
            _chunks_load_error = repr(code)
    except Exception as e:
        _chunks_load_error = str(e)
    finally:
        _chunks_load_done.set()


def _require_rag_loaded() -> None:
    if not _chunks_load_done.is_set():
        raise HTTPException(
            status_code=503,
            detail="Индекс протоколов загружается. Повторите запрос через минуту.",
        )
    if _chunks_load_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Не удалось загрузить корпус: {_chunks_load_error}",
        )


def tokenize_ru(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return [t for t in re.findall(r"[а-яa-z]{2,}", s) if len(t) >= 2]


# Слабые модификаторы без смысла диагноза: совпадение только по ним не должно тянуть чужие протоколы.
RAG_GENERIC_LEX: frozenset[str] = frozenset(
    {
        "правосторонний",
        "левосторонний",
        "двусторонний",
        "односторонний",
        "верхний",
        "нижний",
        "передний",
        "задний",
        "средний",
        "острый",
        "хронический",
        "пациент",
        "пациентка",
        "лет",
        "года",
        "году",
        "женский",
        "мужской",
        "возраст",
        "жалуется",
        "жалобы",
        "жалоб",
        "жалоба",
        "предоставлен",
        "предоставленные",
        "отмечает",
        "считает",
        "наличие",
        "дней",
        "недель",
        "месяц",
        "месяцев",
        "год",
    }
)


def _extra_clinical_tokens(q_raw: str) -> set[str]:
    """Доп. токены по подстрокам запроса (ЛОР, вестибулярная тема) — лучше пересечение с корпусом."""
    rq = _norm_query(q_raw)
    extra: set[str] = set()
    if any(
        x in rq
        for x in (
            "гаймор",
            "синусит",
            "пазух",
            "этмоид",
            "лор",
            "носоглот",
            "аденоид",
            "тонзилл",
            "ангин",
            "фаринг",
            "ларинг",
        )
    ):
        extra.update(
            {
                "гаймор",
                "синусит",
                "пазух",
                "придаточн",
                "носоглот",
                "этмоид",
            }
        )
    if any(x in rq for x in ("вертиго", "дппг", "вестибуляр", "нистагм", "дикс", "холлпайк")):
        extra.update({"вестибуляр", "вертиго", "дппг", "нистагм", "позицион"})
    return extra


def _anchor_tokens(qtok: set[str]) -> list[str]:
    """Токены-якоря: не из общего списка модификаторов."""
    a = [t for t in qtok if t not in RAG_GENERIC_LEX]
    return a


def _cosine_vec(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(s / (na * nb))


def _chunk_text_for_embedding(ch: dict) -> str:
    t = (
        (ch.get("embedding_ready_text") or "").strip()
        or (ch.get("lex_text") or "").strip()
        or (ch.get("text") or "").strip()
    )
    if len(t) > 7500:
        t = t[:7500] + "…"
    if not t:
        t = ((ch.get("title") or "") + " " + (ch.get("path") or "")).strip() or "."
    return t


def _norm_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5] * len(values)
    return [(float(x) - lo) / (hi - lo) for x in values]


def _gemini_embed_one(
    model: str,
    text: str,
    task_type: str | None,
) -> list[float]:
    import google.generativeai as genai

    embed_fn = getattr(genai, "embed_content", None)
    if embed_fn is None:
        from google.generativeai.embedding import embed_content as embed_fn

    kw: dict = {"model": model, "content": text[:8000]}
    if task_type:
        kw["task_type"] = task_type
    try:
        r = embed_fn(**kw)
    except (TypeError, ValueError, KeyError):
        kw.pop("task_type", None)
        r = embed_fn(**kw)
    emb = r.get("embedding")
    if isinstance(emb, dict) and "values" in emb:
        emb = emb["values"]
    if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
        return [float(x) for x in emb]
    raise RuntimeError("unexpected embedding response")


def _gemini_embed_rerank_pool(
    query: str,
    pool_rows: list[tuple[float, float, float, dict]],
    alpha: float,
    model: str,
) -> list[tuple[float, float, float, dict]]:
    """Переранжирование пула чанков: α·lex_norm + (1−α)·cosine(query, chunk)."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key or not pool_rows:
        return pool_rows
    import google.generativeai as genai

    genai.configure(api_key=key)

    q_text = (query or "").strip()[:8000]
    q_vec = _gemini_embed_one(
        model,
        q_text,
        "retrieval_query",
    )

    finals = [float(r[0]) for r in pool_rows]
    lex_norm = _norm_minmax(finals)

    doc_texts = [
        _chunk_text_for_embedding(r[4] if len(r) >= 5 else r[3]) for r in pool_rows
    ]
    max_workers = min(8, max(1, len(doc_texts)))

    def embed_doc(i: int) -> list[float]:
        return _gemini_embed_one(
            model,
            doc_texts[i],
            "retrieval_document",
        )

    timeout = float(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "45"))
    doc_vecs: list[list[float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(embed_doc, i) for i in range(len(pool_rows))]
        for fut in futures:
            doc_vecs.append(fut.result(timeout=timeout))

    out_rows: list[tuple[float, float, float, dict]] = []
    for i, row in enumerate(pool_rows):
        if len(row) >= 5:
            # (final, lex_raw, bm25_raw, routing_mult, ch)
            final, lex, mult, ch = row[0], row[1], row[3], row[4]
        else:
            final, lex, mult, ch = row
        cos = _cosine_vec(q_vec, doc_vecs[i])
        h = alpha * lex_norm[i] + (1.0 - alpha) * cos
        out_rows.append((h, lex, mult, ch))
    out_rows.sort(key=lambda x: -x[0])
    return out_rows


def _icd_embed_rank_candidates(
    rag_query: str,
    pool: list[dict],
    k: int,
    emb_model: str,
) -> list[dict]:
    """k ближайших по косинусу (эмбеддинг запроса vs «код + название ru»)."""
    if k <= 0 or not pool:
        return []
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return []
    rq = (rag_query or "").strip()[:8000]
    try:
        q_vec = _gemini_embed_one(emb_model, rq, "retrieval_query")
    except Exception:
        return []
    doc_texts: list[str] = []
    for p in pool:
        t = f"{p.get('code') or ''} {p.get('title_ru') or ''}".strip()[:4000]
        doc_texts.append(t if t else str(p.get("code") or "."))
    max_workers = min(8, max(1, len(doc_texts)))
    timeout = float(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "45"))

    def embed_doc(i: int) -> list[float]:
        return _gemini_embed_one(emb_model, doc_texts[i], "retrieval_document")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(embed_doc, i) for i in range(len(pool))]
            doc_vecs = [f.result(timeout=timeout) for f in futures]
    except Exception:
        return []
    scored_pairs: list[tuple[float, dict]] = []
    for i, row in enumerate(pool):
        cos = _cosine_vec(q_vec, doc_vecs[i])
        copy = dict(row)
        copy["embed_sim"] = round(float(cos), 4)
        scored_pairs.append((float(cos), copy))
    scored_pairs.sort(key=lambda x: -x[0])
    return [d for _, d in scored_pairs[:k]]


def _merge_icd_allowed_for_gemini(
    lex_top: list[dict], embed_top: list[dict]
) -> list[dict]:
    """Объединение лексического топ-N и k-NN по эмбеддингу; один код — одна строка."""
    by_code: dict[str, dict] = {}
    for it in lex_top:
        c = normalize_icd_code(str(it.get("code") or ""))
        if not c:
            continue
        row = dict(it)
        row["code"] = c
        row["pool_source"] = "lex_top"
        by_code[c] = row
    for it in embed_top:
        c = normalize_icd_code(str(it.get("code") or ""))
        if not c:
            continue
        row = dict(it)
        row["code"] = c
        if c not in by_code:
            row["pool_source"] = "embed_knn"
            by_code[c] = row
        else:
            prev = by_code[c]
            if it.get("embed_sim") is not None:
                prev["embed_sim"] = it.get("embed_sim")
            prev["pool_source"] = "lex_top+embed"
    return list(by_code.values())


def _refine_icd_analysis_with_gemini(
    rag_query: str,
    icd_analysis: dict,
    model,
) -> None:
    """Уточнение suggested и codes_for_retrieval: Gemini выбирает только из лексического топа (без k-NN по умолчанию)."""
    if icd_analysis.get("explicit_icd_in_query"):
        return
    if os.environ.get("RAG_ICD_GEMINI_SELECT", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if not (rag_query or "").strip():
        return
    scored = ru_lexicon_scored_entries(rag_query)
    if not scored:
        return
    n_lex = max(1, min(int(os.environ.get("RAG_ICD_LEX_TOP", "12")), 40))
    n_pool = max(n_lex, min(int(os.environ.get("RAG_ICD_EMBED_POOL", "32")), 120))
    k_embed = max(0, min(int(os.environ.get("RAG_ICD_EMBED_K", "0")), 20))

    lex_top = scored[:n_lex]
    pool = scored[:n_pool]
    emb_model = os.environ.get(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    ).strip()
    embed_top: list[dict] = []
    if k_embed > 0 and os.environ.get("RAG_ICD_EMBED_RANK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        embed_top = _icd_embed_rank_candidates(
            rag_query, pool, k_embed, emb_model
        )
    allowed_list = _merge_icd_allowed_for_gemini(lex_top, embed_top)
    present = {normalize_icd_code(str(x.get("code") or "")) for x in allowed_list}
    present.discard("")
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        dc = normalize_icd_code(str(d.get("code") or ""))
        if not dc or dc in present:
            continue
        found = next((x for x in scored if x["code"] == dc), None)
        if found is None:
            info = describe_code(dc)
            found = {
                "code": dc,
                "title_ru": info.get("title_ru"),
                "title_en": info.get("title_en"),
                "lex_score": None,
            }
        row = dict(found)
        row["pool_source"] = "regex_query"
        allowed_list.append(row)
        present.add(dc)

    allowed_by_code = {
        normalize_icd_code(str(x.get("code") or "")): x for x in allowed_list
    }
    allowed_by_code.pop("", None)

    payload = json.dumps(
        [
            {
                "code": x["code"],
                "title_ru": x.get("title_ru") or "",
                "title_en": x.get("title_en") or "",
            }
            for x in sorted(allowed_list, key=lambda z: str(z.get("code") or ""))
        ],
        ensure_ascii=False,
    )
    prompt = (
        SYSTEM_ICD_POOL_SELECT
        + "\n\n---\n\nЗапрос пользователя:\n"
        + rag_query.strip()[:4000]
        + "\n\nallowed (единственный источник кодов):\n"
        + payload[:14000]
    )
    parsed = None
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException:
        return
    except Exception:
        return
    if not parsed or not isinstance(parsed, dict):
        return
    codes = parsed.get("codes")
    if not isinstance(codes, list):
        return
    selected: list[dict] = []
    for item in codes[:6]:
        if not isinstance(item, dict):
            continue
        raw = normalize_icd_code(str(item.get("code") or ""))
        if not raw or raw not in allowed_by_code:
            continue
        src = allowed_by_code[raw]
        selected.append(
            {
                "code": raw,
                "title_ru": src.get("title_ru"),
                "title_en": src.get("title_en"),
                "match_method": "gemini_from_pool",
                "score": src.get("lex_score"),
                "rationale": (item.get("rationale") or item.get("note") or "")[:320],
            }
        )
    if not selected:
        return
    icd_analysis["suggested"] = selected
    icd_analysis["icd_meta"] = {
        "strategy": (
            "gemini_from_lex_top_and_embed_knn"
            if embed_top
            else "gemini_from_lex_top"
        ),
        "lex_top": n_lex,
        "embed_pool": n_pool,
        "embed_k": k_embed,
        "embedding_used": bool(embed_top),
        "allowed_count": len(allowed_list),
    }
    merged_codes = list(
        dict.fromkeys(
            [normalize_icd_code(str(d.get("code") or "")) for d in icd_analysis.get("detected") or []]
            + [s["code"] for s in selected]
        )
    )
    merged_codes = [c for c in merged_codes if c][:10]
    icd_analysis["codes_for_retrieval"] = merged_codes


def _top_retrieval_score_for_icd_gate(retrieved: list[dict]) -> tuple[float, bool]:
    """После гибридного эмбеддинг-переранжирования чанков поле score ∈ [0,1]."""
    if not retrieved:
        return 0.0, False
    r0 = retrieved[0]
    if not r0.get("embedding_rerank"):
        return 0.0, False
    try:
        sc = float(r0.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0, False
    return max(0.0, min(1.0, sc)), True


def maybe_refine_icd_with_gemini_after_retrieve(
    model,
    rag_query: str,
    icd_analysis: dict,
    retrieved: list[dict],
) -> None:
    """Gemini выбирает коды из пула только при уверенном отборе протоколов (≥ порога) и без явных кодов в тексте."""
    if icd_analysis.get("explicit_icd_in_query"):
        return
    if os.environ.get("RAG_ICD_GEMINI_SELECT", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if not (rag_query or "").strip():
        return
    require_embed = os.environ.get("RAG_ICD_GEMINI_REQUIRE_EMBED_RANK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    top, emb_ok = _top_retrieval_score_for_icd_gate(retrieved)
    if require_embed and not emb_ok:
        return
    min_sc = float(os.environ.get("RAG_ICD_GEMINI_MIN_TOP_SCORE", "0.8"))
    if top < min_sc:
        return
    _refine_icd_analysis_with_gemini(rag_query, icd_analysis, model)


def _norm_query(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


def infer_audience_from_query(q: str, routing: dict) -> str | None:
    """'adult' | 'child' | None — по словам и числам (49 лет, ребёнок …)."""
    nq = _norm_query(q)
    aud = routing.get("audience") or {}
    child_m = aud.get("child_markers") or []
    adult_m = aud.get("adult_markers") or []
    has_ch = any(c in nq for c in child_m)
    has_ad = any(a in nq for a in adult_m)
    if has_ad and not has_ch:
        return "adult"
    if has_ch and not has_ad:
        return "child"

    def age_bucket(age: int) -> str | None:
        if age >= 18:
            return "adult"
        if 0 < age < 18:
            return "child"
        return None

    for m in re.finditer(r"(\d{1,3})\s*лет", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"(\d{1,3})\s*года?\b", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"возраст\s*[:\s]*(\d{1,3})\b", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    for m in re.finditer(r"пациент(?:у|а)?\s+(\d{1,3})\s*лет", nq):
        b = age_bucket(int(m.group(1)))
        if b:
            return b
    return None


def doc_audience_hint(path: str, title: str, routing: dict) -> str | None:
    """pediatric | adult | mixed | None — по названию файла/заголовка."""
    s = f"{path} {title}".lower()
    ped = routing.get("pediatric_title_markers") or []
    adult_t = routing.get("adult_title_markers") or []
    has_p = any(p in s for p in ped)
    has_a = any(a in s for a in adult_t)
    if has_p and has_a:
        return "mixed"
    if has_p:
        return "pediatric"
    if has_a:
        return "adult"
    return None


def filter_retrieval_by_audience(
    rows: list[dict], rq: str, routing: dict
) -> tuple[list[dict], str | None, bool]:
    """Отбрасывает чанки с явно несовпадающей аудиторией (дет/взросл)."""
    aud = infer_audience_from_query(rq, routing)
    if aud is None or not rows:
        return rows, aud, False

    strict = os.environ.get("RAG_AUDIENCE_FILTER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not strict:
        return rows, aud, False

    out: list[dict] = []
    for r in rows:
        hint = doc_audience_hint(
            r.get("path") or "",
            r.get("title") or "",
            routing,
        )
        if hint is None or hint == "mixed":
            out.append(r)
            continue
        if aud == "adult" and hint == "pediatric":
            continue
        if aud == "child" and hint == "adult":
            continue
        out.append(r)

    if not out:
        return rows, aud, True
    return out, aud, False


def routing_multiplier(raw_query: str, ch: dict, routing: dict | None) -> float:
    """Усиление/ослабление релевантности по symptom_routing.json (рубрики, аудитория, path)."""
    if not routing:
        return 1.0
    q = _norm_query(raw_query)
    cat = (ch.get("category") or "").strip()
    title_low = ((ch.get("title") or "") + " " + (ch.get("path") or "")).lower()

    m = 1.0

    for br in routing.get("boost_rules", []):
        kws = br.get("match") or []
        if not any(k in q for k in kws):
            continue
        cats = br.get("categories") or []
        if cat and cat in cats:
            m *= float(br.get("factor", 1.0))

    for pr in routing.get("penalty_rules", []):
        when = pr.get("when") or []
        if not any(w in q for w in when):
            continue
        unless = pr.get("unless") or []
        if unless and any(u in q for u in unless):
            continue
        if cat in (pr.get("categories") or []):
            m *= float(pr.get("factor", 1.0))

    aud = routing.get("audience") or {}
    child_m = aud.get("child_markers") or []
    adult_m = aud.get("adult_markers") or []
    ped_title = routing.get("pediatric_title_markers") or []
    adult_title = routing.get("adult_title_markers") or []

    infer = infer_audience_from_query(raw_query, routing)
    if infer == "adult":
        has_child = False
        has_adult = True
    elif infer == "child":
        has_child = True
        has_adult = False
    else:
        has_child = any(c in q for c in child_m)
        has_adult = any(a in q for a in adult_m)
    if has_adult and not has_child:
        if any(p in title_low for p in ped_title):
            m *= float(aud.get("penalty_adult_query_pediatric_doc", 0.35))
    if has_child and not has_adult:
        if any(a in title_low for a in adult_title):
            m *= float(aud.get("penalty_child_query_adult_doc", 0.4))

    for pp in routing.get("path_penalties", []):
        when_q = pp.get("when_query") or []
        if not when_q or not any(w in q for w in when_q):
            continue
        unless = pp.get("unless_query") or []
        if unless and any(u in q for u in unless):
            continue
        pats = pp.get("path_contains") or []
        if any(p.lower() in title_low for p in pats):
            m *= float(pp.get("factor", 0.5))

    for pb in routing.get("path_boosts", []):
        needed = pb.get("when_query") or []
        min_hits = int(pb.get("when_min_hits", 2))
        hits = sum(1 for w in needed if w in q)
        if hits < min_hits:
            continue
        pats = pb.get("path_contains") or []
        if any(p.lower() in title_low for p in pats):
            m *= float(pb.get("factor", 1.5))

    return max(m, 1e-9)


def clinical_query_for_rag(full_query: str) -> str:
    """Текст для лексического RAG: блок «Жалобы и вопрос» без контекста и без ответов на уточняющие вопросы."""
    sep = "=== Жалобы и вопрос ==="
    if sep in full_query:
        part = full_query.split(sep, 1)[1].strip()
    else:
        part = full_query.strip()
    # Блок ответов содержит слова вопросов (напр. «кровотечение») — подстрока «кров» ложно тянет гематологию и размывает отбор.
    mark = "— Ответы на уточняющие вопросы:"
    if mark in part:
        part = part.split(mark, 1)[0].strip()
    return part if part else full_query.strip()


def gather_protocol_text(path: str, max_chars: int) -> str:
    """Склеивает чанки одного PDF по порядку (до max_chars символов)."""
    parts = _chunks_by_path.get(path) or []
    out: list[str] = []
    n = 0
    for ch in parts:
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        if n + len(t) > max_chars:
            rest = max_chars - n
            if rest > 80:
                out.append(t[:rest])
            break
        out.append(t)
        n += len(t)
    return "\n\n".join(out)


def confidence_display_full(score: object) -> bool:
    """Совпадает с отображением 100% в интерфейсе (округление как в index.html)."""
    try:
        x = float(score)
    except (TypeError, ValueError):
        return False
    x = max(0.0, min(1.0, x))
    return round(100 * x) >= 100


def _confidence_numeric(score: object) -> float | None:
    try:
        x = float(score)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, x))


def confidence_for_detailed_extraction(score: object) -> bool:
    """Развёрнутая выдержка (SYSTEM_EXTRACT_FULL) при оценке ≥80%, не только при 100%."""
    x = _confidence_numeric(score)
    if x is None:
        return False
    min_s = float(os.environ.get("RAG_DETAIL_EXTRACT_MIN_SCORE", "0.8"))
    return x >= min_s


def _protocol_meta_icd_boost(path: str, icd_norms: list[str]) -> float:
    """Усиление, если в protocol_meta заданы icd_codes/mkb_codes и они пересекаются с запросом."""
    if not icd_norms:
        return 1.0
    pm = _protocol_meta.get(path) or {}
    raw = pm.get("icd_codes") or pm.get("mkb_codes") or []
    if not isinstance(raw, list):
        return 1.0
    boost = float(os.environ.get("RAG_ICD_META_BOOST", "1.35"))
    want = {x.strip().lower() for x in icd_norms if isinstance(x, str)}
    for c in raw:
        if not isinstance(c, str):
            continue
        n = normalize_icd_code(c).strip().lower()
        if n and n in want:
            return boost
    return 1.0


def _rag_support_map(retrieved: list[dict]) -> tuple[dict[str, float], float]:
    """path → нормализованный score / max в батче; второе значение — max score."""
    max_s = 0.0
    for r in retrieved:
        try:
            s = float(r.get("score") or 0)
        except (TypeError, ValueError):
            s = 0.0
        if s > max_s:
            max_s = s
    if max_s <= 0:
        max_s = 1.0
    m: dict[str, float] = {}
    for r in retrieved:
        p = str(r.get("path") or "")
        if not p:
            continue
        try:
            s = float(r.get("score") or 0)
        except (TypeError, ValueError):
            s = 0.0
        n = s / max_s
        if p not in m or n > m[p]:
            m[p] = n
    return m, max_s


def apply_protocol_confidence_calibration(
    parsed: dict | None, retrieved: list[dict]
) -> dict[str, float]:
    """Смешивает оценку модели с опорой отбора (rag_support); правит confidence_score."""
    rag_map, _ = _rag_support_map(retrieved)
    if not parsed or not isinstance(parsed, dict):
        return rag_map
    w = float(os.environ.get("RAG_LLM_CONF_BLEND_W", "0.62"))
    cap_low = float(os.environ.get("RAG_MIN_RAG_SUPPORT_CAP", "0.74"))
    min_rag_high = float(os.environ.get("RAG_MIN_RAG_SUPPORT_FOR_HIGH_CONF", "0.2"))
    protos = parsed.get("protocols")
    if not isinstance(protos, list):
        return rag_map
    for pr in protos:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        rag_sup = float(rag_map.get(p, 0.0))
        pr["rag_support"] = round(rag_sup, 4)
        llm_c = _confidence_numeric(pr.get("confidence_score"))
        if llm_c is None:
            llm_c = 0.55
        pr["confidence_score_llm"] = pr.get("confidence_score")
        blended = w * llm_c + (1.0 - w) * rag_sup
        if rag_sup < min_rag_high:
            blended = min(blended, cap_low)
            pr["low_retrieval_support"] = True
        pr["confidence_score"] = round(max(0.0, min(1.0, blended)), 4)
    return rag_map


def _majority_category_from_retrieval(retrieved: list[dict]) -> str | None:
    """Рубрика (slug), чаще всего встречающаяся среди отобранных чанков."""
    cats: list[str] = []
    for r in retrieved:
        c = (r.get("category") or "").strip()
        if c:
            cats.append(c)
            continue
        p = r.get("path") or ""
        pr = _protocols_by_path.get(p) or {}
        pm = _protocol_meta.get(p) or {}
        x = (pr.get("category") or pm.get("category") or "").strip()
        if x:
            cats.append(x)
    if not cats:
        return None
    return Counter(cats).most_common(1)[0][0]


def refine_protocol_confidences_gemini(
    model,
    q: str,
    parsed: dict,
    retrieved: list[dict],
) -> bool:
    """Второй короткий вызов модели для калибровки confidence_score (опционально)."""
    protos = parsed.get("protocols") or []
    if not protos:
        return False
    ex_by_path: dict[str, str] = {}
    for r in retrieved:
        p = r.get("path") or ""
        if p and p not in ex_by_path:
            ex_by_path[p] = str(r.get("excerpt") or "")[:500]
    lines: list[str] = []
    for pr in protos[:8]:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        ex = ex_by_path.get(p, "")
        lines.append(f"path={p}\ntitle={pr.get('title')}\nфрагмент: {ex}\n")
    if not lines:
        return False
    prompt = (
        SYSTEM_CONFIDENCE_REFINE
        + "\n\nЗапрос:\n"
        + (q or "")[:6000]
        + "\n\nПротоколы:\n"
        + "\n---\n".join(lines)
    )
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        pj = _try_parse_json(txt)
    except Exception:
        return False
    if not pj or isinstance(pj, bool) or not isinstance(pj, dict):
        return False
    scores = pj.get("scores") or []
    by_path: dict[str, float] = {}
    for s in scores:
        if not isinstance(s, dict):
            continue
        p = str(s.get("path") or "")
        try:
            c = float(s.get("confidence_score"))
        except (TypeError, ValueError):
            continue
        if p:
            by_path[p] = max(0.0, min(1.0, c))
    if not by_path:
        return False
    mix = float(os.environ.get("RAG_CONFIDENCE_SECOND_BLEND", "0.55"))
    touched = False
    for pr in protos:
        if not isinstance(pr, dict):
            continue
        p = str(pr.get("path") or "")
        if p not in by_path:
            continue
        cur = _confidence_numeric(pr.get("confidence_score"))
        if cur is None:
            cur = 0.55
        new = mix * by_path[p] + (1.0 - mix) * cur
        pr["confidence_score"] = round(max(0.0, min(1.0, new)), 4)
        pr["confidence_second_pass"] = True
        touched = True
    return touched


def _merge_embed_pool_rows(
    scored: list[tuple],
    pool_n: int,
    merge_on: bool,
) -> list[tuple]:
    """Топ по score + доп. кандидаты с высоким BM25, чтобы не терять чанки вне первых N."""
    if not scored:
        return []
    pool_n = min(int(pool_n), len(scored))
    if not merge_on or len(scored) <= pool_n:
        return scored[:pool_n]
    primary = scored[:pool_n]
    primary_ids: set[int] = set()
    for row in primary:
        ch = row[4] if len(row) >= 5 else row[3]
        primary_ids.add(id(ch))
    bm25_i = 2
    by_bm25 = sorted(scored, key=lambda x: -float(x[bm25_i]))
    cap = min(len(scored), max(pool_n * 2, pool_n + 24))
    out = list(primary)
    seen = set(primary_ids)
    for row in by_bm25:
        if len(out) >= cap:
            break
        ch = row[4] if len(row) >= 5 else row[3]
        if id(ch) in seen:
            continue
        seen.add(id(ch))
        out.append(row)
    out.sort(key=lambda x: -float(x[0]))
    return out


_GAP_FIELD_LABELS_RU: dict[str, str] = {
    "investigations": "обследование (диагностика)",
    "medications": "препараты и группы лекарственных средств",
    "treatment_methods": "лечение и методы",
    "monitoring_frequency": "кратность наблюдения",
}

_NON_PROTOCOL_MARK = "[не из протокола]"


def _norm_str_list_ext(val: object) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    return []


def _append_note_field(existing: object, addition: str) -> str:
    e = str(existing or "").strip()
    a = str(addition or "").strip()
    if not a:
        return e
    if not e:
        return a
    return e + " " + a


def _merge_str_lists_unique(a: list[str], b: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in a + b:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _detailed_block_missing_keys(ext: dict) -> list[str]:
    missing: list[str] = []
    if not _norm_str_list_ext(ext.get("investigations")):
        missing.append("investigations")
    if not _norm_str_list_ext(ext.get("medications")):
        missing.append("medications")
    if not _norm_str_list_ext(ext.get("treatment_methods")):
        missing.append("treatment_methods")
    mf = ext.get("monitoring_frequency")
    if not (str(mf).strip() if mf is not None else ""):
        missing.append("monitoring_frequency")
    return missing


def _merge_gap_into_ext(ext: dict, gap: dict, allowed_keys: list[str]) -> None:
    if "investigations" in allowed_keys:
        g = _norm_str_list_ext(gap.get("investigations"))
        if g:
            ext["investigations"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("investigations")), g
            )
    if "medications" in allowed_keys:
        g = _norm_str_list_ext(gap.get("medications"))
        if g:
            ext["medications"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("medications")), g
            )
    if "treatment_methods" in allowed_keys:
        g = _norm_str_list_ext(gap.get("treatment_methods"))
        if g:
            ext["treatment_methods"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("treatment_methods")), g
            )
    if "monitoring_frequency" in allowed_keys:
        g = gap.get("monitoring_frequency")
        s = str(g).strip() if g is not None else ""
        if s and not str(ext.get("monitoring_frequency") or "").strip():
            ext["monitoring_frequency"] = s


def _ensure_non_protocol_prefix(s: str) -> str:
    t = str(s).strip()
    if not t:
        return ""
    if t.startswith(_NON_PROTOCOL_MARK):
        return t
    return _NON_PROTOCOL_MARK + " " + t


def _merge_non_protocol_into_ext(ext: dict, raw: dict, allowed_keys: list[str]) -> None:
    if "investigations" in allowed_keys:
        items = _norm_str_list_ext(raw.get("investigations"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["investigations"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("investigations")), fixed
            )
    if "medications" in allowed_keys:
        items = _norm_str_list_ext(raw.get("medications"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["medications"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("medications")), fixed
            )
    if "treatment_methods" in allowed_keys:
        items = _norm_str_list_ext(raw.get("treatment_methods"))
        if items:
            fixed = [_ensure_non_protocol_prefix(x) for x in items]
            ext["treatment_methods"] = _merge_str_lists_unique(
                _norm_str_list_ext(ext.get("treatment_methods")), fixed
            )
    if "monitoring_frequency" in allowed_keys:
        if not str(ext.get("monitoring_frequency") or "").strip():
            g = raw.get("monitoring_frequency")
            s = str(g).strip() if g is not None else ""
            if s:
                ext["monitoring_frequency"] = _ensure_non_protocol_prefix(s)


def _run_gap_fill_scan(
    *,
    model,
    query: str,
    title_line: str,
    spec: str,
    body: str,
    extra: str,
    missing_keys: list[str],
    plim: int,
) -> dict | None:
    labels = [_GAP_FIELD_LABELS_RU[k] for k in missing_keys if k in _GAP_FIELD_LABELS_RU]
    fields_ru = "; ".join(labels)
    keys_json = json.dumps(missing_keys, ensure_ascii=False)
    head = SYSTEM_EXTRACT_GAP_SCAN.format(
        fields_ru=fields_ru,
        keys_json=keys_json,
    )
    prompt = (
        head
        + "\n\n---\n\n"
        + f"Запрос пользователя:\n{query}\n\n"
        + f"Специальность (рубрика каталога): {spec}\n"
        + f"Название протокола: {title_line}\n\n"
        + "Текст протокола (фрагменты PDF):\n"
        + body
        + extra
    )
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано]"
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except (HTTPException, Exception):
        return None
    if not parsed or not isinstance(parsed, dict):
        return None
    return parsed


def _run_non_protocol_fill(
    *,
    model,
    query: str,
    title_line: str,
    spec: str,
    missing_keys: list[str],
) -> dict | None:
    labels = [_GAP_FIELD_LABELS_RU[k] for k in missing_keys if k in _GAP_FIELD_LABELS_RU]
    fields_ru = "; ".join(labels)
    keys_json = json.dumps(missing_keys, ensure_ascii=False)
    prompt = SYSTEM_EXTRACT_NON_PROTOCOL.format(
        spec=spec,
        title=title_line,
        query=query[:8000],
        fields_ru=fields_ru,
        keys_json=keys_json,
    )
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except (HTTPException, Exception):
        return None
    if not parsed or not isinstance(parsed, dict):
        return None
    return parsed


def _extract_prompt_char_limit() -> int:
    """Лимит символов промпта для извлечения по протоколу (отдельно от общего чата)."""
    v = (os.environ.get("RAG_EXTRACT_PROMPT_MAX_CHARS") or "").strip()
    if v.isdigit():
        return max(4000, int(v))
    return int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))


def _clamp_detail_string_list(vals: list[str], max_item: int) -> list[str]:
    out: list[str] = []
    for x in vals:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > max_item:
            s = s[: max(1, max_item - 1)] + "…"
        out.append(s)
    return out


def _clamp_detail_ext_lists(ext: dict) -> None:
    mic = int(os.environ.get("RAG_EXTRACT_ITEM_MAX_CHARS", "420"))
    mic = max(120, mic)
    for k in ("investigations", "medications", "treatment_methods"):
        if isinstance(ext.get(k), list):
            ext[k] = _clamp_detail_string_list(
                [str(x) for x in ext[k] if str(x).strip()],
                mic,
            )


_DETAIL_FOCUS_EXTRA: dict[str, str] = {
    "investigations": (
        "Приоритет извлечения: максимально полно заполни investigations и при необходимости diagnosis; "
        "medications и treatment_methods — только если явно следуют из текста протокола по запросу."
    ),
    "medications": (
        "Приоритет извлечения: максимально полно medications; не включай сюда пункты обследования (investigations)."
    ),
    "treatment_methods": (
        "Приоритет извлечения: максимально полно treatment_methods (этапы лечения, операции, режим); "
        "отделяй от диагностики."
    ),
    "monitoring_frequency": (
        "Приоритет извлечения: monitoring_frequency (сроки и частота визитов, диспансеризация) "
        "и при необходимости monitoring_followup (срочные ситуации); не дублируй кратность в recommendations."
    ),
    "care_algorithms": (
        "Приоритет извлечения: care_algorithms — пошаговые алгоритмы ведения/неотложной помощи, "
        "ветвления «если/то», критерии эскалации и госпитализации; верни структурированно."
    ),
}


def _normalize_detail_extract_focus(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    x = raw.strip().lower()
    if x == "monitoring":
        x = "monitoring_frequency"
    if x == "algorithms":
        x = "care_algorithms"
    if x in _DETAIL_FOCUS_EXTRA:
        return x
    return None


def _algo_marker_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pats = [
        r"\bалгоритм\w*",
        r"\bэтап\w*",
        r"\bпошаг\w*",
        r"\bесли\b.{0,80}\bто\b",
        r"\bпри\s+отсутствии\s+эффект",
        r"\bпоказания?\b.{0,70}\bгоспитал",
        r"\bнеотложн\w+\s+помощ",
    ]
    score = 0.0
    for p in pats:
        if re.search(p, t, re.IGNORECASE | re.DOTALL):
            score += 0.14
    return max(0.0, min(1.0, score))


def _normalize_algorithm_rows(raw: object) -> list[dict]:
    out: list[dict] = []
    if not raw:
        return out
    rows = raw if isinstance(raw, list) else [raw]
    idx = 0
    for row in rows:
        idx += 1
        if isinstance(row, str):
            s = row.strip()
            if not s:
                continue
            out.append(
                {
                    "id": f"alg_{idx}",
                    "title": f"Алгоритм {idx}",
                    "entry_conditions": [],
                    "steps": [s],
                }
            )
            continue
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip() or f"Алгоритм {idx}"
        ent = row.get("entry_conditions")
        steps = row.get("steps")
        entry_conditions = _norm_str_list_ext(ent)
        steps_norm = _norm_str_list_ext(steps)
        if not steps_norm:
            fallback = _norm_str_list_ext(row.get("actions"))
            if fallback:
                steps_norm = fallback
        if not steps_norm:
            continue
        out.append(
            {
                "id": f"alg_{idx}",
                "title": title[:220],
                "entry_conditions": entry_conditions[:8],
                "steps": steps_norm[:24],
            }
        )
    return out[:12]


def _fallback_algorithms_from_ext(ext: dict) -> list[dict]:
    recs = _norm_str_list_ext(ext.get("recommendations"))
    tms = _norm_str_list_ext(ext.get("treatment_methods"))
    pool = recs + tms
    if not pool:
        return []
    picked: list[str] = []
    for x in pool:
        t = x.strip()
        if not t:
            continue
        if re.search(r"\b(если|то|этап|показан|неотлож|госпитал|алгоритм)\b", t, re.I):
            picked.append(t)
        if len(picked) >= 10:
            break
    if len(picked) < 3:
        return []
    return [
        {
            "id": "alg_1",
            "title": "Алгоритм ведения по протоколу",
            "entry_conditions": [],
            "steps": picked,
        }
    ]


def infer_specialties_gemini(q: str, model) -> list[str]:
    """Опционально: первый короткий вызов LLM — к каким рубрикам относится запрос."""
    if os.environ.get("GEMINI_SPECIALTY_CLASSIFY", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return []
    prompt = SYSTEM_CLASSIFY + "\n\nЗапрос пользователя:\n" + (q or "")[:6000]
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException:
        return []
    except Exception:
        return []
    if not parsed or not isinstance(parsed, dict):
        return []
    cats = parsed.get("categories") or []
    out = [c for c in cats if isinstance(c, str) and c in ALLOWED_SPECIALTY_SLUGS]
    return out[:3]


def extract_clinical_detail(
    path: str,
    query: str,
    title_hint: str,
    model,
    *,
    detailed: bool = False,
    protocol_confidence: float | None = None,
    extract_focus: str | None = None,
    client_rag_support: float | None = None,
) -> dict | None:
    """Второй вызов LLM: факты по протоколу; при detailed — расширенная схема и больший объём текста."""
    focus_key = _normalize_detail_extract_focus(extract_focus)
    detail_prompt_truncated = False
    if detailed:
        max_body = int(os.environ.get("RAG_EXTRACT_FULL_MATCH_MAX_CHARS", "32000"))
        idx_lim = 16000
        summary_lim = min(4096, idx_lim)
        system = SYSTEM_EXTRACT_FULL
    else:
        max_body = int(os.environ.get("RAG_EXTRACT_MAX_CHARS", "16000"))
        idx_lim = 8000
        summary_lim = 4000
        system = SYSTEM_EXTRACT
    body = gather_protocol_text(path, max_body)
    struct = _structured_by_path.get(path) or {}
    extra = ""
    if struct.get("summary"):
        extra += (
            "\n\n[Выдержка индекса: краткое содержание]\n"
            + format_structured_index_text(str(struct["summary"]), summary_lim)
        )
    if struct.get("diagnosis"):
        extra += (
            "\n\n[Выдержка индекса: диагностика]\n"
            + format_structured_index_text(str(struct["diagnosis"]), idx_lim)
        )
    if struct.get("treatment"):
        extra += (
            "\n\n[Выдержка индекса: лечение]\n"
            + format_structured_index_text(str(struct["treatment"]), idx_lim)
        )
    if len(body.strip()) < 120 and not extra.strip():
        return None
    meta = _protocol_meta.get(path) or {}
    spec = meta.get("specialty_ru") or ""
    title_line = title_hint or meta.get("title") or path
    prompt = (
        system
        + "\n\n---\n\n"
        + f"Запрос пользователя:\n{query}\n\n"
        + f"Специальность (рубрика каталога): {spec}\n"
        + f"Название протокола: {title_line}\n\n"
        + "Текст протокола (фрагменты PDF):\n"
        + body
        + extra
    )
    if focus_key:
        prompt += "\n\n---\n" + _DETAIL_FOCUS_EXTRA[focus_key]
    plim = _extract_prompt_char_limit()
    if len(prompt) > plim:
        prompt = prompt[: plim - 80] + "\n…[обрезано для лимита контекста]"
        detail_prompt_truncated = True
    try:
        resp = generate_gemini(model, prompt)
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
    except HTTPException as e:
        return {"error": str(e.detail), "path": path, "title": title_line}
    except Exception as e:
        return {"error": str(e)[:400], "path": path, "title": title_line}
    if not parsed or not isinstance(parsed, dict):
        return None

    def _norm_str_list(val: object) -> list[str]:
        if val is None:
            return []
        if isinstance(val, str):
            s = val.strip()
            return [s] if s else []
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        return []

    ext: dict = {
        "diagnosis": parsed.get("diagnosis") or "",
        "treatment_methods": parsed.get("treatment_methods") or [],
        "medications": parsed.get("medications") or [],
        "note": parsed.get("note") or "",
    }
    if detailed:
        ext["investigations"] = _norm_str_list(parsed.get("investigations"))
        mf = parsed.get("monitoring_frequency")
        ext["monitoring_frequency"] = (
            str(mf).strip() if mf is not None and str(mf).strip() else ""
        )
        ext["recommendations"] = parsed.get("recommendations") or []
        ext["monitoring_followup"] = parsed.get("monitoring_followup") or ""
        ext["care_algorithms"] = _normalize_algorithm_rows(parsed.get("care_algorithms"))
        ext["contraindications"] = parsed.get("contraindications") or ""
        ext["detailed"] = True
        if not ext["care_algorithms"]:
            ext["care_algorithms"] = _fallback_algorithms_from_ext(ext)
        _clamp_detail_ext_lists(ext)
        gap_on = os.environ.get("RAG_EXTRACT_GAP_RETRY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if gap_on:
            missing = _detailed_block_missing_keys(ext)
            if missing:
                gap_parsed = _run_gap_fill_scan(
                    model=model,
                    query=query,
                    title_line=title_line,
                    spec=spec,
                    body=body,
                    extra=extra,
                    missing_keys=missing,
                    plim=plim,
                )
                if gap_parsed:
                    n_before = len(_detailed_block_missing_keys(ext))
                    _merge_gap_into_ext(ext, gap_parsed, missing)
                    _clamp_detail_ext_lists(ext)
                    if len(_detailed_block_missing_keys(ext)) < n_before:
                        ext["note"] = _append_note_field(
                            ext.get("note"),
                            "Выполнен повторный поиск по полному тексту протокола для пустых разделов (обследование, препараты, лечение, кратность наблюдения).",
                        )
            missing2 = _detailed_block_missing_keys(ext)
            np_on = os.environ.get(
                "RAG_EXTRACT_NON_PROTOCOL_FALLBACK", "1"
            ).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            np_mon_only = os.environ.get(
                "RAG_EXTRACT_NON_PROTOCOL_MONITORING_ONLY", "0"
            ).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            np_keys = list(missing2)
            if np_on and np_keys:
                if np_mon_only:
                    np_keys = [k for k in np_keys if k == "monitoring_frequency"]
                if np_keys:
                    np_parsed = _run_non_protocol_fill(
                        model=model,
                        query=query,
                        title_line=title_line,
                        spec=spec,
                        missing_keys=np_keys,
                    )
                    if np_parsed:
                        n_before_np = len(_detailed_block_missing_keys(ext))
                        _merge_non_protocol_into_ext(ext, np_parsed, np_keys)
                        _clamp_detail_ext_lists(ext)
                        if len(_detailed_block_missing_keys(ext)) < n_before_np:
                            ext["note"] = _append_note_field(
                                ext.get("note"),
                                "Формулировки с пометкой «[не из протокола]» — общеклинические ориентиры, не цитата из клинического протокола.",
                            )
    warn_thr = float(os.environ.get("RAG_DETAIL_WARN_RAG_SUPPORT", "0.22"))
    low_sup = False
    if client_rag_support is not None:
        try:
            low_sup = float(client_rag_support) < warn_thr
        except (TypeError, ValueError):
            low_sup = False
    out: dict = {
        "path": path,
        "title": title_line,
        "specialty_ru": spec or None,
        "category": meta.get("category"),
        "extraction": ext,
        "detail_prompt_truncated": detail_prompt_truncated,
        "extract_focus_applied": focus_key,
        "low_protocol_match_support": low_sup,
    }
    algo_src = "\n".join(
        [
            str(ext.get("diagnosis") or ""),
            "\n".join(_norm_str_list_ext(ext.get("recommendations"))),
            "\n".join(_norm_str_list_ext(ext.get("treatment_methods"))),
            body[:12000],
        ]
    )
    algo_conf = _algo_marker_score(algo_src)
    has_algos = bool(_normalize_algorithm_rows(ext.get("care_algorithms")))
    if has_algos:
        algo_conf = max(algo_conf, 0.62)
    out["algorithm_confidence"] = round(float(max(0.0, min(1.0, algo_conf))), 4)
    out["is_algorithmic_protocol"] = bool(has_algos or algo_conf >= 0.42)
    if out["is_algorithmic_protocol"] and not has_algos:
        out["algorithm_warnings"] = [
            "В тексте есть признаки алгоритма, но явная структура шагов извлечена частично."
        ]
    if client_rag_support is not None:
        try:
            out["client_rag_support"] = float(
                max(0.0, min(1.0, float(client_rag_support)))
            )
        except (TypeError, ValueError):
            pass
    if protocol_confidence is not None:
        out["detail_match_score"] = protocol_confidence
    return out


def retrieve(
    query: str,
    max_chunks: int | None = None,
    max_per_path: int = 2,
    routing_query: str | None = None,
    category_boost: list[str] | None = None,
    user_category_slugs: list[str] | None = None,
    icd_codes_for_lex: list[str] | None = None,
) -> list[dict]:
    """Лексический отбор + множители из symptom_routing.json (если RAG_ROUTING=1).

    query — короткий текст для подсчёта совпадений с чанками (обычно только жалобы).
    routing_query — полный запрос для правил возраста/рубрик; если None, берётся query.
    category_boost — slug рубрик из опционального LLM-классификатора запроса.
    user_category_slugs — рубрики, выбранные пользователем в форме: усиление совпадений и штраф нерелевантных чанков.
    icd_codes_for_lex — нормализованные коды МКБ-10: дополнительные лексические токены и усиление чанков, где встречается код.
    """
    if max_chunks is None:
        max_chunks = int(os.environ.get("RAG_MAX_CHUNKS", "6"))
    use_routing = os.environ.get("RAG_ROUTING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    boost_set = frozenset(category_boost or [])
    boost_factor = float(os.environ.get("RAG_CATEGORY_BOOST_FACTOR", "1.45"))
    user_slugs = frozenset(
        s for s in (user_category_slugs or []) if s in ALLOWED_SPECIALTY_SLUGS
    )
    user_boost = float(os.environ.get("RAG_USER_CATEGORY_BOOST", "2.05"))
    user_penalty = float(os.environ.get("RAG_USER_CATEGORY_PENALTY", "0.32"))
    user_uncertain = float(os.environ.get("RAG_USER_CATEGORY_UNCERTAIN", "0.78"))
    rq = routing_query if routing_query is not None else query
    icd_lex = icd_tokens_for_lex(icd_codes_for_lex or [])
    # Коды вида J20.9 в самом запросе не попадают в tokenize_ru — извлекаем отдельно.
    icd_from_query = icd_tokens_for_lex(extract_icd_codes_raw(query))
    qtok = (
        set(tokenize_ru(query))
        | icd_lex
        | icd_from_query
        | _extra_clinical_tokens(rq)
    )
    if not qtok:
        return []
    anchor_list = _anchor_tokens(qtok)
    anchor_set = frozenset(anchor_list)
    generic_w = float(os.environ.get("RAG_GENERIC_LEX_WEIGHT", "0.22"))
    anchor_miss_penalty = float(os.environ.get("RAG_ANCHOR_MISS_PENALTY", "0.045"))
    icd_chunk_boost = float(os.environ.get("RAG_ICD_CHUNK_BOOST", "1.65"))
    icd_norms = [
        c.strip().lower()
        for c in (icd_codes_for_lex or [])
        if isinstance(c, str) and len(c.strip()) >= 3
    ]
    bm25_alpha = float(os.environ.get("RAG_LEX_BM25_ALPHA", "0.55"))
    use_bm25_blend = _bm25_index is not None and bm25_alpha < 0.999
    pool_merge = os.environ.get("RAG_EMBED_POOL_MERGE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    raw_rows: list[tuple[float, float, float, dict, float]] = []
    for ch in _chunks:
        lex_src = (ch.get("lex_text") or ch.get("text") or "") + " " + (
            ch.get("title") or ""
        )
        low = lex_src.lower()
        lex = 0.0
        for t in qtok:
            if t not in low:
                continue
            wt = 1.0 + min(len(t), 10) * 0.02
            if t in RAG_GENERIC_LEX:
                wt *= generic_w
            lex += wt
        if lex <= 0:
            continue
        if anchor_set:
            if not any(t in low for t in anchor_set):
                lex *= anchor_miss_penalty
        bm25_s = 0.0
        if _bm25_index is not None:
            bm25_s = _bm25_index.score_doc(qtok, ch)
        mult = (
            routing_multiplier(rq, ch, _routing)
            if use_routing
            else 1.0
        )
        post = 1.0
        if icd_norms and any(code in low for code in icd_norms):
            post *= icd_chunk_boost
        pth = ch.get("path") or ""
        post *= _protocol_meta_icd_boost(pth, icd_norms)
        cat = (ch.get("category") or "").strip()
        if boost_set and cat in boost_set:
            post *= boost_factor
        if user_slugs:
            if cat and cat in user_slugs:
                post *= user_boost
            elif cat and cat not in user_slugs:
                post *= user_penalty
            else:
                post *= user_uncertain
        if (ch.get("kind") or "").strip() == "table_block":
            ql = (query or "").lower()
            if (
                any(c.isdigit() for c in query)
                or "таблиц" in ql
                or "доз" in ql
                or "мг" in ql
                or "мкг" in ql
                or "мл" in ql
                or "сут" in ql
            ):
                post *= float(os.environ.get("RAG_TABLE_BLOCK_BOOST", "1.14"))
        raw_rows.append((lex, bm25_s, mult, ch, post))
    if not raw_rows:
        return []

    lex_vals = [r[0] for r in raw_rows]
    bm25_vals = [r[1] for r in raw_rows]
    lex_n = _norm_minmax(lex_vals)
    bm25_n = _norm_minmax(bm25_vals)
    scored: list[tuple[float, float, float, float, dict]] = []
    for i, row in enumerate(raw_rows):
        lex, bm25_s, mult, ch, post = row
        ln = lex_n[i]
        bn = bm25_n[i]
        if use_bm25_blend:
            blend = bm25_alpha * ln + (1.0 - bm25_alpha) * bn
        else:
            blend = ln
        final = blend * mult * post
        scored.append((final, lex, bm25_s, mult, ch))
    scored.sort(key=lambda x: -x[0])

    global _retrieval_embed_meta
    _retrieval_embed_meta = {"used": False}

    embed_on = os.environ.get("RAG_GEMINI_EMBED_RERANK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    pool_n = int(os.environ.get("RAG_EMBED_POOL", "44"))
    alpha = float(os.environ.get("RAG_HYBRID_ALPHA", "0.46"))
    emb_model = os.environ.get(
        "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    ).strip()

    work_rows: list[tuple] = scored
    q_embed = query
    if icd_codes_for_lex:
        q_embed = (query + "\n" + " ".join(icd_codes_for_lex)).strip()[:8000]
    ex_emb = _extra_clinical_tokens(rq)
    if ex_emb:
        q_embed = (q_embed + " " + " ".join(sorted(ex_emb))).strip()[:8000]
    if embed_on and api_key and scored:
        pool_n = min(pool_n, len(scored))
        pool_rows = _merge_embed_pool_rows(scored, pool_n, pool_merge)
        try:
            work_rows = _gemini_embed_rerank_pool(q_embed, pool_rows, alpha, emb_model)
            _retrieval_embed_meta = {
                "used": True,
                "model": emb_model,
                "alpha": alpha,
                "pool": len(pool_rows),
            }
        except Exception as e:
            work_rows = scored
            _retrieval_embed_meta = {"used": False, "error": str(e)[:240]}

    per_path: dict[str, int] = {}
    out: list[dict] = []
    rerank_used = bool(_retrieval_embed_meta and _retrieval_embed_meta.get("used"))
    for row in work_rows:
        if len(row) >= 5:
            final, lex, _bm25_s, mult, ch = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            )
        else:
            final, lex, mult, ch = row[0], row[1], row[2], row[3]
        p = ch.get("path") or ""
        if per_path.get(p, 0) >= max_per_path:
            continue
        per_path[p] = per_path.get(p, 0) + 1
        ex_lim = int(os.environ.get("RAG_EXCERPT_CHARS", "700"))
        cat_out = (ch.get("category") or "").strip()
        row_out: dict = {
            "path": p,
            "title": ch.get("title") or "",
            "kind": ch.get("kind") or "general",
            "score": round(final, 3),
            "lexical_score": round(lex, 3),
            "routing_multiplier": round(mult, 4),
            "excerpt": format_excerpt_for_display(ch.get("text") or "", ex_lim),
        }
        if cat_out:
            row_out["category"] = cat_out
        if rerank_used:
            row_out["embedding_rerank"] = True
        out.append(row_out)
        if len(out) >= max_chunks:
            break
    return out


# Большой промпт и вызов модели могут занимать 2–3+ мин; клиент в index.html ждёт дольше сервера
GEMINI_CALL_TIMEOUT = float(os.environ.get("GEMINI_CALL_TIMEOUT", "180"))
GEMINI_SPELLFIX_TIMEOUT = float(os.environ.get("GEMINI_SPELLFIX_TIMEOUT", "45"))
GEMINI_QUERY_REFINE_TIMEOUT = float(os.environ.get("RAG_QUERY_REFINE_TIMEOUT", "45"))


def get_gemini():
    global _model
    if _model is not None:
        return _model
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Задайте переменную окружения GOOGLE_API_KEY",
        )
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Установите: pip install google-generativeai",
        ) from e
    genai.configure(api_key=key)
    name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    _model = genai.GenerativeModel(name)
    return _model


def _extract_gemini_text(resp) -> str:
    """Безопасно: при блокировке/пустом ответе свойство .text бросает ValueError."""
    try:
        t = resp.text
        if t:
            return str(t).strip()
    except (ValueError, AttributeError, TypeError):
        pass
    parts: list[str] = []
    cands = getattr(resp, "candidates", None) or []
    for cand in cands:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts).strip()


def _gemini_finish_reason(resp) -> str | None:
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return None
    fr = getattr(cands[0], "finish_reason", None)
    if fr is None:
        return None
    return str(fr)


def _generate_blocking(model, full_prompt: str):
    import google.generativeai as genai

    max_out = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "16384"))
    use_json = os.environ.get("GEMINI_JSON_MODE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    cfg_kw: dict = {
        "temperature": 0.25,
        "max_output_tokens": max_out,
    }
    if use_json:
        # Снижает обрывы посреди JSON и обрывы «лишнего» текста до/после объекта
        cfg_kw["response_mime_type"] = "application/json"
    return model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(**cfg_kw),
    )


def generate_gemini(model, full_prompt: str):
    """Один поток + таймаут — иначе вызов к API может «висеть» без ответа."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_CALL_TIMEOUT)
        except FuturesTimeout as e:
            raise HTTPException(
                status_code=504,
                detail=f"Таймаут вызова модели ({int(GEMINI_CALL_TIMEOUT)} с). Проверьте сеть или GEMINI_MODEL.",
            ) from e


def _generate_blocking_plain(model, full_prompt: str):
    """Текст без JSON mode — шаблоны заключений и т.п."""
    import google.generativeai as genai

    max_out = int(os.environ.get("GEMINI_TEMPLATE_MAX_TOKENS", "8192"))
    return model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=max_out,
        ),
    )


def generate_gemini_plain(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_plain, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_CALL_TIMEOUT)
        except FuturesTimeout as e:
            raise HTTPException(
                status_code=504,
                detail=f"Таймаут вызова модели ({int(GEMINI_CALL_TIMEOUT)} с).",
            ) from e


def _generate_blocking_spellfix(model, full_prompt: str):
    """Короткий JSON-ответ: исправление опечаток в запросе."""
    import google.generativeai as genai

    return model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )


def generate_gemini_spellfix(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_spellfix, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_SPELLFIX_TIMEOUT)
        except FuturesTimeout:
            return None


def _generate_blocking_query_refine(model, full_prompt: str):
    """JSON: нормализация жалобы под МКБ/протоколы."""
    import google.generativeai as genai

    return model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.15,
            max_output_tokens=3072,
            response_mime_type="application/json",
        ),
    )


def generate_gemini_query_refine(model, full_prompt: str):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_generate_blocking_query_refine, model, full_prompt)
        try:
            return fut.result(timeout=GEMINI_QUERY_REFINE_TIMEOUT)
        except FuturesTimeout:
            return None


def refine_clinical_query_gemini(
    complaint_rag: str, full_query: str, model
) -> tuple[str, dict | None]:
    """Уточнение формулировки жалобы через Gemini для лучшего совпадения с МКБ и RAG."""
    sq = (complaint_rag or "").strip()
    if len(sq) < 3 or len(sq) > 8000:
        return complaint_rag, None
    ctx = (full_query or "").strip()[:4500]
    prompt = (
        SYSTEM_CLINICAL_QUERY_REFINE
        + "\n\n---\n\nТекст жалобы (основной):\n"
        + sq[:6000]
        + "\n\nДополнительный контекст запроса (если есть возраст/пол/шапка — только для согласования формулировок, не выдумывай факты):\n"
        + ctx
    )
    try:
        resp = generate_gemini_query_refine(model, prompt)
        if resp is None:
            return complaint_rag, None
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            return complaint_rag, None
        refined = (parsed.get("refined") or "").strip()
        if not refined or len(refined) > 12000:
            return complaint_rag, None
        # Защита от чрезмерного сжатия/обнуления
        if len(sq) >= 40 and len(refined) < max(12, int(len(sq) * 0.15)):
            return complaint_rag, None
        applied = bool(parsed.get("applied"))
        note = (parsed.get("note") or "").strip()
        if refined == sq and not applied:
            return complaint_rag, None
        if refined == sq:
            applied = False
        meta: dict = {
            "applied": applied,
            "before": sq,
            "after": refined,
        }
        if note:
            meta["note"] = note
        return refined, meta
    except (HTTPException, Exception):
        return complaint_rag, None


def apply_clinical_correction(full_q: str, corrected_rag: str) -> str:
    """Подставляет исправленный клинический текст в полный запрос (контекст + ответы на уточнения сохраняются)."""
    cq = (corrected_rag or "").strip()
    sep = "=== Жалобы и вопрос ==="
    if sep in full_q:
        head = full_q.split(sep, 1)[0] + sep + "\n"
        tail = full_q.split(sep, 1)[1]
        mark = "— Ответы на уточняющие вопросы:"
        if mark in tail:
            return head + cq + "\n\n" + mark + tail.split(mark, 1)[1]
        return head + cq
    return cq


def fix_query_spelling_medical(short_query: str, model) -> tuple[str, bool]:
    """Исправление опечаток для лексического поиска. При сбое API — исходный текст, changed=False."""
    sq = (short_query or "").strip()
    if len(sq) < 2 or len(sq) > 8000:
        return short_query, False
    prompt = SYSTEM_QUERY_SPELLFIX + "\n\nТекст:\n" + sq[:6000]
    try:
        resp = generate_gemini_spellfix(model, prompt)
        if resp is None:
            return short_query, False
        txt = _extract_gemini_text(resp)
        parsed = _try_parse_json(txt)
        if not parsed or not isinstance(parsed, dict):
            return short_query, False
        corrected = (parsed.get("corrected") or "").strip()
        if not corrected:
            return short_query, False
        if corrected == sq:
            return short_query, False
        return corrected, True
    except (HTTPException, Exception):
        return short_query, False


def _try_parse_json(t: str) -> dict | None:
    if not t:
        return None
    s = t.strip()
    if "```" in s:
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.M)
        s = re.sub(r"\s*```\s*$", "", s, flags=re.M)
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_differential_field(parsed: dict | None) -> None:
    """До 5 строк; порядок как у модели (сверху — наиболее вероятное)."""
    if not parsed or not isinstance(parsed, dict):
        return
    d = parsed.get("differential")
    if not isinstance(d, list):
        return
    out: list[str] = []
    for x in d:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        elif isinstance(x, dict):
            t = (x.get("text") or x.get("label") or x.get("diagnosis") or "").strip()
            if t:
                out.append(t)
        if len(out) >= 5:
            break
    parsed["differential"] = out


def _icd_client_payload(icd_analysis: dict) -> dict:
    """Единый JSON для API и llm_json.icd_codes."""
    detected: list[dict] = []
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        detected.append(
            {
                "code": d.get("code"),
                "title_ru": d.get("title_ru"),
                "title_en": d.get("title_en"),
                "role": "detected_in_query",
            }
        )
    suggested: list[dict] = []
    for s in icd_analysis.get("suggested") or []:
        if not isinstance(s, dict):
            continue
        role = (
            "suggested_gemini"
            if s.get("match_method") == "gemini_from_pool"
            else "suggested_lexicon"
        )
        row = {
            "code": s.get("code"),
            "title_ru": s.get("title_ru"),
            "title_en": s.get("title_en"),
            "role": role,
            "score": s.get("score"),
        }
        if role == "suggested_gemini" and s.get("rationale"):
            row["rationale"] = s.get("rationale")
        suggested.append(row)
    du_list: list[dict] = []
    for x in icd_analysis.get("detected_unknown") or []:
        if not isinstance(x, dict):
            continue
        du_list.append(
            {
                "code": x.get("code"),
                "title_ru": x.get("title_ru"),
                "title_en": x.get("title_en"),
            }
        )
    out: dict = {
        "detected": detected,
        "suggested": suggested,
        "codes_for_retrieval": icd_analysis.get("codes_for_retrieval") or [],
        "explicit_icd_in_query": bool(icd_analysis.get("explicit_icd_in_query")),
        "detected_unknown": du_list,
    }
    meta = icd_analysis.get("icd_meta")
    if meta:
        out["meta"] = meta
    return out


def _icd_block_for_prompt(icd_analysis: dict) -> str:
    lines: list[str] = []
    for d in icd_analysis.get("detected_unknown") or []:
        if not isinstance(d, dict):
            continue
        c = d.get("code") or ""
        lines.append(
            f"- {c}: код не найден в справочнике МКБ-10 (Excel→JSON); проверьте написание."
        )
    for d in icd_analysis.get("detected") or []:
        if not isinstance(d, dict):
            continue
        c = d.get("code") or ""
        tr = (d.get("title_ru") or "").strip()
        ten = (d.get("title_en") or "").strip()
        if tr and ten:
            lines.append(f"- {c}: {tr} ({ten})")
        elif tr:
            lines.append(f"- {c}: {tr}")
        elif ten:
            lines.append(f"- {c}: ({ten})")
        else:
            lines.append(f"- {c}")
    for s in icd_analysis.get("suggested") or []:
        if not isinstance(s, dict):
            continue
        c = s.get("code") or ""
        tr = (s.get("title_ru") or "").strip()
        ten = (s.get("title_en") or "").strip()
        sc = s.get("score")
        if s.get("match_method") == "gemini_from_pool":
            rat = (s.get("rationale") or "").strip()
            tail = " [подбор из пула кандидатов]"
            if rat:
                tail = f" [подбор: {rat[:160]}]"
        elif sc is not None:
            tail = f" [лексикон, score={sc}]"
        else:
            tail = " [лексикон]"
        if tr and ten:
            lines.append(f"- {c}: {tr} ({ten}){tail}")
        elif tr:
            lines.append(f"- {c}: {tr}{tail}")
        else:
            lines.append(f"- {c}{tail}")
    if not lines:
        return ""
    return (
        "=== Сопоставление МКБ-10 (автоматически, справочно) ===\n"
        + "Не выдумывай коды вне этого списка. При кратком summary можно упомянуть релевантные коды из списка.\n"
        + "\n".join(lines)
    )


def _diagnostic_mode_summary(icd_payload: dict, retrieved: list[dict]) -> dict:
    explicit = bool(icd_payload.get("explicit_icd_in_query"))
    detected = icd_payload.get("detected") or []
    suggested = icd_payload.get("suggested") or []
    top_score = 0.0
    if retrieved:
        try:
            top_score = float(retrieved[0].get("score") or 0.0)
        except (TypeError, ValueError):
            top_score = 0.0
    top_score = max(0.0, min(1.0, top_score))
    if explicit or detected:
        mode = "diagnosis_or_icd"
        conf = max(0.72, min(0.98, 0.78 + top_score * 0.2))
        notice = (
            "Подбор выполнен с опорой на диагноз/код МКБ-10; "
            "соответствие обычно выше, но всё равно сверяйте с полным текстом протокола."
        )
    elif suggested:
        mode = "symptom_inferred"
        conf = max(0.45, min(0.86, 0.52 + top_score * 0.26))
        notice = (
            "Точный диагноз/код МКБ-10 не указан. Сервис использовал симптомный поиск и "
            "предположительное сопоставление с МКБ-10; результаты ориентировочные."
        )
    else:
        mode = "symptom_only"
        conf = max(0.3, min(0.74, 0.38 + top_score * 0.18))
        notice = (
            "Диагноз/код МКБ-10 не определён. Протоколы подобраны по симптомам; "
            "точность ограничена, рекомендуется уточнить клинические детали."
        )
    return {
        "mode": mode,
        "confidence": round(float(conf), 4),
        "notice": notice,
    }


def _ensure_symptom_followup_questions(parsed: dict | None, diag_mode: str, conf: float) -> None:
    if not parsed or not isinstance(parsed, dict):
        return
    if diag_mode not in ("symptom_inferred", "symptom_only"):
        return
    if conf >= 0.62:
        return
    existing = parsed.get("questions_for_patient")
    questions: list[str] = []
    if isinstance(existing, list):
        for q in existing:
            s = str(q).strip()
            if s:
                questions.append(s)
    extra = [
        "Какова длительность симптомов и динамика ухудшения за последние 24–72 часа?",
        "Есть ли объективные показатели: температура, SpO2, АД, ЧСС или другие измерения?",
        "Какие симптомы тревоги присутствуют сейчас (одышка в покое, боль в груди, выраженная слабость, нарушение сознания)?",
    ]
    seen = set(questions)
    for e in extra:
        if e not in seen:
            questions.append(e)
            seen.add(e)
        if len(questions) >= 4:
            break
    parsed["questions_for_patient"] = questions[:4]


def _normalize_protocol_path_key(p: str) -> str:
    s = (p or "").strip()
    if not s:
        return ""
    try:
        if "%" in s:
            s = unquote(s)
    except Exception:
        pass
    s = s.replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    return s


def _normalize_protocol_title_key(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


def dedupe_protocols_list(protocols: list) -> list:
    """Один path и один title — с максимальным confidence_score (ответ модели без дублей)."""
    if not protocols:
        return []
    by_path: dict[str, dict] = {}
    for pr in protocols:
        if not isinstance(pr, dict):
            continue
        p = _normalize_protocol_path_key(str(pr.get("path") or ""))
        if not p:
            continue
        sc = _confidence_numeric(pr.get("confidence_score")) or 0.0
        prev = by_path.get(p)
        if prev is None:
            by_path[p] = pr
        else:
            psc = _confidence_numeric(prev.get("confidence_score")) or 0.0
            if sc > psc:
                by_path[p] = pr
    merged = list(by_path.values())
    by_title: dict[str, dict] = {}
    for pr in merged:
        tk = _normalize_protocol_title_key(str(pr.get("title") or ""))
        if not tk:
            tk = _normalize_protocol_path_key(str(pr.get("path") or ""))
        sc = _confidence_numeric(pr.get("confidence_score")) or 0.0
        prev = by_title.get(tk)
        if prev is None:
            by_title[tk] = pr
        else:
            psc = _confidence_numeric(prev.get("confidence_score")) or 0.0
            if sc > psc:
                by_title[tk] = pr
    out = list(by_title.values())
    out.sort(
        key=lambda x: -(_confidence_numeric(x.get("confidence_score")) or 0.0)
    )
    return out


def dedupe_parsed_protocols(parsed: dict | None) -> None:
    if not parsed or not isinstance(parsed, dict):
        return
    protos = parsed.get("protocols")
    if not isinstance(protos, list):
        return
    parsed["protocols"] = dedupe_protocols_list(protos)


# --- Выдержки из PDF: склейка переносов, обрезка по границам слов (для UI) ---

_RU_SINGLE_LETTER_WORDS = frozenset(
    "и а в к о с у я ы э ю ё".split()
)

_PDF_HYPHEN_PAIR = re.compile(
    r"([а-яёА-ЯЁa-zA-Z])-\s+([а-яёА-ЯЁa-zA-Z])"
)
_PDF_HYPHEN_NL = re.compile(
    r"([а-яёА-ЯЁa-zA-Z])-\s*\n\s*([а-яёА-ЯЁa-zA-Z])"
)


def _normalize_pdf_hyphenation(text: str) -> str:
    """Склеивает переносы из PDF: «меди- цинской», «Воз- можны» → цельные слова."""
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    for _ in range(24):
        t2 = _PDF_HYPHEN_PAIR.sub(lambda m: m.group(1) + m.group(2), t)
        if t2 == t:
            break
        t = t2
    for _ in range(24):
        t2 = _PDF_HYPHEN_NL.sub(lambda m: m.group(1) + m.group(2), t)
        if t2 == t:
            break
        t = t2
    return t


def _collapse_whitespace_for_excerpt(text: str) -> str:
    """Один блок текста без разрывов строк из верстки PDF."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r" +", " ", t)
    return t.strip()


def _strip_leading_word_fragment(text: str) -> str:
    """Убирает обрезанное первое «слово» (часто 1 буква: «й» от «Настоящий»)."""
    t = text.strip()
    if len(t) < 3:
        return t
    m = re.match(r"^(\S+)(\s+)", t)
    if not m:
        return t
    first = m.group(1)
    if len(first) != 1:
        return t
    if not first.isalpha():
        return t
    if first.lower() in _RU_SINGLE_LETTER_WORDS:
        return t
    return t[m.end() :].lstrip()


def _truncate_excerpt_for_ui(text: str, max_chars: int) -> str:
    """Обрезка по границе слова; без обрыва на середине слова; многоточие при необходимости."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    min_cut = max(24, int(max_chars * 0.5))
    sp = window.rfind(" ")
    if sp >= min_cut:
        window = window[:sp]
    else:
        for sep in (";", ":", ",", "»", ")"):
            ix = window.rfind(sep)
            if ix >= min_cut // 2:
                window = window[: ix + 1]
                break
    window = window.rstrip(" -–—")
    while window.endswith("-") and len(window) > 1:
        window = window[:-1].rstrip()
    if not window:
        window = text[: max_chars - 1].rstrip() + "…"
        return window
    if window[-1] not in ".!?…:;»)]":
        window += "…"
    return window


def format_excerpt_for_display(raw: str, max_chars: int) -> str:
    """Пайплайн для фрагмента КП в ответе API и промпте."""
    t = _normalize_pdf_hyphenation(raw or "")
    t = _collapse_whitespace_for_excerpt(t)
    t = _strip_leading_word_fragment(t)
    return _truncate_excerpt_for_ui(t, max_chars)


def format_structured_index_text(raw: str, max_chars: int) -> str:
    """Текст из structured_index: те же правила, другой лимит."""
    t = _normalize_pdf_hyphenation(raw or "")
    t = _collapse_whitespace_for_excerpt(t)
    t = _strip_leading_word_fragment(t)
    if len(t) <= max_chars:
        return t
    return _truncate_excerpt_for_ui(t, max_chars)


_REDFLAGS_KEYWORDS = (
    "госпитализац",
    "стационар",
    "неотложн",
    "скорой помощ",
    "экстренн",
    "немедленн",
    "угроза жизни",
    "реанимац",
    "орит",
    "интенсивной терапии",
    "показания к госпитал",
    "направлени",
    "жизнеугрожающ",
    "опасн для жизни",
    "срочной медицинской",
)


def _red_flags_from_retrieval(retrieved: list[dict]) -> list[str]:
    """Эвристика по отобранным фрагментам: предложения/строки с маркерами срочности/стационара."""
    if not retrieved:
        return []
    parts: list[str] = []
    for row in retrieved[:6]:
        ex = (row.get("excerpt") or "").strip()
        if not ex:
            continue
        for para in re.split(r"(?<=[.!?])\s+|\n+", ex):
            t = para.strip()
            if len(t) < 30:
                continue
            low = t.lower()
            if any(k in low for k in _REDFLAGS_KEYWORDS):
                parts.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for s in parts:
        s = re.sub(r"\s+", " ", s)
        if len(s) > 240:
            s = s[:237] + "…"
        key = s[:72]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= 5:
            break
    return out


def _protocol_icd_mentions_for_response(protocols: list, *, top_n: int = 5) -> dict[str, list[dict]]:
    """Топ кодов МКБ-10 по числу вхождений в полном тексте structured_index (диагноз/лечение/сводка)."""
    out: dict[str, list[dict]] = {}
    if not _structured_by_path:
        return out
    for pr in protocols:
        if not isinstance(pr, dict):
            continue
        raw = str(pr.get("path") or "").strip()
        if not raw:
            continue
        nk = _normalize_protocol_path_key(raw)
        struct = _structured_by_path.get(raw) or _structured_by_path.get(nk)
        if not struct or not isinstance(struct, dict):
            continue
        parts = [
            str(struct.get("diagnosis") or ""),
            str(struct.get("treatment") or ""),
            str(struct.get("summary") or ""),
        ]
        blob = "\n\n".join(p for p in parts if p.strip()).strip()
        if not blob:
            continue
        rows = count_icd_code_mentions(blob, top_n=top_n)
        if rows:
            out[raw] = rows
    return out


def _finish_hits_max(resp) -> bool:
    fr = (_gemini_finish_reason(resp) or "").upper()
    return "MAX" in fr or "LENGTH" in fr


threading.Thread(
    target=_run_load_data_background,
    daemon=True,
    name="rag-load-chunks",
).start()
app = FastAPI(title="Protocol RAG", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssistIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=12000)
    category_slugs: list[str] = Field(
        default_factory=list,
        description="Рубрики Минздрава (slug), выбранные пользователем — усиливают отбор",
    )


class IcdSuggestIn(BaseModel):
    """Подбор кодов МКБ-10 по жалобам до полного поиска протоколов (шаг 1)."""

    query: str = Field(..., min_length=4, max_length=12000)


class ProtocolDetailIn(BaseModel):
    """Развёрнутая выдержка по протоколу — отдельный запрос (после краткого ответа assist)."""

    query: str = Field(..., min_length=2, max_length=12000)
    path: str = Field(..., min_length=1, max_length=2048)
    title: str = Field(default="", max_length=2000)
    protocol_confidence: float | None = Field(
        default=None,
        description="Оценка соответствия из assist (0–1), для подписи в блоке",
    )
    extract_focus: str | None = Field(
        default=None,
        max_length=32,
        description="Узкий фокус: investigations, medications, treatment_methods, monitoring, algorithms",
    )
    client_rag_support: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="rag_support из assist для предупреждения о слабом отборе",
    )


class ConsultationTemplateIn(BaseModel):
    """Шаблон консультативного заключения по развёрнутой выдержке."""

    query: str = Field(..., min_length=2, max_length=12000)
    clinical_detail: dict = Field(
        ...,
        description="Объект clinical_detail из /api/assist или /api/protocol-detail",
    )
    refine: bool = Field(
        default=False,
        description="Повторная генерация: дополнить черновик по протоколу с учётом подстановок и примечаний",
    )
    previous_template: str | None = Field(
        default=None,
        max_length=120000,
        description="Черновик заключения (после подстановки данных пользователя в плейсхолдеры)",
    )
    additional_notes: str | None = Field(
        default=None,
        max_length=16000,
        description="Дополнительные сведения врача для доработки текста",
    )
    patient_context: str | None = Field(
        default=None,
        max_length=4000,
        description="Возраст, пол и др. из формы — для подстановки в жалобы/анамнез",
    )
    selected_facts_payload: dict | None = Field(
        default=None,
        description="Структурированные выбранные пользователем пункты (sections/items)",
    )


def _normalize_selected_facts_payload(raw: object) -> dict:
    out = {"selected_count": 0, "sections": []}
    if not isinstance(raw, dict):
        return out
    sections = raw.get("sections")
    if not isinstance(sections, list):
        return out
    norm_sections: list[dict] = []
    count = 0
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        title = str(sec.get("title") or "").strip()
        items_raw = sec.get("items")
        if not title or not isinstance(items_raw, list):
            continue
        items = [str(x).strip() for x in items_raw if str(x).strip()]
        if not items:
            continue
        count += len(items)
        norm_sections.append(
            {
                "key": str(sec.get("key") or "").strip(),
                "title": title[:120],
                "items": items[:40],
            }
        )
    out["selected_count"] = count
    out["sections"] = norm_sections
    return out


def _selected_facts_coverage(template_text: str, payload: dict) -> tuple[float, list[str]]:
    txt = (template_text or "").lower()
    if not txt:
        return 0.0, []
    sections = payload.get("sections") if isinstance(payload, dict) else []
    if not isinstance(sections, list) or not sections:
        return 1.0, []
    total = 0
    hit = 0
    missing: list[str] = []
    for sec in sections:
        items = sec.get("items") if isinstance(sec, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            s = str(it).strip()
            if len(s) < 6:
                continue
            total += 1
            toks = [t for t in re.split(r"[\s,;:.()]+", s.lower()) if len(t) >= 5]
            toks = toks[:6]
            ok = False
            for tk in toks:
                if tk in txt:
                    ok = True
                    break
            if ok:
                hit += 1
            else:
                missing.append(s)
    if total <= 0:
        return 1.0, []
    return hit / float(total), missing[:20]


@app.get("/health")
def health() -> dict:
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    icd_ru_n = 0
    try:
        icd_ru_n = len(
            json.loads(
                (ROOT / "data/icd_reference/icd10_ru_mkb10su.json").read_text(encoding="utf-8")
            )
        )
    except (OSError, ValueError, TypeError):
        icd_ru_n = 0
    return {
        "ok": True,
        "rag_ready": _chunks_load_done.is_set(),
        "rag_load_error": _chunks_load_error,
        "chunks": len(_chunks),
        "protocols": len(_protocols_by_path),
        "protocol_meta": len(_protocol_meta),
        "structured_index": len(_structured_by_path),
        "icd_ru_entries": icd_ru_n,
        "gemini_configured": has_key,
        "specialties_count": len(SPECIALTY_LABELS_RU),
        "memory_saver": _memory_saver_enabled(),
        "embedding_rerank": os.environ.get("RAG_GEMINI_EMBED_RERANK", "1"),
        "embedding_model": os.environ.get(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
        ),
    }


@app.get("/api/specialties")
def api_specialties() -> dict:
    """Рубрики каталога клинических протоколов (slug + подпись для формы)."""
    return {
        "specialties": [
            {"slug": s, "label": SPECIALTY_LABELS_RU.get(s, s)}
            for s in sorted(SPECIALTY_LABELS_RU.keys())
        ]
    }


try:
    from gemini_verify import verify_gemini_key as _verify_gemini_key
except ImportError:
    _verify_gemini_key = None


@app.get("/api/verify-key")
def verify_key() -> dict:
    """Один тестовый запрос к модели — проверка ключа из .env."""
    if _verify_gemini_key is None:
        raise HTTPException(
            status_code=501,
            detail="Модуль gemini_verify не найден",
        )
    ok, msg = _verify_gemini_key()
    if not ok:
        raise HTTPException(status_code=502, detail=msg)
    return {
        "ok": True,
        "reply_preview": msg,
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    }


def _infer_icd_pipeline_from_full_query(
    full_query: str,
    model,
) -> tuple[dict | None, str, str, dict | None, str | None]:
    """Та же цепочка МКБ, что в начале api_assist (до retrieve).

    Возвращает:
      icd_analysis, q_эффективный, q_rag, query_clinical_refinement | None, сообщение_об_ошибке | None
    """
    q = (full_query or "").strip()
    q_rag = clinical_query_for_rag(q)
    if not q_rag:
        return (
            None,
            q,
            q_rag,
            None,
            "Пустой текст жалобы — заполните блок «Жалобы и вопрос»",
        )
    query_clinical_refinement: dict | None = None
    if os.environ.get("RAG_GEMINI_QUERY_REFINE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        q_rag_new, rmeta = refine_clinical_query_gemini(q_rag, q, model)
        if rmeta is not None:
            q_rag = q_rag_new
            q = apply_clinical_correction(q, q_rag)
            query_clinical_refinement = rmeta
    icd_analysis = analyze_query_for_icd(q, q_rag)
    pre_icd_infer_on = os.environ.get("RAG_ICD_PRE_RETRIEVE_INFER", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if (
        pre_icd_infer_on
        and not icd_analysis.get("explicit_icd_in_query")
        and not (icd_analysis.get("detected") or [])
    ):
        _refine_icd_analysis_with_gemini(q_rag, icd_analysis, model)
    return icd_analysis, q, q_rag, query_clinical_refinement, None


def _format_icd_append_line(icd_analysis: dict) -> str | None:
    """Строка для добавления в поле запроса перед поиском протокола."""
    codes = icd_analysis.get("codes_for_retrieval") or []
    if not codes:
        return None
    by_code: dict[str, dict] = {}
    for bucket in (icd_analysis.get("detected") or [], icd_analysis.get("suggested") or []):
        for row in bucket:
            if not isinstance(row, dict):
                continue
            c = normalize_icd_code(str(row.get("code") or ""))
            if c:
                by_code[c] = row
    parts: list[str] = []
    for raw in codes[:8]:
        c = normalize_icd_code(str(raw))
        if not c:
            continue
        row = by_code.get(c) or {}
        tr = (row.get("title_ru") or "").strip()
        if tr:
            parts.append(f"{c} ({tr})")
        else:
            parts.append(c)
    if not parts:
        return None
    return "МКБ-10 для поиска протокола: " + "; ".join(parts)


@app.post("/api/assist")
def api_assist(body: AssistIn) -> dict:
    _require_rag_loaded()
    model = get_gemini()
    icd_analysis, q, q_rag, query_clinical_refinement, icd_err = (
        _infer_icd_pipeline_from_full_query(body.query, model)
    )
    if icd_err:
        raise HTTPException(status_code=400, detail=icd_err)
    assert icd_analysis is not None
    icd_codes_for_lex = icd_analysis.get("codes_for_retrieval") or None
    query_specialties = infer_specialties_gemini(q, model)
    user_slugs = [
        s
        for s in (body.category_slugs or [])
        if isinstance(s, str) and s in ALLOWED_SPECIALTY_SLUGS
    ]
    boost_merged = list(dict.fromkeys((query_specialties or []) + user_slugs))
    retrieved = retrieve(
        q_rag,
        routing_query=q,
        category_boost=boost_merged or None,
        user_category_slugs=user_slugs or None,
        icd_codes_for_lex=icd_codes_for_lex,
    )
    query_spelling_correction: dict | None = None
    if not retrieved and os.environ.get("RAG_SPELLFIX_ON_EMPTY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        fixed, changed = fix_query_spelling_medical(q_rag, model)
        if changed and fixed.strip():
            q_llm = apply_clinical_correction(q, fixed)
            retrieved = retrieve(
                fixed,
                routing_query=q_llm,
                category_boost=boost_merged or None,
                user_category_slugs=user_slugs or None,
                icd_codes_for_lex=icd_codes_for_lex,
            )
            if retrieved:
                q = q_llm
                query_spelling_correction = {
                    "applied": True,
                    "rag_query_before": q_rag,
                    "rag_query_after": fixed,
                }

    if not retrieved:
        raise HTTPException(status_code=400, detail="Пустой отбор — уточните запрос")

    retrieved, audience_inferred, audience_fallback = filter_retrieval_by_audience(
        retrieved, q, _routing
    )

    chunk_vote_majority: str | None = None
    if retrieved and os.environ.get("RAG_CHUNK_VOTE_RERETRIEVE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        maj = _majority_category_from_retrieval(retrieved)
        chunk_vote_majority = maj
        if maj and (not boost_merged or maj not in boost_merged):
            boost2 = [maj] + [x for x in (boost_merged or []) if x != maj]
            r2 = retrieve(
                q_rag,
                routing_query=q,
                category_boost=boost2,
                user_category_slugs=user_slugs or None,
                icd_codes_for_lex=icd_codes_for_lex,
            )
            if r2:
                retrieved, audience_inferred, audience_fallback = (
                    filter_retrieval_by_audience(r2, q, _routing)
                )

    maybe_refine_icd_with_gemini_after_retrieve(
        model,
        q_rag,
        icd_analysis,
        retrieved,
    )

    lines = []
    meta_specs: list[str] = []
    for i, r in enumerate(retrieved, 1):
        cat = ""
        p = r["path"]
        if p in _protocols_by_path:
            cat = _protocols_by_path[p].get("category") or ""
        pm = _protocol_meta.get(p)
        if pm and pm.get("specialty_ru"):
            meta_specs.append(pm["specialty_ru"])
        sc = r.get("score")
        lx = r.get("lexical_score")
        rm = r.get("routing_multiplier")
        lines.append(
            f"[{i}] path={p}\n"
            f"рубрика={cat}\n"
            f"тип_фрагмента={r['kind']}\n"
            f"score={sc} lexical_score={lx} routing_multiplier={rm}\n"
            f"текст:\n{r['excerpt']}\n"
        )
    context = "\n---\n".join(lines)

    hint_block = ""
    if meta_specs:
        hint_block = (
            "Справочно рубрики отобранных фрагментов: "
            + ", ".join(sorted(set(meta_specs)))
            + "\n\n"
        )
    icd_block = _icd_block_for_prompt(icd_analysis)
    if icd_block:
        icd_block = icd_block + "\n\n"
    user_block = (
        icd_block
        + hint_block
        + f"Запрос пользователя:\n{q}\n\nФрагменты протоколов:\n{context}\n\n"
        + ASSIST_USER_CONTEXT_GUIDE
    )
    full_prompt = SYSTEM_JSON + "\n\n---\n\n" + user_block
    prompt_limit = int(os.environ.get("GEMINI_PROMPT_MAX_CHARS", "28000"))
    if len(full_prompt) > prompt_limit:
        full_prompt = full_prompt[: prompt_limit - 80] + "\n…[обрезано для лимита контекста]"
    retry_used = False

    def _one_call(prompt: str) -> tuple[object, str, dict | None]:
        try:
            r = generate_gemini(model, prompt)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Модель: {e!s}") from e

        pf = getattr(r, "prompt_feedback", None)
        if pf is not None and getattr(pf, "block_reason", None):
            raise HTTPException(
                status_code=502,
                detail=f"Запрос отклонён моделью: {pf.block_reason}",
            )

        txt = _extract_gemini_text(r)
        if not txt:
            raise HTTPException(
                status_code=502,
                detail="Пустой ответ модели (блокировка контента или сбой). Попробуйте другую формулировку.",
            )
        return r, txt, _try_parse_json(txt)

    try:
        resp, text, parsed = _one_call(full_prompt)
    except HTTPException:
        raise

    finish = _gemini_finish_reason(resp)
    do_retry = os.environ.get("GEMINI_ASSIST_RETRY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if do_retry and (parsed is None or _finish_hits_max(resp)):
        retry_prompt = SYSTEM_JSON_RETRY + "\n\n---\n\n" + user_block
        if len(retry_prompt) > prompt_limit:
            retry_prompt = retry_prompt[: prompt_limit - 80] + "\n…[обрезано]"
        try:
            resp2, text2, parsed2 = _one_call(retry_prompt)
        except HTTPException:
            pass
        else:
            retry_used = True
            resp, text, parsed = resp2, text2, parsed2
            finish = _gemini_finish_reason(resp)

    if parsed and isinstance(parsed, dict):
        apply_protocol_confidence_calibration(parsed, retrieved)
        dedupe_parsed_protocols(parsed)

    icd_payload = _icd_client_payload(icd_analysis)
    diag_mode = _diagnostic_mode_summary(icd_payload, retrieved)
    _ensure_symptom_followup_questions(
        parsed,
        str(diag_mode.get("mode") or ""),
        float(diag_mode.get("confidence") or 0.0),
    )
    if parsed and isinstance(parsed, dict):
        merged_icd: list[dict] = []
        for it in icd_payload.get("detected") or []:
            merged_icd.append(dict(it))
        for it in icd_payload.get("suggested") or []:
            merged_icd.append(dict(it))
        parsed["icd_codes"] = merged_icd

    confidence_second_pass_used = False
    if parsed and isinstance(parsed, dict) and os.environ.get(
        "RAG_CONFIDENCE_SECOND_PASS", "0"
    ).strip().lower() in ("1", "true", "yes"):
        confidence_second_pass_used = bool(
            refine_protocol_confidences_gemini(model, q, parsed, retrieved)
        )

    clinical_detail = None
    clinical_detail_offer: dict | None = None
    if parsed and os.environ.get("GEMINI_EXTRACT_FULL_MATCH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        candidates: list[tuple[float, dict]] = []
        min_detail_rag = float(os.environ.get("RAG_DETAIL_MIN_RAG_SUPPORT", "0.12"))
        for pr in parsed.get("protocols") or []:
            if not confidence_for_detailed_extraction(pr.get("confidence_score")):
                continue
            if float(pr.get("rag_support") or 0.0) < min_detail_rag:
                continue
            raw_p = str(pr.get("path") or "")
            pth = raw_p if raw_p in _chunks_by_path else ""
            if not pth:
                nk = _normalize_protocol_path_key(raw_p)
                if nk in _chunks_by_path:
                    pth = nk
            if not pth:
                continue
            sc = _confidence_numeric(pr.get("confidence_score")) or 0.0
            candidates.append((sc, pr))
        candidates.sort(key=lambda x: -x[0])
        if candidates:
            best_sc, pr = candidates[0]
            pth = pr.get("path") or ""
            score_obj = pr.get("confidence_score")
            if confidence_display_full(score_obj):
                _rs = pr.get("rag_support")
                _rs_f: float | None = None
                if _rs is not None:
                    try:
                        _rs_f = float(_rs)
                    except (TypeError, ValueError):
                        _rs_f = None
                clinical_detail = extract_clinical_detail(
                    pth,
                    q,
                    str(pr.get("title") or ""),
                    model,
                    detailed=True,
                    protocol_confidence=best_sc,
                    client_rag_support=_rs_f,
                )
            else:
                clinical_detail_offer = {
                    "path": pth,
                    "title": str(pr.get("title") or ""),
                    "confidence_score": best_sc,
                    "rag_support": pr.get("rag_support"),
                }

    normalize_differential_field(parsed)

    proto_list = (parsed.get("protocols") or []) if parsed else []
    protocol_icd_mentions = _protocol_icd_mentions_for_response(proto_list, top_n=5)
    red_flags = _red_flags_from_retrieval(retrieved)

    return {
        "query": q,
        "retrieval": retrieved,
        "audience_inferred": audience_inferred,
        "retrieval_audience_fallback": audience_fallback,
        "query_specialties": query_specialties,
        "user_category_slugs": user_slugs,
        "icd": icd_payload,
        "diagnostic_mode": diag_mode.get("mode"),
        "diagnostic_confidence": diag_mode.get("confidence"),
        "diagnostic_notice": diag_mode.get("notice"),
        "llm_text": text,
        "llm_json": parsed,
        "gemini_finish_reason": finish,
        "gemini_retry_used": retry_used,
        "clinical_detail": clinical_detail,
        "clinical_detail_offer": clinical_detail_offer,
        "query_spelling_correction": query_spelling_correction,
        "query_clinical_refinement": query_clinical_refinement,
        "retrieval_embedding": dict(_retrieval_embed_meta)
        if _retrieval_embed_meta
        else {"used": False},
        "red_flags": red_flags,
        "protocol_icd_mentions": protocol_icd_mentions,
        "routing_version": int(_routing.get("version", 1)) if _routing else 1,
        "chunk_vote_majority": chunk_vote_majority,
        "confidence_second_pass_used": confidence_second_pass_used,
    }


@app.post("/api/protocol-detail")
def api_protocol_detail(body: ProtocolDetailIn) -> dict:
    """Развёрнутая выдержка по одному протоколу (второй вызов модели) — по кнопке после краткого ответа."""
    _require_rag_loaded()
    q = body.query.strip()
    pth = body.path.strip()
    if not pth or pth not in _chunks_by_path:
        raise HTTPException(
            status_code=404,
            detail="Протокол не найден в индексе",
        )
    model = get_gemini()
    pc = body.protocol_confidence
    if pc is not None:
        try:
            pc = float(max(0.0, min(1.0, pc)))
        except (TypeError, ValueError):
            pc = None
    clinical_detail = extract_clinical_detail(
        pth,
        q,
        body.title.strip(),
        model,
        detailed=True,
        protocol_confidence=pc,
        extract_focus=body.extract_focus,
        client_rag_support=body.client_rag_support,
    )
    return {"clinical_detail": clinical_detail}


@app.post("/api/icd-suggest")
def api_icd_suggest(body: IcdSuggestIn) -> dict:
    """Та же логика МКБ, что в начале /api/assist, без RAG и без ответа LLM по протоколам."""
    model = get_gemini()
    icd_analysis, q, q_rag, _, err = _infer_icd_pipeline_from_full_query(
        body.query.strip(), model
    )
    if err:
        raise HTTPException(status_code=400, detail=err)
    assert icd_analysis is not None
    payload = _icd_client_payload(icd_analysis)
    append_line = _format_icd_append_line(icd_analysis)
    hint = None
    if not (icd_analysis.get("codes_for_retrieval") or []) and not (
        icd_analysis.get("detected") or []
    ):
        hint = (
            "Коды МКБ-10 по описанию не подобраны — уточните формулировку, рубрику или введите код вручную."
        )
    return {
        "icd": payload,
        "query_effective": q,
        "rag_query": q_rag,
        "append_line": append_line,
        "hint": hint,
    }


@app.post("/api/consultation-template")
def api_consultation_template(body: ConsultationTemplateIn) -> dict:
    """Текстовый шаблон консультативного заключения по выдержке из протокола."""
    cd = body.clinical_detail
    if not isinstance(cd, dict) or cd.get("error"):
        raise HTTPException(
            status_code=400,
            detail="Нет корректной развёрнутой выдержки (clinical_detail)",
        )
    if body.refine and not (body.previous_template or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Для доработки передайте черновик заключения (previous_template).",
        )
    model = get_gemini()
    payload = json.dumps(cd, ensure_ascii=False)
    plim = int(os.environ.get("GEMINI_TEMPLATE_PROMPT_MAX_CHARS", "28000"))
    if len(payload) > plim:
        payload = payload[: plim - 80] + "\n…[обрезано]"
    q = body.query.strip()[:8000]
    selected_payload = _normalize_selected_facts_payload(body.selected_facts_payload)
    selected_payload_json = ""
    if selected_payload.get("sections"):
        selected_payload_json = json.dumps(selected_payload, ensure_ascii=False)
    pctx = ""
    pc = (body.patient_context or "").strip()[:4000]
    if pc:
        pctx = "\n\nКонтекст пациента (из формы пользователя):\n" + pc
    if body.refine:
        draft = (body.previous_template or "").strip()
        dlim = int(os.environ.get("GEMINI_TEMPLATE_DRAFT_MAX_CHARS", "100000"))
        if len(draft) > dlim:
            draft = draft[: dlim - 80] + "\n…[черновик обрезан]"
        notes = (body.additional_notes or "").strip()[:8000]
        notes_block = (
            "\n\nДополнительные сведения от пользователя:\n" + notes
            if notes
            else ""
        )
        full_prompt = (
            SYSTEM_CONSULTATION_REFINE
            + "\n\n---\n\nЗапрос пользователя:\n"
            + q
            + pctx
            + "\n\nЧерновик заключения:\n"
            + draft
            + notes_block
            + "\n\nВыдержка из протокола (JSON):\n"
            + payload
        )
    else:
        notes0 = (body.additional_notes or "").strip()[:8000]
        notes_block0 = (
            "\n\nВыбранные пользователем пункты для включения в заключение (приоритетно отразить в соответствующих разделах):\n"
            + notes0
            if notes0
            else ""
        )
        selected_block = (
            "\n\nselected_facts_payload (структурировано, обязательно отразить):\n"
            + selected_payload_json
            if selected_payload_json
            else ""
        )
        full_prompt = (
            SYSTEM_CONSULTATION_TEMPLATE
            + "\n\n---\n\nЗапрос пользователя:\n"
            + q
            + pctx
            + notes_block0
            + selected_block
            + "\n\nВыдержка (JSON):\n"
            + payload
        )
    try:
        resp = generate_gemini_plain(model, full_prompt)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Модель: {e!s}") from e
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None and getattr(pf, "block_reason", None):
        raise HTTPException(
            status_code=502,
            detail=f"Запрос отклонён моделью: {pf.block_reason}",
        )
    txt = _extract_gemini_text(resp)
    if not txt:
        raise HTTPException(
            status_code=502,
            detail="Пустой ответ модели при формировании шаблона.",
        )
    out: dict = {"template": txt}
    if not body.refine and selected_payload.get("sections"):
        cov, missing = _selected_facts_coverage(txt, selected_payload)
        out["selected_facts_coverage"] = round(float(cov), 4)
        if cov < 0.72 and missing:
            fix_prompt = (
                SYSTEM_CONSULTATION_TEMPLATE
                + "\n\n---\n\nЗапрос пользователя:\n"
                + q
                + pctx
                + "\n\nselected_facts_payload (обязателен к покрытию):\n"
                + selected_payload_json
                + "\n\nПропущенные выбранные пункты (обязательно включить):\n- "
                + "\n- ".join(missing[:12])
                + "\n\nТекущий черновик шаблона:\n"
                + txt[:90000]
                + "\n\nЗадача: верни ПОЛНЫЙ исправленный текст шаблона, включив пропущенные пункты без выдумывания фактов."
                + "\n\nВыдержка (JSON):\n"
                + payload
            )
            try:
                fix_resp = generate_gemini_plain(model, fix_prompt)
                fix_txt = _extract_gemini_text(fix_resp)
                if fix_txt:
                    txt2 = fix_txt.strip()
                    cov2, missing2 = _selected_facts_coverage(txt2, selected_payload)
                    if cov2 >= cov:
                        out["template"] = txt2
                        out["selected_facts_coverage"] = round(float(cov2), 4)
                        out["selected_facts_repair_used"] = True
                        if missing2:
                            out["selected_facts_missing"] = missing2[:8]
                    else:
                        out["selected_facts_repair_used"] = False
                        out["selected_facts_missing"] = missing[:8]
            except Exception:
                out["selected_facts_repair_used"] = False
                out["selected_facts_missing"] = missing[:8]
    return out


# Статика (index.html, protocols.json, PDF) — регистрировать после API-маршрутов.
# Иначе GET / даёт 404 «Not Found» на Render при открытии корня в браузере.
if (ROOT / "index.html").is_file():

    @app.get("/", include_in_schema=False)
    def _serve_index_html() -> FileResponse:
        """Без долгого кэша HTML: после деплоя сразу подхватывается новый JS/разметка."""
        return FileResponse(
            path=str(ROOT / "index.html"),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    app.mount(
        "/",
        StaticFiles(directory=str(ROOT), html=True),
        name="site",
    )
else:

    @app.get("/")
    def root_placeholder() -> dict:
        return {
            "ok": True,
            "service": "Protocol RAG",
            "health": "/health",
            "assist": "POST /api/assist",
            "hint": "В репозитории нет index.html рядом с rag_server.py",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8787)
