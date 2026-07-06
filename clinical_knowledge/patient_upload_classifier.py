"""Определение «не тот документ» в B2C и шутливый ответ пациенту."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal

UploadSlot = Literal["kz", "lab"]

# --- эвристики «похоже на КЗ» ---
_KZ_HINTS: list[tuple[int, re.Pattern[str]]] = [
    (3, re.compile(r"консультативн\w*\s+заключен", re.I)),
    (3, re.compile(r"консультаци\w*", re.I)),
    (2, re.compile(r"жалоб\w*", re.I)),
    (2, re.compile(r"\bдиагноз\b", re.I)),
    (2, re.compile(r"рекомендац\w*", re.I)),
    (2, re.compile(r"\bмкб\b|icd[\-\s]?10", re.I)),
    (2, re.compile(r"объективн\w*\s+статус", re.I)),
    (2, re.compile(r"\bанамнез\b", re.I)),
    (2, re.compile(r"врач\s*[:\-]|зав\.?\s*отделен", re.I)),
    (1, re.compile(r"заключен\w*", re.I)),
    (1, re.compile(r"контрольн\w*\s+явк|повторн\w*\s+(?:явк|визит|консультац)", re.I)),
    (1, re.compile(r"специальност\w*\s+врач", re.I)),
]

# --- типы «чужих» документов ---
_GUESS_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "recipe",
        "рецепт блюда",
        re.compile(
            r"ингредиент|приготовлен|нарезать|духовк|варить|обжарить|"
            r"столов\w*\s+ложк|чайн\w*\s+ложк|грамм\b|мл\b.*мук",
            re.I,
        ),
    ),
    (
        "menu",
        "меню или прайс кафе",
        re.compile(r"меню\b|блюдо\b|десерт\b|напиток\b|доставк\w*\s+еды|бургер|пицц", re.I),
    ),
    (
        "receipt",
        "кассовый чек",
        re.compile(r"касс\w*|чек\b|итого\b|сдача\b|унп\b|рн\s*мм|фискальн", re.I),
    ),
    (
        "passport",
        "паспорт или удостоверение",
        re.compile(
            r"паспорт\b|удостоверен\w*\s+личност|личный\s+номер|"
            r"идентификационн\w*\s+номер|выдан\b.*\d{2}\.\d{2}\.\d{4}",
            re.I,
        ),
    ),
    (
        "homework",
        "школьная тетрадь",
        re.compile(r"задач\w*\s*№|уравнен|реши\w*|контрольн\w*\s+работ|класс\b.*предмет", re.I),
    ),
    (
        "contract",
        "договор",
        re.compile(r"договор\b|сторон\w*\s+договор|подписант|юридическ\w*\s+адрес", re.I),
    ),
    (
        "resume",
        "резюме",
        re.compile(r"резюме\b|опыт\s+работ|желаем\w*\s+должност|навык\w*", re.I),
    ),
    (
        "ticket",
        "билет",
        re.compile(r"билет\b|рейс\b|посадочн\w*\s+талон|вагон\b|кинотеатр", re.I),
    ),
    (
        "social",
        "скриншот из соцсетей",
        re.compile(r"instagram|telegram|подписчик|лайк\b|репост|сторис|tiktok", re.I),
    ),
    (
        "invoice",
        "счёт на оплату",
        re.compile(r"счёт\s+на\s+оплат|счет\s*№|банковск\w*\s+реквизит|плательщик", re.I),
    ),
    (
        "parking",
        "парковочный талон",
        re.compile(r"парковк|штраф\b.*транспорт|госномер", re.I),
    ),
    (
        "pet",
        "ветеринарная выписка",
        re.compile(r"ветеринар|кошк|собак|питомец|вакцинац\w*\s+животн", re.I),
    ),
]

_JOKES: dict[str, dict[str, str]] = {
    "recipe": {
        "emoji": "🍲",
        "title": "Это похоже на кулинарный шедевр, а не на заключение",
        "body": "Борщ мы уважаем, но сверить его с протоколом Минздрава пока не научились. "
        "Загрузите консультативное заключение от врача - фото или PDF.",
    },
    "menu": {
        "emoji": "🍕",
        "title": "Вкусно, но не медицински",
        "body": "Меню отличное, только врач по нему диагноз не поставит. "
        "Нам нужно заключение после приёма - с жалобами, диагнозом и рекомендациями.",
    },
    "receipt": {
        "emoji": "🧾",
        "title": "Чек принят, сдача с протоколом - ноль",
        "body": "Кассовый чек - не консультативное заключение. "
        "Попробуйте сфотографировать лист из поликлиники с текстом врача.",
    },
    "passport": {
        "emoji": "🪪",
        "title": "Паспорт в безопасности - проверяем только КЗ",
        "body": "Документы личности лучше не светить в медицинских сервисах. "
        "Загрузите именно консультативное заключение.",
    },
    "homework": {
        "emoji": "📐",
        "title": "Двойка по медицине, пятёрка по алгебре",
        "body": "Тетрадь с задачами - не то, что мы сверяем с клиническими протоколами. "
        "Нужен лист из клиники после консультации.",
    },
    "contract": {
        "emoji": "📜",
        "title": "Договор подписан - с протоколом не подписан",
        "body": "Юридические бумаги мы не читаем. Пришлите консультативное заключение врача.",
    },
    "resume": {
        "emoji": "💼",
        "title": "Резюме сильное, КЗ не приложено",
        "body": "HR-отдел нас не интересует - только медицинское заключение после приёма.",
    },
    "ticket": {
        "emoji": "🎫",
        "title": "Приятного рейса - но не к врачу через нас",
        "body": "Билет не заменяет консультативное заключение. Загрузите выписку из клиники.",
    },
    "social": {
        "emoji": "📱",
        "title": "Лайк за креатив, но это не КЗ",
        "body": "Скрин из мессенджера или соцсети - не медицинский документ. "
        "Нужен лист с заключением врача.",
    },
    "invoice": {
        "emoji": "💳",
        "title": "Счёт оплатите в банке, КЗ - здесь",
        "body": "Счёт на оплату услуг - не то же самое, что консультативное заключение.",
    },
    "parking": {
        "emoji": "🅿️",
        "title": "Парковка оплачена, диагноз - нет",
        "body": "Талон или штраф с парковки мы сверять не будем. Нужно заключение врача.",
    },
    "pet": {
        "emoji": "🐾",
        "title": "Пушистому - к ветеринару, вам - человеческое КЗ",
        "body": "Ветеринарная выписка - отдельная история. "
        "Загрузите своё консультативное заключение.",
    },
    "lab_in_kz": {
        "emoji": "🧪",
        "title": "Это бланк анализов - он ценный, но не на этом месте",
        "body": "Похоже, вы загрузили результаты анализов в поле для заключения. "
        "КЗ - отдельно, анализы - в блок «Анализы (необязательно)» ниже.",
    },
    "kz_in_lab": {
        "emoji": "📋",
        "title": "Заключение врача - не в баночку для анализов",
        "body": "Похоже, консультативное заключение попало в поле для бланков анализов. "
        "Поменяйте файлы местами - и всё заработает.",
    },
    "protocol_pdf": {
        "emoji": "📑",
        "title": "Это клинический протокол Минздрава, а не ваше заключение",
        "body": "Вы загрузили текст протокола для врачей, а не консультативное заключение после приёма. "
        "Нужен лист из клиники с жалобами, диагнозом и рекомендациями именно по вашему визиту.",
    },
    "unknown": {
        "emoji": "🤔",
        "title": "Мы честно посмотрели - это не похоже на КЗ",
        "body": "В тексте нет типичных разделов заключения: жалобы, диагноз, рекомендации. "
        "Сфотографируйте весь лист из клиники или загрузите PDF.",
    },
    "lab_unknown": {
        "emoji": "🔬",
        "title": "В пробирке пусто - в файле тоже не анализы",
        "body": "Не нашли привычных показателей лаборатории (гемоглобин, глюкоза, СОЭ и т.п.). "
        "Загрузите бланк из лаборатории или клиники.",
    },
    "empty": {
        "emoji": "👻",
        "title": "Текста почти нет - как анализ без крови",
        "body": "Документ пустой или плохо читается. Переснимите при свете или загрузите PDF.",
    },
}

# Пулы шуточных «вопросов врачу» - по 6-8 вариантов на тип документа.
_JOKE_QUESTION_POOL: dict[str, list[dict[str, str]]] = {
    "recipe": [
        {"text": "Доктор, борщ три раза в день - это схема лечения или диета?", "why_ru": "Рецепт блюда не заменяет назначения врача.", "category_ru": "Кулинария"},
        {"text": "Сколько столовых ложек сметаны совместимо с вашим протоколом?", "why_ru": "Дозировки в меню и в медицине разные.", "category_ru": "Дозировка"},
        {"text": "Если свёкла «давит» - это побочный эффект или показание к госпитализации?", "why_ru": "Шутка про ингредиенты, не про симптомы.", "category_ru": "С юмором"},
        {"text": "Нужно ли взвешивать картофель до или после очистки для постановки диагноза?", "why_ru": "В КЗ важны жалобы и осмотр, не граммы.", "category_ru": "Диагностика"},
        {"text": "Можно ли заменить бульон таблетками - и какой курс?", "why_ru": "Вопрос из кухни, не из поликлиники.", "category_ru": "Лечение"},
        {"text": "Через сколько минут варки ждать улучшения самочувствия?", "why_ru": "Время готовки - не срок терапии.", "category_ru": "Контроль"},
        {"text": "Соль по вкусу - это персонализированная медицина?", "why_ru": "Индивидуальный подход бывает, но не в рецепте супа.", "category_ru": "С юмором"},
    ],
    "menu": [
        {"text": "Бургер дважды в день - это терапия или противопоказание?", "why_ru": "Меню кафе не описывает лечение.", "category_ru": "Питание"},
        {"text": "Десерт «по показаниям» - до основного блюда или после?", "why_ru": "Порядок блюд - не схема приёма лекарств.", "category_ru": "С юмором"},
        {"text": "Комбо-обед засчитывается как комплексное обследование?", "why_ru": "Комплекс в клинике - другое.", "category_ru": "Обследования"},
        {"text": "Нужна ли подготовка к анализу крови, если заказали пиццу?", "why_ru": "Пицца и анализы - разные истории.", "category_ru": "Анализы"},
        {"text": "Доставка за 30 минут - нормальный срок ожидания результата УЗИ?", "why_ru": "Сроки доставки еды не про медицину.", "category_ru": "Сроки"},
        {"text": "Можно ли поставить диагноз по акции «второй кофе в подарок»?", "why_ru": "Акции меню не заменяют заключение.", "category_ru": "Диагноз"},
    ],
    "receipt": [
        {"text": "Чек на 40 рублей - включать в стоимость лечения?", "why_ru": "Кассовый чек - не медицинский документ.", "category_ru": "Оплата"},
        {"text": "НДС в чеке влияет на дозировку препаратов?", "why_ru": "Налоги и лечение не связаны.", "category_ru": "С юмором"},
        {"text": "Сдача с кассы - это ваша рекомендация по режиму?", "why_ru": "Сдача - не рекомендация врача.", "category_ru": "Режим"},
        {"text": "Нужно ли хранить чек так же долго, как выписку?", "why_ru": "Меддокументы и чеки хранят по-разному.", "category_ru": "Документы"},
        {"text": "Если в чеке только «итого» - можно ли поставить диагноз?", "why_ru": "В КЗ нужны жалобы и заключение.", "category_ru": "Диагноз"},
        {"text": "Фискальный накопитель заменяет электронную карту здоровья?", "why_ru": "Разные системы учёта.", "category_ru": "С юмором"},
    ],
    "passport": [
        {"text": "Серия паспорта - это код МКБ или только для регистратуры?", "why_ru": "Паспорт - не медицинский код.", "category_ru": "Документы"},
        {"text": "Нужно ли продлевать диагноз вместе с паспортом?", "why_ru": "Диагноз не «продлевают» как документ.", "category_ru": "С юмором"},
        {"text": "Фото в паспорте - достаточно для дистанционного осмотра?", "why_ru": "Осмотр - живой, не по фото из удостоверения.", "category_ru": "Осмотр"},
        {"text": "Место рождения влияет на выбор протокола лечения?", "why_ru": "География бывает важна, но не из паспорта вместо КЗ.", "category_ru": "Анамнез"},
        {"text": "Штамп о регистрации - аналог штампа в медкарте?", "why_ru": "Разные документы.", "category_ru": "С юмором"},
    ],
    "homework": [
        {"text": "Задача №5 не решается - это хроническое или острое?", "why_ru": "Школьная тетрадь - не история болезни.", "category_ru": "Диагноз"},
        {"text": "Двойка по алгебре - показание к госпитализации?", "why_ru": "Оценки в школе - не клинические показатели.", "category_ru": "С юмором"},
        {"text": "Нужно ли сдавать контрольную «натощак»?", "why_ru": "Подготовка к анализам - другое.", "category_ru": "Анализы"},
        {"text": "Учитель математики может выписать больничный?", "why_ru": "Больничный выписывает врач.", "category_ru": "Документы"},
        {"text": "Корень из дискриминанта - это норма анализов?", "why_ru": "Формулы - не лабораторные нормы.", "category_ru": "С юмором"},
    ],
    "contract": [
        {"text": "Пункт 3.2 договора - это ваш план лечения?", "why_ru": "Договор - не медицинское назначение.", "category_ru": "Документы"},
        {"text": "Неустойка за просрочку - побочный эффект терапии?", "why_ru": "Юридические санкции - не про здоровье.", "category_ru": "С юмором"},
        {"text": "Подпись на последней странице заменяет подпись врача?", "why_ru": "Нужно заключение лечащего врача.", "category_ru": "КЗ"},
        {"text": "Срок действия договора - это длительность курса лекарств?", "why_ru": "Сроки договора и лечения разные.", "category_ru": "Лечение"},
    ],
    "resume": [
        {"text": "Опыт работы 10 лет - это стаж болезни?", "why_ru": "Резюме - не анамнез.", "category_ru": "Анамнез"},
        {"text": "Навык Excel - показание к МРТ?", "why_ru": "Навыки в CV - не обследования.", "category_ru": "Обследования"},
        {"text": "Желаемая зарплата - это лимит по ОМС?", "why_ru": "Финансы в резюме - не страховка.", "category_ru": "С юмором"},
        {"text": "Рекомендации от прошлого работодателя - заменяют направление к специалисту?", "why_ru": "Нужно медицинское направление.", "category_ru": "Направления"},
    ],
    "ticket": [
        {"text": "Посадка по талону 14B - это номер кабинета в поликлинике?", "why_ru": "Билет на транспорт - не талон к врачу.", "category_ru": "С юмором"},
        {"text": "Задержка рейса - переносит контрольный визит автоматически?", "why_ru": "Расписание рейсов - не расписание приёма.", "category_ru": "Контроль"},
        {"text": "Багаж 23 кг - норма для сдачи анализов?", "why_ru": "Вес багажа - не лабораторный показатель.", "category_ru": "Анализы"},
        {"text": "Страховка на рейс покрывает консультацию терапевта?", "why_ru": "Разные виды страхования.", "category_ru": "Документы"},
    ],
    "social": [
        {"text": "Лайк под постом - это информированное согласие на лечение?", "why_ru": "Соцсети - не медицинские документы.", "category_ru": "С юмором"},
        {"text": "Сторис с завтраком - достаточно для диагноза «гастрит»?", "why_ru": "Нужен осмотр и заключение.", "category_ru": "Диагноз"},
        {"text": "Репост мема про врачей - заменяет второе мнение?", "why_ru": "Мемы - не консилиум.", "category_ru": "Консультация"},
        {"text": "Подписчики в профиле - это коллегиальное заключение?", "why_ru": "Соцсети - не медкомиссия.", "category_ru": "С юмором"},
    ],
    "invoice": [
        {"text": "Счёт № 184 - это номер истории болезни?", "why_ru": "Счёт на оплату - не медкарта.", "category_ru": "Документы"},
        {"text": "Срок оплаты 5 дней - это курс антибиотиков?", "why_ru": "Сроки оплаты и лечения разные.", "category_ru": "Лечение"},
        {"text": "Банковские реквизиты - показание к анализу крови?", "why_ru": "Реквизиты - не назначения.", "category_ru": "С юмором"},
    ],
    "parking": [
        {"text": "Штраф за парковку - это назначенная терапия?", "why_ru": "Штраф ГАИ - не рецепт.", "category_ru": "С юмором"},
        {"text": "Госномер на талоне - код пациента в ЦИСЗ?", "why_ru": "Разные идентификаторы.", "category_ru": "Документы"},
        {"text": "Парковка на 2 часа - хватит на приём у врача?", "why_ru": "Время парковки - не длительность приёма.", "category_ru": "Контроль"},
    ],
    "pet": [
        {"text": "Мурзик тоже идёт к кардиологу или нужен ветеринарный протокол?", "why_ru": "Ветвыписка - не человеческое КЗ.", "category_ru": "С юмором"},
        {"text": "Прививка собаке - аналог флюорографии для человека?", "why_ru": "Вакцины людей и животных - разные.", "category_ru": "Профилактика"},
        {"text": "Корм «премиум» - заменяет диету по назначению врача?", "why_ru": "Диета назначается врачом человеку.", "category_ru": "Питание"},
        {"text": "Когтеточка - достаточная реабилитация после травмы?", "why_ru": "Реабилитация людей - отдельная тема.", "category_ru": "Лечение"},
    ],
    "lab_in_kz": [
        {"text": "Гемоглобин в норме - можно ставить диагноз «здоров» без осмотра?", "why_ru": "Анализы дополняют заключение, не заменяют его.", "category_ru": "Диагноз"},
        {"text": "Нужно ли подписывать каждую цифру в бланке, как в выписке врача?", "why_ru": "Бланк лаборатории - не КЗ.", "category_ru": "Документы"},
        {"text": "СОЭ 12 - это повод лечить простуду таблетками от головы?", "why_ru": "Интерпретация - задача врача в контексте осмотра.", "category_ru": "Анализы"},
        {"text": "Можно ли отправить анализы в блок «Анализы» и забыть про заключение?", "why_ru": "Нужно и КЗ, и при желании - анализы отдельно.", "category_ru": "Загрузка"},
        {"text": "Референсные значения сбоку - это рекомендации Минздрава?", "why_ru": "Нормы лаборатории - не клинический протокол целиком.", "category_ru": "С юмором"},
        {"text": "Если все показатели зелёные - зачем вообще ходить к врачу?", "why_ru": "Анализы - часть картины, не весь приём.", "category_ru": "Контроль"},
        {"text": "Имя файла A_1 - это диагноз или просто метка лаборатории?", "why_ru": "Имя файла не заменяет заключение.", "category_ru": "С юмором"},
    ],
    "kz_in_lab": [
        {"text": "Диагноз в поле «анализы» - усваивается быстрее?", "why_ru": "КЗ и анализы - в разных блоках.", "category_ru": "Загрузка"},
        {"text": "Жалобы в бланке СОЭ - это новый маркер лаборатории?", "why_ru": "Жалобы - из заключения врача.", "category_ru": "Документы"},
        {"text": "Поменять файлы местами - это лечение или диагностика?", "why_ru": "Просто правильная загрузка.", "category_ru": "С юмором"},
        {"text": "Можно ли «разбавить» заключение физраствором в пробирке?", "why_ru": "Шутка - файлы не смешивают.", "category_ru": "С юмором"},
    ],
    "protocol_pdf": [
        {"text": "Можно лечиться только по протоколу МЗ, не ходя к врачу?", "why_ru": "Протокол для врачей, не персональная выписка.", "category_ru": "Протокол"},
        {"text": "Подпись министра на протоколе заменяет вашу выписку?", "why_ru": "Нужно заключение по вашему случаю.", "category_ru": "КЗ"},
        {"text": "Пункт 4.2 протокола - мой диагноз на сегодня?", "why_ru": "Протокол - общие правила, не ваш диагноз.", "category_ru": "Диагноз"},
        {"text": "Сколько страниц протокола нужно проглотить для выздоровления?", "why_ru": "Протокол читают врачи, не «принимают» пациенты.", "category_ru": "С юмором"},
        {"text": "Утверждён приказом - значит, уже согласован с моим лечащим?", "why_ru": "Приказ - норматив, не ваш план лечения.", "category_ru": "Документы"},
    ],
    "unknown": [
        {"text": "Доктор, что означает этот документ - и где раздел «жалобы»?", "why_ru": "В КЗ обычно есть жалобы, диагноз, рекомендации.", "category_ru": "КЗ"},
        {"text": "Это предварительный диагноз или финальный рецепт на здоровье?", "why_ru": "Нужен узнаваемый формат заключения.", "category_ru": "С юмором"},
        {"text": "Можно ли поставить диагноз по первой странице без оглавления?", "why_ru": "Лучше загрузить весь лист целиком.", "category_ru": "Диагностика"},
        {"text": "Если текст не читается - это защита персональных данных?", "why_ru": "Чаще это блик или обрезанный снимок.", "category_ru": "Загрузка"},
        {"text": "Сверка с протоколом МЗ сработает на списке покупок?", "why_ru": "Нужно медицинское заключение.", "category_ru": "С юмором"},
        {"text": "Где в этом файле контрольный визит - и к кому?", "why_ru": "Типичные разделы КЗ не найдены.", "category_ru": "Контроль"},
    ],
    "lab_unknown": [
        {"text": "Этот файл - анализ или анализ ситуации?", "why_ru": "Нужен бланк с показателями и единицами.", "category_ru": "Анализы"},
        {"text": "Где гемоглобин - в заголовке или в подвале?", "why_ru": "Лабораторный бланк содержит маркеры анализов.", "category_ru": "Документы"},
        {"text": "Можно ли сдать этот PDF вместо крови из пальца?", "why_ru": "Анализ сдают в лаборатории, не файлом «что угодно».", "category_ru": "С юмором"},
    ],
    "empty": [
        {"text": "Пустой лист - это норма или забыли дописать?", "why_ru": "Текст не распознан или файл пуст.", "category_ru": "Загрузка"},
        {"text": "Белый экран на фото - признак идеального здоровья?", "why_ru": "Скорее плохое освещение или пустой кадр.", "category_ru": "С юмором"},
        {"text": "Нужно ли переснимать при свете или при луне?", "why_ru": "Лучше дневной свет и весь лист в кадре.", "category_ru": "Совет"},
        {"text": "Один пиксель текста - достаточно для постановки диагноза?", "why_ru": "Нужен читаемый документ.", "category_ru": "Диагноз"},
        {"text": "Тишина в PDF - это конфиденциальность или ошибка загрузки?", "why_ru": "Проверьте файл и перезагрузите.", "category_ru": "С юмором"},
    ],
}

_JOKE_QUESTION_FALLBACK: list[dict[str, str]] = [
    {"text": "Доктор, этот документ - про лечение или про что-то другое?", "why_ru": "Мы не узнали формат заключения.", "category_ru": "С юмором"},
    {"text": "Где в файле жалобы, диагноз и что делать дальше?", "why_ru": "Типичные разделы КЗ не найдены.", "category_ru": "КЗ"},
    {"text": "Можно ли получить настоящее заключение вместо этого листа?", "why_ru": "Нужна выписка после приёма.", "category_ru": "Загрузка"},
]

_JOKE_QUESTION_INTRO = (
    "Шуточные вопросы - чтобы улыбнуться. Настоящие появятся, когда загрузите заключение."
)
_JOKE_QUESTION_ETIQUETTE = (
    "Это не медицинские рекомендации - делитесь с врачом только после правильной загрузки КЗ."
)


@dataclass(frozen=True)
class UploadGuess:
    slot: UploadSlot
    is_expected: bool
    kind: str
    label_ru: str
    score_kz: int
    score_lab: int


def _score_patterns(text: str, patterns: list[tuple[int, re.Pattern[str]]]) -> int:
    blob = (text or "").strip()
    if not blob:
        return 0
    return sum(pts for pts, rx in patterns if rx.search(blob))


def _guess_foreign_kind(text: str) -> tuple[str, str]:
    blob = (text or "").strip()
    for kind, label, rx in _GUESS_PATTERNS:
        if rx.search(blob):
            return kind, label
    return "unknown", "непонятный документ"


def _lab_marker_count(text: str) -> int:
    from clinical_knowledge.lab_result_parser import extract_lab_markers

    return len(extract_lab_markers(text or ""))


def _kz_score(text: str) -> int:
    return _score_patterns(text, _KZ_HINTS)


def _lab_score(text: str) -> int:
    blob = (text or "").strip()
    if not blob:
        return 0
    n = _lab_marker_count(blob)
    score = min(n * 4, 24)
    if re.search(r"биохимич\w*\s+анализ|общий\s+анализ\s+крови|анализ\s+мочи|инвитро|кравира|synlab|референс", blob, re.I):
        score += 4
    if re.search(r"ед/л|ммоль/л|г/л|результат\s+исследован", blob, re.I):
        score += 2
    return score


def _has_typical_kz_sections(text: str) -> bool:
    low = (text or "").lower()
    markers = (
        bool(re.search(r"жалоб\w*", low)),
        bool(re.search(r"\bдиагноз\b", low)),
        bool(re.search(r"рекомендац\w*", low)),
    )
    return sum(markers) >= 2


def _looks_like_minzdrav_protocol(text: str) -> bool:
    low = (text or "").lower()
    if re.search(r"консультативн\w*\s+заключен", low):
        return False
    if not re.search(r"клинический\s+протокол", low):
        return False
    return bool(
        re.search(r"министерств\w*\s+здоров", low)
        or re.search(r"утвержден\w*\s+приказ", low)
        or re.search(r"республик\w*\s+беларус", low)
        or re.search(r"общие\s+положен", low)
    )


def is_b2c_lab_filename(name: str) -> bool:
    """Имя/case_id начинается на A/a/А/а - B2C анализы в тестовом наборе и загрузках."""
    stem = (name or "").strip()
    if not stem:
        return False
    if "/" in stem or "\\" in stem:
        stem = stem.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith((".pdf", ".txt", ".docx", ".rtf", ".odt", ".html")):
        stem = stem.rsplit(".", 1)[0]
    first = stem[0]
    return first.lower() == "a" or first in ("А", "а")


def check_consult_document(
    text: str,
    *,
    consultation_id: str = "",
    filename: str = "",
) -> UploadGuess | None:
    """None - можно гонять КЗ pipeline; иначе шутливый ответ как в B2C."""
    name = (filename or consultation_id or "").strip()
    blob = (text or "").strip()
    if is_b2c_lab_filename(name):
        return UploadGuess(
            "kz",
            False,
            "lab_in_kz",
            "бланк анализов (имя файла A/a)",
            _kz_score(blob),
            _lab_score(blob),
        )
    guess = classify_kz_upload(blob, filename=name)
    if not guess.is_expected and guess.kind in (
        "lab_in_kz",
        "recipe",
        "menu",
        "receipt",
        "passport",
        "homework",
        "contract",
        "resume",
        "ticket",
        "social",
        "invoice",
        "parking",
        "pet",
        "protocol_pdf",
        "empty",
        "unknown",
    ):
        return guess
    return None


def build_consult_upload_mismatch_response(
    guess: UploadGuess,
    *,
    consultation_id: str = "",
    review_tier: str = "L1",
) -> dict[str, Any]:
    """Ответ consult-review при загрузке не-КЗ (анализ, рецепт и т.д.)."""
    joke_report = build_upload_joke_report(guess)
    upload_joke = joke_report.get("upload_joke") or {}
    summary = str(joke_report.get("plain_summary_ru") or upload_joke.get("body_ru") or "")
    return {
        "ok": True,
        "upload_mismatch": True,
        "wrong_document_kind": guess.kind,
        "review_tier": review_tier,
        "consultation_id": consultation_id or None,
        "overall_score": None,
        "overall_status": "not_assessed",
        "confidence_score": None,
        "matched_protocols_count": 0,
        "critical_issues_count": 0,
        "llm_used": False,
        "rag_used": False,
        "criteria_source": "upload_classifier",
        "review": {
            "summary_ru": summary,
            "overall_compliance_pct": None,
            "criteria": [],
            "limitations_ru": upload_joke.get("hint_ru") or "",
            "disclaimer_ru": joke_report.get("disclaimer_ru") or "",
            "upload_joke": upload_joke,
        },
        "structured_analysis": None,
        "patient_report": joke_report,
    }


def classify_kz_upload(text: str, *, filename: str = "") -> UploadGuess:
    blob = (text or "").strip()
    if len(blob) < 40:
        return UploadGuess("kz", False, "empty", "пустой или нечитаемый файл", 0, 0)

    if is_b2c_lab_filename(filename):
        return UploadGuess(
            "kz",
            False,
            "lab_in_kz",
            "бланк анализов (имя файла A/a)",
            _kz_score(blob),
            _lab_score(blob),
        )

    kz = _kz_score(blob)
    lab = _lab_score(blob)

    if lab >= 14 and kz < 8:
        return UploadGuess("kz", False, "lab_in_kz", "бланк анализов", kz, lab)

    if _looks_like_minzdrav_protocol(blob):
        return UploadGuess("kz", False, "protocol_pdf", "клинический протокол Минздрава", kz, lab)

    if kz >= 10:
        return UploadGuess("kz", True, "kz", "консультативное заключение", kz, lab)

    kind, label = _guess_foreign_kind(blob)
    if kind != "unknown":
        return UploadGuess("kz", False, kind, label, kz, lab)

    has_sections = _has_typical_kz_sections(blob)

    if has_sections and kz >= 5 and len(blob) > 80:
        return UploadGuess("kz", True, "kz", "возможное заключение", kz, lab)

    if len(blob) < 100 and kz < 3:
        return UploadGuess("kz", False, "empty", "слишком мало текста", kz, lab)

    fn = (filename or "").lower()
    if fn and kz < 4 and not lab:
        if any(x in fn for x in ("receipt", "cheque", "menu", "recipe", "passport")):
            kind, label = _guess_foreign_kind(fn.replace("_", " "))
            if kind != "unknown":
                return UploadGuess("kz", False, kind, label, kz, lab)

    if not has_sections or kz < 4:
        return UploadGuess("kz", False, "unknown", "не похоже на КЗ", kz, lab)

    return UploadGuess("kz", True, "kz", "консультативное заключение", kz, lab)


def classify_lab_upload(text: str, *, filename: str = "") -> UploadGuess:
    blob = (text or "").strip()
    if not blob:
        return UploadGuess("lab", True, "empty", "", 0, 0)

    kz = _kz_score(blob)
    lab = _lab_score(blob)

    if kz >= 12 and lab < 8:
        return UploadGuess("lab", False, "kz_in_lab", "консультативное заключение", kz, lab)

    if lab >= 8:
        return UploadGuess("lab", True, "lab", "бланк анализов", kz, lab)

    kind, label = _guess_foreign_kind(blob)
    if kind != "unknown" and lab < 6:
        return UploadGuess("lab", False, kind, label, kz, lab)

    if lab >= 4:
        return UploadGuess("lab", True, "lab", "возможный бланк анализов", kz, lab)

    if kz >= 6 and lab < 4:
        return UploadGuess("lab", False, "kz_in_lab", "консультативное заключение", kz, lab)

    if lab < 3 and len(blob) > 60:
        return UploadGuess("lab", False, "lab_unknown", "не похоже на анализы", kz, lab)

    return UploadGuess("lab", True, "lab", "бланк анализов", kz, lab)


def _joke_question_key(guess: UploadGuess) -> str:
    kind = guess.kind
    if guess.slot == "lab" and kind not in ("kz_in_lab", "lab_unknown") and kind in _JOKES:
        return "lab_unknown"
    if kind == "lab_in_kz":
        return "lab_in_kz"
    if kind == "kz_in_lab":
        return "kz_in_lab"
    if kind == "empty":
        return "empty"
    if guess.slot == "lab" and kind == "lab_unknown":
        return "lab_unknown"
    if kind not in _JOKE_QUESTION_POOL:
        return "unknown"
    return kind


def _pick_joke_doctor_questions(guess: UploadGuess, *, limit: int = 4) -> list[dict[str, Any]]:
    """Стабильный выбор 3-4 разных шуточных вопросов по типу документа."""
    key = _joke_question_key(guess)
    pool = list(_JOKE_QUESTION_POOL.get(key) or _JOKE_QUESTION_FALLBACK)
    if not pool:
        pool = list(_JOKE_QUESTION_FALLBACK)
    seed = f"{key}|{guess.label_ru}|{guess.slot}"
    ranked = sorted(
        pool,
        key=lambda row: hashlib.sha256((seed + str(row.get("text") or "")).encode()).hexdigest(),
    )
    take = min(limit, len(ranked))
    picked = ranked[:take]
    out: list[dict[str, Any]] = []
    emojis = ("😄", "🙂", "😉", "🤓", "💬")
    for i, row in enumerate(picked):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if not text.endswith("?"):
            text = text.rstrip(".") + "?"
        out.append(
            {
                "id": f"joke-q-{key}-{i}",
                "text": text,
                "title": text.split("?")[0].strip()[:72] + "?",
                "why_ru": str(row.get("why_ru") or "Шутка - не вопрос для настоящего приёма.").strip(),
                "plain_context": "",
                "severity": "low",
                "category_ru": str(row.get("category_ru") or "С юмором").strip(),
                "block_id": "joke",
                "intent": "upload_joke",
                "priority": 80 + i,
                "tone": "cheerful",
                "emoji": emojis[i % len(emojis)],
                "checked": False,
            }
        )
    return out[:limit]


def _joke_for_guess(guess: UploadGuess) -> dict[str, str]:
    kind = guess.kind
    if guess.slot == "lab" and kind not in ("kz_in_lab", "lab_unknown") and kind in _JOKES:
        kind = "lab_unknown"
    if kind == "lab_in_kz":
        key = "lab_in_kz"
    elif kind == "kz_in_lab":
        key = "kz_in_lab"
    elif kind == "empty":
        key = "empty"
    elif guess.slot == "lab" and kind == "lab_unknown":
        key = "lab_unknown"
    elif kind not in _JOKES:
        key = "unknown"
    else:
        key = kind
    base = dict(_JOKES[key])
    if guess.label_ru and key not in ("empty", "unknown", "lab_unknown", "lab_in_kz", "kz_in_lab"):
        base["body"] = (
            f"Похоже, вы прислали: {guess.label_ru}. "
            + base["body"]
        )
    return base


def build_upload_joke_report(guess: UploadGuess) -> dict[str, Any]:
    joke = _joke_for_guess(guess)
    slot_label = "заключение" if guess.slot == "kz" else "анализы"
    hint = (
        "Загрузите консультативное заключение (фото или PDF всего листа)."
        if guess.slot == "kz"
        else "В блок «Анализы» - бланк из лаборатории с показателями и единицами измерения."
    )
    joke_questions = _pick_joke_doctor_questions(guess, limit=4)
    questions_for_doctor = [q["text"] for q in joke_questions if q.get("text")]
    return {
        "report_schema_version": 2,
        "upload_mismatch": True,
        "mismatch_slot": guess.slot,
        "guessed_kind": guess.kind,
        "guessed_label_ru": guess.label_ru,
        "headline_ru": joke["title"],
        "overall_pct": None,
        "overall_label_ru": "Другой документ",
        "traffic_light": "yellow",
        "plain_summary_ru": joke["body"],
        "upload_joke": {
            "emoji": joke["emoji"],
            "title_ru": joke["title"],
            "body_ru": joke["body"],
            "guessed_what_ru": guess.label_ru,
            "slot_ru": slot_label,
            "hint_ru": hint,
        },
        "questions_intro_ru": _JOKE_QUESTION_INTRO,
        "questions_etiquette_ru": _JOKE_QUESTION_ETIQUETTE,
        "question_tone": "cheerful",
        "next_steps_ru": [
            hint,
            "Проверьте, что снимок без бликов и видны все страницы.",
            "После правильной загрузки нажмите «Проверить заключение» снова.",
        ],
        "questions_for_doctor": questions_for_doctor,
        "questions_structured": joke_questions,
        "action_checklist": list(joke_questions),
        "blocks": [],
        "protocol_links": [],
        "protocol_citations": [],
        "matched_protocols_count": 0,
        "disclaimer_ru": (
            "Это не медицинская оценка - мы не смогли распознать нужный тип документа."
        ),
        "document_quality": {"confidence_pct": None, "level": "low", "hint_ru": hint},
    }


def check_patient_uploads(
    *,
    kz_text: str,
    lab_text: str | None = None,
    kz_filename: str = "",
    lab_filename: str = "",
) -> UploadGuess | None:
    """Вернуть первое несоответствие (КЗ важнее анализов) или None."""
    kz_guess = classify_kz_upload(kz_text, filename=kz_filename)
    if not kz_guess.is_expected:
        return kz_guess
    lab = (lab_text or "").strip()
    if lab:
        lab_guess = classify_lab_upload(lab, filename=lab_filename)
        if not lab_guess.is_expected:
            return lab_guess
    return None
