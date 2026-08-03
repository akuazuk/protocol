"""Нормализация названий ЛС из свободного текста КЗ к канону INN (+ ATC/группа).

Задача (§4.3 ТЗ): имя препарата, как написал врач (рус., бренд, опечатки) →
канон INN (англ.), чтобы работали детекторы DDI (DDInter), high-alert, формуляр,
STOPP/START/Beers.

Стратегия (без внешних зависимостей, RxNorm - опционально с кэшем):
1. Прямой словарь бренд/рус → INN (курируемый + из high_alert/stopp seed).
2. Транслитерация рус → латиница + нечёткий матч к словарю INN из DDInter (1969).
3. (опц.) RxNorm findRxcuiByString для подтверждения/канона (кэш на диске).

Главный риск (§11) - ложный матч. Поэтому возвращаем confidence и method; при
низкой уверенности вызывающий код помечает finding как needs_human, а не факт.
"""
from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DDINTER = ROOT / "data" / "drug_safety" / "ddinter_pairs.json"
_HIGH_ALERT = ROOT / "data" / "drug_safety" / "high_alert.json"
_STOPP = ROOT / "data" / "drug_safety" / "stopp_start_beers.json"
_RXNORM_CACHE = ROOT / "data" / "drug_safety" / "rxnorm_cache.json"

# Курируемый бренд/рус → INN (нерегулярные случаи, которые транслитерация не поймает).
_BRAND_TO_INN: dict[str, str] = {
    "аспирин": "aspirin", "ацетилсалициловая кислота": "aspirin", "кардиомагнил": "aspirin",
    "тромбоасс": "aspirin", "аспикард": "aspirin",
    "ксарелто": "rivaroxaban", "эликвис": "apixaban", "прадакса": "dabigatran",
    "клексан": "enoxaparin", "эниксум": "enoxaparin", "фраксипарин": "nadroparin",
    "варфарекс": "warfarin", "манинил": "glibenclamide", "амарил": "glimepiride",
    "конкор": "bisoprolol", "эгилок": "metoprolol", "беталок": "metoprolol",
    "лозап": "losartan", "лориста": "losartan", "нольпаза": "pantoprazole",
    "омез": "omeprazole", "ультоп": "omeprazole", "нексиум": "esomeprazole",
    "супрастин": "chloropyramine", "тавегил": "clemastine", "зодак": "cetirizine",
    "зиртек": "cetirizine", "эриус": "desloratadine", "кларитин": "loratadine",
    "сумамед": "azithromycin", "клацид": "clarithromycin", "аугментин": "amoxicillin",
    "амоксициллин": "amoxicillin", "амоксиклав": "amoxicillin",
    "флемоксин": "amoxicillin", "таваник": "levofloxacin",
    "ципролет": "ciprofloxacin", "цифран": "ciprofloxacin",
    "кордарон": "amiodarone", "дигоксин": "digoxin", "мотилиум": "domperidone",
    "мексидол": "ethylmethylhydroxypyridine succinate", "актовегин": "actovegin",
    "но-шпа": "drotaverine", "дротаверин": "drotaverine", "спазмалгон": "metamizole",
    "анальгин": "metamizole", "парацетамол": "paracetamol", "панадол": "paracetamol",
    "нурофен": "ibuprofen", "ибуклин": "ibuprofen", "найз": "nimesulide",
    "нимесил": "nimesulide", "аркоксия": "etoricoxib", "мовалис": "meloxicam",
    "вольтарен": "diclofenac", "ортофен": "diclofenac", "кеторол": "ketorolac",
    "кетанов": "ketorolac", "трамал": "tramadol", "лантус": "insulin glargine",
    "левемир": "insulin detemir", "тресиба": "insulin degludec", "хумулин": "insulin",
    "новорапид": "insulin aspart", "апидра": "insulin glulisine",
    "метформин": "metformin", "сиофор": "metformin", "глюкофаж": "metformin",
    "престариум": "perindopril", "энап": "enalapril", "диротон": "lisinopril",
    "амлодипин": "amlodipine", "норваск": "amlodipine", "верошпирон": "spironolactone",
    "фуросемид": "furosemide", "лазикс": "furosemide", "торасемид": "torasemide",
    "аторис": "atorvastatin", "липримар": "atorvastatin", "крестор": "rosuvastatin",
    "розувастатин": "rosuvastatin",
}

# Транслитерация рус → латиница (для матча к INN-словарю DDInter).
_TRANSLIT = [
    # диграфы фарм-транслитерации - до одиночных букв
    ("кс", "x"),
    ("щ", "sch"), ("ш", "sh"), ("ч", "ch"), ("ц", "c"), ("ю", "yu"), ("я", "ya"),
    ("ж", "zh"), ("х", "kh"), ("э", "e"), ("ё", "e"), ("й", "y"),
    ("а", "a"), ("б", "b"), ("в", "v"), ("г", "g"), ("д", "d"), ("е", "e"),
    ("з", "z"), ("и", "i"), ("к", "k"), ("л", "l"), ("м", "m"), ("н", "n"),
    ("о", "o"), ("п", "p"), ("р", "r"), ("с", "s"), ("т", "t"), ("у", "u"),
    ("ф", "f"), ("ы", "y"), ("ь", ""), ("ъ", ""),
]

_STOPWORDS = {
    "таблетки", "таблетка", "табл", "капсулы", "раствор", "мг", "мл", "сут", "сутки",
    "внутрь", "в/в", "в/м", "п/к", "раз", "день", "дней", "курс", "приём", "прием",
    "по", "на", "и", "или", "утром", "вечером", "ночь", "мкг", "ме", "ед",
}
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{3,}")


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("ё", "е").split())


def transliterate(ru: str) -> str:
    s = _norm(ru)
    for a, b in _TRANSLIT:
        s = s.replace(a, b)
    return s


@lru_cache(maxsize=1)
def _inn_vocab() -> tuple[str, ...]:
    if not _DDINTER.is_file():
        return tuple()
    try:
        d = json.loads(_DDINTER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return tuple()
    return tuple(d.get("drugs") or [])


@lru_cache(maxsize=1)
def _overrides() -> dict[str, str]:
    """RU/бренд → INN из курируемого словаря + seed high_alert/stopp."""
    out = dict(_BRAND_TO_INN)
    for path in (_HIGH_ALERT, _STOPP):
        if not path.is_file():
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = d.get("high_alert") or d.get("rules") or []
        for r in rows:
            inn = r.get("inn")
            inn = inn[0] if isinstance(inn, list) and inn else inn
            if not isinstance(inn, str):
                continue
            for ru in r.get("ru") or []:
                out.setdefault(_norm(ru), _norm(inn))
    return {_norm(k): _norm(v) for k, v in out.items()}


def _load_rxnorm_cache() -> dict[str, str]:
    if _RXNORM_CACHE.is_file():
        try:
            return json.loads(_RXNORM_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_rxnorm_cache(cache: dict[str, str]) -> None:
    _RXNORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _RXNORM_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def normalize_drug(name: str, *, use_rxnorm: bool = False) -> dict:
    """name → {surface, inn, confidence(0..1), method}. inn=None если не распознано."""
    surface = (name or "").strip()
    key = _norm(surface)
    if not key:
        return {"surface": surface, "inn": None, "confidence": 0.0, "method": "empty"}

    ov = _overrides()
    if key in ov:
        return {"surface": surface, "inn": ov[key], "confidence": 0.98, "method": "override"}

    vocab = _inn_vocab()
    vocab_set = set(vocab)

    # прямое совпадение с INN (латиница)
    if key in vocab_set:
        return {"surface": surface, "inn": key, "confidence": 0.95, "method": "exact_inn"}

    # транслитерация + нечёткий матч к INN-словарю
    translit = transliterate(key)
    if translit in vocab_set:
        return {"surface": surface, "inn": translit, "confidence": 0.9, "method": "translit_exact"}
    if vocab:
        close = difflib.get_close_matches(translit, vocab, n=1, cutoff=0.86)
        if close:
            ratio = difflib.SequenceMatcher(None, translit, close[0]).ratio()
            return {"surface": surface, "inn": close[0], "confidence": round(ratio, 3),
                    "method": "translit_fuzzy"}

    if use_rxnorm:
        inn = _rxnorm_lookup(translit)
        if inn:
            return {"surface": surface, "inn": inn, "confidence": 0.8, "method": "rxnorm"}

    return {"surface": surface, "inn": None, "confidence": 0.0, "method": "unresolved"}


def _rxnorm_lookup(name: str) -> str | None:
    cache = _load_rxnorm_cache()
    if name in cache:
        return cache[name] or None
    try:
        import ssl
        import urllib.parse
        import urllib.request

        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            ctx = None
        url = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=" + urllib.parse.quote(name)
        req = urllib.request.Request(url, headers={"User-Agent": "protocol-drugnorm/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            data = json.loads(r.read())
        ids = (data.get("idGroup") or {}).get("rxnormId") or []
        inn = name if ids else ""
        cache[name] = inn
        _save_rxnorm_cache(cache)
        return inn or None
    except Exception:  # noqa: BLE001
        return None


def extract_drugs(text: str, *, use_rxnorm: bool = False) -> list[dict]:
    """Из текста лечения/рекомендаций вытащить кандидаты ЛС и нормализовать.

    Возвращает распознанные (inn != None), отсортированные по confidence.
    Мультисловные бренды/INN из словарей матчатся отдельно (напр. «калия хлорид»).
    """
    blob = _norm(text)
    resolved: dict[str, dict] = {}

    # многословные ключи словарей
    for phrase in list(_overrides().keys()):
        if " " in phrase and phrase in blob:
            r = normalize_drug(phrase, use_rxnorm=False)
            if r["inn"]:
                resolved[r["inn"]] = r

    for m in _TOKEN_RE.finditer(text or ""):
        tok = _norm(m.group(0))
        if tok in _STOPWORDS or len(tok) < 4:
            continue
        r = normalize_drug(tok, use_rxnorm=use_rxnorm)
        if r["inn"] and (r["inn"] not in resolved or r["confidence"] > resolved[r["inn"]]["confidence"]):
            resolved[r["inn"]] = r

    return sorted(resolved.values(), key=lambda x: -x["confidence"])


def clear_cache() -> None:
    _inn_vocab.cache_clear()
    _overrides.cache_clear()
