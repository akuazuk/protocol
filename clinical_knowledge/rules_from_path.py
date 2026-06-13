"""Детерминированные правила по пути PDF (острые/хирургические КП без блока «формулировка диагноза»)."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

# needles в lower(source_path) → condition_id, компоненты диагноза
PATH_CONDITION_TEMPLATES: list[dict[str, Any]] = [
    {
        "needles": ("аппендицит",),
        "condition_id": "acute_appendicitis",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("панкреатит",),
        "condition_id": "acute_pancreatitis",
        "required_components": ["нозология", "форма", "тяжесть"],
    },
    {
        "needles": ("холецистит",),
        "condition_id": "acute_cholecystitis",
        "required_components": ["нозология", "форма", "тяжесть"],
    },
    {
        "needles": ("непроходимост",),
        "condition_id": "intestinal_obstruction",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("инвагинац",),
        "condition_id": "intussusception",
        "required_components": ["нозология", "форма", "осложнения"],
    },
    {
        "needles": ("грыж",),
        "condition_id": "incarcerated_hernia",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("инородн",),
        "condition_id": "foreign_body_gi",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("кровотеч",),
        "condition_id": "gi_bleeding",
        "required_components": ["нозология", "источник", "тяжесть"],
    },
    {
        "needles": ("перфоратив",),
        "condition_id": "perforated_peptic_ulcer",
        "required_components": ["нозология", "локализация", "осложнения"],
    },
    {
        "needles": ("травм", "живот"),
        "condition_id": "abdominal_trauma",
        "required_components": ["нозология", "механизм", "тяжесть"],
        "match_all": True,
    },
    {
        "needles": ("дефекац", "эвакуатор"),
        "condition_id": "defecation_disorder",
        "required_components": ["нозология", "форма", "степень"],
    },
    {
        "needles": ("общехирург",),
        "condition_id": "pediatric_general_surgery",
        "required_components": ["нозология", "форма"],
    },
    {
        "needles": ("целиак",),
        "condition_id": "celiac",
        "required_components": ["нозология", "клиническая форма", "период"],
    },
    {
        "needles": ("прямой_кишки", "доброкач"),
        "condition_id": "rectal_neoplasm",
        "required_components": ["нозология", "локализация", "стадия"],
        "match_all": True,
    },
    {"needles": ("бронхит",), "condition_id": "acute_bronchitis", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("пневмон",), "condition_id": "pneumonia", "required_components": ["нозология", "локализация", "тяжесть"]},
    {"needles": ("астм",), "condition_id": "bronchial_asthma", "required_components": ["нозология", "тяжесть", "контроль"]},
    {"needles": ("хобл",), "condition_id": "copd", "required_components": ["нозология", "стадия", "тяжесть"]},
    {"needles": ("туберкул",), "condition_id": "tuberculosis", "required_components": ["нозология", "локализация", "бактериовыделение"]},
    {"needles": ("инфаркт",), "condition_id": "myocardial_infarction", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("стенокард",), "condition_id": "angina_pectoris", "required_components": ["нозология", "функциональный класс", "стадия"]},
    {"needles": ("аритми",), "condition_id": "cardiac_arrhythmia", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("сердечн", "недостаточ"), "condition_id": "heart_failure", "required_components": ["нозология", "стадия", "функциональный класс"], "match_all": True},
    {"needles": ("гипертон",), "condition_id": "hypertension", "required_components": ["нозология", "стадия", "риск"]},
    {"needles": ("инсульт",), "condition_id": "stroke", "required_components": ["нозология", "локализация", "период"]},
    {"needles": ("эпилепс",), "condition_id": "epilepsy", "required_components": ["нозология", "форма", "контроль"]},
    {"needles": ("рассеян", "склероз"), "condition_id": "multiple_sclerosis", "required_components": ["нозология", "форма", "активность"], "match_all": True},
    {"needles": ("мигрен",), "condition_id": "migraine", "required_components": ["нозология", "частота", "тяжесть"]},
    {"needles": ("карцином",), "condition_id": "carcinoma", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("опухол",), "condition_id": "neoplasm", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("лимфом",), "condition_id": "lymphoma", "required_components": ["нозология", "стадия", "гистология"]},
    {"needles": ("лейкоз",), "condition_id": "leukemia", "required_components": ["нозология", "форма", "стадия"]},
    {"needles": ("диабет",), "condition_id": "diabetes_mellitus", "required_components": ["нозология", "тип", "компенсация"]},
    {"needles": ("сахарн",), "condition_id": "diabetes_mellitus", "required_components": ["нозология", "тип", "компенсация"]},
    {"needles": ("щитовид",), "condition_id": "thyroid_disease", "required_components": ["нозология", "форма", "функция"]},
    {"needles": ("ожирен",), "condition_id": "obesity", "required_components": ["нозология", "степень", "осложнения"]},
    {"needles": ("почечн", "недостат"), "condition_id": "renal_failure", "required_components": ["нозология", "стадия", "тяжесть"], "match_all": True},
    {"needles": ("мочекамен",), "condition_id": "urolithiasis", "required_components": ["нозология", "локализация", "осложнения"]},
    {"needles": ("простат",), "condition_id": "prostate_disease", "required_components": ["нозология", "форма", "стадия"]},
    {"needles": ("артрит",), "condition_id": "arthritis", "required_components": ["нозология", "локализация", "активность"]},
    {"needles": ("остеоартроз",), "condition_id": "osteoarthritis", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("подагр",), "condition_id": "gout", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("перелом",), "condition_id": "fracture", "required_components": ["нозология", "локализация", "тип"]},
    {"needles": ("псориаз",), "condition_id": "psoriasis", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("дерматит",), "condition_id": "dermatitis", "required_components": ["нозология", "форма", "локализация"]},
    {"needles": ("грипп",), "condition_id": "influenza", "required_components": ["нозология", "тяжесть", "осложнения"]},
    {"needles": ("covid",), "condition_id": "covid19", "required_components": ["нозология", "тяжесть", "осложнения"]},
    {"needles": ("катаракт",), "condition_id": "cataract", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("глауком",), "condition_id": "glaucoma", "required_components": ["нозология", "стадия", "риск"]},
    {"needles": ("депресс",), "condition_id": "depression", "required_components": ["нозология", "тяжесть", "эпизод"]},
    {"needles": ("шизофрен",), "condition_id": "schizophrenia", "required_components": ["нозология", "форма", "стадия"]},
    {"needles": ("кариес",), "condition_id": "dental_caries", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("пульпит",), "condition_id": "pulpitis", "required_components": ["нозология", "локализация", "форма"]},
    {"needles": ("анем",), "condition_id": "anemia", "required_components": ["нозология", "тяжесть", "форма"]},
    {"needles": ("беремен",), "condition_id": "pregnancy", "required_components": ["нозология", "срок", "осложнения"]},
    # --- хирургия/ортопедия/сосудистая/торакальная (закрытие непокрытых khirurgiya) ---
    {"needles": ("сколиоз",), "condition_id": "spinal_deformity", "required_components": ["нозология", "локализация", "степень"]},
    {"needles": ("деформаци", "позвоночник"), "condition_id": "spinal_deformity", "required_components": ["нозология", "локализация", "степень"], "match_all": True},
    {"needles": ("позвоночник",), "condition_id": "spine_disease", "required_components": ["нозология", "локализация", "тяжесть"]},
    {"needles": ("эндопротез",), "condition_id": "joint_replacement", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("сустав",), "condition_id": "joint_disease", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("опорно-двигательн",), "condition_id": "musculoskeletal_trauma", "required_components": ["нозология", "локализация", "тип"]},
    {"needles": ("ортопед",), "condition_id": "orthopedic_disease", "required_components": ["нозология", "локализация", "тип"]},
    {"needles": ("остеогенез",), "condition_id": "osteogenesis_imperfecta", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("огнестрельн",), "condition_id": "gunshot_wound", "required_components": ["нозология", "локализация", "тяжесть"]},
    {"needles": ("термическ", "травм"), "condition_id": "burn_injury", "required_components": ["нозология", "площадь", "степень"], "match_all": True},
    {"needles": ("ожог",), "condition_id": "burn_injury", "required_components": ["нозология", "площадь", "степень"]},
    {"needles": ("грудной", "клетк"), "condition_id": "chest_trauma", "required_components": ["нозология", "локализация", "тяжесть"], "match_all": True},
    {"needles": ("аневризм",), "condition_id": "aortic_aneurysm", "required_components": ["нозология", "локализация", "осложнения"]},
    {"needles": ("аортальн", "клапан"), "condition_id": "aortic_stenosis", "required_components": ["нозология", "степень", "тяжесть"], "match_all": True},
    {"needles": ("периферическ", "артери"), "condition_id": "peripheral_artery_disease", "required_components": ["нозология", "локализация", "стадия"], "match_all": True},
    {"needles": ("флеботромбоз",), "condition_id": "phlebothrombosis", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("тромбофлебит",), "condition_id": "superficial_thrombophlebitis", "required_components": ["нозология", "локализация", "срок"]},
    {"needles": ("тромбоз", "глубок"), "condition_id": "deep_vein_thrombosis", "required_components": ["нозология", "локализация", "стадия"], "match_all": True},
    {"needles": ("варикоз",), "condition_id": "varicose_veins", "required_components": ["нозология", "локализация", "осложнения"]},
    {"needles": ("транспозиц", "сосуд"), "condition_id": "great_vessel_transposition", "required_components": ["нозология", "форма", "осложнения"], "match_all": True},
    {"needles": ("абсцесс", "легк"), "condition_id": "lung_abscess", "required_components": ["нозология", "локализация", "осложнения"], "match_all": True},
    {"needles": ("пиоторакс",), "condition_id": "pyothorax", "required_components": ["нозология", "локализация", "тяжесть"]},
    {"needles": ("мягких", "ткан"), "condition_id": "soft_tissue_infection", "required_components": ["нозология", "локализация", "тяжесть"], "match_all": True},
    {"needles": ("эректильн",), "condition_id": "erectile_dysfunction", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("мочевого", "пузыр"), "condition_id": "bladder_dysfunction", "required_components": ["нозология", "форма", "тяжесть"], "match_all": True},
    {"needles": ("атрезией", "пищевод"), "condition_id": "esophageal_atresia", "required_components": ["нозология", "форма", "осложнения"], "match_all": True},
    {"needles": ("челюстно-лицев",), "condition_id": "maxillofacial_disease", "required_components": ["нозология", "локализация", "форма"]},
    {"needles": ("уха_горла_носа",), "condition_id": "ent_protocol", "required_components": ["нозология", "локализация", "форма"]},
    {"needles": ("оториноларинголог",), "condition_id": "ent_protocol", "required_components": ["нозология", "локализация", "форма"]},
    {"needles": ("анестезиолог",), "condition_id": "anesthesiology_protocol", "required_components": ["нозология", "риск", "осложнения"]},
    {"needles": ("злокач",), "condition_id": "oncology_protocol", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("зно_",), "condition_id": "oncology_protocol", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("pulmonolog",), "condition_id": "pulmonology_protocol", "required_components": ["нозология", "форма"]},
    {"needles": ("kardiolog",), "condition_id": "cardiology_protocol", "required_components": ["нозология", "форма", "стадия"]},
    {"needles": ("nevrolog",), "condition_id": "neurology_protocol", "required_components": ["нозология", "форма"]},
    {"needles": ("novoobraz",), "condition_id": "oncology_protocol", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("endokrinolog",), "condition_id": "endocrinology_protocol", "required_components": ["нозология", "форма"]},
    {"needles": ("stomatolog",), "condition_id": "dentistry_protocol", "required_components": ["нозология", "локализация"]},
    {"needles": ("pediatr",), "condition_id": "pediatrics_protocol", "required_components": ["нозология", "форма", "возрастная группа"]},
    {"needles": ("nefrolog",), "condition_id": "nephrology_protocol", "required_components": ["нозология", "стадия", "тяжесть"]},
    {"needles": ("urolog",), "condition_id": "urology_protocol", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("revmatolog",), "condition_id": "rheumatology_protocol", "required_components": ["нозология", "форма", "активность"]},
    {"needles": ("oftalmolog",), "condition_id": "ophthalmology_protocol", "required_components": ["нозология", "локализация", "стадия"]},
    {"needles": ("akusherstvo-ginekolog",), "condition_id": "obgyn_protocol", "required_components": ["нозология", "срок", "осложнения"]},
    {"needles": ("ginekolog",), "condition_id": "obgyn_protocol", "required_components": ["нозология", "срок", "осложнения"]},
    {"needles": ("gematolog",), "condition_id": "hematology_protocol", "required_components": ["нозология", "форма", "стадия"]},
    {"needles": ("infektsion",), "condition_id": "infectious_disease_protocol", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("allergolog",), "condition_id": "allergy_immunology_protocol", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("otorinolaring",), "condition_id": "ent_protocol", "required_components": ["нозология", "локализация", "форма"]},
    {"needles": ("travmatolog",), "condition_id": "traumatology_protocol", "required_components": ["нозология", "локализация", "тип"]},
    {"needles": ("transplant",), "condition_id": "transplant_protocol", "required_components": ["нозология", "стадия", "осложнения"]},
    {"needles": ("palliativ",), "condition_id": "palliative_protocol", "required_components": ["нозология", "стадия", "цели помощи"]},
    {"needles": ("psikhiatr",), "condition_id": "psychiatry_protocol", "required_components": ["нозология", "форма", "тяжесть"]},
    {"needles": ("anesteziolog",), "condition_id": "anesthesiology_protocol", "required_components": ["нозология", "риск", "осложнения"]},
    {"needles": ("dermatovener",), "condition_id": "dermatology_protocol", "required_components": ["нозология", "форма", "локализация"]},
    {"needles": ("bolezni-sistemy-krovoobrashcheniya",), "condition_id": "cardiology_protocol", "required_components": ["нозология", "стадия", "тяжесть"]},
    {"needles": ("perinatal",), "condition_id": "perinatal_protocol", "required_components": ["нозология", "срок", "осложнения"]},
    {"needles": ("immunolog",), "condition_id": "allergy_immunology_protocol", "required_components": ["нозология", "форма", "тяжесть"]},
    # Рубрико-уровневый fallback для хирургии - ловит оставшиеся PDF без специфичного needle (низший приоритет).
    {"needles": ("khirurgiya",), "condition_id": "general_surgery_protocol", "required_components": ["нозология", "локализация", "осложнения"]},
]


def infer_path_condition(source_path: str) -> tuple[str, list[str]] | None:
    low = (source_path or "").lower().replace("\\", "/")
    for tpl in PATH_CONDITION_TEMPLATES:
        needles = tpl.get("needles") or ()
        if tpl.get("match_all"):
            if all(n in low for n in needles):
                return str(tpl["condition_id"]), list(tpl["required_components"])
        elif any(n in low for n in needles):
            return str(tpl["condition_id"]), list(tpl["required_components"])
    return None


def extract_path_rules(
    source_path: str,
    *,
    protocol_id: str,
    rule_id_prefix: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Правила по шаблону пути PDF."""
    inferred = infer_path_condition(source_path)
    if not inferred:
        return {}
    cid, components = inferred
    prefix = (rule_id_prefix + "_") if rule_id_prefix else ""
    rule = {
        "rule_id": f"{prefix}path_{cid}_diagnosis_formula",
        "rule_type": "diagnosis_formula",
        "required_components": components,
        "severity": "warning",
        "description_ru": f"Шаблон по пути КП: полнота диагноза ({cid}).",
        "source": {
            "protocol_id": protocol_id,
            "source_path": source_path.replace("\\", "/"),
            "path_inferred": True,
        },
        "auto_extracted": True,
        "extraction_method": "path_template",
    }
    return {cid: [rule]}


def path_rules_for_uncovered(
    source_paths: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Собрать path-правила для списка PDF без regex-извлечения."""
    merged: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for sp in source_paths:
        pdf_hash = sha256(sp.encode()).hexdigest()[:8]
        protocol_id = f"proto_{pdf_hash}"
        for cid, rules in extract_path_rules(
            sp, protocol_id=protocol_id, rule_id_prefix=pdf_hash
        ).items():
            for rule in rules:
                rid = str(rule.get("rule_id") or "")
                if rid in seen:
                    continue
                seen.add(rid)
                merged.setdefault(cid, []).append(rule)
    return merged
