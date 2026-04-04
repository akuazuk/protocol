#!/usr/bin/env python3
"""
Строит structured_index.json: для каждого протокола — краткие выдержки
по диагностике и лечению (эвристики по абзацам и заголовкам разделов).

Вход: corpus.json (полный извлечённый текст).
Выход: structured_index.json — для поиска и отображения карточек.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.json"
OUT = ROOT / "structured_index.json"

MAX_SNIP = 650
MIN_PARA = 40

# Ключевые темы для отбора абзацев (нижний регистр)
DIAG_PAT = re.compile(
    r"диагност|критери[ий]\s+диагноз|симптом|синдром|клиническ|обследован|анамнез|осмотр|"
    r"лабораторн|инструментальн|рентген|кт\s|мрт|узи|экг|мкб|классификац|"
    r"дифференциальн|показан|жалоб|статус|оценк|физикальн",
    re.I,
)
TREAT_PAT = re.compile(
    r"лечени|терапи|препарат|назначен|дозиров|операц|хирург|консерватив|"
    r"медикамент|немедикамент|не\s*медикамент|неотложн|помощь|режим|реабилитац|"
    r"профилактик|контрол|наблюден|отмен|коррекци|альтернатив|тактик|"
    r"критери[ий]\s+эффективност|длительност",
    re.I,
)

# Начало «клинической» части (после юридической шапки)
CLINICAL_MARKERS = [
    re.compile(r"клиническ(?:ий|ого)\s+протокол", re.I),
    re.compile(r"глава\s+1", re.I),
    re.compile(r"общие\s+положения", re.I),
]


def clinical_start(text: str) -> int:
    t = text
    best = 0
    for rx in CLINICAL_MARKERS:
        m = rx.search(t)
        if m and m.start() < len(t) * 0.35:
            best = max(best, m.start())
    # если ничего — отрезаем типичную шапку (~12%)
    if best == 0 and len(t) > 800:
        return min(len(t) // 8, 2500)
    return best


def split_paragraphs(text: str) -> list[str]:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    parts = re.split(r"(?:\n\s*\n|\n(?=\d+[\.\)]\s)|(?<=[.!?])\s+(?=[А-ЯЁA-Z]))", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= MIN_PARA:
            out.append(p)
    if len(out) < 3 and len(text) > 400:
        # грубое деление по длине
        step = max(400, len(text) // 12)
        for i in range(0, len(text), step):
            chunk = text[i : i + step].strip()
            if len(chunk) >= MIN_PARA:
                out.append(chunk)
    return out


def score_paragraphs(paragraphs: list[str], pattern: re.Pattern) -> list[tuple[float, str]]:
    scored: list[tuple[float, str]] = []
    for p in paragraphs:
        low = p.lower()
        hits = len(pattern.findall(low))
        # бонус за длину в разумных пределах
        bonus = min(len(p), 1200) / 1200.0
        s = hits * 2 + bonus
        if hits:
            scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    return scored


def best_snippets(paragraphs: list[str], pattern: re.Pattern, n: int = 2) -> str:
    scored = score_paragraphs(paragraphs, pattern)
    if not scored:
        return ""
    take = [scored[i][1] for i in range(min(n, len(scored)))]
    merged = " ".join(take)
    if len(merged) > MAX_SNIP:
        merged = merged[: MAX_SNIP - 1] + "…"
    return merged


def extract_by_headings(text: str) -> tuple[str, str]:
    """Пытается вырезать блоки после заголовков Диагностика / Лечение."""
    low = text.lower()
    diag = ""
    treat = ""

    # номерные разделы: "4. диагностика" или "диагностика."
    m_diag = re.search(
        r"(?:^|\n)\s*(?:\d+[\.)]\s*)?(диагностик[аи][^\n]{0,80})\s*\n(.{200,8000}?)"
        r"(?=(?:^|\n)\s*(?:\d+[\.)]\s*)?(лечени|терапи|хирургическ|неотложн))",
        text,
        re.S | re.I,
    )
    if m_diag:
        diag = m_diag.group(2).strip()
        diag = re.sub(r"\s+", " ", diag)[:MAX_SNIP]
        if len(diag) > MAX_SNIP:
            diag = diag[: MAX_SNIP - 1] + "…"

    m_treat = re.search(
        r"(?:^|\n)\s*(?:\d+[\.)]\s*)?((?:лечени|терапи|хирургическ)[^\n]{0,100})\s*\n(.{200,8000}?)"
        r"(?=(?:^|\n)\s*(?:\d+[\.)]\s*)?(приложени|список литератур|контрол))",
        text,
        re.S | re.I,
    )
    if m_treat:
        treat = m_treat.group(2).strip()
        treat = re.sub(r"\s+", " ", treat)[:MAX_SNIP]
        if len(treat) > MAX_SNIP:
            treat = treat[: MAX_SNIP - 1] + "…"

    return diag, treat


def main() -> None:
    if not CORPUS.is_file():
        raise SystemExit(f"Нет {CORPUS}. Запустите: python3 extract_corpus.py")

    rows = json.loads(CORPUS.read_text(encoding="utf-8"))
    out: list[dict] = []

    for r in rows:
        path = r["path"]
        text = (r.get("text") or "").strip()
        parts = path.split("/")
        cat = parts[1] if len(parts) > 1 else ""
        title = Path(path).stem.replace("_", " ")

        if not text:
            out.append(
                {
                    "path": path,
                    "category": cat,
                    "title": title,
                    "summary": "",
                    "diagnosis": "",
                    "treatment": "",
                    "search_text": f"{title} {cat}",
                }
            )
            continue

        start = clinical_start(text)
        clinical = text[start:] if start else text

        d_head, t_head = extract_by_headings(clinical)
        paras = split_paragraphs(clinical)

        diagnosis = d_head or best_snippets(paras, DIAG_PAT)
        treatment = t_head or best_snippets(paras, TREAT_PAT)

        # если оба пустые — взять среднюю часть как summary для поиска
        summary = clinical[:900].replace("\n", " ")
        if len(summary) > 700:
            summary = summary[:699] + "…"

        search_text = " ".join(
            [
                title,
                cat,
                diagnosis,
                treatment,
                summary[:400],
            ]
        )
        search_text = re.sub(r"\s+", " ", search_text).strip()

        out.append(
            {
                "path": path,
                "category": cat,
                "title": title,
                "summary": summary,
                "diagnosis": diagnosis,
                "treatment": treatment,
                "search_text": search_text,
            }
        )

    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Записано: {len(out)} записей → {OUT}")


if __name__ == "__main__":
    main()
