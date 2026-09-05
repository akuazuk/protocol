"""Экранирование в UI врача: инвариант, а не разовая проверка.

Аудит перед промышленной эксплуатацией искал XSS через innerHTML и не нашёл:
все 144 присваивания собирают разметку из уже экранированных частей, подсветка
экранирует текст до вставки <mark>, а ссылки уходят через encodeURIComponent.

Файл на 34 тысячи строк правят часто, и одна интерполяция клинического текста
мимо escapeHtml вернёт дыру. Поэтому инвариант закреплён тестом: он падает на
новом небезопасном месте, а не ждёт ручного аудита.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "frontend" / "web" / "doctor" / "index.html"

# Обёртки, после которых значение безопасно вставлять в разметку.
SAFE_WRAPPERS = (
    "escapeHtml",
    "escapeHtmlAttr",
    "encodeURIComponent",
    "encodeURI",
    # Экранируют текст перед добавлением <mark> - проверяется отдельным тестом ниже.
    "algoHighlight",
    "highlightPlainTextHtml",
    # Локальные псевдонимы escapeHtml внутри двух панелей методиста;
    # их достаточность проверяет test_local_esc_aliases_escape_all_four.
    "esc(",
    # Числовые приведения: результат не содержит < и ".
    "Number(",
    "parseInt",
    "parseFloat",
    "toFixed",
    "String(rank)",
)

# Имена, которые по смыслу числовые или собраны из уже экранированных частей.
SAFE_NAMES = re.compile(
    r"^("
    # индексы и счётчики
    r"i|j|ci|ti|tq|bi|idx|index|n|no|cnt|count|len|pct|num|cols|colspan"
    r"|score|total|step|rank|page|p"
    # уже собранная разметка из экранированных частей
    r"|line|html|body|rows|link|liCls|sumLab|tocHtml|ctxHtml"
    r")$"
    # любое обращение к .length - число (re.match якорит начало, поэтому .*)
    r"|.*\.length$",
    re.IGNORECASE,
)


def _ui_source() -> str:
    assert UI.is_file(), f"не найден UI: {UI}"
    return UI.read_text(encoding="utf-8")


def find_unescaped_html_interpolations(src: str) -> list[tuple[int, str, str]]:
    """Интерполяции в строку с разметкой без безопасной обёртки."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(src.split("\n"), start=1):
        if "<" not in line:
            continue
        if any(w in line for w in SAFE_WRAPPERS):
            continue
        for m in re.finditer(r"\$\{([^{}]+)\}", line):
            expr = m.group(1).strip()
            if SAFE_NAMES.match(expr):
                continue
            out.append((lineno, "template", expr))
        for m in re.finditer(
            r"""["'][^"']*<[^"']*["']\s*\+\s*([A-Za-z_$][\w.$\[\]]*)""", line
        ):
            expr = m.group(1)
            if SAFE_NAMES.match(expr):
                continue
            out.append((lineno, "concat", expr))
    return out


def test_no_unescaped_interpolation_into_markup() -> None:
    findings = find_unescaped_html_interpolations(_ui_source())
    detail = "\n".join(f"  строка {ln} [{kind}]: {expr}" for ln, kind, expr in findings)
    assert not findings, (
        "Значение вставляется в разметку без экранирования.\n"
        "Оберните в escapeHtml (текст), escapeHtmlAttr (атрибут) или "
        "encodeURIComponent (URL):\n" + detail
    )


def test_guard_detects_an_injected_violation() -> None:
    """Проверка самого сторожа: без неё он мог бы просто ничего не находить."""
    bad = 'html += "<li>" + patientComment + "</li>";'
    assert find_unescaped_html_interpolations(bad), "сторож не увидел явную дыру"

    bad_template = "el.innerHTML = `<b>${doctorNote}</b>`;"
    assert find_unescaped_html_interpolations(bad_template), "сторож пропустил шаблонную дыру"

    good = 'html += "<li>" + escapeHtml(patientComment) + "</li>";'
    assert not find_unescaped_html_interpolations(good), "ложное срабатывание на escapeHtml"


@pytest.mark.parametrize(
    "func",
    ["escapeHtml", "escapeHtmlAttr", "algoHighlight", "highlightPlainTextHtml"],
)
def test_escaping_helpers_exist(func: str) -> None:
    assert re.search(rf"function {func}\s*\(", _ui_source()), f"пропал хелпер {func}"


def _function_body(src: str, func: str, *, lines: int = 30) -> str:
    """Окно строк после объявления функции.

    Ищем именно окном, а не по закрывающей скобке: в этом файле вложенность
    разная, и привязка к отступу закрывающей скобки уже давала ложный провал.
    """
    all_lines = src.split("\n")
    for idx, line in enumerate(all_lines):
        if re.search(rf"function {func}\s*\(", line):
            return "\n".join(all_lines[idx : idx + lines])
    return ""


def test_highlight_helpers_escape_before_adding_markup() -> None:
    """Подсветка добавляет <mark>, поэтому обязана экранировать текст первой.

    Если escapeHtml уедет ниже вставки разметки, клинический текст с тегами
    попадёт в документ как разметка.
    """
    src = _ui_source()
    for func in ("algoHighlight", "highlightPlainTextHtml"):
        text = _function_body(src, func)
        assert text, f"не удалось прочитать тело {func}"
        esc_at = text.find("escapeHtml")
        mark_at = text.find("<mark")
        assert esc_at >= 0, f"{func} не экранирует вход"
        if mark_at >= 0:
            assert esc_at < mark_at, f"{func} вставляет разметку до экранирования"


def test_local_esc_aliases_escape_all_four() -> None:
    """Сторож доверяет локальным esc(), значит их достаточность нужно проверить.

    Панели методиста определяют свой esc вместо общего escapeHtml. Пока он
    закрывает & < > ", доверие оправдано; иначе таблицы методиста станут
    точкой инъекции.
    """
    src = _ui_source()
    positions = [m.start() for m in re.finditer(r"function esc\s*\(", src)]
    assert positions, "локальный esc пропал - обновите SAFE_WRAPPERS"

    all_lines = src.split("\n")
    for pos in positions:
        start = src[:pos].count("\n")
        text = "\n".join(all_lines[start : start + 10])
        for pattern, name in (("/&/g", "&"), ("/</g", "<"), ("/>/g", ">"), ('/"/g', '"')):
            assert pattern in text, f"локальный esc не экранирует {name}"


def test_attr_escaper_covers_quote_and_angle() -> None:
    """Достаточность escapeHtmlAttr: без кавычки возможен выход из атрибута."""
    text = _function_body(_ui_source(), "escapeHtmlAttr", lines=10)
    assert text, "не удалось прочитать escapeHtmlAttr"
    assert '/"/g' in text or '/"/' in text, "escapeHtmlAttr не экранирует двойную кавычку"
    assert "/</g" in text or "/</" in text, "escapeHtmlAttr не экранирует <"
    assert "/&/g" in text or "/&/" in text, "escapeHtmlAttr не экранирует &"


class _FieldNameChecker(HTMLParser):
    """Поля без доступного имени.

    Имя даёт либо aria-label/aria-labelledby, либо <label for=id>, либо
    обёртка <label>...</label>. Регулярка обёртку не видит и даёт 30 ложных
    срабатываний на этом файле, поэтому нужен разбор с учётом вложенности.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.label_depth = 0
        self.label_for: set[str] = set()
        self.unnamed: list[str] = []
        self._pending: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "label":
            self.label_depth += 1
            if a.get("for"):
                self.label_for.add(str(a["for"]))
        elif tag in ("input", "select", "textarea"):
            if a.get("type") == "hidden":
                return
            if a.get("aria-label") or a.get("aria-labelledby"):
                return
            if self.label_depth > 0:
                return
            self._pending.append((tag, a))

    def handle_endtag(self, tag: str) -> None:
        if tag == "label" and self.label_depth > 0:
            self.label_depth -= 1

    def finish(self) -> list[str]:
        for tag, a in self._pending:
            ident = a.get("id")
            if ident and str(ident) in self.label_for:
                continue
            desc = " ".join(f'{k}="{v}"' for k, v in a.items() if k in ("id", "type", "name"))
            self.unnamed.append(f"<{tag} {desc}>")
        return self.unnamed


def test_every_form_field_has_accessible_name() -> None:
    """Без доступного имени поле не читается экранным диктором.

    Для врача, который заполняет КЗ с клавиатуры, это разница между работой и
    угадыванием, какое поле сейчас в фокусе.
    """
    checker = _FieldNameChecker()
    checker.feed(_ui_source())
    unnamed = checker.finish()
    assert not unnamed, "поля без доступного имени:\n  " + "\n  ".join(unnamed)


def test_page_declares_language_and_viewport() -> None:
    src = _ui_source()
    assert re.search(r'<html[^>]*lang="ru"', src), "не объявлен язык страницы"
    assert 'name="viewport"' in src, "нет viewport - страница непригодна на телефоне"


def test_resource_href_blocks_dangerous_schemes() -> None:
    """safeResourceHref - единственный путь для ссылок из данных."""
    text = _function_body(_ui_source(), "safeResourceHref", lines=25)
    assert text, "не удалось прочитать safeResourceHref"
    assert "javascript" in text and "data" in text, "не отсечены javascript:/data: схемы"
    assert ".." in text, "не отсечён обход каталогов"
    assert "encodeURI" in text, "путь не кодируется"
