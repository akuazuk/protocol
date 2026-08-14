"""HTTP-клиент Refbank с throttle, жёстким timeout и retry."""
from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


BASE = "https://www.rceth.by"
SEARCH_URL = f"{BASE}/Refbank/reestr_lekarstvennih_sredstv/results"
HOME_URL = f"{BASE}/Refbank/"


class RefbankClient:
    def __init__(
        self,
        *,
        throttle_sec: float = 0.6,
        insecure_ssl: bool = False,
        timeout: float = 30.0,
        retries: int = 3,
        user_agent: str = "ProtocolRcethSync/1.0 (+internal)",
    ) -> None:
        self.throttle_sec = max(0.0, float(throttle_sec))
        self.timeout = max(5.0, float(timeout))
        self.retries = max(1, int(retries))
        self._last = 0.0
        ctx = ssl._create_unverified_context() if insecure_ssl else ssl.create_default_context()
        self._cj = CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx),
            urllib.request.HTTPCookieProcessor(self._cj),
        )
        self._opener.addheaders = [("User-Agent", user_agent)]

    def _sleep(self) -> None:
        if self.throttle_sec <= 0:
            return
        gap = self.throttle_sec - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)

    def request(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        method: str | None = None,
        max_read: int | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        hdrs = {"Referer": HOME_URL, "Origin": BASE}
        if headers:
            hdrs.update(headers)
        if data is not None and "Content-Type" not in hdrs:
            hdrs["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"

        last_exc: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            self._sleep()
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            try:
                resp = self._opener.open(req, timeout=self.timeout)
                try:
                    if max_read is not None:
                        body = resp.read(max(0, int(max_read)))
                    else:
                        body = resp.read()
                    code = getattr(resp, "status", 200) or 200
                    rh = {k.lower(): v for k, v in resp.headers.items()}
                finally:
                    try:
                        resp.close()
                    except Exception:
                        pass
                self._last = time.monotonic()
                return int(code), body, rh
            except urllib.error.HTTPError as e:
                body = e.read() if e.fp else b""
                self._last = time.monotonic()
                # 503 / 429 - retry
                if int(e.code) in {429, 503} and attempt < self.retries:
                    time.sleep(min(12.0, attempt * 2.0))
                    last_exc = e
                    continue
                return int(e.code), body, {k.lower(): v for k, v in (e.headers or {}).items()}
            except (TimeoutError, urllib.error.URLError, OSError) as e:
                last_exc = e
                self._last = time.monotonic()
                if attempt >= self.retries:
                    break
                time.sleep(min(10.0, attempt * 1.5))
        raise TimeoutError(f"rceth request failed after {self.retries} tries: {url} ({last_exc})")

    def get_text(self, url: str) -> tuple[int, str]:
        code, body, _ = self.request(url if url.startswith("http") else BASE + url)
        return code, body.decode("utf-8", errors="replace")

    def get_bytes(
        self,
        url: str,
        *,
        max_read: int | None = None,
        range_bytes: tuple[int, int] | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        full = url if url.startswith("http") else BASE + url
        headers: dict[str, str] = {}
        if range_bytes is not None:
            start, end = range_bytes
            headers["Range"] = f"bytes={int(start)}-{int(end)}"
        return self.request(full, headers=headers or None, max_read=max_read)

    def ensure_session(self) -> None:
        self.get_text(HOME_URL)

    def post_search(self, pairs: list[tuple[str, str]]) -> tuple[int, str]:
        body = urllib.parse.urlencode(pairs).encode("utf-8")
        code, raw, _ = self.request(SEARCH_URL, data=body)
        return code, raw.decode("utf-8", errors="replace")


def active_search_pairs(name_val: str, name_crit: str = "Start") -> list[tuple[str, str]]:
    """Форма поиска: только действующие (без expired/annul/pause)."""
    pairs: list[tuple[str, str]] = [
        ("QueryStringFind", ""),
        ("IsPostBack", ""),
        ("PropSubmit", ""),
        ("ValueSubmit", ""),
        ("VFiles", ""),
        ("FProps[0].IsText", "True"),
        ("FProps[0].Name", "N_LP"),
        ("FProps[0].CritElems[0].Num", "1"),
        ("FProps[0].CritElems[0].Val", name_val),
        ("FProps[0].CritElems[0].Crit", name_crit),
        ("FProps[0].CritElems[0].Excl", "false"),
        ("FProps[1].IsText", "True"),
        ("FProps[1].Name", "N_MP"),
        ("FProps[1].CritElems[0].Num", "1"),
        ("FProps[1].CritElems[0].Val", ""),
        ("FProps[1].CritElems[0].Crit", "Like"),
        ("FProps[1].CritElems[0].Excl", "false"),
        ("FProps[2].IsDrop", "True"),
        ("FProps[2].Name", "isVersionP"),
        ("FProps[2].CritElems[0].Num", "1"),
        ("FProps[2].CritElems[0].Val", ""),
        ("FProps[2].CritElems[0].Excl", "false"),
        ("FProps[3].IsText", "True"),
        ("FProps[3].Name", "N_FR"),
        ("FProps[3].CritElems[0].Num", "1"),
        ("FProps[3].CritElems[0].Val", ""),
        ("FProps[3].CritElems[0].Crit", "Like"),
        ("FProps[3].CritElems[0].Excl", "false"),
        ("FProps[4].IsText", "True"),
        ("FProps[4].Name", "N_FV"),
        ("FProps[4].CritElems[0].Num", "1"),
        ("FProps[4].CritElems[0].Val", ""),
        ("FProps[4].CritElems[0].Crit", "Like"),
        ("FProps[4].CritElems[0].Excl", "false"),
        ("FProps[5].IsText", "True"),
        ("FProps[5].Name", "Company_Declarant"),
        ("FProps[5].CritElems[0].Num", "1"),
        ("FProps[5].CritElems[0].Val", ""),
        ("FProps[5].CritElems[0].Crit", "Like"),
        ("FProps[5].CritElems[0].Excl", "false"),
        ("FProps[6].IsText", "True"),
        ("FProps[6].Name", "NREG"),
        ("FProps[6].CritElems[0].Num", "1"),
        ("FProps[6].CritElems[0].Val", ""),
        ("FProps[6].CritElems[0].Crit", "Like"),
        ("FProps[6].CritElems[0].Excl", "false"),
        ("FProps[7].IsDate", "True"),
        ("FProps[7].Name", "Data"),
        ("FProps[7].CritElemsD.Val1", ""),
        ("FProps[7].CritElemsD.Crit", "Equal"),
        ("FProps[7].CritElemsD.Val2", ""),
        ("FProps[8].IsDate", "True"),
        ("FProps[8].Name", "TERM"),
        ("FProps[8].CritElemsD.Val1", ""),
        ("FProps[8].CritElemsD.Crit", "Equal"),
        ("FProps[8].CritElemsD.Val2", ""),
        ("FOpt.VUnTerm", "false"),
        ("FOpt.VAn", "false"),
        ("FOpt.VPause", "false"),
        ("FOpt.VFiles", "true"),
        ("FOpt.VFiles", "false"),
        ("FOpt.VEField1", "true"),
        ("FOpt.VEField1", "false"),
        ("FOpt.VPPV", "false"),
        ("FOpt.VDiscountinued", "true"),
        ("FOpt.VDiscountinued", "false"),
    ]
    return pairs


def page_pairs_from_html(html: str, page_n: int) -> list[tuple[str, str]]:
    """Пагинация Refbank: QueryStringFind + IsPostBack=true + FOpt_PageN.

    site.js: при клике по `.page-view a` ставит IsPostBack='true',
    PropSubmit=id (FOpt_PageN), ValueSubmit=propval. Без IsPostBack=true
    сервер игнорирует смену страницы и отдаёт ту же выдачу.
    """
    import re

    qsf = ""
    m = re.search(r'id="QueryStringFind"[^>]*value="([^"]*)"', html)
    if m:
        qsf = m.group(1)
    if not qsf:
        raise ValueError("QueryStringFind missing; cannot paginate without prior search state")
    return [
        ("QueryStringFind", qsf),
        ("IsPostBack", "true"),
        ("PropSubmit", "FOpt_PageN"),
        ("ValueSubmit", str(int(page_n))),
    ]
