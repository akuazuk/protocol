"""Кабинетный профиль жёсткости оценок МО (зоны, статусы, risk-caps).

SSOT на диске данных: ``/var/data/medical_exams/config/mo_scoring_profile.json``.
Не коммитится в git; дефолты = текущий standard из YAML.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent

PRESETS: dict[str, dict[str, Any]] = {
    "soft": {
        "label_ru": "Мягкая",
        "zone_bands": {"bad_below": 40.0, "ok_at_or_above": 75.0},
        "status_thresholds": {"good": 70.0, "acceptable": 50.0},
        "risk_caps": {"P0": 50.0, "P1": 70.0},
        "attention_score_below": 60.0,
        "shadow": {"poor_max": 40.0, "critical_max": 25.0},
    },
    "standard": {
        "label_ru": "Стандарт",
        "zone_bands": {"bad_below": 50.0, "ok_at_or_above": 85.0},
        "status_thresholds": {"good": 78.0, "acceptable": 58.0},
        "risk_caps": {"P0": 40.0, "P1": 60.0},
        "attention_score_below": 70.0,
        "shadow": {"poor_max": 45.0, "critical_max": 30.0},
    },
    "strict": {
        "label_ru": "Жёсткая",
        "zone_bands": {"bad_below": 60.0, "ok_at_or_above": 90.0},
        "status_thresholds": {"good": 85.0, "acceptable": 65.0},
        "risk_caps": {"P0": 35.0, "P1": 55.0},
        "attention_score_below": 80.0,
        "shadow": {"poor_max": 50.0, "critical_max": 35.0},
    },
}

_JOB_LOCK = threading.Lock()
_ACTIVE_JOB: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def data_root() -> Path:
    raw = (os.environ.get("MO_DATA_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser()
    var = Path("/var/data/medical_exams")
    if var.is_dir():
        return var
    return ROOT / "data" / "medical_exams"


def profile_path(*, root: Path | None = None) -> Path:
    override = (os.environ.get("MO_SCORING_PROFILE_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    return (root or data_root()) / "config" / "mo_scoring_profile.json"


def job_status_path(*, root: Path | None = None) -> Path:
    return (root or data_root()) / "config" / "mo_recompute_job.json"


def _clip_pct(value: Any, *, default: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed != parsed:
        parsed = default
    return max(lo, min(hi, parsed))


def _normalize_knobs(raw: Mapping[str, Any] | None, *, base: Mapping[str, Any]) -> dict[str, Any]:
    src = dict(base)
    if isinstance(raw, Mapping):
        src.update({k: v for k, v in raw.items() if v is not None})
    zb = src.get("zone_bands") if isinstance(src.get("zone_bands"), Mapping) else {}
    st = src.get("status_thresholds") if isinstance(src.get("status_thresholds"), Mapping) else {}
    rc = src.get("risk_caps") if isinstance(src.get("risk_caps"), Mapping) else {}
    sh = src.get("shadow") if isinstance(src.get("shadow"), Mapping) else {}
    bad = _clip_pct(zb.get("bad_below"), default=50.0)
    ok_at = _clip_pct(zb.get("ok_at_or_above"), default=85.0)
    if ok_at <= bad:
        ok_at = min(100.0, bad + 5.0)
    good = _clip_pct(st.get("good"), default=78.0)
    acc = _clip_pct(st.get("acceptable"), default=58.0)
    if good <= acc:
        good = min(100.0, acc + 5.0)
    return {
        "zone_bands": {"bad_below": bad, "ok_at_or_above": ok_at},
        "status_thresholds": {"good": good, "acceptable": acc},
        "risk_caps": {
            "P0": _clip_pct(rc.get("P0"), default=40.0),
            "P1": _clip_pct(rc.get("P1"), default=60.0),
        },
        "attention_score_below": _clip_pct(src.get("attention_score_below"), default=70.0),
        "shadow": {
            "poor_max": _clip_pct(sh.get("poor_max"), default=45.0),
            "critical_max": _clip_pct(sh.get("critical_max"), default=30.0),
        },
    }


def default_profile() -> dict[str, Any]:
    knobs = _normalize_knobs(PRESETS["standard"], base=PRESETS["standard"])
    return {
        "schema_version": 1,
        "preset": "standard",
        "profile_version": 1,
        "apply_on_next_load": False,
        "last_applied_version": None,
        "last_applied_at": None,
        "updated_at": None,
        "updated_by": None,
        **knobs,
        "presets": {
            key: {
                "label_ru": value["label_ru"],
                **_normalize_knobs(value, base=value),
            }
            for key, value in PRESETS.items()
        },
    }


def _merge_loaded(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    base = default_profile()
    if not isinstance(raw, Mapping):
        return base
    preset = str(raw.get("preset") or "standard").strip().lower()
    if preset not in PRESETS and preset != "custom":
        preset = "standard"
    if preset in PRESETS:
        knobs = _normalize_knobs(raw, base=PRESETS[preset])
    else:
        knobs = _normalize_knobs(raw, base=PRESETS["standard"])
    out = {**base, **knobs}
    out["preset"] = preset
    try:
        out["profile_version"] = max(1, int(raw.get("profile_version") or 1))
    except (TypeError, ValueError):
        out["profile_version"] = 1
    out["apply_on_next_load"] = bool(raw.get("apply_on_next_load"))
    if raw.get("last_applied_version") is not None:
        try:
            out["last_applied_version"] = int(raw["last_applied_version"])
        except (TypeError, ValueError):
            out["last_applied_version"] = None
    if raw.get("last_applied_at"):
        out["last_applied_at"] = str(raw["last_applied_at"])
    if raw.get("updated_at"):
        out["updated_at"] = str(raw["updated_at"])
    if raw.get("updated_by"):
        out["updated_by"] = str(raw["updated_by"])[:80]
    pending = raw.get("pending_recompute")
    if isinstance(pending, Mapping) and (
        (pending.get("date_from") and pending.get("date_to")) or pending.get("whole_range")
    ):
        out["pending_recompute"] = {
            "date_from": str(pending.get("date_from") or "")[:10],
            "date_to": str(pending.get("date_to") or "")[:10],
            "whole_range": bool(pending.get("whole_range")),
            "mode": str(pending.get("mode") or "warehouse_zones"),
            "requested_at": str(pending.get("requested_at") or ""),
            "requested_by": str(pending.get("requested_by") or "")[:80],
        }
    else:
        out["pending_recompute"] = None
    return out


def load_scoring_profile(*, root: Path | None = None) -> dict[str, Any]:
    path = profile_path(root=root)
    if not path.is_file():
        return default_profile()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_profile()
    return _merge_loaded(raw if isinstance(raw, dict) else None)


def _invalidate_caches() -> None:
    """Сброс YAML-кешей (профиль читается каждый раз поверх них)."""
    try:
        from clinical_knowledge.mo_zone_scores import _load_zone_bands_yaml

        _load_zone_bands_yaml.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from clinical_knowledge.kz_evaluation_v4 import _load_v4_config_yaml

        _load_v4_config_yaml.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from clinical_knowledge.kz_deep_eval import _load_deep_config_yaml

        _load_deep_config_yaml.cache_clear()
    except Exception:  # noqa: BLE001
        pass


def save_scoring_profile(
    patch: Mapping[str, Any] | None,
    *,
    actor: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    current = load_scoring_profile(root=root)
    patch = dict(patch or {})
    preset = str(patch.get("preset") or current.get("preset") or "standard").strip().lower()
    if preset in PRESETS and not any(
        key in patch
        for key in ("zone_bands", "status_thresholds", "risk_caps", "attention_score_below", "shadow")
    ):
        merged_src = {**PRESETS[preset], "preset": preset}
    else:
        merged_src = {**current, **patch, "preset": preset if preset in PRESETS or preset == "custom" else "custom"}
        if any(
            key in patch
            for key in ("zone_bands", "status_thresholds", "risk_caps", "attention_score_below", "shadow")
        ):
            merged_src["preset"] = "custom" if preset not in PRESETS else preset
            # if user tweaked knobs while selecting a named preset, mark custom
            if preset in PRESETS:
                knobs_now = _normalize_knobs(merged_src, base=PRESETS[preset])
                knobs_preset = _normalize_knobs(PRESETS[preset], base=PRESETS[preset])
                if knobs_now != knobs_preset:
                    merged_src["preset"] = "custom"
    knobs = _normalize_knobs(merged_src, base=PRESETS.get(str(merged_src.get("preset")), PRESETS["standard"]))
    out = {
        "schema_version": 1,
        "preset": str(merged_src.get("preset") or "custom"),
        "profile_version": int(current.get("profile_version") or 1) + 1,
        "apply_on_next_load": bool(
            patch["apply_on_next_load"]
            if "apply_on_next_load" in patch
            else current.get("apply_on_next_load")
        ),
        "last_applied_version": current.get("last_applied_version"),
        "last_applied_at": current.get("last_applied_at"),
        "pending_recompute": current.get("pending_recompute"),
        "updated_at": _utc_now(),
        "updated_by": (actor or current.get("updated_by") or "")[:80],
        **knobs,
    }
    if "pending_recompute" in patch:
        pending = patch.get("pending_recompute")
        if pending is None:
            out["pending_recompute"] = None
        elif isinstance(pending, Mapping):
            out["pending_recompute"] = {
                "date_from": str(pending.get("date_from") or "")[:10],
                "date_to": str(pending.get("date_to") or "")[:10],
                "whole_range": bool(pending.get("whole_range")),
                "mode": str(pending.get("mode") or "warehouse_zones"),
                "requested_at": str(pending.get("requested_at") or _utc_now()),
                "requested_by": str(pending.get("requested_by") or actor or "")[:80],
            }
    path = profile_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # persist without nested presets catalog (rebuilt on load)
    persist = {k: v for k, v in out.items() if k != "presets"}
    path.write_text(json.dumps(persist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    _invalidate_caches()
    return load_scoring_profile(root=root)


def mark_profile_applied(*, root: Path | None = None, version: int | None = None) -> dict[str, Any]:
    current = load_scoring_profile(root=root)
    persist = {k: v for k, v in current.items() if k != "presets"}
    persist["last_applied_version"] = int(version or current.get("profile_version") or 1)
    persist["last_applied_at"] = _utc_now()
    persist["apply_on_next_load"] = False
    persist["pending_recompute"] = None
    path = profile_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(persist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _invalidate_caches()
    return load_scoring_profile(root=root)


def effective_zone_bands(base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    profile = load_scoring_profile()
    zb = profile.get("zone_bands") or {}
    out = dict(base or {})
    out["bad_below"] = float(zb.get("bad_below") or out.get("bad_below") or 50)
    out["ok_at_or_above"] = float(zb.get("ok_at_or_above") or out.get("ok_at_or_above") or 85)
    return out


def apply_profile_to_v4_config(config: dict[str, Any]) -> dict[str, Any]:
    profile = load_scoring_profile()
    out = deepcopy(config)
    caps = dict(out.get("risk_caps") or {})
    for key, value in (profile.get("risk_caps") or {}).items():
        caps[str(key)] = float(value)
    out["risk_caps"] = caps
    out["scoring_profile_version"] = profile.get("profile_version")
    out["scoring_preset"] = profile.get("preset")
    return out


def apply_profile_to_deep_config(config: dict[str, Any]) -> dict[str, Any]:
    profile = load_scoring_profile()
    out = dict(config)
    st = profile.get("status_thresholds") or {}
    if st.get("good") is not None:
        out["t_good"] = float(st["good"])
    if st.get("acceptable") is not None:
        out["t_acc"] = float(st["acceptable"])
    return out


def shadow_thresholds() -> tuple[float, float]:
    profile = load_scoring_profile()
    sh = profile.get("shadow") or {}
    return float(sh.get("poor_max") or 45.0), float(sh.get("critical_max") or 30.0)


def discover_scored_days(*, root: Path | None = None) -> list[str]:
    base = root or data_root()
    secure = base / "secure_cases"
    days: list[str] = []
    if not secure.is_dir():
        return days
    for year_dir in sorted(secure.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for path in sorted(month_dir.glob("kz_l1_*_cases.jsonl")):
                name = path.name
                # kz_l1_YYYY-MM-DD_cases.jsonl
                mid = name[len("kz_l1_") : -len("_cases.jsonl")]
                if len(mid) == 10 and mid[4] == "-" and mid[7] == "-":
                    days.append(mid)
    return days


def _read_job(*, root: Path | None = None) -> dict[str, Any] | None:
    path = job_status_path(root=root)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_job(job: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    path = job_status_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(job)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return payload


def get_recompute_job(*, root: Path | None = None) -> dict[str, Any] | None:
    global _ACTIVE_JOB
    with _JOB_LOCK:
        disk = _read_job(root=root)
        if _ACTIVE_JOB and (not disk or disk.get("job_id") == _ACTIVE_JOB.get("job_id")):
            return dict(_ACTIVE_JOB)
        return disk


def scoring_config_public(*, root: Path | None = None) -> dict[str, Any]:
    profile = load_scoring_profile(root=root)
    days = discover_scored_days(root=root)
    job = get_recompute_job(root=root)
    return {
        "ok": True,
        "profile": profile,
        "effective": {
            "zone_bands": profile.get("zone_bands"),
            "status_thresholds": profile.get("status_thresholds"),
            "risk_caps": profile.get("risk_caps"),
            "attention_score_below": profile.get("attention_score_below"),
            "shadow": profile.get("shadow"),
            "preset": profile.get("preset"),
            "profile_version": profile.get("profile_version"),
        },
        "available_days": {
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
            "n": len(days),
        },
        "recompute_job": job,
        "notes_ru": [
            "Потолки ограничивают итоговый балл, если замечание уже выставлено. Не меняют правила, какое замечание считать Важно или Критично.",
            "«Пересчитать период/всё» обновляет витрину и полосы зон (Оформление / Диагноз / План) без повторного LLM и без rewrite deep в cases.jsonl.",
            "«Deep-пересчёт периода» заново считает findings/deep и может обновить overall/status в cases, затем пересобирает витрину.",
            "«На следующую загрузку» применит профиль при ночном прогоне или inbound score + recompute.",
        ],
        "recompute_mode": "warehouse_zones",
        "recompute_modes": [
            {"mode": "warehouse_zones", "label_ru": "Витрина и зоны"},
            {"mode": "deep_rescore", "label_ru": "Deep-пересчёт + витрина"},
        ],
    }


def _run_recompute_job(job: dict[str, Any], *, root: Path) -> None:
    global _ACTIVE_JOB
    from scripts.recompute_mo_days import recompute_day

    warehouse = root / "warehouse" / "mo_analytics.sqlite"
    try:
        from clinical_knowledge.mo_daily import initialize_warehouse

        initialize_warehouse(warehouse)
    except Exception:  # noqa: BLE001
        pass

    first = date.fromisoformat(str(job["date_from"])[:10])
    last = date.fromisoformat(str(job["date_to"])[:10])
    if last < first:
        first, last = last, first
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    results: list[dict[str, Any]] = []
    success_n = 0
    mode = str(job.get("mode") or "warehouse_zones").strip().lower()
    deep_modes = {"deep_rescore", "deep", "cases_deep"}
    for index, day in enumerate(days, start=1):
        try:
            if mode in deep_modes:
                from scripts.rescore_mo_deep_days import rescore_day as deep_rescore_day

                deep_item = deep_rescore_day(
                    day, data_root=root, update_primary_score=True
                )
                if deep_item.get("status") not in {"success", "missing_cases"}:
                    # Не гоняем warehouse поверх сорванного deep (например без DDInter).
                    item = {
                        "date": day.isoformat(),
                        "status": deep_item.get("status") or "error",
                        "error": deep_item.get("error"),
                        "deep_rescore": deep_item,
                        "mode": mode,
                    }
                else:
                    item = recompute_day(
                        day, data_root=root, warehouse=warehouse, write_reports=True
                    )
                    item = {
                        **item,
                        "deep_rescore": deep_item,
                        "mode": mode,
                    }
            else:
                item = recompute_day(day, data_root=root, warehouse=warehouse, write_reports=True)
        except Exception as exc:  # noqa: BLE001
            item = {"date": day.isoformat(), "status": "error", "error": f"{type(exc).__name__}: {exc}"[:240]}
        results.append(item)
        if item.get("status") == "success":
            success_n += 1
        with _JOB_LOCK:
            job = {
                **job,
                "status": "running",
                "progress": {"done": index, "total": len(days), "success_n": success_n},
                "updated_at": _utc_now(),
            }
            _ACTIVE_JOB = job
            _write_job(job, root=root)

    mark_profile_applied(root=root)
    with _JOB_LOCK:
        job = {
            **job,
            "status": "done",
            "finished_at": _utc_now(),
            "updated_at": _utc_now(),
            "progress": {"done": len(days), "total": len(days), "success_n": success_n},
            "results_tail": results[-20:],
        }
        _ACTIVE_JOB = job
        _write_job(job, root=root)


def start_recompute(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    whole_range: bool = False,
    actor: str = "",
    root: Path | None = None,
    mode: str = "warehouse_zones",
) -> dict[str, Any]:
    global _ACTIVE_JOB
    base = root or data_root()
    days = discover_scored_days(root=base)
    if whole_range:
        if not days:
            raise ValueError("Нет дней с оценками для пересчёта.")
        date_from, date_to = days[0], days[-1]
    if not date_from or not date_to:
        raise ValueError("Укажите период date_from/date_to или whole_range=true.")
    try:
        first = date.fromisoformat(str(date_from)[:10])
        last = date.fromisoformat(str(date_to)[:10])
    except ValueError as exc:
        raise ValueError("Некорректная дата периода.") from exc
    if last < first:
        first, last = last, first
    span = (last - first).days + 1
    if span > 400:
        raise ValueError("Слишком длинный период (максимум 400 дней).")

    with _JOB_LOCK:
        current = _read_job(root=base) or _ACTIVE_JOB
        if current and current.get("status") in {"queued", "running"}:
            raise RuntimeError("Пересчёт уже выполняется. Дождитесь завершения.")
        job = {
            "job_id": uuid.uuid4().hex[:12],
            "status": "queued",
            "mode": mode or "warehouse_zones",
            "date_from": first.isoformat(),
            "date_to": last.isoformat(),
            "requested_by": (actor or "")[:80],
            "requested_at": _utc_now(),
            "updated_at": _utc_now(),
            "progress": {"done": 0, "total": span, "success_n": 0},
        }
        _ACTIVE_JOB = job
        _write_job(job, root=base)

    thread = threading.Thread(
        target=_run_recompute_job,
        kwargs={"job": job, "root": base},
        name=f"mo-recompute-{job['job_id']}",
        daemon=True,
    )
    thread.start()
    with _JOB_LOCK:
        job = {**job, "status": "running", "updated_at": _utc_now()}
        _ACTIVE_JOB = job
        _write_job(job, root=base)
    return job


def consume_next_load_recompute(*, root: Path | None = None, actor: str = "pipeline") -> dict[str, Any] | None:
    """После штатного recompute пайплайна: досчитать pending-диапазон и снять флаги.

    Если только ``apply_on_next_load`` без pending - считаем, что только что
    пересчитанные дни пайплайна уже увидели новый профиль; штампуем applied.
    Если есть ``pending_recompute`` с датами или whole - запускаем отдельный job.
    """
    base = root or data_root()
    profile = load_scoring_profile(root=base)
    pending = profile.get("pending_recompute") if isinstance(profile.get("pending_recompute"), Mapping) else None
    wants_apply = bool(profile.get("apply_on_next_load") or pending)
    if not wants_apply:
        if profile.get("last_applied_version") != profile.get("profile_version"):
            mark_profile_applied(root=base)
        return None

    date_from = str((pending or {}).get("date_from") or "").strip()[:10]
    date_to = str((pending or {}).get("date_to") or "").strip()[:10]
    whole = bool((pending or {}).get("whole_range")) or (
        bool(profile.get("apply_on_next_load")) and not date_from and not date_to and bool(pending)
    )
    # pending с датами → исторический догон; иначе только stamp (дни пайплайна уже пересчитаны)
    if date_from and date_to:
        try:
            return start_recompute(
                date_from=date_from,
                date_to=date_to,
                whole_range=False,
                actor=actor,
                root=base,
            )
        except RuntimeError:
            mark_profile_applied(root=base)
            return get_recompute_job(root=base)
    if whole or (pending or {}).get("whole_range"):
        try:
            return start_recompute(whole_range=True, actor=actor, root=base)
        except RuntimeError:
            mark_profile_applied(root=base)
            return get_recompute_job(root=base)

    mark_profile_applied(root=base)
    return None
