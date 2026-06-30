"""Тесты целостности базы знаний онкориска (data/onco_risk/*.yaml)."""
from __future__ import annotations

import pytest

from clinical_knowledge import onco_risk as orisk

FORBIDDEN = ["рак", "онколог", "злокачествен", "опухол", "метастаз", "карцином", "саркома", "меланом"]


def _all_features():
    d = orisk._lr_data()
    return list(d.get("features") or []) + list(d.get("lab_features") or [])


def test_every_feature_has_source_and_signal():
    for f in _all_features():
        assert f.get("id"), f
        assert f.get("source"), f"no source: {f.get('id')}"
        assert f.get("lr") or f.get("ppv_single"), f"no lr/ppv: {f.get('id')}"
        assert f.get("keywords"), f"no keywords: {f.get('id')}"


def test_feature_sites_exist_in_priors():
    prior_sites = set((orisk._priors().get("sites") or {}).keys())
    for f in _all_features():
        site = f.get("cancer_site")
        if site and site != "unknown":
            assert site in prior_sites, f"site '{site}' missing in priors ({f.get('id')})"


def test_priors_baselines_in_range():
    for site, cfg in (orisk._priors().get("sites") or {}).items():
        b = cfg.get("baseline_symptomatic")
        assert b is not None, f"no baseline: {site}"
        assert 0 < float(b) < 0.5, f"baseline out of range: {site}={b}"


def test_thresholds_present():
    th = orisk._thresholds()
    assert abs(float(th.get("referral_ppv_threshold")) - 0.03) < 1e-9
    assert float(th.get("pediatric_ppv_threshold")) <= 0.03


def test_b2c_templates_have_no_forbidden_words():
    cfg = orisk._b2c()
    buckets = []
    for k in ("by_feature", "by_context", "intake_hints_b2c"):
        for _, arr in (cfg.get(k) or {}).items():
            buckets.extend(arr or [])
    buckets.extend(cfg.get("safety_netting") or [])
    for q in buckets:
        low = q.lower()
        assert not any(w in low for w in FORBIDDEN), f"scary word in template: {q}"
        assert "%" not in q


def test_required_inputs_weights_sum_to_one():
    w = orisk._required().get("weights") or {}
    assert abs(sum(w.values()) - 1.0) < 1e-6, w


def test_strata_sex_values_valid():
    for f in _all_features():
        strata = f.get("strata") or {}
        s = str(strata.get("sex") or "any")
        assert s in ("any", "male", "female"), f"{f.get('id')}: bad sex strata {s}"
