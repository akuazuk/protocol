import pytest
from clinical_knowledge.mo_dual_score import dual_admission_scores, dual_scores_from_result, form_content_matrix


@pytest.mark.parametrize('readiness', [{'pct': 0, 'score': 90}, {'pct': None, 'score': 0}])
def test_zero_readiness_is_preserved(readiness):
    result = dual_scores_from_result({'overall_pct': 90, 'readiness': readiness})
    assert result['document_ready_pct'] == 0
    assert result['admission_pct'] == 0
    assert result['status'] == 'complete'


@pytest.mark.parametrize('clinical,ready', [(None, 90), (90, None), (None, None), (None, 0)])
def test_incomplete_assessment_does_not_issue_admission(clinical, ready):
    result = dual_admission_scores(clinical_pct=clinical, document_ready_pct=ready)
    assert result['admission_pct'] is None
    assert result['clinical_pct'] == clinical
    assert result['document_ready_pct'] == ready
    assert result['status'] == 'incomplete'
    assert form_content_matrix(clinical_pct=clinical, document_ready_pct=ready)['cell'] == 'unknown'


def test_real_zero_is_not_unknown():
    assert form_content_matrix(clinical_pct=0, document_ready_pct=0)['cell'] == 'both'
    assert dual_admission_scores(clinical_pct=0, document_ready_pct=0)['admission_pct'] == 0
