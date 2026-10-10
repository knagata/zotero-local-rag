import pytest
from scripts.update_status_cache import classify_structure_status


@pytest.mark.parametrize('saved,artifact,group', [
    ('flat_fallback', 'degraded', 'flat_fallback'),
    ('flat_fallback', 'failed', 'failed'),
    ('flat_fallback', 'blocked', 'failed'),
    ('flat_fallback', 'running', 'outdated'),
    ('flat_fallback', 'stale', 'outdated'),
    ('unavailable', 'degraded', 'failed'),
    ('exact', 'failed', 'failed'),
    ('recovered', 'blocked', 'failed'),
    ('degraded', 'success', 'degraded'),
    ('exact', 'degraded', 'degraded'),
    ('stale', 'success', 'outdated'),
    ('running', 'success', 'outdated'),
])
def test_structure_states_are_disjoint_and_failures_take_precedence(saved, artifact, group):
    result = classify_structure_status({'A'}, {'A': ('fingerprint', saved)}, {('A', 'structure'): artifact})
    assert result[group] == 1
    assert sum(result[k] for k in ('flat_fallback', 'failed', 'outdated', 'unprocessed', 'degraded')) == 1
    assert result['pending'] == (0 if group == 'flat_fallback' else 1)
    assert result['attention'] == (1 if group in {'failed', 'degraded'} else 0)
    assert result['sample_keys'] == ([] if group == 'flat_fallback' else ['A'])


def test_missing_and_successful_structures_and_inventory_scope():
    result = classify_structure_status({'NEW', 'GOOD', 'RECOVERED'}, {
        'GOOD': ('f', 'exact'), 'RECOVERED': ('f', 'recovered'), 'EXCLUDED': ('f', 'flat_fallback')}, {})
    assert result['unprocessed'] == result['pending'] == 1
    assert result['failed'] == result['flat_fallback'] == 0
    assert result['sample_keys'] == ['NEW']


def test_many_usable_flat_documents_are_not_processing_failures():
    keys = {f'ITEM{i:03}' for i in range(213)}
    result = classify_structure_status(keys, {k: ('f', 'flat_fallback') for k in keys},
                                       {(k, 'structure'): 'degraded' for k in keys})
    assert result['flat_fallback'] == 213
    assert result['pending'] == result['failed'] == result['attention'] == 0
    assert len(result['flat_fallback_sample_keys']) == 20
