from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import pytest
from citation_graph import admin_jobs
from citation_graph.admin_plan import build_plan, public_stages
from citation_graph.admin_routes import StartJobRequest
from pydantic import ValidationError


def definitions(tmp_path):
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    return admin_jobs.job_definitions(tmp_path)


def test_default_plan_matches_quick_and_server_orders_reverse_selection(tmp_path):
    defs = definitions(tmp_path)
    keys = [s['key'] for s in public_stages(defs) if s['default']]
    plan = build_plan(list(reversed(keys)), defs)
    assert plan.steps == defs['quick_update'].steps
    assert plan.paid


@pytest.mark.parametrize('selected', [None, [], ['unknown'], ['database_audit'] * 2, [42], ['structure_update'], ['summary_batch'], ['library_update', 'summary_batch', 'database_audit']])
def test_invalid_selection_rejected(tmp_path, selected):
    with pytest.raises(ValueError):
        build_plan(selected, definitions(tmp_path))


def test_all_stages_do_not_duplicate_structure_or_audit_and_ocr_first(tmp_path):
    defs = definitions(tmp_path)
    plan = build_plan([s['key'] for s in public_stages(defs)], defs)
    names = [Path(cmd[1]).name for _, cmd in plan.steps]
    assert names == ['check_admin_update_status.py', 'run_mistral_ocr_maintenance.py', 'run_searchable_index.py', 'rebuild_document_structure.py', 'run_searchable_index.py', 'run_db_audit.py', 'build_structure_summaries.py', 'update_citations.py']


def test_audit_only_is_unpaid_and_independent(tmp_path):
    plan = build_plan(['database_audit'], definitions(tmp_path))
    assert not plan.paid
    assert len(plan.steps) == 1


def test_consent_is_strict_boolean():
    for value in ['true', 1, 'yes']:
        with pytest.raises(ValidationError):
            StartJobRequest(job_type='selected_steps', paid_consent=value)


def test_plan_saved_and_no_process_spawn_without_consent(tmp_path, monkeypatch):
    definitions(tmp_path)
    monkeypatch.setattr(admin_jobs, 'active_job', lambda root: None)
    with patch.object(admin_jobs.subprocess, 'Popen', return_value=SimpleNamespace(pid=123)) as spawn:
        with pytest.raises(ValueError, match='同意'):
            admin_jobs.start_job('selected_steps', '', 'test', tmp_path, selected_steps=['library_update'])
        spawn.assert_not_called()
        record = admin_jobs.start_job('selected_steps', '', 'test', tmp_path, selected_steps=['library_update'], paid_consent=True)
    assert record['selected_steps'] == ['library_update']
    assert record['paid_consent'] is True


def test_worker_rejects_paid_plan_without_recorded_consent(tmp_path):
    definitions(tmp_path)
    admin_jobs.write_record(dict(id='test-job', type='selected_steps', pid=os.getpid(), token='token', selected_steps=['library_update'], paid_consent=False), tmp_path)
    assert admin_jobs.run_job('test-job', 'selected_steps', 'token', tmp_path) == 2


def test_selected_worker_executes_all_in_order_and_stops_on_error(tmp_path, monkeypatch):
    import sys
    defs = definitions(tmp_path)
    calls = []
    for key in ['update_check', 'database_audit']:
        defs[key] = admin_jobs.JobDefinition(key, key, key, ((key, (sys.executable, '-c', 'pass')),))
    monkeypatch.setattr(admin_jobs, 'job_definitions', lambda root: defs)
    for failure in [False, True]:
        calls.clear()
        record = dict(id='plan-job', type='selected_steps', pid=os.getpid(), token='token', selected_steps=['database_audit', 'update_check'], paid_consent=False)
        admin_jobs.write_record(record, tmp_path)
        def spawn(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(pid=123, wait=lambda: 7 if failure else 0)
        with patch.object(admin_jobs.subprocess, 'Popen', side_effect=spawn):
            assert admin_jobs.run_job('plan-job', 'selected_steps', 'token', tmp_path) == (7 if failure else 0)
        assert len(calls) == (1 if failure else 2)
        saved = admin_jobs.read_record('plan-job', tmp_path)
        assert saved['status'] == ('failed' if failure else 'completed')


def test_browser_checkbox_and_consent_interactions():
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable; DOM interaction harness also runs in browser runtime')
    subprocess.run([node, 'tests/admin_plan_dom.cjs'], check=True)
