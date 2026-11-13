import json
from unittest.mock import Mock


def test_summary_cli_filters_unapproved_items_before_selection(tmp_path, monkeypatch, capsys):
    from scripts import build_structure_summaries as cli
    import sys
    report = {'gate': {'passed': False}, 'indexed_gate': {'passed': True, 'eligible_item_keys': ['GOOD']}}
    validator = Mock(return_value=report)
    monkeypatch.setattr(cli, 'validate_database_gate', validator)
    monkeypatch.setattr(cli, 'list_item_keys', lambda **kw: ['BAD', 'GOOD', 'WAIT'])
    monkeypatch.setattr(cli, '_select_batch_keys', lambda keys, **kw: keys)
    writer = Mock()
    monkeypatch.setattr(cli, 'build_structure_summaries', writer)
    monkeypatch.setattr(sys, 'argv', ['summary', '--all', '--mode', 'llm', '--dry-run',
                                    '--database-gate', str(tmp_path/'gate.json'), '--allow-pending-ocr'])
    cli.main()
    assert json.loads(capsys.readouterr().out)['items'] == ['GOOD']
    assert validator.call_args.kwargs['allow_pending_ocr'] is True
    writer.assert_not_called()


def test_audit_partial_success_publishes_only_indexed_subset(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from scripts import run_db_audit as cli
    paths = tuple(tmp_path / name for name in ['gate','zotero','source','status'])
    report = {'gate': {'passed': False}, 'indexed_gate': {'passed': True,
              'eligible_item_keys': ['GOOD'], 'excluded_item_keys': ['BAD'],
              'pending_ocr_attachments': ['WAIT']}}
    monkeypatch.setattr(cli, '_validated_report_paths', lambda: paths)
    monkeypatch.setattr(cli, 'held', lambda path: nullcontext())
    def audit():
        raise cli.AuditFailure(2, report)
    monkeypatch.setattr(cli, '_audit_once', audit)
    monkeypatch.setattr('src.audit_repair.repair_derived_data', lambda *a: {'repaired': False})
    publish = Mock()
    monkeypatch.setattr(cli, '_refresh_search_snapshot', publish)
    cli.main(['--allow-partial-index'])
    assert json.loads(paths[3].read_text())['status'] == 'completed_partial'
    publish.assert_called_once_with(partial=True)
    assert report['gate']['passed'] is False


def test_summary_writer_lock_precedes_validation_and_paid_work(monkeypatch):
    import pytest
    import sys
    import src.indexing_lock as lock
    from scripts import build_structure_summaries as cli
    monkeypatch.setattr(sys, 'argv', ['summary', '--all', '--mode', 'llm'])
    def busy(path):
        raise RuntimeError('indexing is active')
    monkeypatch.setattr(lock, 'held', busy)
    validator = Mock()
    monkeypatch.setattr(cli, 'validate_database_gate', validator)
    with pytest.raises(RuntimeError, match='indexing is active'):
        cli.main()
    validator.assert_not_called()


def test_partial_report_from_crashing_child_cannot_authorize_paid_work(tmp_path, monkeypatch):
    import pytest
    from contextlib import nullcontext
    from scripts import run_db_audit as cli
    paths = tuple(tmp_path / name for name in ['gate','zotero','source','status'])
    report = {'gate': {'passed': False}, 'indexed_gate': {'passed': True, 'pending_ocr_attachments': ['WAIT']}}
    monkeypatch.setattr(cli, '_validated_report_paths', lambda: paths)
    monkeypatch.setattr(cli, 'held', lambda path: nullcontext())
    def crash():
        paths[0].write_text(json.dumps(report))
        raise cli.AuditFailure(-11, report)
    monkeypatch.setattr(cli, '_audit_once', crash)
    monkeypatch.setattr('src.audit_repair.repair_derived_data', lambda *a: {'repaired': False})
    with pytest.raises(SystemExit):
        cli.main(['--allow-pending-ocr'])
    assert not paths[0].exists()
    assert json.loads(paths[3].read_text())['status'] == 'failed'
