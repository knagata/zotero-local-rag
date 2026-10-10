from citation_graph.admin_jobs import job_definitions
from citation_graph.admin_plan import public_stages


def test_summary_job_processes_all_differences_without_forcing_regeneration(tmp_path):
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    definitions = job_definitions(tmp_path)
    job = definitions['summary_batch']
    command = job.steps[-1][1]
    assert '--all' in command
    assert '--limit' not in command
    assert '--force' not in command
    assert command[command.index('--workers') + 1] == '10'
    assert '--database-gate' in command and '--embed' in command
    assert job.paid is True
    stage = next(s for s in public_stages(definitions) if s['key'] == 'summary_batch')
    assert stage['label'] == '階層要約の差分更新（全件）'
    assert 'すべて' in stage['description']
