import json

from scripts.run_pdf_batch_supervisor import build_command, parse_batch_result


def test_parse_batch_result_uses_machine_readable_event():
    event = {"event": "index_batch_result", "processed_parent_items": 3}
    output = "progress\n" + json.dumps(event) + "\n"
    assert parse_batch_result(output) == event


def test_parse_batch_result_ignores_other_json():
    assert parse_batch_result('{"event":"something_else"}\n') is None


def test_command_keeps_rapidocr_safe_batch_size():
    command = build_command(3)
    assert command[command.index("--limit") + 1] == "3"
    assert command[:2] == ["caffeinate", "-i"]
