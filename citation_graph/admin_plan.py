"""Selectable maintenance stages; commands always come from server definitions."""
from __future__ import annotations

STAGES = (
    ("update_check", "更新状況を確認", (), False),
    ("mistral_ocr", "Mistral OCR Batch", (), False),
    ("library_update", "ライブラリ差分更新", (), True),
    ("structure_update", "文書構造・目次の差分更新", ("library_update",), True),
    ("database_audit", "DB監査", (), True),
    ("summary_batch", "階層要約の差分更新（全件）", ("structure_update", "database_audit"), False),
    ("citation_update", "Citation Network更新", (), True),
)


def public_stages(definitions):
    return [dict(key=key, label=label, requires=list(requires), default=default,
                 paid=definitions[key].paid or key in {"library_update", "structure_update", "citation_update"},
                 description=definitions[key].description)
            for key, label, requires, default in STAGES]


def build_plan(selected, definitions):
    from citation_graph.admin_jobs import JobDefinition
    if not isinstance(selected, list) or not selected or any(not isinstance(x, str) for x in selected):
        raise ValueError("実行する工程を選択してください。")
    keys = set(selected)
    available = {stage["key"]: stage for stage in public_stages(definitions)}
    if len(keys) != len(selected) or keys - available.keys():
        raise ValueError("不明または重複した工程です。")
    steps = []
    for key, stage in available.items():
        if key not in keys:
            continue
        missing = set(stage["requires"]) - keys
        if missing:
            raise ValueError(f"{stage['label']}には前工程の選択が必要です。")
        definition = definitions[key]
        if key == "database_audit" and {"library_update", "structure_update"} <= keys:
            steps.append(definitions["quick_update"].steps[2])
        steps.extend(definition.steps[-1:] if key == "summary_batch" else definition.steps[:1])
    return JobDefinition("selected_steps", "選択した工程を実行", " → ".join(available[k]["label"] for k in available if k in keys),
                         tuple(steps), paid=any(available[k]["paid"] for k in keys))
