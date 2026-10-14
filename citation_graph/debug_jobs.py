"""Fixed maintenance plan for an explicitly selected review draft."""
from __future__ import annotations
import re


def build_debug_job(root, definitions, selection):
    from citation_graph.admin_jobs import JobDefinition, _python
    if (not isinstance(selection,dict) or set(selection)!={'attachment_key','revision'}
            or not isinstance(selection['attachment_key'],str)
            or not re.fullmatch(r'[A-Z0-9]{8}',selection['attachment_key'])
            or isinstance(selection['revision'],bool) or not isinstance(selection['revision'],int)
            or selection['revision']<1):
        raise ValueError('修正案の添付ID・版番号が不正です。')
    return JobDefinition('debug_apply','資料の手動修正を反映',
        '保存済みの修正を反映し、全DB監査で確認します。OCR・LLMは呼びません。',
        (('修正案を反映',(_python(root),str(root/'scripts/apply_debug_review.py'),
                         '--attachment',selection['attachment_key'],'--revision',str(selection['revision']))),
         definitions['database_audit'].steps[0]))
