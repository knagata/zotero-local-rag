"""Shared prompt builders for legacy-compatible and V3 node summaries."""
from __future__ import annotations


def cited_section_summary_prompt(source: str) -> str:
    return (
        "以下の学術資料の節を、入力された内容だけに基づいて日本語で要約してください。"
        "3〜5文、全体で200〜400字を目安にし、中心的な主張、対象、方法、具体的知見を"
        "優先してください。各文には、その文全体を直接支持するevidence_unit_idsを1〜3個"
        "付けてください。一つの文を支持できない複数箇所の寄せ集めは禁止します。"
        "入力にない人物、場所、年代、数量、因果関係、評価を補わないでください。"
        "構造化項目や引用文の抽出は行わず、要約文だけを返してください。\n\n"
        + source
    )


def cited_item_summary_prompt(title: str, source: str) -> str:
    return (
        f"次の資料『{title}』の検証済み節要約を統合してください。"
        "日本語4〜7文、400〜700字程度で、資料全体の主題、中心的主張、方法・対象、"
        "主要な展開や結論を簡潔にまとめてください。各文に、その文全体を直接支持する"
        "evidence_unit_idsを1〜3個付けてください。入力にない事実・数値・固有名詞を"
        "補わず、著者評価やメタコメントも書かないでください。\n\n" + source
    )


__all__ = ["cited_item_summary_prompt", "cited_section_summary_prompt"]
