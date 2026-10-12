"""User-facing explanations shared by logs and browser diagnostics."""

REASONS = {
    "lexical_content_mismatch": ("全文索引の本文・書誌内容が保存済み本文と一致していません。", "対象本文の内容を比較して修復してください。ID一致だけでは合格にしません。"),
    "missing_v3_structure": ("文書構造が保存されていません。", "保存済み本文から構造を修復し、DB監査を再実行してください。"),
    "stale_structure_version": ("文書構造の形式が旧版です。", "文書構造を更新し、DB監査を再実行してください。"),
    "structure_attachment_coverage_mismatch": ("文書構造が索引済み添付をすべて含んでいません。", "対象資料の構造を修復し、DB監査を再実行してください。"),
    "incomplete_node_coverage": ("本文と文書構造の参照が一致していません。", "対象資料の構造・本文メタデータを同期してください。"),
    "no_retrievable_chunks": ("資料の本文が検索対象から外れています。", "本文の領域判定と検索ポリシーを調査してください。"),
    "unavailable_v3_structure": ("利用可能な文書構造がありません。", "対象資料の本文・構造生成ログを確認してください。"),
    "duplicate_v3_chunk_ids": ("本文の識別子が重複しています。", "取込記録を調査し、対象限定で修復してください。"),
    "manifest_chroma_attachment_mismatch": ("取込記録と本文索引の添付が一致していません。", "ライブラリ差分更新を実行してください。残る場合は添付単位の調査が必要です。"),
    "chroma_lexical_id_mismatch": ("全文索引とベクトル索引が一致していません。", "全文索引の欠落だけは自動補完します。余分な本文や重複は自動削除せず調査が必要です。"),
    "hnsw_not_validated": ("ベクトル索引の検証が未完了です。", "取込完了と索引の実検索検証を確認してください。検証フラグだけの変更は行いません。"),
    "inflight_attachments": ("取込が途中の添付があります。", "実行中なら完了を待ち、停止済みなら対象添付の取込を再開してください。記録だけは消しません。"),
    "post_index_pending": ("索引後の処理が未完了です。", "ライブラリ差分更新で後処理を完了し、DB監査を再実行してください。"),
    "incomplete_source_coverage": ("原本の抽出範囲が不足、または確認できません。", "対象添付の抽出記録と欠落範囲を確認してください。OCRが必要なら有料処理は別途承認が必要です。"),
    "unapproved_truncated_pdf": ("PDFの抽出が途中で打ち切られています。", "対象PDFを再処理してください。欠落を自動承認することはありません。"),
    "missing_v3_pipeline_provenance": ("索引を作成した設定・モデルの記録が不足しています。", "設定と取込記録の調査が必要です。来歴を推測して埋めません。"),
    "chunks_without_item": ("資料との対応を失った本文があります。", "取込記録を調査し、本文の所属を確定してください。自動削除は行いません。"),
    "chunks_without_attachment": ("添付との対応を失った本文があります。", "本文と原本の対応を調査してください。自動削除は行いません。"),
}


def reason_detail(reason: str) -> dict:
    message, action = REASONS.get(reason, (
        f"資料の整合性検査が不合格です：{reason}",
        "対象資料と実行ログを確認し、修復後にDB監査を再実行してください。",
    ))
    return {"reason": reason, "message": message, "action": action}


def item_failure_details(report: dict) -> list[dict]:
    details = []
    for item in report.get("items", []):
        for reason in item.get("failures", []):
            stale = reason == "stale_structure_fingerprint"
            details.append({
                "item_key": item.get("item_key", ""),
                "title": item.get("title") or item.get("item_key", "名称不明"),
                "reason": reason,
                "message": ("文書構造と本文索引の更新状態が一致していません。"
                            if stale else reason_detail(reason)["message"]),
                "action": ("「文書構造・目次の差分更新」を実行後、DB監査を再実行してください。"
                           "この判定だけでは再OCRは必要ありません。"
                           if stale else reason_detail(reason)["action"]),
            })
    return details


def report_failure_details(report: dict) -> list[dict]:
    details = item_failure_details(report)
    details.extend(reason_detail(reason) for reason in report.get("gate", {}).get("global_failures", []))
    for document in report.get("documents", []):
        if document.get("lost_pages") or document.get("error"):
            details.append({
                "title": document.get("title") or "名称不明",
                "attachment_key": document.get("attachment_key", ""),
                "message": (f"索引に本文がないページ：{document['lost_pages'][:20]}"
                            if document.get("lost_pages") else f"原本を読み取れません：{document['error']}"),
                "action": "原本と抽出状態を確認してください。必要な再抽出・OCRは対象を特定して実行してください。",
            })
    for row in report.get("missing_required_from_manifest", []):
        key = row.get("attachment_key", "") if isinstance(row, dict) else row
        title = row.get("filename") or key if isinstance(row, dict) else key
        details.append({"attachment_key": key, "title": title,
                        "message": "未処理の添付が残っています。全DB監査は未合格です。",
                        "action": "同期待ちの資料は自動同期・ダウンロード完了後にクイック実行を再実行してください。"
                        "同期済みなら対象資料の取込ログを確認してください。"})
    for name, message in (("orphan_chunks", "資料に対応しない本文"),
                          ("dangling_node_chunks", "文書構造への参照不整合"),
                          ("unretrievable_items", "検索できない資料")):
        values = report.get(name, [])
        if values:
            details.append({"message": f"{message}：{len(values)}件。対象例：{values[:10]}",
                            "action": "自動修復で解消しなければ対象の本文・構造を調査してください。"})
    return details
