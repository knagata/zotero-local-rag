from __future__ import annotations

import unittest

from src.document_structure import attach_structure_metadata, build_document_structure, validate_structure


def _chunk(chunk_id: str, text: str, **metadata: object) -> dict:
    return {"id": chunk_id, "text": text, "metadata": metadata}


class DocumentStructureV3Tests(unittest.TestCase):
    def test_attaches_leaf_policy_and_provenance_to_every_chunk(self):
        chunks = [
            _chunk("A:p1", "body", attachmentKey="A", structure_path=["One"], zone="body",
                   extraction_engine="epub_dom", extraction_version="3"),
            _chunk("A:p2", "note", attachmentKey="A", structure_path=["One"], zone="footnote",
                   extraction_engine="epub_dom", extraction_version="3"),
        ]
        result = build_document_structure("ITEM", chunks)
        annotated = attach_structure_metadata(chunks, result["nodes"])
        self.assertEqual({row["metadata"]["chunk_scheme"] for row in annotated}, {3})
        self.assertNotEqual(annotated[0]["metadata"]["node_id"], annotated[1]["metadata"]["node_id"])
        self.assertEqual(annotated[1]["metadata"]["retrieval_policy"], "explicit_only")

    def test_validation_detects_noncontiguous_leaf_assignment(self):
        chunks = [_chunk("A:p1", "a"), _chunk("A:p2", "b"), _chunk("A:p3", "c")]
        nodes = [{
            "node_id": "leaf", "parent_node_id": None, "depth": 0, "ordinal": 0,
            "zone": "body", "summary_policy": "include", "retrieval_policy": "normal",
            "citation_policy": "none", "first_chunk_id": "A:p1", "last_chunk_id": "A:p3",
            "chunks": [{"chunk_id": "A:p1"}, {"chunk_id": "A:p3"}],
        }]
        audit = validate_structure(nodes, chunks)
        self.assertFalse(audit["valid"])
        self.assertTrue(any(error.startswith("noncontiguous_leaf") for error in audit["errors"]))

    def test_validation_keeps_duplicate_input_ids_as_a_hard_error(self):
        chunks = [_chunk("A:p1", "first"), _chunk("A:p1", "duplicate")]
        nodes = [{
            "node_id": "leaf", "parent_node_id": None, "depth": 0, "ordinal": 0,
            "zone": "body", "summary_policy": "include", "retrieval_policy": "normal",
            "citation_policy": "none", "first_chunk_id": "A:p1", "last_chunk_id": "A:p1",
            "chunks": [{"chunk_id": "A:p1"}],
        }]

        audit = validate_structure(nodes, chunks)

        self.assertFalse(audit["valid"])
        self.assertIn("duplicate_input_chunk_id", audit["errors"])

    def test_recovers_attachment_namespace_for_legacy_enrichment_chunks(self):
        # Older Docling reference-enrichment rows retained their attachment key
        # in the chunk-id namespace but not in metadata.  They can be
        # interleaved with regular chunks from the same attachment, so treating
        # all missing keys as one anonymous attachment makes leaves noncontiguous.
        attachment = "EW6GPNQH"
        chunks = [
            _chunk(f"{attachment}:p1:para0:part0", "body", attachmentKey=attachment),
            _chunk(f"{attachment}:docref:{attachment}:p1:para1:part0", "note",
                   chapter="Notes", zone="endnote"),
            _chunk(f"{attachment}:p2:para0:part0", "body", attachmentKey=attachment),
        ]

        result = build_document_structure("FUDKAS3M", chunks)

        self.assertTrue(result["diagnostics"]["valid"])
        self.assertEqual(result["diagnostics"]["attachment_count"], 1)
        attachment_root = next(node for node in result["nodes"] if node["node_type"] == "attachment_root")
        self.assertEqual(attachment_root["attachment_key"], attachment)

    def test_splits_interleaved_unknown_attachment_runs_to_preserve_contiguity(self):
        # If identity is genuinely unavailable, the builder must still keep the
        # validation invariant instead of combining non-adjacent anonymous rows.
        chunks = [
            _chunk("legacy-a:1", "a"),
            _chunk("legacy-a:2", "b", attachmentKey="KNOWN001"),
            _chunk("legacy-a:3", "c"),
        ]

        result = build_document_structure("ITEM", chunks)

        self.assertTrue(result["diagnostics"]["valid"])
        self.assertEqual(result["diagnostics"]["attachment_count"], 3)


if __name__ == "__main__":
    unittest.main()
