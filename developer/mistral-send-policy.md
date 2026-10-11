# Per-item Mistral OCR send exclusion

The user-facing tag is documented in docs/daily-use.md. Its only purpose is to
exclude an attachment from paid Mistral OCR sends; it never asserts that OCR is
correct or removes an item from indexing, local search, summary generation, or
quality audits.

`src/mistral_send_policy.py` defines the tag once and reads fresh attachment and
parent tags from Zotero Local API for paid-send checks. Tag matching trims spaces
and ignores case. The Batch resolver filters normalized attachment/parent tags;
submission checks fresh tags before upload and again before creating the paid
job. An already-uploaded saved Batch cannot bypass the check. Zotero errors fail
closed. If an uploaded batch contains a newly excluded source, no paid job is
created; prepare a fresh batch with a new state file to omit that source.

The direct OCR extractor checks immediately before the external API call,
including use_cache=False/refetch. A successful archived-result replay returns
before this gate, and staged adoption parses its local result without this
extractor, so neither needs permission to send. CLI force/reparse does not
silently override the tag. Removing the tag is the explicit way to permit a new
send. Jobs already submitted to the provider cannot be recalled by a later tag.

An externally OCRed PDF has no trustworthy internal request history. The tag is
therefore an explicit user's instruction, not a PDF producer-name heuristic.
Existing quality flags and pending diagnostics can remain: exclusion must never
turn an unsuccessful quality gate into success. No vendor calls are made by the
policy itself. tests/test_mistral_send_policy.py verifies tag filtering, fresh
parent checks, refusal before vendor access, a tag added during upload, direct
refetch refusal, and cache replay preserving degraded quality.
