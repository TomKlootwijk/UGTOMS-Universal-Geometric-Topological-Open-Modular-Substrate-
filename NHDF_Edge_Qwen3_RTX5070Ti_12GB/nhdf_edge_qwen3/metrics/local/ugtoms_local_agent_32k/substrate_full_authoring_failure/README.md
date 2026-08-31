# Full substrate-authoring negative result

This directory preserves the failed 2026-08-31 live acceptance attempt against
the 32K local Qwen3-30B-A3B runtime. The run passed the server, context,
configuration, and native-tool-call prechecks; verified the expected 4/4
failing unimplemented clean-room baseline; then timed out after 1,200 seconds
while attempting the broad four-file SCLP application task.

The raw run JSONL showed repeated oversized malformed Edit calls. This result
therefore does **not** establish broad autonomous substrate authoring. It is
retained as a negative result and must not be collapsed into the passing direct
reference-app, generic coding-agent, or focused substrate-repair gates.

The raw `opencode.stdout.jsonl` is intentionally omitted from the public tracked
snapshot because it contains host-specific absolute temporary paths and opaque
session, message, and call identifiers. Host-local paths in `evidence.json` are
replaced with stable `<PROJECT_ROOT>` placeholders. `SOURCE_RAW_SHA256SUMS`
preserves the original ignored-run digests, including the omitted JSONL.

The retained committed evidence uses repository-normalized LF line endings.
`SHA256SUMS` binds only those retained normalized evidence payloads. The source
digests may differ because the ignored originals used different line-ending
bytes. Empty stderr files are represented only by their source digest.
