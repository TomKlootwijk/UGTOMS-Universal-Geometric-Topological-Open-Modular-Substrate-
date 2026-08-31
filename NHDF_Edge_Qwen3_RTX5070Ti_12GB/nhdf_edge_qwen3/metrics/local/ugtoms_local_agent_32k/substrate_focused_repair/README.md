# Focused substrate semantic-repair result

This directory preserves the passing 31 August 2026 live acceptance result for
one fully specified SCLP defect: a same-generation feedback target `n -> n`
had to become the required bounded transition `n -> n + 1`.

The local agent read the committed kernel, registry, SCLP profile, graph and
runtime sources, made exactly one one-line edit, ran exactly one declared
pytest command, and passed 3/3 independent fixture tests. The two deterministic
replays were byte-identical, only `src/sclp_repair.py` changed, and Git HEAD was
unchanged. Agent wall time was 34.241295 seconds.

This proves instruction-following and tool compliance for that one declared
repair. The answer was disclosed by the fixture and prompt, so it does not
prove independent diagnosis, broad substrate understanding, general coding
competence, compression, or preventive sandboxing. The separately retained
full four-file substrate-authoring attempt remains a failed 1,200-second run.

`evidence.json` is the exact live record. Its SHA-256 is recorded in
`SHA256SUMS`.
