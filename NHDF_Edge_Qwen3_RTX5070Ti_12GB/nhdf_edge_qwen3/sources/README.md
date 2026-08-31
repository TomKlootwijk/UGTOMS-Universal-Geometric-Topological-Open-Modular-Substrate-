# Source-material policy

Both supplied formal-specification revisions are retained for lineage. Version
0.3 is the current source. It is primarily a general-purpose continuous
collision-detection specification; its Edge-AI section states explicitly that
dense model weights do not disappear and that compression, quality, latency,
memory and equal-budget baselines must be measured. It does not define a tensor
codec. Accordingly, this repository uses v0.3 only as a validation and resource
contract for an Edge-AI adapter, not as evidence that the earlier scalar weight
format is an NHDF/CCD implementation.

The earlier 30-page prompt-chain transcript was reviewed for traceability but
is not duplicated because it contains personal identifiers and superseded,
unvalidated hardware claims. Its SHA-256 is recorded in `SOURCE_SHA256.txt` so
the user's original can be matched without redistributing it.
