# Vendored OpenCode client

This directory contains the pinned Windows x64 OpenCode 1.18.25 client used by
the local UGTOMS coding application. The archive is committed so first use does
not require npm or an OpenCode download.

The setup path must verify the archive's byte count and SHA-256 before
extraction, allow exactly the `opencode.exe` member, then verify the extracted
executable's byte count and SHA-256 before its first execution. The expected
records are in `MANIFEST.json`.

The 62,030,007-byte archive is below GitHub's 100 MB per-file limit. The
179,651,624-byte executable is not committed as one blob; it is reconstructed
from the verified archive during setup. OpenCode is redistributed under the
included MIT license.
