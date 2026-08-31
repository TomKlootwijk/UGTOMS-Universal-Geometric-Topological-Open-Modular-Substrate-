# UGTOMS Local Coder GUI

`START_LOCAL_CODER.cmd` opens a native Windows desktop front end for the verified
local-coder path. The application keeps one `llama-server` process resident across
prompts, while each prompt runs the pinned local OpenCode client against that same
loopback endpoint.

## First run

1. Double-click `START_LOCAL_CODER.cmd` at the repository root. The launcher resolves
   its own directory and checks for Python 3.10 or newer with tkinter.
2. Click **Install Client** if the Client card is not ready. The GUI runs
   `scripts/setup_local_coder.ps1`, which verifies and extracts the vendored OpenCode
   1.18.25 archive. It then rechecks the executable's exact size, SHA-256, and version.
3. Click **Download Model** if the Model card is missing. This is the only large
   first-run download in the release layout. The controlled file is
   `Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf`: 9,870,270,464 bytes (9.19 GiB),
   SHA-256 `f2dc78edd3ec0171904f1945d8c05a948131b1103172b1710b763db2eb65f52a`.
4. Choose the Git repository you want the agent to inspect or change.
5. Close VRAM-heavy applications, click **Refresh**, then click **Start Resident
   Model** when all five readiness cards are green.

The downloader reads only the digest-pinned `CONTROL_SOURCE.json` record. It uses the
record's immutable HTTPS revision URL, supports byte-range resume, rejects a wrong
range or size, verifies the complete SHA-256, and atomically promotes the temporary
file only after verification. **Cancel** keeps a partial download for the next resume.
An existing invalid final model is never overwritten automatically.

## Daily use

The default **Review (read-only)** mode does not pass OpenCode's `--auto` switch.
Read, glob, grep, list, and language-server inspection remain available; edit and
shell requests that need approval are rejected by noninteractive OpenCode.

**Work (scoped edits + tests)** is opt-in. The GUI shows a clear confirmation once per
session. After confirmation, it internally passes `--auto` so ask-level edits and
focused test commands can run. The pinned config continues to deny built-in network,
external-directory, destructive Git, commit, push, and delegation operations.

That Work-mode boundary is not an operating-system sandbox. A general shell command
can theoretically invoke Python, PowerShell, or another program to bypass tool-level
network, external-path, or destructive-command rules. The scope prompt and pinned
denies reduce risk but do not guarantee containment. Use Review for untrusted tasks;
enable Work only for a repository and request you are willing to let the local model
change, then review its tool events and Git diff.

OpenCode's session identifier is accepted only from its JSONL output and is never
typed or overridden by the user. Later prompts receive that exact internally held
identifier, preserving conversation continuity without reloading the model. **New
Session** clears conversation continuity and Work authorization but leaves the model
resident. **Stop** releases the resident process and GPU memory.

**Cancel** terminates the owned OpenCode process tree on Windows and drains its output
readers; it does not unload the model. You can send a different prompt afterward.
Closing the window cancels current work and stops the resident server before exit.

## What the readiness cards prove

- **Client:** the canonical project-local OpenCode executable has the pinned version
  and SHA-256.
- **Model:** the GGUF has the exact controlled byte length and SHA-256.
- **Runtime:** every pinned llama.cpp file is exact; on Windows, the required CUDA
  12.8 DLLs are also checked against pinned sizes and hashes.
- **GPU:** the measured deployment profile sees the exact validated RTX 5070 Ti
  Laptop GPU identity, at least 12,227 MiB total VRAM, and at least 10,712 MiB free.
- **Artifact:** the canonical 32K/q4-KV manifest and its evidence references pass the
  hardened launcher validation.

Green cards are a preflight display, not a replacement for launch validation. Start
re-runs the canonical config, client, contract, artifact, payload, runtime, GPU, and
loopback ownership checks. The server receives the launcher's externally approved
artifact identity and remains the sole owner of the model process.

## Diagnostics and recovery

Use **Show diagnostics** to see raw JSON event records, client stderr, session binding,
and actionable validation errors. Common recovery steps are:

- **Client missing:** click Install Client. A public release must include the verified
  `vendor/opencode/opencode-windows-x64-1.18.25.zip` archive.
- **Model missing or partial:** click Download Model; the `.download.part` file resumes.
- **GPU blocked:** close other GPU programs until the card reports enough free VRAM.
- **Runtime or Artifact blocked:** do not bypass the check. Restore the tracked release
  files from Git and refresh.
- **Port or health failure:** stop any previous local-coder instance, then Start again.
- **Prompt failure:** inspect diagnostics, keep the model running, and send a narrower
  request. Use New Session if prior conversation context is no longer useful.

The GUI never accepts arbitrary OpenCode flags, alternate artifacts, alternate model
URLs, public bind addresses, or user account/provider configuration.
