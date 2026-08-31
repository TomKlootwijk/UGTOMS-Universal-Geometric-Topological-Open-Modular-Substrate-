#!/usr/bin/env python3
"""Launch the pinned, project-local OpenCode client against an owned NHDF server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nhdf_edge import hybrid  # noqa: E402
from nhdf_edge.server import (  # noqa: E402
    DEFAULT_PORT,
    ApprovedArtifactFile,
    ArtifactApproval,
    HybridServer,
)


PINNED_OPENCODE_VERSION = "1.18.25"
EXPECTED_OPENCODE_SHA256 = (
    "ef06e41a35795066e95acde276a42fbbf85d7a683c2787f6a19ed20bcde9b6ff"
)
DEFAULT_ARTIFACT = PROJECT_ROOT / "packs" / "qwen3-30b-a3b-iq2m-32k-q4kv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "opencode_nhdf_local.json"
SUBSTRATE_CONTRACT = PROJECT_ROOT / "substrate" / "AGENT_CONTRACT.md"
EXPECTED_CONFIG_SHA256 = "66a6e5adcc98fc80921bc4e0386f341c80a4f824597e26309b8499344f440ab1"
EXPECTED_CONTRACT_SHA256 = "54bbf57e0b26154df0c417c909dc756bdfbfad3df74622dc456f94d0f46a0035"
REQUIRED_CONTEXT_TOKENS = 32_768
REQUIRED_KV_CACHE = "q4_0"
CANONICAL_AGENT_PROMPT = (
    "Work as a careful local-only general-purpose coding agent. Inspect the selected "
    "Git repository, make scoped edits, run focused local checks, and report measured "
    "results and uncertainty. The digest-verified UGTOMS contract installed in global "
    "AGENTS.md is mandatory for substrate work. Never use network services, external "
    "directories, destructive actions, task delegation, commits, pushes, or external systems."
)
OPENCODE_EXE = (
    PROJECT_ROOT
    / ".local-coder"
    / "node_modules"
    / "opencode-ai"
    / "bin"
    / "opencode.exe"
)
LOCAL_STATE_ROOT = PROJECT_ROOT / ".local-coder" / "state"
MODEL_ID = "local-runtime/local-qwen3-30b-a3b"
CANONICAL_PAYLOAD_PATH = (
    "models/Qwen3-30B-A3B-Instruct-2507-IQ2_M/"
    "Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf"
)
CANONICAL_SOURCE_RECORD_PATH = (
    "models/Qwen3-30B-A3B-Instruct-2507-IQ2_M/CONTROL_SOURCE.json"
)
CANONICAL_RUNTIME_ENTRYPOINT = (
    "tools/llama.cpp-f8dbcd61/bin/llama-completion.exe"
)
CANONICAL_BENCHMARK_ENTRYPOINT = (
    "tools/llama.cpp-f8dbcd61/bin/llama-bench.exe"
)
CANONICAL_SERVER_ENTRYPOINT = "tools/llama.cpp-f8dbcd61/bin/llama-server.exe"
CANONICAL_SPECIFICATION_PATH = "substrate/kernel/contract.json"
CANONICAL_ASSURANCE_PATHS = (
    "substrate/profiles/registry.json",
    "substrate/evidence/application_proofs.json",
    "substrate/AGENT_CONTRACT.md",
)
CANONICAL_VALIDATION_EVIDENCE_PATH = (
    "packs/qwen3-30b-a3b-iq2m-32k-q4kv/evidence/functional_gate.json"
)
CANONICAL_VALIDATION_SNAPSHOT_PATH = (
    "metrics/local/ugtoms_local_agent_32k/functional_gate.json"
)
EXPECTED_VALIDATION_EVIDENCE_BYTES = 32_876
EXPECTED_VALIDATION_EVIDENCE_SHA256 = (
    "d56140c0a4bc97fb9fab5d3930222a494e681744570363d0f810a8e872aa01c1"
)
CANONICAL_RUNTIME_PATHS = (
    CANONICAL_RUNTIME_ENTRYPOINT,
    CANONICAL_BENCHMARK_ENTRYPOINT,
    CANONICAL_SERVER_ENTRYPOINT,
    "tools/llama.cpp-f8dbcd61/bin/ggml-base.dll",
    "tools/llama.cpp-f8dbcd61/bin/ggml-cpu.dll",
    "tools/llama.cpp-f8dbcd61/bin/ggml-cuda.dll",
    "tools/llama.cpp-f8dbcd61/bin/ggml.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama-bench-impl.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama-cli-impl.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama-common.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama-completion-impl.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama-server-impl.dll",
    "tools/llama.cpp-f8dbcd61/bin/llama.dll",
    "tools/llama.cpp-f8dbcd61/bin/mtmd.dll",
)
CANONICAL_REFERENCE_RECORDS: Mapping[str, tuple[int, str]] = {
    CANONICAL_PAYLOAD_PATH: (
        9_870_270_464,
        "f2dc78edd3ec0171904f1945d8c05a948131b1103172b1710b763db2eb65f52a",
    ),
    CANONICAL_SOURCE_RECORD_PATH: (
        948,
        "9b184e403b591bd4fc9a6adc176ff47bbe0f372cd529168dc999fa09b4f6543a",
    ),
    CANONICAL_SPECIFICATION_PATH: (
        10_187,
        "9b5fa7cff4483129e80e1c234055d57e0f79a574dcd2e131c418bbe0259448c3",
    ),
    "substrate/profiles/registry.json": (
        1_003,
        "aa1d788808ebd624dcff435538cd4e63f13004bf84101956ffe4119f14b14152",
    ),
    "substrate/evidence/application_proofs.json": (
        4_856,
        "94f815f5b3e28592077e13005dbfd9d1c5cdb22253ca6a3d0f1273511817de44",
    ),
    "substrate/AGENT_CONTRACT.md": (
        9_796,
        EXPECTED_CONTRACT_SHA256,
    ),
    CANONICAL_VALIDATION_EVIDENCE_PATH: (
        EXPECTED_VALIDATION_EVIDENCE_BYTES,
        EXPECTED_VALIDATION_EVIDENCE_SHA256,
    ),
    CANONICAL_RUNTIME_ENTRYPOINT: (
        10_752,
        "d687544bc1e82d3fc18e4da0d64dc5a94f08e8a8bd358bc96a3243d00df38f31",
    ),
    CANONICAL_BENCHMARK_ENTRYPOINT: (
        10_752,
        "48b29766bdb58bbec86804589decf2b12049a3eb8ca62899a57d063cb989280a",
    ),
    CANONICAL_SERVER_ENTRYPOINT: (
        10_752,
        "3f1041dcc2e797a05b80324e0b04a20021cd14fba272f6e2945d16b2ead2364c",
    ),
    "tools/llama.cpp-f8dbcd61/bin/ggml-base.dll": (
        671_744,
        "5debdf44b6a8a30cbc9e733b41712bac29b9dbc68c94b6f9ecfde9a5cce82acb",
    ),
    "tools/llama.cpp-f8dbcd61/bin/ggml-cpu.dll": (
        907_264,
        "9230df69ab901f24fd6742949bf04c8f725bebacae189aac54d349c773c7ad1d",
    ),
    "tools/llama.cpp-f8dbcd61/bin/ggml-cuda.dll": (
        52_752_896,
        "734b80b60afeb373e91ad876f03698a713324102c63f0e755a417d56147e4536",
    ),
    "tools/llama.cpp-f8dbcd61/bin/ggml.dll": (
        67_072,
        "bfa741c63302c8051b80a717749ca796c2e65d8ae6267b776d512391b51f1cf9",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama-bench-impl.dll": (
        904_704,
        "08d8e3a0a03e4d4d3d48c6315385ee312653109c575e9c0ef11cf466838243ac",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama-cli-impl.dll": (
        873_472,
        "25cc56a47561d5c153731a6f783b512fac3c9cb0222a70321d06c9cbc31bde64",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama-common.dll": (
        8_201_216,
        "4ed30d4d551f42c4b5f8dc761a06172402922bab01e2fce03d34f489d6a65cf8",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama-completion-impl.dll": (
        247_808,
        "a2539189c876a5f31afa4066650476b31800f0bdb494ed527808e396ce3777b5",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama-server-impl.dll": (
        13_089_280,
        "666adac5159f65338a897a1bb3e404ab207796f6653f25ee24fb1ee4220876a2",
    ),
    "tools/llama.cpp-f8dbcd61/bin/llama.dll": (
        2_567_680,
        "216dc4e66b6505766f531d1539d87f6a66d018cdb82d5955c65cc130b19c916d",
    ),
    "tools/llama.cpp-f8dbcd61/bin/mtmd.dll": (
        2_377_728,
        "b816fd936992b81e3ce29a170ad4983cbf7d6db3ea71ebb55119e3c6c1c3224b",
    ),
}
FORBIDDEN_RUN_OPTIONS = frozenset(
    {
        "--agent",
        "--attach",
        "--auto",
        "--command",
        "--continue",
        "--cors",
        "--dir",
        "--file",
        "--fork",
        "--hostname",
        "--mdns",
        "--mdns-domain",
        "--model",
        "--password",
        "--port",
        "--pure",
        "--session",
        "--share",
        "--username",
        "-c",
        "-f",
        "-m",
        "-p",
        "-s",
        "-u",
    }
)
FORBIDDEN_SHORT_RUN_FLAGS = frozenset("cfmpsu")


class LocalCoderError(RuntimeError):
    """Raised when the safe local-coder launch contract cannot be met."""


@dataclass(frozen=True)
class LaunchOptions:
    target: Path
    artifact: Path
    config: Path
    port: int
    threads: int | None
    quick: bool
    startup_timeout: float
    run_args: tuple[str, ...] | None


@dataclass(frozen=True)
class ValidatedConfig:
    """The exact pinned config bytes parsed during validation."""

    path: Path
    raw: bytes
    sha256: str


@dataclass(frozen=True)
class ValidatedAgentArtifact:
    """A validated manifest snapshot plus its external server trust anchor."""

    artifact: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    server_approval: ArtifactApproval


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise LocalCoderError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _read_pinned_file(path: Path, expected_sha256: str, label: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        data = resolved.read_bytes()
    except OSError as exc:
        raise LocalCoderError(f"could not read pinned {label}: {exc}") from exc
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise LocalCoderError(
            f"pinned {label} digest mismatch: expected {expected_sha256}, got {actual}"
        )
    return data


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as exc:
        raise LocalCoderError(f"could not inspect path {path}: {exc}") from exc
    return path.is_symlink() or bool(attributes & 0x400)


def _require_unaliased_executable(
    path: Path,
    *,
    label: str,
    trusted_root: Path,
) -> Path:
    """Resolve an executable without accepting symlink/reparse indirection."""

    try:
        root = trusted_root.resolve(strict=True)
        lexical = Path(os.path.abspath(path))
        relative = lexical.relative_to(root)
    except (OSError, ValueError) as exc:
        raise LocalCoderError(
            f"{label} must be an absolute file below its trusted root: {path}"
        ) from exc

    current = root
    for component in relative.parts:
        current = current / component
        if _is_reparse_point(current):
            raise LocalCoderError(
                f"{label} traverses a symlink, junction, or reparse point: {current}"
            )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"{label} does not resolve: {path}") from exc
    if os.path.normcase(str(resolved)) != os.path.normcase(str(lexical)):
        raise LocalCoderError(f"{label} resolves through an aliased path: {path}")
    if not resolved.is_file():
        raise LocalCoderError(f"{label} is not a regular file: {resolved}")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise LocalCoderError(f"{label} is not executable: {resolved}")
    return resolved


def _windows_git_install_roots() -> tuple[Path, ...]:
    """Read machine-owned Git for Windows locations without consulting PATH."""

    if os.name != "nt":
        return ()
    import winreg

    roots: list[Path] = []
    views = tuple(
        flag
        for flag in (
            getattr(winreg, "KEY_WOW64_64KEY", 0),
            getattr(winreg, "KEY_WOW64_32KEY", 0),
        )
    )
    for view in views:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\GitForWindows",
                0,
                winreg.KEY_READ | view,
            ) as key:
                value, value_type = winreg.QueryValueEx(key, "InstallPath")
        except OSError:
            continue
        if value_type == winreg.REG_SZ and isinstance(value, str) and value:
            candidate = Path(value)
            if candidate.is_absolute():
                roots.append(candidate)

    system_directory = hybrid._windows_system_directory()
    roots.append(Path(system_directory.anchor) / "Program Files" / "Git")
    unique: dict[str, Path] = {}
    for root in roots:
        unique.setdefault(os.path.normcase(str(root)), root)
    return tuple(unique.values())


def _resolve_git_executable() -> Path:
    """Resolve Git only from a machine-owned location, never PATH or the target."""

    if os.name == "nt":
        candidates = tuple(
            root / "cmd" / "git.exe" for root in _windows_git_install_roots()
        )
    else:
        candidates = (Path("/usr/bin/git"), Path("/bin/git"))
    failures: list[str] = []
    for candidate in candidates:
        try:
            anchor = Path(candidate.anchor)
            return _require_unaliased_executable(
                candidate,
                label="trusted Git executable",
                trusted_root=anchor,
            )
        except LocalCoderError as exc:
            failures.append(str(exc))
    detail = "; ".join(failures) if failures else "no machine-owned candidates"
    raise LocalCoderError(f"could not locate a trusted Git executable: {detail}")


def _native_probe_environment(executable: Path) -> dict[str, str]:
    """Build a credential-free environment for native identity probes."""

    environment = hybrid._minimal_subprocess_environment(
        executable_directory=executable.parent
    )
    environment["NoDefaultCurrentDirectoryInExePath"] = "1"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GCM_INTERACTIVE"] = "Never"
    return environment


def _ensure_safe_project_directory(path: Path) -> Path:
    """Create a directory below the project without traversing reparse points."""

    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"project root does not resolve: {exc}") from exc
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as exc:
        raise LocalCoderError(
            f"local-coder state path must remain inside the project: {candidate}"
        ) from exc
    current = project_root
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if _is_reparse_point(current):
                raise LocalCoderError(
                    f"local-coder state path traverses a symlink/reparse point: {current}"
                )
            if not current.is_dir():
                raise LocalCoderError(
                    f"local-coder state path component is not a directory: {current}"
                )
        else:
            try:
                current.mkdir()
            except OSError as exc:
                raise LocalCoderError(
                    f"could not create local-coder state directory {current}: {exc}"
                ) from exc
        try:
            resolved = current.resolve(strict=True)
        except OSError as exc:
            raise LocalCoderError(f"local-coder state path does not resolve: {exc}") from exc
        if _is_reparse_point(current) or not _within(resolved, project_root):
            raise LocalCoderError(
                f"local-coder state path escaped the project: {current}"
            )
    return current


def _atomic_write_local(path: Path, data: bytes) -> None:
    directory = _ensure_safe_project_directory(path.parent)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if _is_reparse_point(directory):
            raise LocalCoderError(
                f"local-coder destination became a reparse point: {directory}"
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def install_substrate_contract(xdg_config_home: Path) -> Path:
    """Install the exact tracked contract into isolated global OpenCode rules."""

    data = _read_pinned_file(
        SUBSTRATE_CONTRACT, EXPECTED_CONTRACT_SHA256, "substrate contract"
    )
    destination_directory = xdg_config_home / "opencode"
    _ensure_safe_project_directory(destination_directory)
    destination = destination_directory / "AGENTS.md"
    try:
        _atomic_write_local(destination, data)
    except OSError as exc:
        raise LocalCoderError(f"could not install isolated substrate contract: {exc}") from exc
    if _sha256_bytes(destination.read_bytes()) != EXPECTED_CONTRACT_SHA256:
        raise LocalCoderError("installed substrate contract failed its post-write digest check")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned local OpenCode client against the validated NHDF Qwen model. "
            "No account or API key is used."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(Path.cwd()),
        help="Git working tree to work in (default: current directory)",
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT),
        help="validated NHDF hybrid artifact directory",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="OpenCode configuration file",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="compatibility flag; strict final payload verification remains mandatory",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the local model to load (default: 300)",
    )
    parser.add_argument(
        "--run",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help=(
            "run noninteractively; every remaining argument is passed to `opencode run` "
            "(place --run last)"
        ),
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> LaunchOptions:
    args = _parser().parse_args(argv)
    if not 1 <= args.port <= 65_535:
        raise LocalCoderError("--port must be between 1 and 65535")
    if args.threads is not None and args.threads <= 0:
        raise LocalCoderError("--threads must be positive")
    if args.startup_timeout <= 0:
        raise LocalCoderError("--startup-timeout must be positive")
    if args.run is not None and not args.run:
        raise LocalCoderError("--run requires a prompt or OpenCode run arguments")
    if args.run is not None:
        for argument in args.run:
            option = argument.split("=", 1)[0]
            canonical_option = (
                f"--{option[5:]}" if option.startswith("--no-") else option
            )
            compact_short_option = (
                argument.startswith("-")
                and not argument.startswith("--")
                and any(
                    character in FORBIDDEN_SHORT_RUN_FLAGS for character in argument[1:]
                )
            )
            if canonical_option in FORBIDDEN_RUN_OPTIONS or compact_short_option:
                raise LocalCoderError(
                    f"{option} cannot override the local-coder isolation contract"
                )
    return LaunchOptions(
        target=Path(args.target).expanduser(),
        artifact=Path(args.artifact).expanduser(),
        config=Path(args.config).expanduser(),
        port=args.port,
        threads=args.threads,
        quick=args.quick,
        startup_timeout=args.startup_timeout,
        run_args=None if args.run is None else tuple(args.run),
    )


def validate_git_target(target: Path) -> tuple[Path, Path]:
    """Return the target directory and its Git worktree root, or fail closed."""

    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"target does not exist: {target}") from exc
    if not resolved.is_dir():
        raise LocalCoderError(f"target is not a directory: {resolved}")

    git_executable = _resolve_git_executable()
    try:
        result = subprocess.run(
            [
                str(git_executable),
                "-C",
                str(resolved),
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=str(git_executable.parent),
            env=_native_probe_environment(git_executable),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalCoderError(f"could not validate target with Git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "not a Git working tree"
        raise LocalCoderError(f"target must be inside a Git working tree: {detail}")

    root_text = result.stdout.strip()
    if not root_text:
        raise LocalCoderError("Git returned an empty worktree root")
    try:
        worktree_root = Path(root_text).resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"Git worktree root does not exist: {root_text}") from exc
    return resolved, worktree_root


def validate_local_install(executable: Path) -> Path:
    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(
            "project-local OpenCode is not installed; run scripts/setup_local_coder.ps1"
        ) from exc
    expected = _require_unaliased_executable(
        OPENCODE_EXE,
        label="canonical project-local OpenCode executable",
        trusted_root=project_root,
    )
    try:
        resolved = _require_unaliased_executable(
            executable,
            label="requested OpenCode executable",
            trusted_root=project_root,
        )
    except LocalCoderError as exc:
        raise LocalCoderError(
            f"only the canonical project-local OpenCode executable is permitted: {expected}"
        ) from exc
    if resolved != expected:
        raise LocalCoderError(
            f"only the canonical project-local OpenCode executable is permitted: {expected}"
        )
    actual_digest = _sha256_file(resolved)
    if actual_digest != EXPECTED_OPENCODE_SHA256:
        raise LocalCoderError(
            "project-local OpenCode executable digest mismatch: "
            f"expected {EXPECTED_OPENCODE_SHA256}, got {actual_digest}"
        )
    try:
        result = subprocess.run(
            [str(resolved), "--version"],
            cwd=str(resolved.parent),
            env=_native_probe_environment(resolved),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalCoderError(f"could not run project-local OpenCode: {exc}") from exc
    installed = result.stdout.strip().splitlines()
    version = installed[0].strip() if installed else ""
    if result.returncode != 0 or version != PINNED_OPENCODE_VERSION:
        raise LocalCoderError(
            f"OpenCode {PINNED_OPENCODE_VERSION} is required; found {version or 'unknown'}"
        )
    post_run_digest = _sha256_file(resolved)
    if post_run_digest != EXPECTED_OPENCODE_SHA256:
        raise LocalCoderError("project-local OpenCode executable changed during validation")
    return resolved


def validate_config(config_path: Path) -> ValidatedConfig:
    try:
        resolved = config_path.resolve(strict=True)
        raw = resolved.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalCoderError(f"could not load OpenCode config: {exc}") from exc
    actual_digest = _sha256_bytes(raw)
    if actual_digest != EXPECTED_CONFIG_SHA256:
        raise LocalCoderError(
            "OpenCode config digest mismatch: "
            f"expected {EXPECTED_CONFIG_SHA256}, got {actual_digest}"
        )
    _validate_config_contract(
        value,
        expected_base_url="{env:UGTOMS_LOCAL_CODER_BASE_URL}/v1",
        source_config=True,
    )
    return ValidatedConfig(path=resolved, raw=raw, sha256=actual_digest)


def _validate_config_contract(
    value: object,
    *,
    expected_base_url: str,
    expected_instructions: list[str] | None = None,
    source_config: bool = False,
) -> None:
    if not isinstance(value, dict):
        raise LocalCoderError("OpenCode config must contain a JSON object")
    try:
        provider = value["provider"]["local-runtime"]
        model = provider["models"]["local-qwen3-30b-a3b"]
        base_url = provider["options"]["baseURL"]
    except (KeyError, TypeError) as exc:
        raise LocalCoderError(f"OpenCode config is missing the local provider contract: {exc}") from exc
    if not all(isinstance(item, dict) for item in (provider, model)):
        raise LocalCoderError("OpenCode local provider and model entries must be objects")
    problems: list[str] = []
    if source_config:
        allowed_keys = {
            "$schema", "model", "small_model", "default_agent", "subagent_depth",
            "share", "autoupdate", "enabled_providers", "provider", "agent",
            "permission", "compaction", "formatter", "mcp", "plugin",
        }
        unexpected = sorted(set(value) - allowed_keys)
        if unexpected:
            problems.append(f"unexpected top-level config fields: {unexpected!r}")
        if set(value.get("provider", {})) != {"local-runtime"}:
            problems.append("the source config must define exactly one local provider")
        models = provider.get("models")
        if not isinstance(models, dict) or set(models) != {"local-qwen3-30b-a3b"}:
            problems.append("the source config must define exactly one local model")
    if value.get("model") != MODEL_ID or value.get("small_model") != MODEL_ID:
        problems.append("primary and small models must both use the local runtime model")
    if value.get("enabled_providers") != ["local-runtime"]:
        problems.append("the local provider must be the only enabled provider")
    if base_url != expected_base_url:
        problems.append(f"provider baseURL must be {expected_base_url!r}")
    if model.get("tool_call") is not True:
        problems.append("tool calling must be enabled")
    if model.get("limit") != {"context": REQUIRED_CONTEXT_TOKENS, "output": 4_096}:
        problems.append("model limits must be context=32768 and output=4096")
    if value.get("share") != "disabled" or value.get("autoupdate") is not False:
        problems.append("sharing and automatic updates must be disabled")
    if value.get("subagent_depth") != 0:
        problems.append("subagent depth must be zero")
    if value.get("default_agent") != "local-coder":
        problems.append("the local-coder agent must be the default")
    if value.get("mcp") != {} or value.get("plugin") != [] or value.get("formatter") is not False:
        problems.append("MCP servers and plugins must be absent")
    instructions = value.get("instructions")
    if expected_instructions is None:
        if instructions not in (None, []):
            problems.append("source config instruction paths are not permitted")
    elif instructions != expected_instructions:
        problems.append("resolved config must load exactly the digest-verified substrate contract")
    if value.get("references") not in (None, {}):
        problems.append("remote or external config references are not permitted")
    agents = value.get("agent")
    local_agent = agents.get("local-coder") if isinstance(agents, dict) else None
    if not isinstance(local_agent, dict):
        local_agent = {}
        problems.append("the local-coder agent configuration is missing")
    agent_permission = local_agent.get("permission")
    if (
        local_agent.get("mode") != "primary"
        or not isinstance(agent_permission, dict)
        or agent_permission.get("task") != "deny"
    ):
        problems.append("the primary local-coder agent must deny subagents")
    if local_agent.get("prompt") != CANONICAL_AGENT_PROMPT:
        problems.append("agent prompt must exactly match the pinned local-coder prompt")
    permission = value.get("permission")
    if not isinstance(permission, dict):
        permission = {}
        problems.append("global permissions must be an object")
    if permission.get("*") != "deny":
        problems.append("unknown tools must be denied")
    for allowed in ("read", "glob", "grep", "list", "lsp"):
        if permission.get(allowed) != "allow":
            problems.append(f"permission {allowed!r} must be allowed")
    for denied in ("task", "external_directory", "webfetch", "websearch", "skill"):
        if permission.get(denied) != "deny":
            problems.append(f"permission {denied!r} must be denied")
    shell_rules = permission.get("bash")
    if permission.get("edit") != "ask" or not isinstance(shell_rules, dict):
        problems.append("edits and shell commands must require approval")
    elif shell_rules.get("*") != "ask":
        problems.append("ordinary shell commands must require approval")
    else:
        for command in (
            "git commit*",
            "git push*",
            "git reset --hard*",
            "git clean*",
            "rm *",
            "Remove-Item *",
        ):
            if shell_rules.get(command) != "deny":
                problems.append(f"destructive shell rule {command!r} must be denied")
    if problems:
        raise LocalCoderError("unsafe OpenCode config: " + "; ".join(problems))


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_canonical_artifact(artifact: Path) -> Path:
    """Resolve one artifact and require the sole sealed project artifact root."""

    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
        expected = DEFAULT_ARTIFACT.resolve(strict=True)
        resolved = artifact.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"could not resolve canonical local-agent artifact: {exc}") from exc
    if not expected.is_dir():
        raise LocalCoderError(f"canonical artifact is not a directory: {expected}")
    if not _within(expected, project_root):
        raise LocalCoderError("canonical artifact resolves outside the intended project repository")
    if resolved != expected:
        raise LocalCoderError(
            "only the canonical sealed local-agent artifact is permitted: "
            f"expected {expected}, got {resolved}"
        )
    return resolved


def _canonical_project_file(project_relative: str, label: str) -> Path:
    relative = Path(project_relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise LocalCoderError(f"internal canonical {label} path is unsafe: {project_relative!r}")
    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
        resolved = (project_root / relative).resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"canonical {label} does not resolve: {exc}") from exc
    if not _within(resolved, project_root):
        raise LocalCoderError(f"canonical {label} resolves outside the intended project repository")
    if not resolved.is_file():
        raise LocalCoderError(f"canonical {label} is not a regular file: {resolved}")
    return resolved


def _resolve_artifact_reference(artifact: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, str) or not reference or reference != reference.strip():
        raise LocalCoderError(f"artifact {label} path must be a non-empty normalized string")
    relative = Path(reference)
    if relative.is_absolute() or relative.drive:
        raise LocalCoderError(f"artifact {label} path must be repository-relative")
    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
        resolved = (artifact / relative).resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"artifact {label} reference does not resolve: {exc}") from exc
    if not _within(resolved, project_root):
        raise LocalCoderError(
            f"artifact {label} reference resolves outside the intended project repository"
        )
    if not resolved.is_file():
        raise LocalCoderError(f"artifact {label} reference is not a regular file: {resolved}")
    return resolved


def _expected_reference_text(artifact: Path, project_relative: str, label: str) -> str:
    expected = _canonical_project_file(project_relative, label)
    try:
        return Path(os.path.relpath(expected, artifact)).as_posix()
    except ValueError as exc:
        raise LocalCoderError(f"canonical {label} is not addressable from the artifact") from exc


def _validate_pinned_record(
    artifact: Path,
    record: object,
    *,
    project_relative: str,
    label: str,
    verify_file: bool = True,
) -> Path:
    if not isinstance(record, Mapping):
        raise LocalCoderError(f"artifact {label} record must be an object")
    expected_reference = _expected_reference_text(artifact, project_relative, label)
    resolved = _resolve_artifact_reference(artifact, record.get("path"), label)
    expected_path = _canonical_project_file(project_relative, label)
    if record.get("path") != expected_reference or resolved != expected_path:
        raise LocalCoderError(
            f"artifact {label} must reference exactly {expected_reference!r}"
        )
    expected_bytes, expected_digest = CANONICAL_REFERENCE_RECORDS[project_relative]
    if record.get("bytes") != expected_bytes:
        raise LocalCoderError(
            f"artifact {label} byte contract changed: expected {expected_bytes}, "
            f"got {record.get('bytes')!r}"
        )
    digest = record.get("sha256")
    if not isinstance(digest, str) or digest.lower() != expected_digest:
        raise LocalCoderError(f"artifact {label} does not bind the canonical SHA-256")
    if verify_file and (
        resolved.stat().st_size != expected_bytes
        or _sha256_file(resolved) != expected_digest
    ):
        raise LocalCoderError(f"canonical {label} file does not match its pinned record")
    return resolved


def _validate_evidence_snapshot(artifact: Path, record: object) -> Mapping[str, Any]:
    """Bind deployment status to the committed, externally pinned gate snapshot."""

    evidence_path = _validate_pinned_record(
        artifact,
        record,
        project_relative=CANONICAL_VALIDATION_EVIDENCE_PATH,
        label="validation evidence",
    )
    snapshot = _canonical_project_file(
        CANONICAL_VALIDATION_SNAPSHOT_PATH, "validation evidence snapshot"
    )
    for path, label in (
        (evidence_path, "artifact validation evidence"),
        (snapshot, "committed validation evidence snapshot"),
    ):
        if path.stat().st_size != EXPECTED_VALIDATION_EVIDENCE_BYTES:
            raise LocalCoderError(f"{label} byte length differs from the pinned snapshot")
        if _sha256_file(path) != EXPECTED_VALIDATION_EVIDENCE_SHA256:
            raise LocalCoderError(f"{label} SHA-256 differs from the pinned snapshot")
    try:
        evidence = json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalCoderError(f"could not parse pinned validation evidence: {exc}") from exc
    if not isinstance(evidence, Mapping):
        raise LocalCoderError("pinned validation evidence must be a JSON object")
    return evidence


def _validate_evidence_claims(
    evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Check the measured pass fields the local coding claim depends on."""

    aggregate = evidence.get("aggregate")
    benchmark = evidence.get("benchmark")
    payload = evidence.get("payload")
    thresholds = evidence.get("thresholds")
    if not all(
        isinstance(value, Mapping)
        for value in (aggregate, benchmark, payload, thresholds)
    ):
        raise LocalCoderError("pinned validation evidence is missing measured sections")
    prompt = benchmark.get("prompt")
    generation = benchmark.get("generation")
    if not isinstance(prompt, Mapping) or not isinstance(generation, Mapping):
        raise LocalCoderError("pinned validation benchmark sections are incomplete")
    expected = {
        "experiment": "nhdf_hybrid_full_model_functional_gate",
        "artifact_format": hybrid.HYBRID_FORMAT,
        "passed": True,
        "status": "functional-hybrid-pass",
        "runtime_revision": "f8dbcd61893702976f9ab03be89c2b9f436d532c",
        "runtime_build_number": 10_720,
        "runtime_argument_profile": "current-2026",
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise LocalCoderError(
                f"pinned validation evidence field {field!r} does not match {value!r}"
            )
    if evidence.get("execution_profile_sha256") != hybrid._execution_profile_sha256(
        dict(manifest)
    ):
        raise LocalCoderError("validation evidence is not bound to this execution profile")
    if payload.get("bytes") != CANONICAL_REFERENCE_RECORDS[CANONICAL_PAYLOAD_PATH][0] or (
        payload.get("sha256")
        != CANONICAL_REFERENCE_RECORDS[CANONICAL_PAYLOAD_PATH][1]
    ):
        raise LocalCoderError("validation evidence is not bound to the canonical payload")
    aggregate_expected = {
        "functional_prompts_passed": 4,
        "functional_prompts_total": 4,
        "allocated_context_tokens": REQUIRED_CONTEXT_TOKENS,
        "allocated_context_passed": True,
        "full_offload_passed": True,
        "peak_gpu_memory_mib": 11_068,
        "target_vram_mib": 12_227,
        "contract_target_vram_mib": 12_227,
        "headroom_mib": 1_159,
        "resource_gate_passed": True,
        "throughput_gate_passed": True,
    }
    if any(aggregate.get(field) != value for field, value in aggregate_expected.items()):
        raise LocalCoderError("validation evidence aggregate does not match the measured 32K gate")
    if thresholds.get("minimum_generation_tokens_per_second") != 80.0:
        raise LocalCoderError("validation evidence throughput threshold changed")
    if thresholds.get("full_offload_required") != [49, 49]:
        raise LocalCoderError("validation evidence offload threshold changed")
    if prompt.get("tokens") != 64 or prompt.get("average_tokens_per_second") != 442.151809:
        raise LocalCoderError("validation evidence prompt measurement changed")
    if generation.get("tokens") != 64 or (
        generation.get("average_tokens_per_second") != 132.502673
    ):
        raise LocalCoderError("validation evidence generation measurement changed")


def _validate_exact_record_sequence(
    artifact: Path,
    records: object,
    *,
    expected_paths: Sequence[str],
    label: str,
    verify_files: bool = True,
) -> None:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise LocalCoderError(
            f"artifact {label} must contain exactly {len(expected_paths)} canonical records"
        )
    for index, (record, expected) in enumerate(zip(records, expected_paths)):
        _validate_pinned_record(
            artifact,
            record,
            project_relative=expected,
            label=f"{label}:{index}",
            verify_file=verify_files,
        )


def _validate_exact_entrypoint(
    artifact: Path,
    reference: object,
    *,
    project_relative: str,
    label: str,
) -> None:
    expected_reference = _expected_reference_text(artifact, project_relative, label)
    resolved = _resolve_artifact_reference(artifact, reference, label)
    expected_path = _canonical_project_file(project_relative, label)
    if reference != expected_reference or resolved != expected_path:
        raise LocalCoderError(
            f"artifact {label} must reference exactly {expected_reference!r}"
        )


def _validate_artifact_references(
    artifact: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate every path-bearing manifest field against the pinned project layout."""

    try:
        runtime = manifest["runtime"]
        substrate = manifest["substrate"]
        validation = manifest["validation"]
        if not all(isinstance(value, Mapping) for value in (runtime, substrate, validation)):
            raise TypeError("runtime, substrate, and validation must be objects")
        _validate_pinned_record(
            artifact,
            manifest["payload"],
            project_relative=CANONICAL_PAYLOAD_PATH,
            label="model payload",
            verify_file=False,
        )
        _validate_pinned_record(
            artifact,
            manifest["source_record"],
            project_relative=CANONICAL_SOURCE_RECORD_PATH,
            label="source record",
        )
        _validate_pinned_record(
            artifact,
            substrate["specification"],
            project_relative=CANONICAL_SPECIFICATION_PATH,
            label="substrate specification",
        )
        _validate_exact_record_sequence(
            artifact,
            runtime["files"],
            expected_paths=CANONICAL_RUNTIME_PATHS,
            label="runtime files",
            verify_files=False,
        )
        _validate_exact_record_sequence(
            artifact,
            manifest["assurance_evidence"],
            expected_paths=CANONICAL_ASSURANCE_PATHS,
            label="assurance evidence",
        )
        _validate_exact_entrypoint(
            artifact,
            runtime["entrypoint"],
            project_relative=CANONICAL_RUNTIME_ENTRYPOINT,
            label="runtime entrypoint",
        )
        _validate_exact_entrypoint(
            artifact,
            runtime["benchmark_entrypoint"],
            project_relative=CANONICAL_BENCHMARK_ENTRYPOINT,
            label="benchmark entrypoint",
        )
        _validate_exact_entrypoint(
            artifact,
            runtime["server_entrypoint"],
            project_relative=CANONICAL_SERVER_ENTRYPOINT,
            label="server entrypoint",
        )
        evidence = _validate_evidence_snapshot(artifact, validation["evidence"])
    except (KeyError, TypeError) as exc:
        raise LocalCoderError(f"artifact reference contract is incomplete: {exc}") from exc
    return evidence


def _approved_artifact_file(project_relative: str) -> ApprovedArtifactFile:
    expected_bytes, expected_sha256 = CANONICAL_REFERENCE_RECORDS[project_relative]
    return ApprovedArtifactFile(
        path=_canonical_project_file(project_relative, "approved execution file"),
        bytes=expected_bytes,
        sha256=expected_sha256,
    )


def validate_agent_artifact(artifact: Path) -> ValidatedAgentArtifact:
    """Require the one canonical measured 32K/q4 local-agent artifact."""

    artifact = resolve_canonical_artifact(artifact)

    try:
        manifest, manifest_sha256 = hybrid.load_hybrid_manifest_snapshot(artifact)
        profile = manifest["execution_profile"]
        validation = manifest["validation"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LocalCoderError(f"could not load local-agent artifact manifest: {exc}") from exc
    evidence = _validate_artifact_references(artifact, manifest)
    _validate_evidence_claims(evidence, manifest)
    event_chain = hybrid._verify_event_chain(manifest)
    if event_chain.get("ok") is not True:
        raise LocalCoderError(
            "artifact event chain is invalid: " + str(event_chain.get("error"))
        )
    problems: list[str] = []
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    payload = manifest.get("payload")
    codec = manifest.get("weight_codec")
    if manifest.get("artifact_kind") != "external-codec-reference":
        problems.append("artifact kind must remain the external-codec reference")
    if not isinstance(model, Mapping) or (
        model.get("id") != "Qwen/Qwen3-30B-A3B-Instruct-2507"
        or model.get("source_revision") != "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        or model.get("parameters") != 30_532_122_624
    ):
        problems.append("artifact model identity or source revision changed")
    if not isinstance(runtime, Mapping) or (
        runtime.get("implementation") != "llama.cpp"
        or runtime.get("revision") != "f8dbcd61893702976f9ab03be89c2b9f436d532c"
        or runtime.get("build_number") != 10_720
        or runtime.get("argument_profile") != "current-2026"
    ):
        problems.append("artifact runtime identity or argument profile changed")
    if not isinstance(payload, Mapping) or payload.get("link_mode") != "workspace-relative-reference":
        problems.append("artifact payload must remain a workspace-relative reference")
    if not isinstance(codec, Mapping) or (
        codec.get("container") != "GGUF"
        or codec.get("profile") != "IQ2_M mixed-bit"
        or codec.get("nhdf_native_codec") is not False
    ):
        problems.append("artifact codec identity changed")
    if validation.get("status") != "VALIDATED" or validation.get("deployment_loadable") is not True:
        problems.append("artifact must have measured VALIDATED deployment status")
    if profile.get("maximum_context_tokens") != REQUIRED_CONTEXT_TOKENS:
        problems.append(f"artifact context must be exactly {REQUIRED_CONTEXT_TOKENS}")
    if profile.get("kv_cache_k") != REQUIRED_KV_CACHE or profile.get("kv_cache_v") != REQUIRED_KV_CACHE:
        problems.append(f"both artifact KV caches must use {REQUIRED_KV_CACHE}")
    if profile.get("expected_offloaded_layers") != [49, 49]:
        problems.append("artifact must declare complete 49/49 GPU offload")
    if problems:
        raise LocalCoderError("artifact is not the validated local-agent profile: " + "; ".join(problems))
    runtime_files = tuple(
        _approved_artifact_file(path) for path in CANONICAL_RUNTIME_PATHS
    )
    server_file = next(
        item
        for item in runtime_files
        if item.path == _canonical_project_file(
            CANONICAL_SERVER_ENTRYPOINT, "approved server entrypoint"
        )
    )
    approval = ArtifactApproval(
        artifact_dir=artifact,
        reference_root=PROJECT_ROOT,
        manifest_sha256=manifest_sha256,
        payload=_approved_artifact_file(CANONICAL_PAYLOAD_PATH),
        runtime_files=runtime_files,
        server=server_file,
    )
    return ValidatedAgentArtifact(
        artifact=artifact,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        server_approval=approval,
    )


def isolated_environment(config: ValidatedConfig, base_url: str) -> dict[str, str]:
    """Build an environment that cannot inherit OpenCode account/config state."""

    if not isinstance(config, ValidatedConfig):
        raise LocalCoderError("isolated environment requires a validated config snapshot")
    state_paths = {
        "XDG_CONFIG_HOME": LOCAL_STATE_ROOT / "xdg-config",
        "XDG_DATA_HOME": LOCAL_STATE_ROOT / "xdg-data",
        "XDG_CACHE_HOME": LOCAL_STATE_ROOT / "xdg-cache",
        "XDG_STATE_HOME": LOCAL_STATE_ROOT / "xdg-state",
    }
    for path in state_paths.values():
        _ensure_safe_project_directory(path)
    home = LOCAL_STATE_ROOT / "home"
    appdata = LOCAL_STATE_ROOT / "appdata"
    local_appdata = LOCAL_STATE_ROOT / "local-appdata"
    temporary_directory = LOCAL_STATE_ROOT / "temp"
    for path in (home, appdata, local_appdata, temporary_directory):
        _ensure_safe_project_directory(path)
    config_directory = LOCAL_STATE_ROOT / "config-dir"
    _ensure_safe_project_directory(config_directory)
    isolated_config = config_directory / "opencode.json"
    _atomic_write_local(isolated_config, config.raw)
    if _sha256_file(isolated_config) != config.sha256:
        raise LocalCoderError("isolated OpenCode config failed its post-write digest check")
    installed_contract = install_substrate_contract(state_paths["XDG_CONFIG_HOME"])

    try:
        inline_config = json.loads(config.raw.decode("utf-8"))
        inline_config["provider"]["local-runtime"]["options"]["baseURL"] = (
            f"{base_url}/v1"
        )
        inline_config["instructions"] = [str(installed_contract)]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LocalCoderError(f"could not prepare isolated OpenCode config: {exc}") from exc

    allowed_parent_keys = {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
        "OS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
        "NUMBER_OF_PROCESSORS", "LANG", "LC_ALL", "TERM", "COLORTERM",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_parent_keys
    }
    environment.update({key: str(value) for key, value in state_paths.items()})
    environment["HOME"] = str(home)
    environment["USERPROFILE"] = str(home)
    environment["APPDATA"] = str(appdata)
    environment["LOCALAPPDATA"] = str(local_appdata)
    environment["TEMP"] = str(temporary_directory)
    environment["TMP"] = str(temporary_directory)
    environment["HOMEDRIVE"] = home.drive
    environment["HOMEPATH"] = str(home)[len(home.drive) :]
    environment["OPENCODE_CONFIG"] = str(isolated_config)
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        inline_config, separators=(",", ":")
    )
    environment["OPENCODE_CONFIG_DIR"] = str(config_directory)
    environment["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
    environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "1"
    environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "1"
    environment["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    environment["OPENCODE_PURE"] = "1"
    environment["UGTOMS_LOCAL_CODER_BASE_URL"] = base_url
    environment["UGTOMS_SUBSTRATE_CONTRACT_SHA256"] = EXPECTED_CONTRACT_SHA256
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    return environment


def verify_execution_inputs(
    executable: Path,
    config: ValidatedConfig,
    environment: Mapping[str, str],
) -> None:
    """Rehash every local-client trust anchor immediately before execution."""

    try:
        project_root = PROJECT_ROOT.resolve(strict=True)
    except OSError as exc:
        raise LocalCoderError(f"project root changed after validation: {exc}") from exc
    expected_executable = _require_unaliased_executable(
        OPENCODE_EXE,
        label="canonical project-local OpenCode executable",
        trusted_root=project_root,
    )
    resolved_executable = _require_unaliased_executable(
        executable,
        label="requested OpenCode executable",
        trusted_root=project_root,
    )
    if resolved_executable != expected_executable:
        raise LocalCoderError("OpenCode executable path changed after validation")
    if _sha256_file(resolved_executable) != EXPECTED_OPENCODE_SHA256:
        raise LocalCoderError("OpenCode executable changed after validation")
    if _sha256_file(config.path.resolve(strict=True)) != config.sha256:
        raise LocalCoderError("canonical OpenCode config changed after validation")
    isolated_config = Path(environment.get("OPENCODE_CONFIG", ""))
    if not isolated_config.is_file() or _sha256_file(isolated_config) != config.sha256:
        raise LocalCoderError("isolated OpenCode config changed after installation")
    source_contract = SUBSTRATE_CONTRACT.resolve(strict=True)
    if _sha256_file(source_contract) != EXPECTED_CONTRACT_SHA256:
        raise LocalCoderError("canonical AGENT_CONTRACT changed after validation")
    installed_contract = (
        Path(environment.get("XDG_CONFIG_HOME", "")) / "opencode" / "AGENTS.md"
    )
    if not installed_contract.is_file() or (
        _sha256_file(installed_contract) != EXPECTED_CONTRACT_SHA256
    ):
        raise LocalCoderError("installed AGENT_CONTRACT changed after installation")


def validate_resolved_config(
    executable: Path,
    target: Path,
    environment: dict[str, str],
    *,
    expected_base_url: str,
    expected_contract_path: Path,
) -> None:
    """Ask the pinned client to resolve all config layers, then recheck safety."""

    try:
        result = subprocess.run(
            [str(executable), "--pure", "debug", "config"],
            cwd=str(target),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalCoderError(f"could not resolve the OpenCode config: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise LocalCoderError(f"OpenCode rejected the isolated config: {detail}")
    try:
        resolved = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LocalCoderError("OpenCode returned an invalid resolved config") from exc
    _validate_config_contract(
        resolved,
        expected_base_url=f"{expected_base_url}/v1",
        expected_instructions=[str(expected_contract_path)],
    )


def build_opencode_command(
    executable: Path, run_args: tuple[str, ...] | None
) -> list[str]:
    command = [str(executable), "--pure"]
    if run_args is None:
        command.extend(["--model", MODEL_ID, "--agent", "local-coder"])
    else:
        command.extend(
            ["run", "--model", MODEL_ID, "--agent", "local-coder", *run_args]
        )
    return command


def run_local_coder(
    options: LaunchOptions,
    *,
    server_factory: Callable[..., HybridServer] = HybridServer,
    client_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    target, worktree_root = validate_git_target(options.target)
    config = validate_config(options.config)
    opencode_executable = validate_local_install(OPENCODE_EXE)
    artifact = resolve_canonical_artifact(options.artifact)
    validated_artifact = validate_agent_artifact(artifact)

    server = server_factory(
        artifact,
        port=options.port,
        threads=options.threads,
        startup_timeout_seconds=options.startup_timeout,
        request_timeout_seconds=600.0,
        verify_payload_hash=True,
        artifact_approval=validated_artifact.server_approval,
    )
    try:
        environment = isolated_environment(config, server.base_url)
        verify_execution_inputs(opencode_executable, config, environment)
        validate_resolved_config(
            opencode_executable,
            target,
            environment,
            expected_base_url=server.base_url,
            expected_contract_path=(
                Path(environment.get("XDG_CONFIG_HOME", LOCAL_STATE_ROOT / "xdg-config"))
                / "opencode"
                / "AGENTS.md"
            ),
        )
        server.start()
        command = build_opencode_command(opencode_executable, options.run_args)
        print(f"Local coder target: {target}", file=sys.stderr)
        print(f"Git worktree: {worktree_root}", file=sys.stderr)
        print(f"Local model endpoint: {server.base_url}/v1", file=sys.stderr)
        if options.quick:
            print(
                "Verification: strict sealed payload check (--quick is compatibility-only)",
                file=sys.stderr,
            )
        else:
            print("Verification: strict sealed payload check", file=sys.stderr)
        verify_execution_inputs(opencode_executable, config, environment)
        completed = client_runner(
            command,
            cwd=str(target),
            env=environment,
            check=False,
        )
        return int(completed.returncode)
    finally:
        server.stop()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_local_coder(parse_options(argv))
    except (LocalCoderError, OSError) as exc:
        print(f"local-coder error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Local coder interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
