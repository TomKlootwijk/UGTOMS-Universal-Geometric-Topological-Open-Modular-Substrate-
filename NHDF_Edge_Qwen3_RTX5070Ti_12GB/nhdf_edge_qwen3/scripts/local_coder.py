#!/usr/bin/env python3
"""Launch the pinned, project-local OpenCode client against an owned NHDF server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nhdf_edge import hybrid  # noqa: E402
from nhdf_edge.server import DEFAULT_PORT, HybridServer  # noqa: E402


PINNED_OPENCODE_VERSION = "1.18.25"
DEFAULT_ARTIFACT = PROJECT_ROOT / "packs" / "qwen3-30b-a3b-iq2m-32k-q4kv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "opencode_nhdf_local.json"
SUBSTRATE_CONTRACT = PROJECT_ROOT / "substrate" / "AGENT_CONTRACT.md"
EXPECTED_CONFIG_SHA256 = "66a6e5adcc98fc80921bc4e0386f341c80a4f824597e26309b8499344f440ab1"
EXPECTED_CONTRACT_SHA256 = "ad3fc7963c5ab4f222892816a73078458e4d7714cede8a960fb1dcf37df28b40"
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
FORBIDDEN_RUN_OPTIONS = frozenset(
    {
        "--agent",
        "--attach",
        "--auto",
        "--dir",
        "--file",
        "--model",
        "--password",
        "--port",
        "--share",
        "--username",
        "-f",
        "-m",
        "-p",
        "-u",
    }
)


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def install_substrate_contract(xdg_config_home: Path) -> Path:
    """Install the exact tracked contract into isolated global OpenCode rules."""

    data = _read_pinned_file(
        SUBSTRATE_CONTRACT, EXPECTED_CONTRACT_SHA256, "substrate contract"
    )
    destination_directory = xdg_config_home / "opencode"
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / "AGENTS.md"
    temporary = destination_directory / f".AGENTS.md.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
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
        help="skip the large model-payload rehash for a routine daily launch",
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
            compact_short_option = any(
                argument.startswith(prefix) and argument != prefix
                for prefix in ("-f", "-m", "-p", "-u")
            )
            if option in FORBIDDEN_RUN_OPTIONS or compact_short_option:
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

    try:
        result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
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


def validate_local_install(executable: Path) -> None:
    if not executable.is_file():
        raise LocalCoderError(
            "project-local OpenCode is not installed; run scripts/setup_local_coder.ps1"
        )
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalCoderError(f"could not run project-local OpenCode: {exc}") from exc
    installed = result.stdout.strip().splitlines()
    version = installed[0].strip() if installed else ""
    if result.returncode != 0 or version != PINNED_OPENCODE_VERSION:
        raise LocalCoderError(
            f"OpenCode {PINNED_OPENCODE_VERSION} is required; found {version or 'unknown'}"
        )


def validate_config(config_path: Path) -> Path:
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
    return resolved


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


def validate_agent_artifact(artifact: Path) -> dict[str, object]:
    """Require the measured 32K/q4 local-agent execution profile."""

    try:
        manifest = hybrid.load_hybrid_manifest(artifact)
        profile = manifest["execution_profile"]
        validation = manifest["validation"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LocalCoderError(f"could not load local-agent artifact manifest: {exc}") from exc
    problems: list[str] = []
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
    return manifest


def isolated_environment(config: Path, base_url: str) -> dict[str, str]:
    """Build an environment that cannot inherit OpenCode account/config state."""

    state_paths = {
        "XDG_CONFIG_HOME": LOCAL_STATE_ROOT / "xdg-config",
        "XDG_DATA_HOME": LOCAL_STATE_ROOT / "xdg-data",
        "XDG_CACHE_HOME": LOCAL_STATE_ROOT / "xdg-cache",
        "XDG_STATE_HOME": LOCAL_STATE_ROOT / "xdg-state",
    }
    for path in state_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    home = LOCAL_STATE_ROOT / "home"
    appdata = LOCAL_STATE_ROOT / "appdata"
    local_appdata = LOCAL_STATE_ROOT / "local-appdata"
    temporary_directory = LOCAL_STATE_ROOT / "temp"
    for path in (home, appdata, local_appdata, temporary_directory):
        path.mkdir(parents=True, exist_ok=True)
    config_directory = LOCAL_STATE_ROOT / "config-dir"
    config_directory.mkdir(parents=True, exist_ok=True)
    installed_contract = install_substrate_contract(state_paths["XDG_CONFIG_HOME"])

    try:
        inline_config = json.loads(config.read_text(encoding="utf-8"))
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
    environment["OPENCODE_CONFIG"] = str(config)
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
    validate_local_install(OPENCODE_EXE)
    artifact = options.artifact.resolve(strict=True)
    if not artifact.is_dir():
        raise LocalCoderError(f"artifact is not a directory: {artifact}")
    validate_agent_artifact(artifact)

    server = server_factory(
        artifact,
        port=options.port,
        threads=options.threads,
        startup_timeout_seconds=options.startup_timeout,
        request_timeout_seconds=600.0,
        verify_payload_hash=not options.quick,
    )
    try:
        environment = isolated_environment(config, server.base_url)
        validate_resolved_config(
            OPENCODE_EXE,
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
        command = build_opencode_command(OPENCODE_EXE, options.run_args)
        print(f"Local coder target: {target}", file=sys.stderr)
        print(f"Git worktree: {worktree_root}", file=sys.stderr)
        print(f"Local model endpoint: {server.base_url}/v1", file=sys.stderr)
        if options.quick:
            print("Verification: quick metadata/runtime check (payload rehash skipped)", file=sys.stderr)
        else:
            print("Verification: full sealed payload check", file=sys.stderr)
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
