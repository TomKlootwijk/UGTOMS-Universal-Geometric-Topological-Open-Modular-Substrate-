#!/usr/bin/env python3
"""Resident, verified local-coding desktop UI for the canonical UGTOMS profile.

The display layer is intentionally thin.  Artifact, client, configuration, target,
and server validation remain owned by :mod:`scripts.local_coder` and
``HybridServer``.  The classes above ``LocalCoderApp`` are display-free so the
security-sensitive lifecycle and downloader can be tested without a Tk display.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import re
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
    build_opener,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "local_coder.py"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_local_coder.ps1"
CONTROL_SOURCE = (
    PROJECT_ROOT
    / "models"
    / "Qwen3-30B-A3B-Instruct-2507-IQ2_M"
    / "CONTROL_SOURCE.json"
)
GUI_VERSION = "0.1"
MAX_PROMPT_CHARACTERS = 16_000
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class GuiError(RuntimeError):
    """Actionable failure surfaced by the display-free GUI services."""


class DownloadCancelled(GuiError):
    """The user cancelled a resumable model download."""


class PromptCancelled(GuiError):
    """The user cancelled a running OpenCode prompt."""


class OperationCancelled(GuiError):
    """The user cancelled a non-prompt GUI operation."""


class InteractionMode(Enum):
    READ_ONLY = "read-only"
    SCOPED_EDITS = "scoped-edits"


class ReadinessState(Enum):
    READY = "ready"
    MISSING = "missing"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ModelControl:
    source_path: Path
    artifact: str
    expected_bytes: int
    sha256: str
    immutable_url: str
    source_revision: str

    @property
    def destination(self) -> Path:
        return self.source_path.parent / self.artifact


@dataclass(frozen=True)
class DownloadProgress:
    phase: str
    completed_bytes: int
    total_bytes: int
    resumed: bool = False


@dataclass(frozen=True)
class ReadinessCard:
    key: str
    label: str
    state: ReadinessState
    detail: str


@dataclass(frozen=True)
class ReadinessSnapshot:
    cards: tuple[ReadinessCard, ...]

    @property
    def ready_to_start(self) -> bool:
        return bool(self.cards) and all(
            card.state is ReadinessState.READY for card in self.cards
        )

    def by_key(self, key: str) -> ReadinessCard:
        for card in self.cards:
            if card.key == key:
                return card
        raise KeyError(key)


@dataclass(frozen=True)
class TranscriptEvent:
    kind: str
    title: str
    text: str
    detail: str = ""
    session_id: str | None = None


@dataclass(frozen=True)
class ClientRunResult:
    returncode: int
    session_id: str | None


@dataclass
class _ResidentContext:
    target: Path
    worktree_root: Path
    executable: Path
    config: Any
    environment: dict[str, str]
    server: Any
    session_id: str | None = None
    scoped_work_authorized: bool = False


class _Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(self, *args: object) -> object: ...


class _UrlOpener(Protocol):
    def open(self, request: Request, timeout: float) -> _Response: ...


class _ClientExecutor(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        cancel_event: threading.Event,
        on_event: Callable[[TranscriptEvent], None],
    ) -> ClientRunResult: ...

    def cancel(self) -> None: ...


_LAUNCHER: ModuleType | None = None
_LAUNCHER_LOCK = threading.Lock()


def load_launcher() -> ModuleType:
    """Load the hardened launcher without turning ``scripts`` into a package."""

    global _LAUNCHER
    with _LAUNCHER_LOCK:
        if _LAUNCHER is not None:
            return _LAUNCHER
        spec = importlib.util.spec_from_file_location(
            "ugtoms_local_coder", LAUNCHER_PATH
        )
        if spec is None or spec.loader is None:
            raise GuiError(f"Could not load the verified launcher at {LAUNCHER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _LAUNCHER = module
        return module


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is prohibited: {value}")


def _strict_json_loads(raw: str | bytes) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_json_constant,
    )


def _sha256_file(
    path: Path,
    *,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int], None] | None = None,
) -> str:
    digest = hashlib.sha256()
    completed = 0
    try:
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled(
                        "Operation cancelled; no file was promoted."
                    )
                digest.update(block)
                completed += len(block)
                if progress is not None:
                    progress(completed)
    except DownloadCancelled:
        raise
    except OSError as exc:
        raise GuiError(f"Could not read {path}: {exc}") from exc
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        information = path.lstat()
    except OSError as exc:
        raise GuiError(f"Could not inspect path {path}: {exc}") from exc
    attributes = int(getattr(information, "st_file_attributes", 0))
    return stat.S_ISLNK(information.st_mode) or bool(attributes & 0x400)


def _require_safe_path(
    path: Path,
    *,
    trusted_root: Path,
    must_exist: bool,
    require_file: bool = False,
) -> Path:
    """Reject traversal through symlinks, junctions, and Windows reparse points."""

    try:
        root = trusted_root.resolve(strict=True)
        lexical = Path(os.path.abspath(path))
        relative = lexical.relative_to(root)
    except (OSError, ValueError) as exc:
        raise GuiError(f"Path must remain below the project root: {path}") from exc

    current = root
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if _is_reparse_or_symlink(current):
                raise GuiError(
                    f"Path traverses a symlink, junction, or reparse point: {current}"
                )
        elif current != lexical or must_exist:
            raise GuiError(f"Required path does not exist: {current}")

    if must_exist:
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise GuiError(f"Required path does not resolve: {lexical}") from exc
        if os.path.normcase(str(resolved)) != os.path.normcase(str(lexical)):
            raise GuiError(f"Path resolves through an alias: {lexical}")
        if require_file and not resolved.is_file():
            raise GuiError(f"Required path is not a regular file: {resolved}")
    return lexical


class _ArtifactDownloadLock:
    """Serialize one artifact download across GUI processes.

    Windows holds a no-sharing ``CreateFileW`` handle. POSIX uses a nonblocking
    advisory lock and ``O_NOFOLLOW`` when available. The lock is cooperative on
    POSIX; promotion still uses an atomic no-replace hard link and verifies the
    installed bytes before returning.
    """

    def __init__(self, path: Path, *, trusted_root: Path) -> None:
        self.path = path
        self.trusted_root = trusted_root
        self._windows_handle: int | None = None
        self._posix_descriptor: int | None = None

    def __enter__(self) -> "_ArtifactDownloadLock":
        safe = _require_safe_path(
            self.path,
            trusted_root=self.trusted_root,
            must_exist=False,
        )
        try:
            if os.name == "nt":
                self._windows_handle = self._open_windows(safe)
            else:
                import fcntl

                flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(safe, flags, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(descriptor)
                    raise
                self._posix_descriptor = descriptor
            _require_safe_path(
                safe,
                trusted_root=self.trusted_root,
                must_exist=True,
                require_file=True,
            )
        except BaseException as exc:
            self.close()
            if isinstance(exc, GuiError):
                raise
            raise GuiError(
                "Another model download is active, or its lock could not be acquired. "
                "Wait for that download to finish and retry."
            ) from exc
        return self

    @staticmethod
    def _open_windows(path: Path) -> int:
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0xC0000000,  # GENERIC_READ | GENERIC_WRITE
            0,  # deliberately deny read/write/delete sharing
            None,
            4,  # OPEN_ALWAYS
            0x00000080 | 0x00200000,  # NORMAL | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            error = ctypes.get_last_error()
            raise OSError(error, f"could not lock model download: {path}")
        return int(handle)

    def close(self) -> None:
        if self._windows_handle is not None:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            close_handle(self._windows_handle)
            self._windows_handle = None
        if self._posix_descriptor is not None:
            try:
                os.close(self._posix_descriptor)
            except OSError:
                pass
            self._posix_descriptor = None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def load_model_control(
    source_path: Path,
    *,
    trusted_root: Path,
    expected_source_bytes: int | None = None,
    expected_source_sha256: str | None = None,
) -> ModelControl:
    source = _require_safe_path(
        source_path,
        trusted_root=trusted_root,
        must_exist=True,
        require_file=True,
    )
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise GuiError(f"Could not read the model control record: {exc}") from exc
    if expected_source_bytes is not None and len(raw) != expected_source_bytes:
        raise GuiError(
            "Model control record size mismatch: "
            f"expected {expected_source_bytes}, found {len(raw)}"
        )
    actual_source_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        expected_source_sha256 is not None
        and actual_source_sha256 != expected_source_sha256
    ):
        raise GuiError(
            "Model control record digest mismatch: "
            f"expected {expected_source_sha256}, found {actual_source_sha256}"
        )
    try:
        value = _strict_json_loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GuiError(f"Model control record is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GuiError("Model control record must contain a JSON object.")

    artifact = value.get("artifact")
    expected_bytes = value.get("bytes")
    digest = value.get("sha256")
    immutable_url = value.get("immutable_url")
    source_revision = value.get("source_revision")
    source_repository = value.get("source_repository")
    if (
        not isinstance(artifact, str)
        or not artifact.endswith(".gguf")
        or Path(artifact).name != artifact
    ):
        raise GuiError("Model control artifact must be one concrete GGUF filename.")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise GuiError("Model control byte count must be a positive integer.")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise GuiError(
            "Model control SHA-256 must be 64 lowercase hexadecimal characters."
        )
    if (
        not isinstance(source_revision, str)
        or REVISION_PATTERN.fullmatch(source_revision) is None
    ):
        raise GuiError("Model source revision must be a 40-character Git digest.")
    if not isinstance(source_repository, str) or not source_repository.strip():
        raise GuiError("Model source repository is missing.")
    if not isinstance(immutable_url, str):
        raise GuiError("Model control immutable URL is missing.")

    parsed = urlparse(immutable_url)
    decoded_path = unquote(parsed.path)
    expected_prefix = f"/{source_repository}/resolve/{source_revision}/"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
        or not decoded_path.startswith(expected_prefix)
        or decoded_path.rsplit("/", 1)[-1] != artifact
    ):
        raise GuiError(
            "Model URL must be the immutable HTTPS Hugging Face revision and artifact "
            "declared by the control record."
        )
    return ModelControl(
        source_path=source,
        artifact=artifact,
        expected_bytes=expected_bytes,
        sha256=digest,
        immutable_url=immutable_url,
        source_revision=source_revision,
    )


def canonical_model_control(launcher: ModuleType | None = None) -> ModelControl:
    launcher = launcher or load_launcher()
    record = launcher.CANONICAL_REFERENCE_RECORDS[launcher.CANONICAL_SOURCE_RECORD_PATH]
    return load_model_control(
        CONTROL_SOURCE,
        trusted_root=Path(launcher.PROJECT_ROOT),
        expected_source_bytes=int(record[0]),
        expected_source_sha256=str(record[1]),
    )


class _HttpsOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> Request | None:
        parsed = urlparse(new_url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise HTTPError(
                request.full_url,
                code,
                "Refusing a non-HTTPS or credential-bearing model redirect",
                headers,
                file_pointer,
            )
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _default_model_opener() -> OpenerDirector:
    return build_opener(ProxyHandler({}), _HttpsOnlyRedirect())


class ModelDownloader:
    """Resumable immutable-model downloader with verified atomic promotion."""

    def __init__(
        self,
        control: ModelControl,
        *,
        trusted_root: Path,
        opener: _UrlOpener | None = None,
        timeout_seconds: float = 60.0,
        chunk_bytes: int = DOWNLOAD_CHUNK_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("download timeout must be positive")
        if chunk_bytes <= 0:
            raise ValueError("download chunk size must be positive")
        self.control = control
        self.trusted_root = trusted_root
        self.opener = opener or _default_model_opener()
        self.timeout_seconds = timeout_seconds
        self.chunk_bytes = chunk_bytes

    @classmethod
    def canonical(cls, launcher: ModuleType | None = None) -> "ModelDownloader":
        launcher = launcher or load_launcher()
        return cls(
            canonical_model_control(launcher),
            trusted_root=Path(launcher.PROJECT_ROOT),
        )

    def _validate_destination_layout(self) -> tuple[Path, Path]:
        destination = _require_safe_path(
            self.control.destination,
            trusted_root=self.trusted_root,
            must_exist=False,
        )
        _require_safe_path(
            destination.parent,
            trusted_root=self.trusted_root,
            must_exist=True,
        )
        partial = destination.with_name(destination.name + ".download.part")
        _require_safe_path(
            partial,
            trusted_root=self.trusted_root,
            must_exist=False,
        )
        return destination, partial

    def _verify_exact_file(
        self,
        path: Path,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[DownloadProgress], None],
        phase: str,
        resumed: bool,
    ) -> None:
        safe = _require_safe_path(
            path,
            trusted_root=self.trusted_root,
            must_exist=True,
            require_file=True,
        )
        size = safe.stat().st_size
        if size != self.control.expected_bytes:
            raise GuiError(
                f"{phase.capitalize()} file has {size:,} bytes; "
                f"expected {self.control.expected_bytes:,}."
            )

        def report(completed: int) -> None:
            on_progress(
                DownloadProgress(
                    phase=phase,
                    completed_bytes=completed,
                    total_bytes=self.control.expected_bytes,
                    resumed=resumed,
                )
            )

        digest = _sha256_file(
            safe,
            cancel_event=cancel_event,
            progress=report,
        )
        if digest != self.control.sha256:
            raise GuiError(
                f"{phase.capitalize()} model SHA-256 mismatch: expected "
                f"{self.control.sha256}, found {digest}."
            )

    @staticmethod
    def _response_status(response: _Response) -> int:
        status = getattr(response, "status", None)
        if isinstance(status, int):
            return status
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            code = getcode()
            if isinstance(code, int):
                return code
        raise GuiError("Model server response did not include an HTTP status.")

    def _validate_range_response(
        self,
        response: _Response,
        *,
        offset: int,
    ) -> bool:
        status = self._response_status(response)
        final_url = response.geturl() if hasattr(response, "geturl") else ""
        if final_url and urlparse(final_url).scheme != "https":
            raise GuiError("Model download resolved to a non-HTTPS URL.")
        if offset and status == 200:
            return False
        if status not in ({206} if offset else {200, 206}):
            raise GuiError(f"Model server returned unexpected HTTP status {status}.")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
            if match is None:
                raise GuiError(
                    "Resumed model response has no valid Content-Range header."
                )
            start, end, total = (int(item) for item in match.groups())
            if (
                start != offset
                or end < start
                or end >= self.control.expected_bytes
                or total != self.control.expected_bytes
            ):
                raise GuiError(
                    "Resumed model response does not match the exact requested byte range."
                )
        return offset > 0

    def download(
        self,
        *,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[DownloadProgress], None] | None = None,
    ) -> Path:
        cancel_event = cancel_event or threading.Event()
        on_progress = on_progress or (lambda _progress: None)
        if cancel_event.is_set():
            raise DownloadCancelled("Model download was cancelled before it started.")
        destination, _partial = self._validate_destination_layout()
        lock_path = destination.with_name(destination.name + ".download.lock")
        with _ArtifactDownloadLock(lock_path, trusted_root=self.trusted_root):
            return self._download_locked(
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

    def _promote_verified_partial(
        self,
        partial: Path,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[DownloadProgress], None],
        resumed: bool,
    ) -> Path:
        destination, expected_partial = self._validate_destination_layout()
        if os.path.normcase(str(partial)) != os.path.normcase(str(expected_partial)):
            raise GuiError("Internal partial-model path changed before promotion.")
        partial = _require_safe_path(
            partial,
            trusted_root=self.trusted_root,
            must_exist=True,
            require_file=True,
        )
        source_identity = partial.stat()
        if source_identity.st_size != self.control.expected_bytes:
            raise GuiError("Verified partial model changed size before promotion.")
        if cancel_event.is_set():
            raise DownloadCancelled(
                "Model download cancelled; verified partial bytes were kept."
            )
        self._validate_destination_layout()
        try:
            os.link(partial, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise GuiError(
                "Model destination appeared during promotion; refusing overwrite."
            ) from exc
        except (NotImplementedError, OSError) as exc:
            raise GuiError(
                "Could not atomically install the verified model without overwriting an "
                f"existing file: {exc}"
            ) from exc

        installed: Path | None = None
        try:
            installed = _require_safe_path(
                destination,
                trusted_root=self.trusted_root,
                must_exist=True,
                require_file=True,
            )
            if not os.path.samestat(source_identity, installed.stat()):
                raise GuiError(
                    "Installed model identity differs from the verified partial; refusing it."
                )
            self._verify_exact_file(
                installed,
                cancel_event=cancel_event,
                on_progress=on_progress,
                phase="verifying installed model",
                resumed=resumed,
            )
            if cancel_event.is_set():
                raise DownloadCancelled(
                    "Model download cancelled; verified partial bytes were kept."
                )
            if not os.path.samestat(source_identity, partial.stat()):
                raise GuiError(
                    "Partial model identity changed during promotion; refusing cleanup."
                )
        except BaseException:
            # The destination name was created by our no-replace link. Remove only that
            # link when it still names the same source identity; otherwise leave the
            # unexpected path untouched for explicit operator inspection.
            try:
                if installed is not None and os.path.samestat(
                    source_identity, installed.stat()
                ):
                    installed.unlink()
            except OSError:
                pass
            raise

        try:
            partial.unlink()
        except OSError as exc:
            raise GuiError(
                "The model was verified and installed, but the resumable partial name "
                f"could not be removed: {exc}"
            ) from exc
        on_progress(
            DownloadProgress(
                phase="installed",
                completed_bytes=self.control.expected_bytes,
                total_bytes=self.control.expected_bytes,
                resumed=resumed,
            )
        )
        return destination

    def _download_locked(
        self,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[DownloadProgress], None],
    ) -> Path:
        destination, partial = self._validate_destination_layout()

        if destination.exists() or destination.is_symlink():
            self._verify_exact_file(
                destination,
                cancel_event=cancel_event,
                on_progress=on_progress,
                phase="verifying installed model",
                resumed=False,
            )
            return destination

        offset = 0
        if partial.exists() or partial.is_symlink():
            partial = _require_safe_path(
                partial,
                trusted_root=self.trusted_root,
                must_exist=True,
                require_file=True,
            )
            offset = partial.stat().st_size
            if offset > self.control.expected_bytes:
                raise GuiError(
                    "Partial model file is larger than the pinned artifact. Remove only "
                    f"this rejected partial file and retry: {partial}"
                )
            if offset == self.control.expected_bytes:
                try:
                    self._verify_exact_file(
                        partial,
                        cancel_event=cancel_event,
                        on_progress=on_progress,
                        phase="verifying resumed download",
                        resumed=True,
                    )
                except DownloadCancelled:
                    raise
                except GuiError:
                    partial.unlink(missing_ok=True)
                    raise GuiError(
                        "Completed partial model failed SHA-256 and was discarded; retry "
                        "Download Model."
                    ) from None
                if cancel_event.is_set():
                    raise DownloadCancelled(
                        "Model download cancelled; verified partial bytes were kept."
                    )
                return self._promote_verified_partial(
                    partial,
                    cancel_event=cancel_event,
                    on_progress=on_progress,
                    resumed=True,
                )

        if cancel_event.is_set():
            raise DownloadCancelled(
                "Model download cancelled; partial bytes were kept."
            )
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": f"UGTOMS-Local-Coder-GUI/{GUI_VERSION}",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(self.control.immutable_url, headers=headers, method="GET")
        try:
            response_context = self.opener.open(request, timeout=self.timeout_seconds)
            with response_context as response:
                resumed = self._validate_range_response(response, offset=offset)
                if offset and not resumed:
                    # A server may ignore Range.  Reuse its complete 200 response, but
                    # explicitly restart the owned partial rather than append corruption.
                    offset = 0
                mode = "ab" if offset else "wb"
                completed = offset
                on_progress(
                    DownloadProgress(
                        phase="downloading",
                        completed_bytes=completed,
                        total_bytes=self.control.expected_bytes,
                        resumed=resumed,
                    )
                )
                with partial.open(mode) as handle:
                    while True:
                        if cancel_event.is_set():
                            raise DownloadCancelled(
                                "Model download cancelled; partial bytes were kept for resume."
                            )
                        block = response.read(self.chunk_bytes)
                        if not block:
                            break
                        completed += len(block)
                        if completed > self.control.expected_bytes:
                            raise GuiError(
                                "Model server sent more bytes than the pinned artifact; "
                                "nothing was promoted."
                            )
                        handle.write(block)
                        on_progress(
                            DownloadProgress(
                                phase="downloading",
                                completed_bytes=completed,
                                total_bytes=self.control.expected_bytes,
                                resumed=resumed,
                            )
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
        except DownloadCancelled:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise GuiError(
                "Model download failed; verified partial bytes remain resumable. "
                f"Details: {exc}"
            ) from exc

        if partial.stat().st_size != self.control.expected_bytes:
            raise GuiError(
                "Model download ended early at "
                f"{partial.stat().st_size:,}/{self.control.expected_bytes:,} bytes; "
                "press Download Model again to resume."
            )
        if cancel_event.is_set():
            raise DownloadCancelled(
                "Model download cancelled; partial bytes were kept."
            )
        try:
            self._verify_exact_file(
                partial,
                cancel_event=cancel_event,
                on_progress=on_progress,
                phase="verifying download",
                resumed=offset > 0,
            )
        except DownloadCancelled:
            raise
        except GuiError:
            partial.unlink(missing_ok=True)
            raise GuiError(
                "Downloaded model failed the pinned SHA-256 and was discarded; retry "
                "Download Model."
            ) from None

        if cancel_event.is_set():
            raise DownloadCancelled(
                "Model download cancelled; verified partial bytes were kept."
            )
        return self._promote_verified_partial(
            partial,
            cancel_event=cancel_event,
            on_progress=on_progress,
            resumed=offset > 0,
        )


def _compact_json(value: object, *, limit: int = 12_000) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + "\n… [display truncated]"


def _first_string(mapping: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _session_ids(value: object, *, depth: int = 0) -> set[str]:
    if depth > 3 or not isinstance(value, Mapping):
        return set()
    found: set[str] = set()
    for key in ("sessionID", "sessionId", "session_id"):
        candidate = value.get(key)
        if isinstance(candidate, str) and SESSION_ID_PATTERN.fullmatch(candidate):
            found.add(candidate)
    for key in ("part", "info", "properties"):
        child = value.get(key)
        if isinstance(child, Mapping):
            found.update(_session_ids(child, depth=depth + 1))
    return found


class OpenCodeEventParser:
    """Convert pinned OpenCode JSONL events into readable transcript records."""

    def parse_line(self, line: str) -> TranscriptEvent | None:
        if not line.strip():
            return None
        try:
            value = _strict_json_loads(line)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return TranscriptEvent(
                kind="diagnostic",
                title="Unparsed OpenCode output",
                text=line.rstrip(),
                detail=str(exc),
            )
        if not isinstance(value, dict):
            return TranscriptEvent(
                kind="diagnostic",
                title="Unexpected OpenCode record",
                text=_compact_json(value),
            )
        identifiers = _session_ids(value)
        session_id = next(iter(identifiers)) if len(identifiers) == 1 else None
        event_type = str(value.get("type", "unknown")).strip().lower()
        part = value.get("part")
        source = part if isinstance(part, Mapping) else value
        raw_detail = _compact_json(value)

        if len(identifiers) > 1:
            return TranscriptEvent(
                kind="error",
                title="Conflicting OpenCode session identifiers",
                text="The client emitted one event bound to multiple sessions.",
                detail=raw_detail,
            )
        if event_type in {"text", "assistant", "message"}:
            text = _first_string(source, ("text", "content", "message"))
            if text is None:
                text = _first_string(value, ("text", "content", "message")) or ""
            return TranscriptEvent(
                kind="assistant",
                title="Local coder",
                text=text,
                detail=raw_detail,
                session_id=session_id,
            )
        if event_type in {"tool", "tool_use"}:
            name = _first_string(source, ("tool", "name")) or "unknown tool"
            state = source.get("state")
            state_mapping = state if isinstance(state, Mapping) else {}
            status_text = _first_string(state_mapping, ("status",)) or _first_string(
                source, ("status",)
            )
            arguments = state_mapping.get(
                "input", source.get("input", source.get("arguments", {}))
            )
            output = state_mapping.get("output", source.get("output"))
            pieces = [_compact_json(arguments)] if arguments not in ({}, None) else []
            if output not in (None, ""):
                pieces.append("Result\n" + _compact_json(output))
            return TranscriptEvent(
                kind="tool",
                title=f"Tool · {name}" + (f" · {status_text}" if status_text else ""),
                text="\n\n".join(pieces) or "No arguments reported.",
                detail=raw_detail,
                session_id=session_id,
            )
        if event_type == "error":
            error = value.get("error", source.get("error"))
            if isinstance(error, Mapping):
                message = _first_string(error, ("message", "name")) or _compact_json(
                    error
                )
            else:
                message = str(
                    error or _first_string(value, ("message",)) or "Unknown error"
                )
            return TranscriptEvent(
                kind="error",
                title="OpenCode error",
                text=message,
                detail=raw_detail,
                session_id=session_id,
            )
        if event_type in {"step_start", "step_finish", "reasoning"}:
            return TranscriptEvent(
                kind="status" if event_type != "reasoning" else "diagnostic",
                title=event_type.replace("_", " ").title(),
                text="",
                detail=raw_detail,
                session_id=session_id,
            )
        return TranscriptEvent(
            kind="diagnostic",
            title=f"OpenCode event · {event_type or 'unknown'}",
            text="",
            detail=raw_detail,
            session_id=session_id,
        )


class SubprocessClientExecutor:
    """Run one JSONL OpenCode request and allow cancellation from another thread."""

    def __init__(self, parser: OpenCodeEventParser | None = None) -> None:
        self.parser = parser or OpenCodeEventParser()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                from nhdf_edge import hybrid

                taskkill = hybrid._trusted_system_executable("taskkill.exe")
                if taskkill is None:
                    raise OSError("machine-owned taskkill.exe was not found")
                completed = subprocess.run(
                    [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                    cwd=str(taskkill.parent),
                    env=hybrid._minimal_subprocess_environment(
                        executable_directory=taskkill.parent
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode != 0 and process.poll() is None:
                    raise OSError(f"taskkill exited with status {completed.returncode}")
                return
            except (OSError, subprocess.TimeoutExpired):
                # Last-resort termination still unblocks pipe readers.  The caller's
                # pinned Work-mode deny rules reduce child creation, but the trusted
                # taskkill path above is the required normal Windows cancellation path.
                process.kill()
                return
        try:
            process.terminate()
        except OSError:
            return

    def cancel(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            self._terminate_process_tree(process)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        cancel_event: threading.Event,
        on_event: Callable[[TranscriptEvent], None],
    ) -> ClientRunResult:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise GuiError("A local-coder prompt is already running.")
            if cancel_event.is_set():
                raise PromptCancelled(
                    "Prompt cancelled before the client process started."
                )
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=str(cwd),
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise GuiError(
                    f"Could not start the pinned OpenCode client: {exc}"
                ) from exc
            self._process = process
        if cancel_event.is_set():
            self.cancel()

        stderr_tail: list[str] = []
        reader_errors: list[BaseException] = []

        def drain_stderr() -> None:
            if process.stderr is None:
                return
            for raw_line in process.stderr:
                line = raw_line.rstrip()
                if not line:
                    continue
                stderr_tail.append(line)
                del stderr_tail[:-20]
                try:
                    on_event(
                        TranscriptEvent(
                            kind="diagnostic",
                            title="OpenCode diagnostic",
                            text=line,
                        )
                    )
                except BaseException as exc:
                    reader_errors.append(exc)
                    self._terminate_process_tree(process)

        stderr_thread = threading.Thread(
            target=drain_stderr,
            name="local-coder-stderr",
            daemon=True,
        )
        stderr_thread.start()
        observed_session: str | None = None
        try:
            if process.stdout is None:
                raise GuiError("Pinned OpenCode client did not expose JSONL output.")
            for line in process.stdout:
                event = self.parser.parse_line(line)
                if event is None:
                    continue
                if event.session_id is not None:
                    if observed_session is None:
                        observed_session = event.session_id
                    elif observed_session != event.session_id:
                        raise GuiError(
                            "OpenCode changed session identifier during one prompt; "
                            "continuity was rejected."
                        )
                on_event(event)
            returncode = process.wait()
            stderr_thread.join(timeout=2)
            if reader_errors:
                raise GuiError(
                    "The transcript callback failed while reading OpenCode diagnostics: "
                    f"{reader_errors[0]}"
                ) from reader_errors[0]
            if cancel_event.is_set():
                raise PromptCancelled(
                    "Prompt cancelled. The resident model is still running."
                )
            if returncode != 0:
                detail = (
                    "\n".join(stderr_tail[-8:]) or "No diagnostic output was returned."
                )
                raise GuiError(f"OpenCode exited with status {returncode}.\n{detail}")
            return ClientRunResult(returncode=returncode, session_id=observed_session)
        finally:
            if process.poll() is None:
                self._terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.stdout is not None:
                try:
                    for _discarded in process.stdout:
                        pass
                except (OSError, ValueError):
                    pass
                process.stdout.close()
            stderr_thread.join(timeout=5)
            if process.stderr is not None:
                process.stderr.close()
            with self._lock:
                if self._process is process:
                    self._process = None


class VerifiedSetupRunner:
    """Invoke the repository's pinned, digest-verifying client installer."""

    def __init__(self, launcher: ModuleType | None = None) -> None:
        self.launcher = launcher or load_launcher()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            SubprocessClientExecutor._terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass

    def run(
        self,
        *,
        cancel_event: threading.Event,
        on_output: Callable[[str], None],
    ) -> Path:
        if cancel_event.is_set():
            raise OperationCancelled(
                "Client installation was cancelled before it started."
            )
        if os.name != "nt":
            raise GuiError("Install Client currently requires Windows PowerShell.")
        project_root = Path(self.launcher.PROJECT_ROOT)
        script = _require_safe_path(
            SETUP_SCRIPT,
            trusted_root=project_root,
            must_exist=True,
            require_file=True,
        )
        powershell = resolve_windows_powershell(self.launcher)
        environment = self.launcher.hybrid._minimal_subprocess_environment(
            executable_directory=powershell.parent
        )
        setup_temp = Path(self.launcher.LOCAL_STATE_ROOT) / "setup-temp"
        self.launcher._ensure_safe_project_directory(setup_temp)
        environment.update(
            {
                "TEMP": str(setup_temp),
                "TMP": str(setup_temp),
                "NoDefaultCurrentDirectoryInExePath": "1",
            }
        )
        command = [
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ]
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise GuiError("A verified client installation is already running.")
            if cancel_event.is_set():
                raise OperationCancelled(
                    "Client installation was cancelled before it started."
                )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(project_root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                raise GuiError(
                    f"Could not start the verified client installer: {exc}"
                ) from exc
            self._process = process
        if cancel_event.is_set():
            self.cancel()
        output_tail: list[str] = []
        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if line:
                        output_tail.append(line)
                        del output_tail[:-20]
                        on_output(line)
                    if cancel_event.is_set():
                        self.cancel()
            returncode = process.wait()
            if cancel_event.is_set():
                raise OperationCancelled(
                    "Client installation was cancelled; run Install Client again."
                )
            if returncode != 0:
                details = (
                    "\n".join(output_tail[-8:]) or "No installer output was returned."
                )
                raise GuiError(
                    f"Verified client installer exited with status {returncode}.\n{details}"
                )
            return self.launcher.validate_local_install(self.launcher.OPENCODE_EXE)
        finally:
            if process.poll() is None:
                SubprocessClientExecutor._terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.stdout is not None:
                try:
                    for _discarded in process.stdout:
                        pass
                except (OSError, ValueError):
                    pass
                process.stdout.close()
            with self._lock:
                if self._process is process:
                    self._process = None


def resolve_windows_powershell(launcher: ModuleType) -> Path:
    """Resolve legacy Windows PowerShell from its exact machine-owned subdirectory."""

    system_directory = launcher.hybrid._windows_system_directory()
    return launcher._require_unaliased_executable(
        system_directory / "WindowsPowerShell" / "v1.0" / "powershell.exe",
        label="machine-owned Windows PowerShell",
        trusted_root=Path(system_directory.anchor),
    )


def verify_runtime_readiness(
    launcher: ModuleType,
    *,
    include_cuda: bool,
) -> str:
    """Verify the pinned project runtime and, on Windows, its pinned CUDA DLLs."""

    root = Path(launcher.PROJECT_ROOT)
    for relative in launcher.CANONICAL_RUNTIME_PATHS:
        expected_bytes, expected_digest = launcher.CANONICAL_REFERENCE_RECORDS[relative]
        path = _require_safe_path(
            root / relative,
            trusted_root=root,
            must_exist=True,
            require_file=True,
        )
        if path.stat().st_size != expected_bytes:
            raise GuiError(f"Runtime size mismatch: {relative}")
        if _sha256_file(path) != expected_digest:
            raise GuiError(f"Runtime digest mismatch: {relative}")

    cuda_count = 0
    cuda_version = ""
    if include_cuda:
        preflight = launcher.hybrid.preflight_windows_cuda_dependencies()
        cuda_count = int(preflight.verified_dependency_count)
        cuda_version = str(preflight.version)
        if cuda_count <= 0 or len(preflight.dependency_names) != cuda_count:
            raise GuiError(
                "Pinned CUDA dependency records were not available on Windows."
            )
    detail = f"llama.cpp · {len(launcher.CANONICAL_RUNTIME_PATHS)} files verified"
    if cuda_count:
        detail += f" · CUDA {cuda_version} ({cuda_count} DLLs)"
    return detail


class ReadinessProbe:
    CARD_ORDER = (
        ("client", "Client"),
        ("model", "Model"),
        ("runtime", "Runtime"),
        ("gpu", "GPU"),
        ("artifact", "Artifact"),
    )

    def __init__(self, checks: Mapping[str, Callable[[], str]]) -> None:
        self.checks = dict(checks)

    @classmethod
    def canonical(cls, launcher: ModuleType | None = None) -> "ReadinessProbe":
        launcher = launcher or load_launcher()

        def client() -> str:
            executable = launcher.validate_local_install(launcher.OPENCODE_EXE)
            return f"OpenCode {launcher.PINNED_OPENCODE_VERSION} · {executable.name}"

        def model() -> str:
            control = canonical_model_control(launcher)
            downloader = ModelDownloader(
                control,
                trusted_root=Path(launcher.PROJECT_ROOT),
            )
            destination, _partial = downloader._validate_destination_layout()
            if not destination.exists():
                raise FileNotFoundError("Use Download Model (9.87 GB).")
            downloader._verify_exact_file(
                destination,
                cancel_event=threading.Event(),
                on_progress=lambda _progress: None,
                phase="verifying installed model",
                resumed=False,
            )
            return f"{control.expected_bytes / (1024**3):.2f} GiB · SHA-256 verified"

        def runtime() -> str:
            return verify_runtime_readiness(
                launcher,
                include_cuda=os.name == "nt",
            )

        def gpu() -> str:
            from nhdf_edge import hybrid

            manifest, _digest = hybrid.load_hybrid_manifest_snapshot(
                launcher.DEFAULT_ARTIFACT
            )
            contract = manifest["resource_contract"]
            name = hybrid._gpu_name()
            sample = hybrid._gpu_sample()
            if name is None or sample is None:
                raise GuiError("nvidia-smi did not return GPU identity and memory.")
            total, used, utilization = sample
            expected_name = str(contract["target_gpu"])
            required_total = int(contract["target_vram_mib"])
            required_free = int(contract["required_free_vram_mib"])
            free = total - used
            if name != expected_name:
                raise GuiError(f"Detected {name}; profile requires {expected_name}.")
            if total < required_total or free < required_free:
                raise GuiError(
                    f"{total:,} MiB total / {free:,} MiB free; need "
                    f"{required_total:,} total / {required_free:,} free."
                )
            return (
                f"{name.replace('NVIDIA GeForce ', '')} · {free:,} MiB free · "
                f"{utilization}% busy"
            )

        def artifact() -> str:
            validated = launcher.validate_agent_artifact(launcher.DEFAULT_ARTIFACT)
            context = validated.manifest["execution_profile"]["maximum_context_tokens"]
            return f"VALIDATED · {int(context):,}-token profile"

        return cls(
            {
                "client": client,
                "model": model,
                "runtime": runtime,
                "gpu": gpu,
                "artifact": artifact,
            }
        )

    def probe(self) -> ReadinessSnapshot:
        cards: list[ReadinessCard] = []
        for key, label in self.CARD_ORDER:
            check = self.checks.get(key)
            if check is None:
                cards.append(
                    ReadinessCard(
                        key=key,
                        label=label,
                        state=ReadinessState.BLOCKED,
                        detail="No readiness check is configured.",
                    )
                )
                continue
            try:
                detail = check()
            except FileNotFoundError as exc:
                cards.append(
                    ReadinessCard(
                        key=key,
                        label=label,
                        state=ReadinessState.MISSING,
                        detail=str(exc),
                    )
                )
            except Exception as exc:  # readiness must report every independent card
                cards.append(
                    ReadinessCard(
                        key=key,
                        label=label,
                        state=ReadinessState.BLOCKED,
                        detail=str(exc),
                    )
                )
            else:
                cards.append(
                    ReadinessCard(
                        key=key,
                        label=label,
                        state=ReadinessState.READY,
                        detail=detail,
                    )
                )
        return ReadinessSnapshot(cards=tuple(cards))


class LocalCoderController:
    """Display-free owner of install, resident server, session, and prompt lifecycle."""

    def __init__(
        self,
        *,
        launcher: ModuleType | None = None,
        server_factory: Callable[..., Any] | None = None,
        readiness_probe: ReadinessProbe | None = None,
        downloader: ModelDownloader | None = None,
        setup_runner: VerifiedSetupRunner | None = None,
        client_executor: _ClientExecutor | None = None,
        port: int | None = None,
        startup_timeout_seconds: float = 120.0,
        startup_poll_seconds: float = 0.1,
    ) -> None:
        self.launcher = launcher or load_launcher()
        if server_factory is None:
            server_factory = self.launcher.HybridServer
        self.server_factory = server_factory
        self.readiness_probe = readiness_probe or ReadinessProbe.canonical(
            self.launcher
        )
        self.downloader = downloader or ModelDownloader.canonical(self.launcher)
        self.setup_runner = setup_runner or VerifiedSetupRunner(self.launcher)
        self.client_executor = client_executor or SubprocessClientExecutor()
        self.port = int(port if port is not None else self.launcher.DEFAULT_PORT)
        if startup_timeout_seconds <= 0 or startup_poll_seconds <= 0:
            raise ValueError("startup timeout and poll interval must be positive")
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.startup_poll_seconds = float(startup_poll_seconds)
        self._lock = threading.RLock()
        self._context: _ResidentContext | None = None
        self._prompt_cancel: threading.Event | None = None
        self._start_cancel: threading.Event | None = None
        self._starting_server: Any | None = None
        self._start_done = threading.Event()
        self._start_done.set()
        self._closing = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            context = self._context
        return bool(context is not None and context.server.is_running)

    @property
    def has_resident_context(self) -> bool:
        """Whether Start has bound a server/repository context, even if it exited."""

        with self._lock:
            return self._context is not None

    @property
    def bound_target(self) -> Path | None:
        with self._lock:
            return self._context.target if self._context is not None else None

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._context.session_id if self._context is not None else None

    @property
    def base_url(self) -> str | None:
        with self._lock:
            return self._context.server.base_url if self._context is not None else None

    def probe_readiness(self) -> ReadinessSnapshot:
        return self.readiness_probe.probe()

    def install_client(
        self,
        *,
        cancel_event: threading.Event,
        on_output: Callable[[str], None],
    ) -> Path:
        return self.setup_runner.run(
            cancel_event=cancel_event,
            on_output=on_output,
        )

    def download_model(
        self,
        *,
        cancel_event: threading.Event,
        on_progress: Callable[[DownloadProgress], None],
    ) -> Path:
        if self.has_resident_context:
            raise GuiError(
                "Stop the resident model context before changing its payload file."
            )
        return self.downloader.download(
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

    def start(
        self,
        repository: Path,
        *,
        cancel_event: threading.Event | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        cancel_event = cancel_event or threading.Event()
        on_status = on_status or (lambda _message: None)
        with self._lock:
            if self._closing:
                raise GuiError("The local coder is shutting down.")
            if self._context is not None:
                raise GuiError(
                    "A resident model context already exists. Use Stop to release or reset "
                    "it before starting again."
                )
            if self._start_cancel is not None:
                raise GuiError("A resident model start is already in progress.")
            self._start_cancel = cancel_event
            self._start_done.clear()
        try:
            return self._start_resident(
                repository,
                cancel_event=cancel_event,
                on_status=on_status,
            )
        finally:
            with self._lock:
                if self._start_cancel is cancel_event:
                    self._start_cancel = None
                    self._starting_server = None
                self._start_done.set()

    def _start_resident(
        self,
        repository: Path,
        *,
        cancel_event: threading.Event,
        on_status: Callable[[str], None],
    ) -> str:
        target, worktree_root = self.launcher.validate_git_target(repository)
        config = self.launcher.validate_config(self.launcher.DEFAULT_CONFIG)
        executable = self.launcher.validate_local_install(self.launcher.OPENCODE_EXE)
        artifact = self.launcher.resolve_canonical_artifact(
            self.launcher.DEFAULT_ARTIFACT
        )
        validated = self.launcher.validate_agent_artifact(artifact)
        server = self.server_factory(
            artifact,
            port=self.port,
            threads=None,
            startup_timeout_seconds=self.startup_timeout_seconds,
            request_timeout_seconds=600.0,
            verify_payload_hash=True,
            artifact_approval=validated.server_approval,
        )
        with self._lock:
            if self._closing or cancel_event.is_set():
                server.stop()
                raise OperationCancelled("Resident model start was cancelled.")
            self._starting_server = server
        environment = self.launcher.isolated_environment(config, server.base_url)
        self.launcher.verify_execution_inputs(executable, config, environment)
        expected_contract = (
            Path(environment["XDG_CONFIG_HOME"]) / "opencode" / "AGENTS.md"
        )
        self.launcher.validate_resolved_config(
            executable,
            target,
            environment,
            expected_base_url=server.base_url,
            expected_contract_path=expected_contract,
        )
        try:
            if cancel_event.is_set():
                raise OperationCancelled("Resident model start was cancelled.")
            on_status("Launching the verified resident runtime…")
            server.start(wait_ready=False)
            deadline = time.monotonic() + self.startup_timeout_seconds
            last_error: BaseException | None = None
            from nhdf_edge.server import HybridServerUnavailableError

            while time.monotonic() < deadline:
                if cancel_event.is_set():
                    raise OperationCancelled(
                        "Resident model start was cancelled and the process was stopped."
                    )
                try:
                    server.health(
                        timeout_seconds=min(2.0, max(0.1, deadline - time.monotonic()))
                    )
                    break
                except HybridServerUnavailableError as exc:
                    last_error = exc
                    if not server.is_running:
                        raise GuiError(
                            f"Resident runtime exited before becoming ready: {exc}"
                        ) from exc
                    on_status("Model is loading into GPU memory…")
                    cancel_event.wait(
                        min(
                            self.startup_poll_seconds,
                            max(0.0, deadline - time.monotonic()),
                        )
                    )
            else:
                detail = f" Last health result: {last_error}" if last_error else ""
                raise GuiError(
                    "Resident model did not become healthy within "
                    f"{self.startup_timeout_seconds:g} seconds.{detail}"
                )
            self.launcher.verify_execution_inputs(executable, config, environment)
        except BaseException:
            server.stop()
            raise
        context = _ResidentContext(
            target=target,
            worktree_root=worktree_root,
            executable=executable,
            config=config,
            environment=environment,
            server=server,
        )
        with self._lock:
            if self._closing or self._context is not None:
                server.stop()
                raise GuiError(
                    "Resident start was superseded by shutdown or another start."
                )
            self._context = context
        return server.base_url

    def stop(self) -> None:
        self.cancel_prompt()
        with self._lock:
            context = self._context
            self._context = None
        if context is not None:
            context.server.stop()

    def new_session(self) -> None:
        with self._lock:
            if self._prompt_cancel is not None:
                raise GuiError(
                    "Cancel or finish the current prompt before starting a new session."
                )
            if self._context is None:
                raise GuiError("Start the resident model before creating a session.")
            self._context.session_id = None
            self._context.scoped_work_authorized = False

    def authorize_scoped_work(self, *, confirmed: bool) -> None:
        """Record an explicit Work-mode confirmation for the current GUI session."""

        if not confirmed:
            raise GuiError("Work mode was not authorized.")
        with self._lock:
            if self._prompt_cancel is not None:
                raise GuiError(
                    "Finish the current prompt before changing authorization."
                )
            if self._context is None or not self._context.server.is_running:
                raise GuiError("Start the resident model before authorizing Work mode.")
            self._context.scoped_work_authorized = True

    @property
    def scoped_work_authorized(self) -> bool:
        with self._lock:
            return bool(
                self._context is not None and self._context.scoped_work_authorized
            )

    def cancel_prompt(self) -> None:
        with self._lock:
            event = self._prompt_cancel
        if event is not None:
            event.set()
            self.client_executor.cancel()

    @staticmethod
    def _mode_prompt(prompt: str, mode: InteractionMode, worktree: Path) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise GuiError("Enter a non-empty request.")
        if "\x00" in prompt:
            raise GuiError("Prompt cannot contain a NUL character.")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise GuiError(
                f"Prompt is too long ({len(prompt):,} characters); limit is "
                f"{MAX_PROMPT_CHARACTERS:,}."
            )
        if mode is InteractionMode.READ_ONLY:
            boundary = (
                "GUI MODE — READ ONLY. The user has not authorized file changes or shell "
                "commands. Inspect with read/glob/grep/list/LSP only. Explain findings and "
                "ask the user to switch modes before editing or testing."
            )
        elif mode is InteractionMode.SCOPED_EDITS:
            boundary = (
                "GUI MODE — USER-AUTHORIZED SCOPED EDITS AND TESTS. The user explicitly "
                f"authorizes edits and focused local tests only inside {worktree}. Preserve "
                "the pinned deny rules: no network, external directories, destructive "
                "commands, delegation, commits, pushes, or unrelated expansion."
            )
        else:
            raise GuiError(f"Unsupported interaction mode: {mode!r}")
        return f"{boundary}\n\nUSER REQUEST\n{prompt.strip()}"

    def send_prompt(
        self,
        prompt: str,
        *,
        mode: InteractionMode,
        cancel_event: threading.Event | None = None,
        on_event: Callable[[TranscriptEvent], None] | None = None,
    ) -> ClientRunResult:
        on_event = on_event or (lambda _event: None)
        cancel_event = cancel_event or threading.Event()
        with self._lock:
            if self._closing:
                raise GuiError("The local coder is shutting down.")
            context = self._context
            if context is None or not context.server.is_running:
                raise GuiError("Start the resident model before sending a request.")
            if self._prompt_cancel is not None:
                raise GuiError("A local-coder prompt is already running.")
            self._prompt_cancel = cancel_event
            current_session = context.session_id
            scoped_work_authorized = context.scoped_work_authorized

        try:
            if cancel_event.is_set():
                raise PromptCancelled("Prompt cancelled before validation started.")
            target, worktree_root = self.launcher.validate_git_target(context.target)
            if target != context.target or worktree_root != context.worktree_root:
                raise GuiError("Selected Git target changed after the resident start.")
            context.server.health()
            self.launcher.verify_execution_inputs(
                context.executable,
                context.config,
                context.environment,
            )
            if cancel_event.is_set():
                raise PromptCancelled(
                    "Prompt cancelled before the client process started."
                )
            run_args: list[str] = ["--format", "json"]
            if mode is InteractionMode.SCOPED_EDITS:
                if not scoped_work_authorized:
                    raise GuiError(
                        "Work mode requires explicit confirmation for this session."
                    )
                # OpenCode's noninteractive JSON run rejects ask-level permissions unless
                # --auto is present.  The pinned config's deny rules remain authoritative.
                run_args.append("--auto")
            if current_session is not None:
                if SESSION_ID_PATTERN.fullmatch(current_session) is None:
                    raise GuiError("Stored OpenCode session identifier is invalid.")
                run_args.extend(["--session", current_session])
            run_args.append(self._mode_prompt(prompt, mode, context.worktree_root))
            command = self.launcher.build_opencode_command(
                context.executable,
                tuple(run_args),
            )
            result = self.client_executor.run(
                command,
                cwd=context.target,
                environment=context.environment,
                cancel_event=cancel_event,
                on_event=on_event,
            )
            if current_session is not None and result.session_id not in (
                None,
                current_session,
            ):
                raise GuiError(
                    "OpenCode returned a different session than the internally bound session."
                )
            with self._lock:
                if self._context is context and result.session_id is not None:
                    context.session_id = result.session_id
            if current_session is None and result.session_id is None:
                on_event(
                    TranscriptEvent(
                        kind="diagnostic",
                        title="Session continuity unavailable",
                        text=(
                            "The client returned no valid session identifier; the next prompt "
                            "will start a new OpenCode session."
                        ),
                    )
                )
            return result
        finally:
            with self._lock:
                if self._prompt_cancel is cancel_event:
                    self._prompt_cancel = None

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            start_cancel = self._start_cancel
            starting_server = self._starting_server
        if start_cancel is not None:
            start_cancel.set()
        self.setup_runner.cancel()
        if starting_server is not None:
            starting_server.stop()
        self.stop()
        self._start_done.wait()
        # A start that was still validating when shutdown began cannot publish a
        # context because _closing is set, but stop once more to cover the boundary.
        self.stop()


class LocalCoderApp:
    """Tk/ttk presentation for :class:`LocalCoderController`."""

    NAVY = "#0B1F33"
    NAVY_LIGHT = "#173A56"
    TEAL = "#168C88"
    TEAL_LIGHT = "#DDF3F1"
    BACKGROUND = "#F2F6F8"
    CARD = "#FFFFFF"
    TEXT = "#18313F"
    MUTED = "#607886"
    ERROR = "#B34444"
    AMBER = "#A56B12"

    def __init__(self, root: Any, controller: LocalCoderController) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = root
        self.controller = controller
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.operation_thread: threading.Thread | None = None
        self.operation_cancel: threading.Event | None = None
        self.closing = False
        self.diagnostics_visible = False
        self.card_widgets: dict[str, tuple[Any, Any, Any]] = {}

        root.title("UGTOMS Local Coder")
        root.geometry("1280x820")
        root.minsize(980, 680)
        root.configure(background=self.BACKGROUND)
        self._configure_style()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self._request_close)
        root.after(80, self._drain_events)
        root.after(150, self.refresh_readiness)

    def _configure_style(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure("App.TFrame", background=self.BACKGROUND)
        style.configure("Header.TFrame", background=self.NAVY)
        style.configure(
            "Header.TLabel",
            background=self.NAVY,
            foreground="#FFFFFF",
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=self.NAVY,
            foreground="#B9D8E1",
            font=("Segoe UI", 10),
        )
        style.configure("Card.TFrame", background=self.CARD, relief="flat")
        style.configure(
            "CardTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "CardText.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton",
            background=self.TEAL,
            foreground="#FFFFFF",
            padding=(16, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Primary.TButton", background=[("active", "#117A76")])
        style.configure("Action.TButton", padding=(13, 8), font=("Segoe UI", 10))
        style.configure(
            "Section.TLabel",
            background=self.BACKGROUND,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Status.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(28, 20))
        header.pack(fill="x")
        ttk.Label(header, text="UGTOMS Local Coder", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text=(
                "Verified 30.5B MoE coding model · 32K context · resident between prompts"
            ),
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 20))
        body.pack(fill="both", expand=True)

        repo_row = ttk.Frame(body, style="App.TFrame")
        repo_row.pack(fill="x")
        ttk.Label(repo_row, text="Coding repository", style="Section.TLabel").pack(
            side="left"
        )
        self.repo_var = tk.StringVar(value=str(PROJECT_ROOT))
        self.repo_entry = ttk.Entry(repo_row, textvariable=self.repo_var)
        self.repo_entry.pack(side="left", fill="x", expand=True, padx=(16, 8))
        self.repo_button = ttk.Button(
            repo_row, text="Browse…", command=self._choose_repo
        )
        self.repo_button.pack(side="left")

        cards = ttk.Frame(body, style="App.TFrame")
        cards.pack(fill="x", pady=(16, 14))
        for column, (key, label) in enumerate(ReadinessProbe.CARD_ORDER):
            cards.columnconfigure(column, weight=1, uniform="readiness")
            frame = ttk.Frame(cards, style="Card.TFrame", padding=(14, 12))
            frame.grid(
                row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0)
            )
            title = ttk.Label(frame, text=label, style="CardTitle.TLabel")
            title.pack(anchor="w")
            state = ttk.Label(frame, text="CHECKING", style="CardTitle.TLabel")
            state.pack(anchor="w", pady=(7, 2))
            detail = ttk.Label(
                frame,
                text="Waiting for readiness scan…",
                style="CardText.TLabel",
                wraplength=205,
                justify="left",
            )
            detail.pack(anchor="w")
            self.card_widgets[key] = (frame, state, detail)

        action_row = ttk.Frame(body, style="App.TFrame")
        action_row.pack(fill="x", pady=(0, 12))
        self.install_button = ttk.Button(
            action_row,
            text="Install Client",
            style="Action.TButton",
            command=self.install_client,
        )
        self.install_button.pack(side="left")
        self.download_button = ttk.Button(
            action_row,
            text="Download Model",
            style="Action.TButton",
            command=self.download_model,
        )
        self.download_button.pack(side="left", padx=(8, 0))
        self.start_button = ttk.Button(
            action_row,
            text="Start Resident Model",
            style="Primary.TButton",
            command=self.start_server,
        )
        self.start_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(
            action_row,
            text="Stop",
            style="Action.TButton",
            command=self.stop_server,
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        self.new_session_button = ttk.Button(
            action_row,
            text="New Session",
            style="Action.TButton",
            command=self.new_session,
        )
        self.new_session_button.pack(side="left", padx=(8, 0))
        self.refresh_button = ttk.Button(
            action_row,
            text="Refresh",
            style="Action.TButton",
            command=self.refresh_readiness,
        )
        self.refresh_button.pack(side="right")

        self.progress = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(value="Checking local readiness…")
        ttk.Label(body, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor="w", pady=(5, 12)
        )

        pane = ttk.Panedwindow(body, orient="horizontal")
        pane.pack(fill="both", expand=True)
        conversation = ttk.Frame(pane, style="Card.TFrame", padding=(14, 12))
        compose = ttk.Frame(pane, style="Card.TFrame", padding=(14, 12))
        pane.add(conversation, weight=3)
        pane.add(compose, weight=2)

        ttk.Label(conversation, text="Conversation", style="CardTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        transcript_frame = ttk.Frame(conversation, style="Card.TFrame")
        transcript_frame.pack(fill="both", expand=True)
        self.transcript = tk.Text(
            transcript_frame,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background="#FBFDFE",
            foreground=self.TEXT,
            padx=12,
            pady=10,
            font=("Segoe UI", 10),
            state="disabled",
        )
        transcript_scroll = ttk.Scrollbar(
            transcript_frame, orient="vertical", command=self.transcript.yview
        )
        self.transcript.configure(yscrollcommand=transcript_scroll.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        transcript_scroll.pack(side="right", fill="y")
        self.transcript.tag_configure(
            "user", foreground=self.NAVY, font=("Segoe UI Semibold", 10)
        )
        self.transcript.tag_configure("assistant", foreground=self.TEXT)
        self.transcript.tag_configure("tool", foreground=self.TEAL)
        self.transcript.tag_configure("error", foreground=self.ERROR)
        self.transcript.tag_configure("status", foreground=self.MUTED)

        ttk.Label(compose, text="Request", style="CardTitle.TLabel").pack(anchor="w")
        self.prompt = tk.Text(
            compose,
            height=10,
            wrap="word",
            relief="solid",
            borderwidth=1,
            background="#FFFFFF",
            foreground=self.TEXT,
            padx=10,
            pady=8,
            font=("Segoe UI", 10),
        )
        self.prompt.pack(fill="both", expand=True, pady=(8, 10))
        self.mode_var = tk.StringVar(value="Review (read-only)")
        ttk.Label(compose, text="Permission mode", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        self.mode_box = ttk.Combobox(
            compose,
            state="readonly",
            textvariable=self.mode_var,
            values=("Review (read-only)", "Work (scoped edits + tests)"),
        )
        self.mode_box.pack(fill="x", pady=(5, 5))
        self.mode_note = ttk.Label(
            compose,
            text=(
                "Work mode requires one confirmation per session, then auto-approves "
                "ask-level edit/test tools. Pinned denies reduce risk; this is not an "
                "OS sandbox."
            ),
            style="CardText.TLabel",
            wraplength=390,
            justify="left",
        )
        self.mode_note.pack(anchor="w", pady=(0, 10))
        send_row = ttk.Frame(compose, style="Card.TFrame")
        send_row.pack(fill="x")
        self.send_button = ttk.Button(
            send_row,
            text="Send",
            style="Primary.TButton",
            command=self.send_prompt,
        )
        self.send_button.pack(side="left")
        self.cancel_button = ttk.Button(
            send_row,
            text="Cancel",
            style="Action.TButton",
            command=self.cancel_operation,
        )
        self.cancel_button.pack(side="left", padx=(8, 0))
        self.diagnostics_button = ttk.Button(
            send_row,
            text="Show diagnostics",
            style="Action.TButton",
            command=self.toggle_diagnostics,
        )
        self.diagnostics_button.pack(side="right")

        self.diagnostics_frame = ttk.Frame(body, style="Card.TFrame", padding=(12, 8))
        self.diagnostics = tk.Text(
            self.diagnostics_frame,
            height=8,
            wrap="word",
            relief="flat",
            background="#071725",
            foreground="#B9D8E1",
            insertbackground="#FFFFFF",
            font=("Consolas", 9),
            state="disabled",
        )
        self.diagnostics.pack(fill="both", expand=True)
        self._update_controls()

    def _choose_repo(self) -> None:
        if self.closing or self.controller.has_resident_context:
            return
        selected = self.filedialog.askdirectory(
            title="Choose a Git repository",
            initialdir=self.repo_var.get() or str(PROJECT_ROOT),
            mustexist=True,
        )
        if selected:
            self.repo_var.set(selected)

    def _append_text(self, widget: Any, text: str, tag: str | None = None) -> None:
        widget.configure(state="normal")
        widget.insert("end", text, tag or ())
        widget.see("end")
        widget.configure(state="disabled")

    def _append_transcript(self, event: TranscriptEvent) -> None:
        if event.kind == "diagnostic":
            if event.text:
                self._append_diagnostic(f"{event.title}\n{event.text}\n")
            if event.detail:
                self._append_diagnostic(event.detail + "\n")
            return
        if event.kind == "status" and not event.text:
            if event.detail:
                self._append_diagnostic(f"{event.title}\n{event.detail}\n")
            return
        tag = (
            event.kind
            if event.kind in {"user", "assistant", "tool", "error"}
            else "status"
        )
        heading = event.title + "\n" if event.title else ""
        self._append_text(self.transcript, "\n" + heading, tag)
        if event.text:
            self._append_text(self.transcript, event.text.rstrip() + "\n", tag)
        if event.detail:
            self._append_diagnostic(f"{event.title}\n{event.detail}\n")

    def _append_diagnostic(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._append_text(self.diagnostics, f"[{stamp}] {text.rstrip()}\n")

    def _run_operation(
        self,
        name: str,
        worker: Callable[[threading.Event], Any],
        *,
        success: Callable[[Any], None] | None = None,
    ) -> None:
        if self.closing:
            return
        if self.operation_thread is not None and self.operation_thread.is_alive():
            self.messagebox.showinfo(
                "Local coder busy", "Finish or cancel the current operation first."
            )
            return
        cancel_event = threading.Event()
        self.operation_cancel = cancel_event
        self.status_var.set(name)
        self.progress.configure(value=0)

        def run() -> None:
            try:
                result = worker(cancel_event)
            except (DownloadCancelled, PromptCancelled, OperationCancelled) as exc:
                self.events.put(("cancelled", str(exc)))
            except BaseException as exc:
                self.events.put(("error", (name, exc)))
            else:
                self.events.put(("success", (name, result, success)))
            finally:
                self.events.put(("operation_done", None))

        self.operation_thread = threading.Thread(
            target=run,
            name="local-coder-gui-operation",
            daemon=True,
        )
        self.operation_thread.start()
        self._update_controls()

    def refresh_readiness(self) -> None:
        self._run_operation(
            "Verifying client, model, runtime, GPU, and artifact…",
            lambda _cancel: self.controller.probe_readiness(),
            success=lambda result: self.events.put(("readiness", result)),
        )

    def install_client(self) -> None:
        self._run_operation(
            "Installing and verifying the pinned local client…",
            lambda cancel: self.controller.install_client(
                cancel_event=cancel,
                on_output=lambda line: self.events.put(("diagnostic", line)),
            ),
            success=lambda _result: self.root.after(100, self.refresh_readiness),
        )

    def download_model(self) -> None:
        def progress(value: DownloadProgress) -> None:
            self.events.put(("download_progress", value))

        self._run_operation(
            "Downloading the immutable model…",
            lambda cancel: self.controller.download_model(
                cancel_event=cancel,
                on_progress=progress,
            ),
            success=lambda _result: self.root.after(100, self.refresh_readiness),
        )

    def start_server(self) -> None:
        repository = Path(self.repo_var.get()).expanduser()
        self._run_operation(
            "Verifying and loading the resident model (this can take a few minutes)…",
            lambda cancel: self.controller.start(
                repository,
                cancel_event=cancel,
                on_status=lambda message: self.events.put(("status", message)),
            ),
            success=self._resident_started,
        )

    def _resident_started(self, url: str) -> None:
        target = self.controller.bound_target
        if target is not None:
            self.repo_var.set(str(target))
        self._append_transcript(
            TranscriptEvent(
                kind="status",
                title="Resident model ready",
                text=f"Verified loopback endpoint: {url}/v1",
            )
        )

    def stop_server(self) -> None:
        self._run_operation(
            "Stopping the resident model…",
            lambda _cancel: self.controller.stop(),
            success=lambda _result: self._append_transcript(
                TranscriptEvent(
                    kind="status",
                    title="Resident model stopped",
                    text="GPU memory has been released.",
                )
            ),
        )

    def new_session(self) -> None:
        try:
            self.controller.new_session()
        except Exception as exc:
            self.messagebox.showerror("Could not start a new session", str(exc))
            return
        self._append_transcript(
            TranscriptEvent(
                kind="status",
                title="New session",
                text="The resident model stayed loaded; conversation context was reset.",
            )
        )

    def send_prompt(self) -> None:
        prompt = self.prompt.get("1.0", "end-1c")
        mode = (
            InteractionMode.SCOPED_EDITS
            if self.mode_var.get() == "Work (scoped edits + tests)"
            else InteractionMode.READ_ONLY
        )
        if not prompt.strip():
            self.messagebox.showinfo("Request needed", "Enter a request first.")
            return
        if (
            mode is InteractionMode.SCOPED_EDITS
            and not self.controller.scoped_work_authorized
        ):
            confirmed = self.messagebox.askyesno(
                "Authorize Work mode for this session?",
                (
                    "Work mode passes OpenCode's internal --auto switch so ask-level file "
                    "edits and local test commands can run without an interactive terminal.\n\n"
                    "The pinned config still denies its listed commit, push, reset --hard, "
                    "clean, restore, built-in network, external-directory, and delegation "
                    "operations. A general shell can bypass tool-level patterns, so this is "
                    "not an operating-system sandbox. Review the selected repository and "
                    "request scope before continuing.\n\nAuthorize Work mode until New Session?"
                ),
                icon="warning",
            )
            if not confirmed:
                return
            try:
                self.controller.authorize_scoped_work(confirmed=True)
            except Exception as exc:
                self.messagebox.showerror("Could not authorize Work mode", str(exc))
                return
        self._append_transcript(
            TranscriptEvent(kind="user", title="You", text=prompt.strip())
        )
        self.prompt.delete("1.0", "end")
        self._run_operation(
            "Local coder is working…",
            lambda cancel: self.controller.send_prompt(
                prompt,
                mode=mode,
                cancel_event=cancel,
                on_event=lambda event: self.events.put(("transcript", event)),
            ),
            success=lambda result: self._append_diagnostic(
                "Prompt complete"
                + (f" · session {result.session_id}" if result.session_id else "")
            ),
        )

    def cancel_operation(self) -> None:
        if self.operation_cancel is not None:
            self.operation_cancel.set()
        self.controller.cancel_prompt()
        self.controller.setup_runner.cancel()
        self.status_var.set("Cancelling safely…")

    def toggle_diagnostics(self) -> None:
        self.diagnostics_visible = not self.diagnostics_visible
        if self.diagnostics_visible:
            self.diagnostics_frame.pack(fill="x", pady=(12, 0))
            self.diagnostics_button.configure(text="Hide diagnostics")
        else:
            self.diagnostics_frame.pack_forget()
            self.diagnostics_button.configure(text="Show diagnostics")

    def _apply_readiness(self, snapshot: ReadinessSnapshot) -> None:
        colors = {
            ReadinessState.READY: self.TEAL,
            ReadinessState.MISSING: self.AMBER,
            ReadinessState.BLOCKED: self.ERROR,
        }
        for card in snapshot.cards:
            _frame, state, detail = self.card_widgets[card.key]
            state.configure(
                text=card.state.value.upper(),
                foreground=colors[card.state],
            )
            detail.configure(text=card.detail)
        self.status_var.set(
            "Ready to start the resident model."
            if snapshot.ready_to_start
            else "Resolve the highlighted readiness items, then refresh."
        )

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "transcript":
                    self._append_transcript(payload)
                elif kind == "diagnostic":
                    self._append_diagnostic(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "download_progress":
                    progress: DownloadProgress = payload
                    percent = 100 * progress.completed_bytes / progress.total_bytes
                    self.progress.configure(value=percent)
                    resume = " · resumed" if progress.resumed else ""
                    self.status_var.set(
                        f"{progress.phase.capitalize()}{resume} · {percent:.1f}% · "
                        f"{progress.completed_bytes / (1024**3):.2f}/"
                        f"{progress.total_bytes / (1024**3):.2f} GiB"
                    )
                elif kind == "readiness":
                    self._apply_readiness(payload)
                elif kind == "success":
                    name, result, callback = payload
                    if callback is not None and not self.closing:
                        callback(result)
                    if not name.startswith("Verifying"):
                        self.status_var.set("Completed.")
                elif kind == "cancelled":
                    self.status_var.set(str(payload))
                    self._append_transcript(
                        TranscriptEvent(
                            kind="status", title="Cancelled", text=str(payload)
                        )
                    )
                elif kind == "error":
                    name, error = payload
                    self.status_var.set(f"{name} failed.")
                    self._append_diagnostic(f"{name}\n{error!r}\n")
                    if not self.closing:
                        self.messagebox.showerror(name, str(error))
                elif kind == "operation_done":
                    self.operation_cancel = None
                    self.progress.configure(value=0)
                    self._update_controls()
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(80, self._drain_events)

    def _update_controls(self) -> None:
        busy = self.operation_thread is not None and self.operation_thread.is_alive()
        running = self.controller.is_running
        has_context = self.controller.has_resident_context
        normal = "normal"
        disabled = "disabled"
        if self.closing:
            for control in (
                self.repo_entry,
                self.repo_button,
                self.install_button,
                self.download_button,
                self.start_button,
                self.stop_button,
                self.new_session_button,
                self.send_button,
                self.cancel_button,
                self.refresh_button,
                self.mode_box,
                self.diagnostics_button,
            ):
                control.configure(state=disabled)
            self.prompt.configure(state=disabled)
            return

        repository_locked = busy or has_context
        self.repo_entry.configure(state=disabled if repository_locked else normal)
        self.repo_button.configure(state=disabled if repository_locked else normal)
        self.install_button.configure(state=disabled if busy or has_context else normal)
        self.download_button.configure(
            state=disabled if busy or has_context else normal
        )
        self.start_button.configure(state=disabled if busy or has_context else normal)
        self.stop_button.configure(
            state=normal if has_context and not busy else disabled
        )
        self.new_session_button.configure(
            state=normal if running and not busy else disabled
        )
        self.send_button.configure(state=normal if running and not busy else disabled)
        self.cancel_button.configure(state=normal if busy else disabled)
        self.refresh_button.configure(state=disabled if busy else normal)
        self.mode_box.configure(state="readonly")
        self.prompt.configure(state=normal)
        self.diagnostics_button.configure(state=normal)

    def _request_close(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.operation_cancel is not None:
            self.operation_cancel.set()
        self.status_var.set("Stopping client and resident model…")
        self._update_controls()

        def shutdown() -> None:
            try:
                self.controller.shutdown()
            finally:
                self.events.put(("closed", None))

        thread = threading.Thread(
            target=shutdown, name="local-coder-shutdown", daemon=True
        )
        thread.start()

        def wait_for_shutdown() -> None:
            if thread.is_alive() or (
                self.operation_thread is not None and self.operation_thread.is_alive()
            ):
                self.root.after(100, wait_for_shutdown)
            else:
                self.root.destroy()

        self.root.after(100, wait_for_shutdown)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=PROJECT_ROOT,
        help="initial Git repository shown by the picker",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        controller = LocalCoderController()
        app = LocalCoderApp(root, controller)
        app.repo_var.set(str(options.repository.expanduser()))
        root.mainloop()
        return 0
    except Exception as exc:
        try:
            messagebox.showerror("UGTOMS Local Coder could not start", str(exc))
        except Exception:
            print(f"UGTOMS Local Coder could not start: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
