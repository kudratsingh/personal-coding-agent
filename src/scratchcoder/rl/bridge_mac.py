"""The RL reward bridge, Mac side: poll the Pod over SSH, score on the Mac, push the reply.

``BridgeServer`` serves one Pod. Each poll lists the Pod's outbox and inbox (one SSH call)
and, for every request without a reply:

1. pulls it with a size bound and checks its schema, ID, count and payload SHA-256
   (``scratchcoder.rl.bridge.decode_request``);
2. answers a request it has answered before from its stored reply, keyed by request ID and
   payload SHA-256, and never scores it again;
3. refuses (a ``refused`` reply, not stored) a request older than one already answered in
   its session (``stale``), a reused request ID (``conflict``), a Pod task-file SHA-256
   other than the configured one (``task_file``) and a task ID outside the phase's allowed
   list (``unknown_task``): the kept screening list for rewards, the 200-task dev subset for
   dev checks;
4. scores it against the Mac's full task files: rewards with ``LocalRewardSource``, dev
   verdicts with the evaluation runner (``isolated_execution`` through the evaluation
   pipeline's ``run_calls``). A host failure the retries cannot resolve becomes a
   ``host_failure`` reply, a Mac-side reward timeout a ``timeout`` reply;
5. stores the reply on the Mac, then pushes it by atomic rename over SSH.

One Mac server serves a Pod at a time: the server takes a lease on the Pod (``server.lease``
in its bridge root, renewed by every listing and by the progress ticker while it scores,
reads a request or pushes a reply, so a slow SSH call never lets it lapse) and a second
server is refused with ``BridgeLeaseHeld``, so a request is never scored twice by two Macs
to different rewards.

With a ``window_check`` the server repeats the Mac execution-window checks
(``scratchcoder.sandbox.window``) on its own process while it runs: at its start and before
it scores a request once the last pass is ``window_check_seconds`` old. A failed or errored
pass ends the server (``WindowCheckFailed``) before anything more is scored, its receipt in
``window-checks/`` in the state directory.

SSH calls are retried a bounded number of times with doubling backoff; a request read whose
size differs from the listing is looked at again on the next poll, and an unreadable one is
refused only when the second look finds it unchanged (a non-atomic writer may still be at
it). ``progress.json`` in the state directory records the last request seen, the one being
scored and the last one answered, with times, so a watchdog can count reward waits as
progress. Replies carry no hidden test or expected value.

The SSH channel follows the rehearsal's pattern: pinned host keys, no config file, batch
mode, the same keep-alive options, one fixed remote program whose inputs travel as JSON on
stdin (never interpolated into a shell command), and a child environment with no ``R2_``
variable, no Runpod key, no SSH agent and no ``.env`` name. See
``docs/walkthroughs/rl-bridge.md``.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scratchcoder.common.logging import EventLogger
from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.eval.harness import HOST_RETRIES, isolated_execution
from scratchcoder.eval.parallel import Call, Execute, HostUnavailableError, Outcome, run_calls
from scratchcoder.eval.tasks import Task
from scratchcoder.rl.bridge import (
    DEV,
    FILE_NAME,
    HOST_FAILURE,
    KINDS,
    MAX_REPLY_BYTES,
    MAX_REQUEST_BYTES,
    REWARD,
    TIMEOUT,
    BridgeError,
    BridgeReply,
    BridgeRequest,
    BridgeSchemaError,
    DevItem,
    RequestName,
    atomic_write,
    decode_reply,
    decode_request,
    failure_reply,
    parse_request_id,
    read_bounded,
    refusal_reply,
    require_sha256,
    scored_reply,
)
from scratchcoder.rl.dev import DevExecutor
from scratchcoder.rl.rewards import (
    DEFAULT_WORKERS,
    Attempt,
    LocalRewardSource,
    RewardHostFailure,
    RewardSource,
    RewardTimeout,
    check_rewards,
)
from scratchcoder.sandbox.runner import Result
from scratchcoder.sandbox.window import failed_checks, pass_seconds_bound

PROGRESS_FORMAT = "rl-bridge-progress-v1"
LEASE_FORMAT = "rl-bridge-lease-v1"
LEASE_FILE = "server.lease"
LEASE_LOCK = "server.lease.lock"
LEASE_SECONDS = 60.0
"""How long a server's claim on the Pod holds without renewal. A poll or the progress ticker
(while a request is read, scored or answered) renews it every few seconds, so another
server can take over only after this server has been silent for this long."""
MAX_LEASE_SECONDS = 3600.0
SERVER_ID = re.compile(r"[0-9a-f]{32}\Z")
EXIT_TOO_LARGE = 3
EXIT_MISSING = 4
EXIT_REFUSED = 5
MAX_LISTING_BYTES = 4 * 1024 * 1024
PHASES = {REWARD: "kept screening list", DEV: "dev subset"}
WINDOW_CHECK_INTERVAL_SECONDS = 900.0
"""How old the bridge's last full execution-window pass may be when it scores a request: an
older one is repeated first (``BridgeServer.window_check``). A pass is 11 ``docker`` calls (five
container starts that fail to mount) and three local calls, seconds on an idle Docker
Desktop. At the planning rates RL sends a reward request about every 60 s (7 s of rollouts at
batch 256, 5 s of training, 48 s of scoring), so between its dev checks a pass runs before about
one request in fifteen and costs the Mac about 1% of its time; one before every request would
add its seconds to each of RL's 100 reward waits. A change to the Mac's sharing, agents, image
or environment is caught before the first request scored more than 15 minutes after the last
pass began. The Pod's timeouts hold one pass at its worst
(``scratchcoder.sandbox.window.pass_seconds_bound``). It is also the longest interval a server
accepts: a shorter one only repeats the passes more often, a longer one (a typed 9000 for 900)
would quietly turn the repeat into a start-only check."""
WINDOW_RECEIPTS = "window-checks"
"""The state directory's folder of window-pass receipts, one ``pass-NNNNN.json`` per pass. A
server restarted over the same state numbers its passes after the highest receipt there, so
an earlier server's receipts (the failed pass that stopped it among them) are never
overwritten."""
WINDOW_RECEIPT = re.compile(r"pass-(\d{5,})\.json\Z")
IMAGE_CHECK_RETRIES = 3
"""Retries of the pinned image's ``docker image inspect`` before each request, after the first
try (4 tries, waiting 1, 2 and 4 s)."""
IMAGE_CHECK_TIMEOUT_SECONDS = 30.0
"""The bound on one ``docker image inspect`` of the check before each request
(``scratchcoder.rl.bridge_command.image_check``)."""
WINDOW_DOCKER_TIMEOUT_SECONDS = 60.0
"""The bound on one ``docker`` call of the bridge's window passes; one pass is at most
``window.pass_seconds_bound(60)`` = 730 s."""


def image_check_seconds_bound(
    retries: int = IMAGE_CHECK_RETRIES,
    *,
    timeout: float = IMAGE_CHECK_TIMEOUT_SECONDS,
    backoff_seconds: float = 1.0,
) -> float:
    """The longest the check before a request can take: every try to its bound and every
    backoff wait, 4 x 30 + 1 + 2 + 4 = 127 s."""
    return (retries + 1) * timeout + sum(backoff_seconds * 2.0**n for n in range(retries))


def window_check_seconds_bound() -> float:
    """The longest one of the bridge's window passes can take: 730 s."""
    return pass_seconds_bound(WINDOW_DOCKER_TIMEOUT_SECONDS)


class BridgeTransportError(RuntimeError):
    """An SSH call (or its fake) failed; retried with backoff, never a verdict."""


class RequestMissing(Exception):
    """The request left the outbox between the listing and the read (consumed or withdrawn)."""


class RequestTooLarge(Exception):
    """The request is larger than the read bound."""


class BridgeLeaseHeld(BridgeError):
    """Another bridge server holds the Pod's lease: two servers would score the same
    request twice, possibly to different rewards, and the Pod uses whichever lands first."""

    def __init__(self, holder: str, age_seconds: float) -> None:
        super().__init__(
            f"another bridge server ({holder}) renewed its lease on this Pod {age_seconds:.0f} s "
            "ago; stop it first, or wait for its lease to lapse"
        )
        self.holder = holder
        self.age_seconds = age_seconds


class WindowCheckFailed(BridgeError):
    """A Mac execution-window pass failed or could not be made (``BridgeServer.window_check``):
    the server scores nothing more and stops, its receipt naming the failed checks."""


WindowCheck = Callable[[], Mapping[str, Any]]
"""One full pass of the execution-window checks (``scratchcoder.sandbox.window.check``): a
record whose ``status`` is ``passed``, or anything else for a failed pass."""


class BridgeRemoteError(BridgeError):
    """The Pod's program refused its input (exit 5): a wrong bridge root or name. It would
    refuse again, so it is raised at once instead of being retried as an outage."""


@dataclass(frozen=True)
class PodListing:
    """Requests in the Pod's outbox (name to size in bytes) and replies in its inbox.

    ``ready`` is false while the Pod has no bridge root yet (RL has not started there): an
    idle poll, not a transport failure."""

    requests: Mapping[str, int]
    replies: frozenset[str]
    ready: bool = True


class PodTransport(Protocol):
    """How the Mac reaches the Pod's bridge root: list, read one request, write one reply.

    ``claim`` registers the server's lease: from then on every ``listing`` of a ready Pod
    takes or renews the lease in the same call, and raises ``BridgeLeaseHeld`` when another
    server holds it."""

    def claim(self, server: str, seconds: float) -> None: ...

    def listing(self) -> PodListing: ...

    def read_request(self, name: str, *, max_bytes: int) -> bytes: ...

    def write_reply(self, name: str, data: bytes) -> None: ...


def _checked_name(name: object) -> str:
    if not isinstance(name, str) or not FILE_NAME.fullmatch(name):
        raise ValueError("not a bridge file name")
    return name


def take_lease(
    root: Path, server: str, seconds: float, now: float
) -> tuple[bool, str | None, float]:
    """Take or renew the lease in ``root`` for ``server``; ``(held, holder, age)``.

    The lease is ``root/server.lease`` (JSON: server, renewed time, length), written by atomic
    rename under an exclusive ``flock`` of ``root/server.lease.lock``. Another server's lease
    holds until ``seconds`` after it was last renewed; an unreadable lease, or one dated in
    the future (a clock step), counts as lapsed. This is the logic of the remote program's
    ``take_lease``, for the directory-pair fake."""
    lock = os.open(root / LEASE_LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        holder: tuple[str, float, float] | None = None
        try:
            raw = read_bounded(root / LEASE_FILE, 4096)
            if raw is not None:
                value = json.loads(raw)
                holder = (
                    str(value["server"])[:200],
                    float(value["renewed"]),
                    float(value["seconds"]),
                )
        except (BridgeSchemaError, ValueError, KeyError, TypeError):
            holder = None
        if holder is not None and holder[0] != server and 0 <= now - holder[1] < holder[2]:
            return False, holder[0], now - holder[1]
        body = {"format": LEASE_FORMAT, "server": server, "renewed": now, "seconds": seconds}
        atomic_write(root / LEASE_FILE, json.dumps(body, sort_keys=True).encode())
        return True, server, 0.0
    finally:
        os.close(lock)


class DirectoryTransport:
    """A transport over a local directory pair (``root/outbox``, ``root/inbox``): the fake
    SSH transport for tests, with the same semantics as the remote program. ``wall`` is the
    Pod's clock, which dates the lease."""

    def __init__(self, root: Path, *, wall: Callable[[], float] = time.time) -> None:
        self.root = root
        self._wall = wall
        self._claim: tuple[str, float] | None = None

    def claim(self, server: str, seconds: float) -> None:
        self._claim = (server, seconds)

    def listing(self) -> PodListing:
        requests: dict[str, int] = {}
        replies: set[str] = set()
        for box in ("outbox", "inbox"):
            try:
                names = os.listdir(self.root / box)
            except FileNotFoundError:
                return PodListing({}, frozenset(), ready=False)
            except OSError as error:
                raise BridgeTransportError(f"cannot list {box}: {error}") from error
            for name in names:
                if not FILE_NAME.fullmatch(name):
                    continue
                try:
                    info = os.lstat(self.root / box / name)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                if box == "outbox":
                    requests[name] = info.st_size
                else:
                    replies.add(name)
        if self._claim is not None:
            try:
                held, holder, age = take_lease(self.root, *self._claim, self._wall())
            except OSError as error:
                raise BridgeTransportError(f"cannot take the lease: {error}") from error
            if not held:
                assert holder is not None
                raise BridgeLeaseHeld(holder, age)
        return PodListing(requests, frozenset(replies))

    def read_request(self, name: str, *, max_bytes: int) -> bytes:
        try:
            data = read_bounded(self.root / "outbox" / _checked_name(name), max_bytes)
        except BridgeSchemaError as error:
            raise RequestTooLarge(str(error)) from error
        except OSError as error:
            raise BridgeTransportError(f"cannot read {name}: {error}") from error
        if data is None:
            raise RequestMissing(name)
        return data

    def write_reply(self, name: str, data: bytes) -> None:
        try:
            atomic_write(self.root / "inbox" / _checked_name(name), data)
        except OSError as error:
            raise BridgeTransportError(f"cannot write {name}: {error}") from error


REMOTE_SCRIPT = r"""
import base64, fcntl, hashlib, json, os, re, stat, sys, time

NAME = re.compile(r'[0-9a-f]{12}-[0-9]{6}-(?:reward|dev)-[0-9]{6}\.json\Z')
NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
NONBLOCK = getattr(os, 'O_NONBLOCK', 0)
DIRECTORY = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | NOFOLLOW
MAX_FILES = 10000
MAX_WRITE = 16 * 1024 * 1024
SERVER = re.compile(r'[0-9a-f]{32}\Z')
LEASE = 'server.lease'
LEASE_LOCK = 'server.lease.lock'

def open_root(path):
    if not isinstance(path, str) or not path.startswith('/') or '\x00' in path:
        raise ValueError('the bridge root must be an absolute path')
    parts = path.split('/')[1:]
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('the bridge root has an unsafe component')
    fd = os.open('/', DIRECTORY)
    try:
        for part in parts:
            next_fd = os.open(part, DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise

def checked(name):
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError('not a bridge file name')
    return name

def not_ready():
    empty = {'ready': False, 'requests': {}, 'replies': []}
    sys.stdout.write(json.dumps(empty, sort_keys=True, separators=(',', ':')))
    return 0

def take_lease(root, claim):
    server, seconds = claim.get('server'), claim.get('seconds')
    if not isinstance(server, str) or not SERVER.match(server):
        raise ValueError('invalid lease server')
    if type(seconds) not in (int, float) or not 0 < seconds <= 3600:
        raise ValueError('invalid lease length')
    lock = os.open(LEASE_LOCK, os.O_RDWR | os.O_CREAT | NOFOLLOW, 0o600, dir_fd=root)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = time.time()
        holder = None
        try:
            fd = os.open(LEASE, os.O_RDONLY | NOFOLLOW | NONBLOCK, dir_fd=root)
        except FileNotFoundError:
            fd = None
        if fd is not None:
            try:
                raw = os.read(fd, 4096)
            finally:
                os.close(fd)
            try:
                value = json.loads(raw)
                holder = (str(value['server'])[:200], float(value['renewed']),
                          float(value['seconds']))
            except (ValueError, KeyError, TypeError):
                holder = None
        if holder is not None and holder[0] != server and 0 <= now - holder[1] < holder[2]:
            return {'held': False, 'holder': holder[0], 'age': now - holder[1]}
        body = {'format': 'rl-bridge-lease-v1', 'server': server, 'renewed': now,
                'seconds': seconds}
        data = json.dumps(body, sort_keys=True).encode()
        temp = '.' + LEASE + '.' + os.urandom(8).hex() + '.tmp'
        tmp = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600, dir_fd=root)
        try:
            try:
                os.write(tmp, data)
                os.fsync(tmp)
            finally:
                os.close(tmp)
            os.rename(temp, LEASE, src_dir_fd=root, dst_dir_fd=root)
        except BaseException:
            try:
                os.unlink(temp, dir_fd=root)
            except OSError:
                pass
            raise
        return {'held': True, 'holder': server, 'age': 0.0}
    finally:
        os.close(lock)

def listing(root, claim):
    result = {'ready': True, 'requests': {}, 'replies': []}
    for box, key in (('outbox', 'requests'), ('inbox', 'replies')):
        try:
            fd = os.open(box, DIRECTORY, dir_fd=root)
        except FileNotFoundError:
            return not_ready()
        try:
            names = os.listdir(fd)
            if len(names) > MAX_FILES:
                raise ValueError('too many files in ' + box)
            for name in names:
                if not NAME.match(name):
                    continue
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                if key == 'requests':
                    result[key][name] = info.st_size
                else:
                    result[key].append(name)
        finally:
            os.close(fd)
    result['replies'].sort()
    if claim is not None:
        result['lease'] = take_lease(root, claim)
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return 0

def read(root, name, limit):
    if type(limit) is not int or limit < 0:
        raise ValueError('invalid bound')
    box = os.open('outbox', DIRECTORY, dir_fd=root)
    try:
        try:
            fd = os.open(name, os.O_RDONLY | NOFOLLOW | NONBLOCK, dir_fd=box)
        except FileNotFoundError:
            return 4
    finally:
        os.close(box)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('not a regular file')
        if info.st_size > limit:
            return 3
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                return 3
            chunks.append(chunk)
    finally:
        os.close(fd)
    sys.stdout.buffer.write(b''.join(chunks))
    return 0

def write(root, name, encoded):
    data = base64.b64decode(encoded, validate=True)
    if len(data) > MAX_WRITE:
        raise ValueError('reply too large')
    box = os.open('inbox', DIRECTORY, dir_fd=root)
    try:
        temp = '.' + name + '.' + os.urandom(8).hex() + '.tmp'
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW
        fd = os.open(temp, flags, 0o600, dir_fd=box)
        try:
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(temp, name, src_dir_fd=box, dst_dir_fd=box)
        except BaseException:
            try:
                os.unlink(temp, dir_fd=box)
            except OSError:
                pass
            raise
        try:
            os.fsync(box)
        except OSError:
            pass
    finally:
        os.close(box)
    digest = hashlib.sha256(data).hexdigest()
    sys.stdout.write(json.dumps({'sha256': digest, 'size': len(data)}, sort_keys=True))
    return 0

def main():
    try:
        return serve()
    except ValueError as error:
        sys.stderr.write('refused: ' + str(error)[:200])
        return 5

def serve():
    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise ValueError('invalid request')
    action = request.get('action')
    try:
        root = open_root(request.get('root'))
    except FileNotFoundError:
        if action == 'list':
            return not_ready()
        raise
    try:
        if action == 'list':
            claim = request.get('lease')
            if claim is not None and not isinstance(claim, dict):
                raise ValueError('invalid lease claim')
            return listing(root, claim)
        if action == 'read':
            return read(root, checked(request.get('name')), request.get('max_bytes'))
        if action == 'write':
            return write(root, checked(request.get('name')), request.get('data'))
        raise ValueError('unsupported action')
    finally:
        os.close(root)

sys.exit(main())
"""
"""The one program the Mac runs on the Pod, as ``python3 -I -c``; its inputs (the bridge
root, the action, a file name, a bound, a base64 reply) arrive as JSON on stdin. It lists
only regular files with bridge request names, reads only from ``outbox/`` and writes only to
``inbox/``, opening every path component without following symlinks. The one other file it
writes is the lease: a listing that carries a claim takes or renews ``server.lease`` in the
root (``take_lease``), so a second Mac server is refused. Exit 3 means the request is over
the bound, exit 4 that it is gone, exit 5 that the program refused its input (an unsafe root
or name, a bad bound): a defect of the call, not an outage, so never retried."""


# ------------------------------------------------------------------------------- SSH

_SAFE_HOST = re.compile(r"[A-Za-z0-9._-]+\Z")
_SAFE_USER = re.compile(r"[A-Za-z0-9._-]+\Z")
_PROXY_SUFFIX = "ssh.runpod.io"
SSH_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
"""The only variables an ssh child inherits: no ``R2_`` variable, Runpod key, SSH agent
socket or ``.env`` name can reach it."""
SECRET_NAMES = frozenset({"RUNPOD_API_KEY"})
SECRET_PREFIXES = ("R2_",)
AGENT_NAMES = ("SSH_AUTH_SOCK", "SSH_AGENT_PID")


def validate_host(host: object) -> str:
    """``user@host`` or ``host`` of safe characters; the Runpod SSH proxy is refused."""
    if not isinstance(host, str) or not host or host.startswith("-") or host.count("@") > 1:
        raise ValueError("SSH host must be a user@host or host value without a leading dash")
    user, _, hostname = host.rpartition("@")
    if (user and not _SAFE_USER.fullmatch(user)) or not _SAFE_HOST.fullmatch(hostname):
        raise ValueError("SSH user and host must contain only letters, digits, dot, _ or -")
    if hostname.startswith("-"):
        raise ValueError("SSH hostname must not start with a dash")
    if hostname.lower().endswith(_PROXY_SUFFIX):
        raise ValueError("the Runpod SSH proxy cannot carry the bridge; use the direct public IP")
    return host


def validate_port(port: object) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("SSH port must be an integer from 1 to 65535")
    return port


def validate_local_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().absolute()
    try:
        info = resolved.lstat()
    except OSError as error:
        raise ValueError(f"{name} file is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular, non-symlink file")
    return resolved


def validate_remote_root(root: object) -> str:
    if not isinstance(root, str) or not root.startswith("/") or "\x00" in root:
        raise ValueError("the bridge root must be an absolute POSIX path")
    # The remote program's rule, exactly: no empty (a trailing or doubled slash), "." or ".."
    # component. A root it would refuse would otherwise look like an SSH outage.
    if any(part in ("", ".", "..") for part in root.split("/")[1:]):
        raise ValueError(
            "the bridge root must be a normalised path: no trailing or doubled slash and "
            "no . or .. component"
        )
    return root


def ssh_options(port: int, identity: Path, known_hosts: Path) -> list[str]:
    """The rehearsal's SSH options (``tools/pilot_transfer.py``): pinned host key, no
    config file, the given key only, batch mode, 10 s connect, 15 s keep-alive x 3."""
    return [
        "-T",
        "-F",
        "/dev/null",
        "-p",
        str(port),
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


def ssh_command(host: str, port: int, identity_file: Path, known_hosts: Path) -> list[str]:
    """The argument vector that runs ``REMOTE_SCRIPT`` on the Pod; nothing is interpolated."""
    host = validate_host(host)
    port = validate_port(port)
    identity = validate_local_file(identity_file, "identity")
    hosts = validate_local_file(known_hosts, "known-hosts")
    remote = "python3 -I -c " + shlex.quote(REMOTE_SCRIPT)
    return ["ssh", *ssh_options(port, identity, hosts), host, remote]


def ssh_child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The allowlisted environment of an ssh child (``SSH_ENVIRONMENT`` only)."""
    source = os.environ if environ is None else environ
    return {key: source[key] for key in SSH_ENVIRONMENT if key in source}


def dotenv_names(path: Path) -> set[str]:
    """The variable names a ``.env`` file defines; its values are never kept."""
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export ") :].lstrip()
        name = text.split("=", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and "=" in text:
            names.add(name)
    return names


def environment_refusals(
    environ: Mapping[str, str], dotenv: Path | Iterable[Path] | None = None
) -> list[str]:
    """Names (never values) of set variables the bridge must not run with: any ``R2_``
    variable, the Runpod key, and every name the ``dotenv`` file or files define. Docker and
    ssh children inherit the bridge's environment, so it is refused rather than filtered.
    A ``dotenv`` file that does not exist raises ``FileNotFoundError``: a check that finds
    nothing to read must not pass silently."""
    names = {
        key
        for key, value in environ.items()
        if value and (key in SECRET_NAMES or key.startswith(SECRET_PREFIXES))
    }
    files = [dotenv] if isinstance(dotenv, Path) else list(dotenv or ())
    for path in files:
        names |= {key for key in dotenv_names(path) if environ.get(key)}
    return sorted(names)


Run = Callable[..., subprocess.CompletedProcess[bytes]]


class SSHTransport:
    """The Pod's bridge root over SSH: each call runs ``command`` (from ``ssh_command``)
    with a JSON request on stdin, under ``timeout`` seconds, with ``environment``
    (``ssh_child_environment()`` by default). ``run`` is ``subprocess.run`` or a test fake;
    tests can also point ``command`` at a local ``python -I -c REMOTE_SCRIPT``."""

    def __init__(
        self,
        command: Sequence[str],
        root: str,
        *,
        timeout: float = 120.0,
        list_timeout: float = 30.0,
        run: Run | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("the SSH command is empty")
        for name, value in (("timeout", timeout), ("list_timeout", list_timeout)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number of seconds")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.command = list(command)
        self.root = validate_remote_root(root)
        self.timeout = float(timeout)
        self.list_timeout = float(list_timeout)
        self._run: Run = run or subprocess.run
        self._claim: tuple[str, float] | None = None
        self.environment = dict(ssh_child_environment() if environment is None else environment)
        leaked = [
            key
            for key in self.environment
            if key in SECRET_NAMES or key.startswith(SECRET_PREFIXES) or key in AGENT_NAMES
        ]
        if leaked:
            raise ValueError(f"the ssh child environment must not hold {leaked}")

    def _call(self, request: dict[str, Any], *, timeout: float, limit: int) -> bytes:
        payload = json.dumps({"root": self.root, **request}, separators=(",", ":")).encode()
        try:
            result = self._run(
                self.command,
                input=payload,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=self.environment,
            )
        except subprocess.TimeoutExpired as error:
            raise BridgeTransportError(f"SSH {request['action']} timed out") from error
        except OSError as error:
            raise BridgeTransportError(f"SSH {request['action']} did not start: {error}") from error
        if result.returncode == EXIT_TOO_LARGE:
            raise RequestTooLarge(str(request.get("name")))
        if result.returncode == EXIT_MISSING:
            raise RequestMissing(str(request.get("name")))
        if result.returncode == EXIT_REFUSED:
            tail = (result.stderr or b"")[-300:].decode("utf-8", errors="replace").strip()
            raise BridgeRemoteError(
                f"the Pod's bridge program refused the {request['action']}: {tail}"
            )
        if result.returncode != 0:
            tail = (result.stderr or b"")[-300:].decode("utf-8", errors="replace").strip()
            raise BridgeTransportError(
                f"SSH {request['action']} failed (exit {result.returncode}): {tail}"
            )
        stdout = result.stdout or b""
        if len(stdout) > limit:
            raise BridgeTransportError(f"SSH {request['action']} returned too many bytes")
        return stdout

    def claim(self, server: str, seconds: float) -> None:
        if not isinstance(server, str) or not SERVER_ID.fullmatch(server):
            raise ValueError("a bridge server ID is 32 lowercase hex digits")
        self._claim = (server, _lease_seconds(seconds))

    def listing(self) -> PodListing:
        request: dict[str, Any] = {"action": "list"}
        if self._claim is not None:
            request["lease"] = {"server": self._claim[0], "seconds": self._claim[1]}
        raw = self._call(request, timeout=self.list_timeout, limit=MAX_LISTING_BYTES)
        try:
            value = json.loads(raw)
            requests, replies, ready = value["requests"], value["replies"], value["ready"]
            lease = value.get("lease")
            leased = self._claim is not None and ready is True
            if (
                set(value) != {"ready", "requests", "replies"} | ({"lease"} if leased else set())
                or type(ready) is not bool
                or not isinstance(requests, dict)
                or not isinstance(replies, list)
                or any(not FILE_NAME.fullmatch(name) for name in [*requests, *replies])
                or any(type(size) is not int or size < 0 for size in requests.values())
                or (leased and not _valid_lease_reply(lease))
            ):
                raise ValueError("listing shape")
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
            raise BridgeTransportError("the Pod returned a malformed listing") from error
        if leased and not lease["held"]:
            raise BridgeLeaseHeld(lease["holder"], float(lease["age"]))
        return PodListing(dict(requests), frozenset(replies), ready)

    def read_request(self, name: str, *, max_bytes: int) -> bytes:
        request = {"action": "read", "name": _checked_name(name), "max_bytes": max_bytes}
        return self._call(request, timeout=self.timeout, limit=max_bytes)

    def write_reply(self, name: str, data: bytes) -> None:
        request = {
            "action": "write",
            "name": _checked_name(name),
            "data": base64.b64encode(data).decode("ascii"),
        }
        raw = self._call(request, timeout=self.timeout, limit=4096)
        try:
            receipt = json.loads(raw)
            ok = receipt == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        except (ValueError, UnicodeDecodeError):
            ok = False
        if not ok:
            raise BridgeTransportError(f"the Pod did not confirm the reply {name}")


def _lease_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("the lease length must be a number of seconds")
    if not math.isfinite(value) or not 0 < value <= MAX_LEASE_SECONDS:
        raise ValueError(f"the lease length must lie in (0, {MAX_LEASE_SECONDS:g}] seconds")
    return float(value)


def _valid_lease_reply(lease: object) -> bool:
    return (
        isinstance(lease, dict)
        and set(lease) == {"held", "holder", "age"}
        and type(lease["held"]) is bool
        and isinstance(lease["holder"], str)
        and len(lease["holder"]) <= 200
        and isinstance(lease["age"], (int, float))
        and not isinstance(lease["age"], bool)
        and math.isfinite(lease["age"])
    )


# ----------------------------------------------------------------------- Mac scoring


class _PastDeadline(Exception):
    """Raised inside the dev executor's workers once the check's deadline has passed."""


def _encodable(code: str) -> bool:
    """Whether the sandbox can be sent ``code``: an unpaired surrogate cannot be written as
    UTF-8, so the runner would raise instead of running it."""
    try:
        code.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class SandboxDevExecutor:
    """Dev verdicts on the Mac: each ``(task, program)`` runs with the task's hidden tests
    through the evaluation runner (``isolated_execution``, one fresh container per call),
    in parallel through ``run_calls``, host failures retried ``retries`` times. A pair still
    unresolved makes the whole check a host failure (``RewardHostFailure``): a dev verdict
    is never guessed. After ``max_consecutive_unresolved`` unresolved calls in a row (the
    sandbox host looks down) no further call starts, so a dead Docker fails the check in
    seconds rather than after every program's retries.

    A program that cannot be written as UTF-8 (an unpaired surrogate) fails without running,
    as the reward channel scores it ``invalid_source``: it is the candidate's failure, not
    the host's. ``timeout`` (seconds, ``None`` for none) bounds one check: once it has
    passed no further call or retry starts and ``RewardTimeout`` is raised, after the calls
    already running finish at the runner's own deadlines. Without it a check the Pod has
    already given up on would keep the single-threaded server busy while the next request
    waits behind it."""

    def __init__(
        self,
        *,
        workers: int = DEFAULT_WORKERS,
        retries: int = HOST_RETRIES,
        execute: Execute = isolated_execution,
        sleep: Callable[[float], None] = time.sleep,
        max_consecutive_unresolved: int | None = 16,
        timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive seconds or None")
        self.workers, self.retries, self.execute, self.sleep = workers, retries, execute, sleep
        self.max_consecutive_unresolved = max_consecutive_unresolved
        self.timeout = None if timeout is None else float(timeout)
        self.clock = clock

    def __call__(self, pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        deadline = None if self.timeout is None else self.clock() + self.timeout
        outcomes: dict[int, Outcome] = {}
        unrunnable = {position for position, (_, code) in enumerate(pairs) if not _encodable(code)}

        def keep(outcome: Outcome) -> None:
            assert isinstance(outcome.key, int)
            outcomes[outcome.key] = outcome

        def execute(code: str, tests: str, mode: str) -> Result:
            if deadline is not None and self.clock() >= deadline:
                raise _PastDeadline
            return self.execute(code, tests, mode)

        def sleep(seconds: float) -> None:
            if deadline is not None and self.clock() + seconds > deadline:
                raise _PastDeadline
            self.sleep(seconds)

        calls = (
            Call(position, code, task.hidden_tests, task.mode)
            for position, (task, code) in enumerate(pairs)
            if position not in unrunnable
        )
        try:
            run_calls(
                calls,
                execute,
                on_outcome=keep,
                workers=self.workers,
                retries=self.retries,
                sleep=sleep,
                max_consecutive_unresolved=self.max_consecutive_unresolved,
            )
        except _PastDeadline as error:
            raise RewardTimeout(
                f"{len(outcomes)} of {len(pairs) - len(unrunnable)} dev programs ran within "
                f"{self.timeout:g} s"
            ) from error
        except HostUnavailableError as error:
            raise RewardHostFailure(f"the sandbox host looks down: {error}") from error
        unresolved = [outcome for outcome in outcomes.values() if not outcome.resolved]
        if unresolved:
            raise RewardHostFailure(
                f"{len(unresolved)} of {len(pairs)} dev programs stayed unresolved after "
                f"{self.retries} retries (last status {unresolved[0].status})"
            )
        return [
            False if position in unrunnable else outcomes[position].passed
            for position in range(len(pairs))
        ]


# ------------------------------------------------------------------------ Mac state


class ReplyStore:
    """Replies the Mac sent, by request ID, exactly as pushed (``<request id>.json``)."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

    def get(self, request_id: str) -> tuple[bytes, BridgeReply] | None:
        path = self.directory / f"{request_id}.json"
        raw = read_bounded(path, MAX_REPLY_BYTES)
        if raw is None:
            return None
        try:
            reply = decode_reply(raw)
        except BridgeSchemaError as error:
            raise BridgeError(f"the Mac's stored reply {path} is unreadable") from error
        if reply.request_id != request_id:
            raise BridgeError(f"the Mac's stored reply {path} names another request")
        return raw, reply

    def put(self, request_id: str, raw: bytes) -> None:
        atomic_write(self.directory / f"{request_id}.json", raw)

    def count(self) -> int:
        return sum(1 for path in self.directory.iterdir() if FILE_NAME.fullmatch(path.name))

    def highest_seq(self) -> dict[str, int]:
        """The highest answered sequence number of each session."""
        highest: dict[str, int] = {}
        for path in self.directory.iterdir():
            if FILE_NAME.fullmatch(path.name):
                name = parse_request_id(path.name)
                highest[name.session] = max(highest.get(name.session, 0), name.seq)
        return highest


def _request_row(name: RequestName, unix: float) -> dict[str, Any]:
    return {"request_id": name.request_id, "kind": name.kind, "index": name.index, "unix": unix}


class BridgeProgress:
    """``progress.json``: what the Mac last saw, is scoring and last answered, with times.

    Rewritten atomically on every change and every ``touch`` (the server's ticker touches
    it while a long scoring call runs), so its ``updated_unix`` shows the Mac is alive."""

    def __init__(self, path: Path, *, wall: Callable[[], float], fields: Mapping[str, Any]) -> None:
        self.path = path
        self._wall = wall
        self._lock = threading.Lock()
        now = wall()
        self.value: dict[str, Any] = {
            "format": PROGRESS_FORMAT,
            "state": "starting",
            "pid": os.getpid(),
            "started_unix": now,
            "updated_unix": now,
            "updated_at": utc_now(),
            "polls": 0,
            "pod_ready": None,
            "last_poll_unix": None,
            "current": None,
            "last_seen": None,
            "last_answered": None,
            "answered": 0,
            "replayed": 0,
            "refused": 0,
            "deferred": 0,
            "transport_failures": 0,
            "consecutive_failures": 0,
            "last_error": None,
            **fields,
        }
        self.write()

    def update(self, **changes: Any) -> None:
        with self._lock:
            self.value.update(changes)
            self._write_locked()

    def bump(self, counter: str, **changes: Any) -> None:
        with self._lock:
            self.value[counter] += 1
            self.value.update(changes)
            self._write_locked()

    def write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        self.value["updated_unix"] = self._wall()
        self.value["updated_at"] = utc_now()
        atomic_json(self.path, self.value)


def _short(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]


def server_identity(state: Path, server_id: str | None = None) -> str:
    """This server's ID: 32 hex digits, kept in ``state/server.id`` so a restart over the
    same state directory is the same server and takes its own lease back at once."""
    path = state / "server.id"
    if server_id is None:
        raw = read_bounded(path, 128)
        if raw is not None:
            server_id = raw.decode("ascii", errors="replace").strip()
        else:
            server_id = secrets.token_hex(16)
            atomic_write(path, server_id.encode("ascii") + b"\n")
    if not isinstance(server_id, str) or not SERVER_ID.fullmatch(server_id):
        raise ValueError(f"the bridge server ID in {path} is not 32 lowercase hex digits")
    return server_id


class BridgeServer:
    """The Mac's end of the bridge for one Pod (see the module docstring).

    ``reward_tasks`` are the full records of the kept screening list (train split) and
    ``dev_tasks`` those of the 200-task dev subset (dev split); their IDs are each phase's
    allowed list. ``pod_task_sha256`` maps ``reward`` and ``dev`` to the SHA-256 of the
    Pod's task file for that phase. ``reward_source`` defaults to a ``LocalRewardSource``
    over ``reward_tasks`` with ``workers`` threads and ``reward_timeout``; ``dev_execute``
    to a ``SandboxDevExecutor`` with ``dev_timeout``. Both timeouts are Mac-side bounds on
    one request (``None`` for none); the single-threaded server scores one request at a
    time, so without them a request the Pod has withdrawn keeps the next one waiting. Set
    them to the Pod's timeout less the bridge's allowance (``docs/rl.md``). ``before_scoring``
    runs before each request is scored (for example a check of the sandbox image); if it
    raises, the request gets a ``host_failure`` reply and nothing runs. State lives under
    ``state``: ``replies/``, ``progress.json``, ``events.jsonl``, ``window-checks/`` and an
    exclusive ``server.lock``.

    ``window_check`` is one full pass of the Mac execution-window checks (``None`` for none).
    It runs when ``serve`` starts and again before a request is scored whenever the last pass
    began ``window_check_seconds`` ago or more; each pass leaves a receipt in
    ``window-checks/``. A pass that fails, or raises, ends the server with
    ``WindowCheckFailed`` before anything more is scored: the request stays unanswered, the
    lease lapses, and the Pod gives its wait up as a lost bridge. While a pass runs the ticker
    renews the lease, so a slow pass never reads as a dead Mac.
    """

    def __init__(
        self,
        transport: PodTransport,
        *,
        reward_tasks: Sequence[Task],
        dev_tasks: Sequence[Task],
        pod_task_sha256: Mapping[str, str],
        state: Path,
        reward_source: RewardSource | None = None,
        dev_execute: DevExecutor | None = None,
        before_scoring: Callable[[BridgeRequest], None] | None = None,
        workers: int = DEFAULT_WORKERS,
        reward_timeout: float | None = None,
        dev_timeout: float | None = None,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        transport_retries: int = 4,
        backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 16.0,
        server_id: str | None = None,
        lease_seconds: float | None = LEASE_SECONDS,
        inputs: Mapping[str, Any] | None = None,
        window_check: WindowCheck | None = None,
        window_check_seconds: float = WINDOW_CHECK_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.reward_tasks = _phase_tasks(reward_tasks, "train", REWARD)
        self.dev_tasks = _phase_tasks(dev_tasks, "dev", DEV)
        if set(pod_task_sha256) != set(KINDS):
            raise ValueError(f"pod_task_sha256 names exactly {KINDS}")
        self.pod_task_sha256 = {
            kind: require_sha256(value, f"the Pod's {kind} task-file SHA-256")
            for kind, value in pod_task_sha256.items()
        }
        self.allowed = {
            REWARD: frozenset(self.reward_tasks),
            DEV: frozenset(self.dev_tasks),
        }
        if type(transport_retries) is not int or transport_retries < 0:
            raise ValueError("transport_retries must be a nonnegative integer")
        if type(max_request_bytes) is not int or max_request_bytes < 1:
            raise ValueError("max_request_bytes must be a positive integer")
        if not 0 <= backoff_seconds <= max_backoff_seconds:
            raise ValueError("backoff must satisfy 0 <= backoff_seconds <= max_backoff_seconds")
        self.transport = transport
        self.reward_source: RewardSource = reward_source or LocalRewardSource(
            self.reward_tasks.values(), timeout=reward_timeout, workers=workers
        )
        self.dev_execute: DevExecutor = dev_execute or SandboxDevExecutor(
            workers=workers, timeout=dev_timeout
        )
        self.before_scoring = before_scoring
        if (
            isinstance(window_check_seconds, bool)
            or not isinstance(window_check_seconds, (int, float))
            or not math.isfinite(window_check_seconds)
            or window_check_seconds <= 0
            or window_check_seconds > WINDOW_CHECK_INTERVAL_SECONDS
        ):
            raise ValueError(
                "window_check_seconds must be positive and at most "
                f"{WINDOW_CHECK_INTERVAL_SECONDS:.0f} (the justified bound)"
            )
        self.window_check = window_check
        self.window_check_seconds = float(window_check_seconds)
        self._window_passes = highest_window_pass(state)
        self._window_started: float | None = None
        """When the last passed window pass began (this server's monotonic clock)."""
        self.max_request_bytes = max_request_bytes
        self.transport_retries = transport_retries
        self.backoff_seconds = float(backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self._clock, self._wall, self._sleep = clock, wall, sleep
        state.mkdir(parents=True, exist_ok=True)
        self.state = state
        self._lock = os.open(state / "server.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._lock)
            raise BridgeError(f"another bridge server holds {state}") from error
        try:
            self.store = ReplyStore(state / "replies")
            self.highest = self.store.highest_seq()
            self._deferred: dict[str, int] = {}
            self.server_id = server_identity(state, server_id)
            self.lease_seconds = None if lease_seconds is None else _lease_seconds(lease_seconds)
            if self.lease_seconds is not None:
                transport.claim(self.server_id, self.lease_seconds)
            self._scoring = threading.Event()
            self._transferring = threading.Event()
            self._checking = threading.Event()
            self._lease_lost: BridgeLeaseHeld | None = None
            self.log = EventLogger(state / "events.jsonl", "rl-bridge")  # last: nothing to close
        except BaseException:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)
            raise
        self._closed = False
        self.progress = BridgeProgress(
            state / "progress.json",
            wall=wall,
            fields={
                "server_id": self.server_id,
                "lease_seconds": self.lease_seconds,
                "pod_task_sha256": dict(self.pod_task_sha256),
                "allowed_tasks": {kind: len(ids) for kind, ids in self.allowed.items()},
                "inputs": dict(inputs or {}),
                "window_check": (
                    None
                    if window_check is None
                    else {
                        "interval_seconds": self.window_check_seconds,
                        "passes": 0,
                        "last": None,
                    }
                ),
            },
        )
        self.log.event(
            "bridge_started",
            server_id=self.server_id,
            lease_seconds=self.lease_seconds,
            pod_task_sha256=dict(self.pod_task_sha256),
            allowed_tasks={kind: len(ids) for kind, ids in self.allowed.items()},
            stored_replies=self.store.count(),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.log.close()
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)

    def __enter__(self) -> BridgeServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------------- transport

    def _retry[T](self, what: str, operation: Callable[[], T]) -> T:
        for attempt in range(self.transport_retries + 1):
            try:
                return operation()
            except BridgeTransportError as error:
                self.progress.bump("transport_failures", last_error=_short(error))
                self.log.event("transport_failure", operation=what, attempt=attempt + 1)
                if attempt == self.transport_retries:
                    raise
                self._sleep(min(self.max_backoff_seconds, self.backoff_seconds * 2.0**attempt))
        raise AssertionError("unreachable")

    def _transfer[T](self, what: str, operation: Callable[[], T]) -> T:
        """``_retry`` for a read or a write, with the ticker renewing the lease meanwhile:
        at its worst (every try to its timeout) one such call outlasts the lease and the
        Pod's lost-bridge rule, though the Mac is alive. A lease another server took ends
        this server before anything is pushed."""
        self._transferring.set()
        try:
            return self._retry(what, operation)
        finally:
            self._transferring.clear()

    def _push(self, name: RequestName, raw: bytes) -> None:
        if self._lease_lost is not None:
            raise self._lease_lost
        self._transfer("write", lambda: self.transport.write_reply(name.file_name, raw))

    # ----------------------------------------------------------------------- polling

    def poll_once(self) -> list[str]:
        """List the Pod's boxes once and handle every request without a reply; returns
        the IDs of the requests answered, replayed or refused in this poll."""
        listing = self._retry("list", self.transport.listing)
        now = self._wall()
        if not listing.ready:
            self.progress.bump("polls", last_poll_unix=now, consecutive_failures=0, pod_ready=False)
            return []
        pending: list[tuple[RequestName, int]] = []
        for file_name, size in listing.requests.items():
            if file_name not in listing.replies:
                pending.append((parse_request_id(file_name), size))
        self.progress.bump("polls", last_poll_unix=now, consecutive_failures=0, pod_ready=True)
        names = {name.file_name for name, _ in pending}
        self._deferred = {key: size for key, size in self._deferred.items() if key in names}
        newest: dict[str, int] = {}
        for name, _ in pending:
            newest[name.session] = max(newest.get(name.session, 0), name.seq)
        handled: list[str] = []
        for name, size in sorted(pending, key=lambda row: (row[0].session, row[0].seq)):
            if self._handle(name, size, superseded=name.seq < newest[name.session]):
                handled.append(name.request_id)
        return handled

    def _handle(self, name: RequestName, size: int, *, superseded: bool) -> bool:
        self.progress.update(last_seen=_request_row(name, self._wall()))
        if size > self.max_request_bytes:
            return self._refuse(name, "oversized", f"{size} bytes, over {self.max_request_bytes}")
        try:
            raw = self._transfer(
                "read",
                lambda: self.transport.read_request(
                    name.file_name, max_bytes=self.max_request_bytes
                ),
            )
        except RequestMissing:
            return False  # consumed or withdrawn since the listing
        except RequestTooLarge:
            return self._refuse(name, "oversized", f"over {self.max_request_bytes} bytes")
        if self._lease_lost is not None:
            raise self._lease_lost  # another server took the Pod during the read
        if len(raw) != size:
            self.log.event(
                "request_changed", request_id=name.request_id, listed=size, read=len(raw)
            )
            return False  # it changed since the listing: look again at the next poll
        try:
            request = decode_request(
                raw, max_bytes=self.max_request_bytes, expected_id=name.request_id
            )
        except BridgeSchemaError as error:
            if error.reason == "corrupt" and self._deferred.get(name.file_name) != size:
                self._deferred[name.file_name] = size
                self.progress.bump("deferred")
                self.log.event("request_deferred", request_id=name.request_id, size=size)
                return False
            return self._refuse(name, error.reason, str(error))
        self._deferred.pop(name.file_name, None)
        stored = self.store.get(name.request_id)
        if stored is not None:
            raw_reply, reply = stored
            if (reply.payload_sha256, reply.task_file_sha256, reply.count) == (
                request.payload_sha256,
                request.task_file_sha256,
                request.count,
            ):
                self._push(name, raw_reply)
                self.progress.bump(
                    "replayed",
                    last_answered={
                        **_request_row(name, self._wall()),
                        "outcome": reply.outcome,
                        "replayed": True,
                    },
                )
                self.log.event("request_replayed", request_id=name.request_id)
                return True
            return self._refuse(
                name, "conflict", "this request ID was answered for another payload", request
            )
        if superseded or name.seq <= self.highest.get(name.session, 0):
            return self._refuse(
                name, "stale", "a later request of this session exists or was answered", request
            )
        expected = self.pod_task_sha256[name.kind]
        if request.task_file_sha256 != expected:
            return self._refuse(
                name,
                "task_file",
                f"the Pod's {name.kind} task file is {request.task_file_sha256}, "
                f"configured {expected}",
                request,
            )
        unknown = sorted(request.task_ids - self.allowed[name.kind])
        if unknown:
            return self._refuse(
                name,
                "unknown_task",
                f"{len(unknown)} task IDs outside the {PHASES[name.kind]}, "
                f"e.g. {unknown[0][:100]!r}",
                request,
            )
        reply = self._score(request)
        if self._lease_lost is not None:
            raise self._lease_lost  # another server took the Pod while this one scored
        raw_reply = reply.to_bytes()
        if len(raw_reply) > MAX_REPLY_BYTES:
            raise BridgeError(f"the reply to {name.request_id} is over the reply bound")
        self.store.put(name.request_id, raw_reply)  # before the push: never scored twice
        self.highest[name.session] = max(self.highest.get(name.session, 0), name.seq)
        self._push(name, raw_reply)
        self.progress.bump(
            "answered",
            last_answered={
                **_request_row(name, self._wall()),
                "outcome": reply.outcome,
                "scoring_seconds": reply.seconds,
                "replayed": False,
            },
        )
        self.log.event(
            "request_answered",
            request_id=name.request_id,
            count=request.count,
            outcome=reply.outcome,
            seconds=reply.seconds,
        )
        return True

    def _refuse(
        self,
        name: RequestName,
        reason: str,
        detail: str,
        request: BridgeRequest | None = None,
    ) -> bool:
        reply = refusal_reply(name, reason, detail, request=request, answered_unix=self._wall())
        self._push(name, reply.to_bytes())
        self.progress.bump(
            "refused",
            last_answered={
                **_request_row(name, self._wall()),
                "outcome": reply.outcome,
                "reason": reason,
                "replayed": False,
            },
        )
        self.log.event("request_refused", request_id=name.request_id, reason=reason, detail=detail)
        return True

    def _renew_lease(self) -> None:
        """Renew the lease from the progress ticker while a long scoring call, a read or a
        write runs (the polls that normally renew it are paused). A trouble is recorded and
        left to the next poll; a lease another server took ends this server before it pushes
        anything."""
        if self.lease_seconds is None:
            return
        try:
            self.transport.listing()
        except BridgeLeaseHeld as error:
            self._lease_lost = error
            self.progress.update(last_error=_short(error))
        except Exception as error:
            self.progress.update(last_error=_short(error))

    def window_due(self) -> bool:
        """Whether a window pass must run before the next request is scored."""
        if self.window_check is None:
            return False
        started = self._window_started
        return started is None or self._clock() - started >= self.window_check_seconds

    def _window_pass(self, reason: str) -> None:
        """One full pass of the execution-window checks, its receipt written first; a failed
        or errored pass raises ``WindowCheckFailed``. Receipts and the log carry the checks'
        own records (names, exit codes, image IDs), never a value of the environment."""
        assert self.window_check is not None
        self._window_passes += 1
        number = self._window_passes
        started, started_unix = self._clock(), self._wall()
        try:
            record: dict[str, Any] = dict(self.window_check())
            passed = record.get("status") == "passed"
            failed = [] if passed else failed_checks(record)
        except Exception as error:
            record = {"status": "error", "error": _short(error)}
            passed, failed = False, ["error"]
        receipt = self.state / WINDOW_RECEIPTS / f"pass-{number:05d}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if receipt.exists() or receipt.is_symlink():
            raise WindowCheckFailed(f"{receipt} already exists; a receipt is never overwritten")
        summary = {
            "pass": number,
            "reason": reason,
            "status": "passed" if passed else str(record.get("status") or "failed"),
            "failed": failed,
            "started_unix": started_unix,
            "seconds": self._clock() - started,
            "receipt": str(receipt),
        }
        atomic_json(receipt, {**record, "bridge_pass": summary})
        self.progress.update(
            window_check={
                "interval_seconds": self.window_check_seconds,
                "passes": number,
                "last": summary,
            }
        )
        self.log.event("window_check", **summary)
        if not passed:
            raise WindowCheckFailed(
                f"the Mac execution-window check failed ({', '.join(failed)}) at pass {number} "
                f"({reason}); nothing more is scored: see {receipt}"
            )
        self._window_started = started

    def _score(self, request: BridgeRequest) -> BridgeReply:
        started = self._clock()
        self._scoring.set()
        self.progress.update(
            state="scoring",
            current={**_request_row(request.name, self._wall()), "count": request.count},
        )
        try:
            if self.window_due():
                # Before anything of this request runs; a failure ends the server unanswered.
                self._window_pass(f"before scoring {request.request_id}")
            try:
                if self.before_scoring is not None:
                    self.before_scoring(request)
            except Exception as error:
                self.log.event("check_failed", request_id=request.request_id, error=_short(error))
                return failure_reply(
                    request,
                    HOST_FAILURE,
                    "the Mac's pre-scoring check failed; nothing ran (see the Mac's bridge log)",
                    seconds=self._clock() - started,
                    answered_unix=self._wall(),
                )
            try:
                if request.kind == REWARD:
                    attempts = [item for item in request.items if isinstance(item, Attempt)]
                    rewards = list(self.reward_source.rewards(request.index, attempts))
                    check_rewards(attempts, rewards)
                    return scored_reply(
                        request,
                        rewards=rewards,
                        seconds=self._clock() - started,
                        answered_unix=self._wall(),
                    )
                items = [item for item in request.items if isinstance(item, DevItem)]
                passed = list(
                    self.dev_execute([(self.dev_tasks[item.task_id], item.code) for item in items])
                )
                if len(passed) != len(items) or any(type(flag) is not bool for flag in passed):
                    raise ValueError("the dev executor must return one boolean per program")
                return scored_reply(
                    request,
                    passed=passed,
                    seconds=self._clock() - started,
                    answered_unix=self._wall(),
                )
            except (RewardHostFailure, RewardTimeout) as error:
                self.log.event("scoring_failed", request_id=request.request_id, error=_short(error))
                outcome = HOST_FAILURE if isinstance(error, RewardHostFailure) else TIMEOUT
                detail = (
                    "a host failure persisted through the Mac's retries"
                    if outcome == HOST_FAILURE
                    else "the Mac's scoring timeout passed"
                )
                return failure_reply(
                    request,
                    outcome,
                    f"{detail} (see the Mac's bridge log)",
                    seconds=self._clock() - started,
                    answered_unix=self._wall(),
                )
        finally:
            self._scoring.clear()
            self.progress.update(state="polling", current=None)

    def serve(
        self,
        *,
        poll_seconds: float = 2.0,
        should_stop: Callable[[], bool] | None = None,
        max_polls: int | None = None,
        max_consecutive_failures: int = 30,
        progress_seconds: float | None = 10.0,
    ) -> None:
        """Poll until ``should_stop`` returns true or ``max_polls`` polls ran.

        A poll that answered something is followed at once by the next; otherwise the
        server sleeps ``poll_seconds``. A poll whose SSH calls failed through their retries
        counts as a failed cycle; ``max_consecutive_failures`` of them in a row raise the
        last ``BridgeTransportError``. Any other error is recorded in ``progress.json``
        (state ``failed``) and raised; nothing unscored is stored, so a restarted server
        scores that request again. ``progress_seconds`` is the ticker that rewrites
        ``progress.json`` and, while a request is read, scored or answered or a window pass
        runs, renews the lease (``None`` for no ticker). With a ``window_check`` the first
        pass runs before the first poll; a failed pass raises ``WindowCheckFailed``.
        """
        if type(max_consecutive_failures) is not int or max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be a positive integer")
        stop = threading.Event()
        ticker: threading.Thread | None = None
        if progress_seconds is not None:
            interval = float(progress_seconds)

            def tick() -> None:
                while not stop.wait(interval):
                    self.progress.write()
                    if (
                        self._scoring.is_set()
                        or self._transferring.is_set()
                        or self._checking.is_set()
                    ):
                        self._renew_lease()

            ticker = threading.Thread(target=tick, name="rl-bridge-progress", daemon=True)
            ticker.start()
        failures = polls = 0
        try:
            if self.window_check is not None:
                self.progress.update(state="checking")
                self._checking.set()
                try:
                    self._window_pass("start")
                finally:
                    self._checking.clear()
            self.progress.update(state="polling")
            while not (should_stop is not None and should_stop()):
                if max_polls is not None and polls >= max_polls:
                    break
                polls += 1
                try:
                    handled = self.poll_once()
                except BridgeTransportError as error:
                    failures += 1
                    self.progress.update(consecutive_failures=failures, last_error=_short(error))
                    if failures >= max_consecutive_failures:
                        raise
                    handled = []
                else:
                    failures = 0
                if not handled:
                    self._sleep(poll_seconds)
            self.progress.update(state="stopped", current=None)
            self.log.event("bridge_stopped", polls=polls)
        except BaseException as error:
            self.progress.update(state="failed", last_error=_short(error))
            self.log.event("bridge_failed", error=_short(error))
            raise
        finally:
            stop.set()
            if ticker is not None:
                ticker.join(timeout=5)


def highest_window_pass(state: Path) -> int:
    """The highest pass number among the window receipts in ``state`` (0 for none)."""
    directory = state / WINDOW_RECEIPTS
    if not directory.is_dir():
        return 0
    numbers = [
        int(match.group(1))
        for path in directory.iterdir()
        if (match := WINDOW_RECEIPT.fullmatch(path.name))
    ]
    return max(numbers, default=0)


def _phase_tasks(tasks: Iterable[Task], split: str, kind: str) -> dict[str, Task]:
    by_id: dict[str, Task] = {}
    for task in tasks:
        if task.split != split:
            raise ValueError(f"{kind} tasks must be {split}-split tasks; {task.id} is {task.split}")
        if task.id in by_id:
            raise ValueError(f"duplicate {kind} task ID: {task.id}")
        by_id[task.id] = task
    if not by_id:
        raise ValueError(f"the bridge needs at least one {kind} task")
    return by_id
