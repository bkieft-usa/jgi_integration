"""Where the explorer listens, and starting/stopping it in the background from a notebook.

Kept free of heavy imports so ``tools.cli`` can use it before loading the pipeline.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

DEFAULT_PORT = 8050
_STATE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "jgi_integration")


def in_docker() -> bool:
    """True inside a Docker container (its own network namespace, reached through a published port).

    Apptainer/Singularity/Shifter share the host network, so they are deliberately not treated as
    containers here: binding all interfaces on a shared HPC node would expose the app to other users.
    """
    return os.path.exists("/.dockerenv")


def default_host() -> str:
    """Interface to bind: ``JGI_EXPLORER_HOST`` if set, all interfaces inside Docker (so the published
    port works), otherwise localhost only."""
    env = os.environ.get("JGI_EXPLORER_HOST")
    if env:
        return env
    return "0.0.0.0" if in_docker() else "127.0.0.1"


def browser_url(host: str, port: int) -> str:
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1", "::") else host
    return f"http://{shown}:{port}"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _pid_file(port: int) -> str:
    return os.path.join(_STATE_DIR, f"explorer_{port}.pid")


def _log_file(port: int) -> str:
    return os.path.join(_STATE_DIR, f"explorer_{port}.log")


def _alive(pid: int) -> bool:
    try:  # reap it if it is our own exited child (e.g. started from this notebook kernel)
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_explorer(port: int = DEFAULT_PORT) -> bool:
    """Stop an explorer started by :func:`start_explorer` on ``port``. Returns True if one was stopped."""
    path = _pid_file(port)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        pid = int(fh.read().strip() or 0)
    os.remove(path)
    if not pid or not _alive(pid):
        return False
    os.killpg(pid, signal.SIGTERM)  # started in its own session: pid == process-group id
    for _ in range(50):
        if not _alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    print(f"Stopped the explorer on port {port}.")
    return True


def start_explorer(analysis_dir: str, port: int = DEFAULT_PORT, host: Optional[str] = None,
                   timeout: float = 600.0) -> str:
    """Start the explorer for one run as a background process and return its URL.

    The server runs in its own session, so it keeps running after the notebook kernel restarts or
    stops; call :func:`stop_explorer` (or start again, which replaces it) to stop it. Output goes to
    ``~/.cache/jgi_integration/explorer_<port>.log``. If the bundle is missing or stale it is built
    first, which can take a minute or two.
    """
    host = host or default_host()
    stop_explorer(port)
    if _port_open(port):
        raise RuntimeError(f"Port {port} is already in use by another program. Stop it, or pass port=<other> "
                           f"(the port must also be published by the container).")
    os.makedirs(_STATE_DIR, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (repo, os.environ.get("PYTHONPATH")) if p),
               MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "tools.cli", "explore", "--analysis-dir", os.path.abspath(analysis_dir),
           "--host", host, "--port", str(port)]
    log_path = _log_file(port)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=repo,
                                start_new_session=True)
    with open(_pid_file(port), "w") as fh:
        fh.write(str(proc.pid))

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            os.remove(_pid_file(port))
            with open(log_path) as fh:
                tail = "".join(fh.readlines()[-20:])
            raise RuntimeError(f"The explorer exited during startup. Last log lines:\n{tail}")
        if _port_open(port):
            url = browser_url(host, port)
            print(f"Explorer running at {url}\n"
                  f"It keeps running in the background; stop it with stop_explorer(port={port}).\n"
                  f"Log: {log_path}")
            return url
        time.sleep(1)
    raise RuntimeError(f"The explorer did not start within {timeout:.0f} s; see {log_path}.")
