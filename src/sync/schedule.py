"""Render platform scheduler units so ``mailrag sync`` runs unattended.

``mailrag sync`` is the portable unit — one-shot, idempotent, resumable. A daemon
would be needless complexity for a 1–2 day freshness target. What is needed is
something that starts it periodically and survives a laptop's life:

* **launchd** on macOS. Not cron: cron silently skips a tick that falls while the
  machine is asleep, whereas launchd runs the job on wake. For a laptop that
  spends most nights closed, that is the difference between a fresh index and a
  stale one.
* **systemd user timer** on Linux, with ``Persistent=true`` for the same reason.

The single most common way this feature fails is a scheduler that runs mailrag
outside its conda environment and dies on the first import, silently, for weeks.
So the rendered units use absolute paths, invoke the environment explicitly, and
always write stdout/stderr to a log file — with ``sync --status`` warning about
staleness as the backstop.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional

LAUNCHD_LABEL = "eu.fmasi.mailrag.sync"
SYSTEMD_UNIT = "mailrag-sync"


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def scheduler_environment(overrides: Optional[dict] = None) -> dict:
    """Environment a scheduled run needs that it will NOT inherit.

    A launchd/systemd job starts from an almost-empty environment: no shell
    profile, no login session. Two values matter here and neither can be left to
    chance, because the project's ``.env`` is written for the DEVCONTAINER —
    ``host.docker.internal`` resolves inside a container and nowhere else. A
    scheduled run on the host would fail every tick, forever, reporting the LLM
    and vector store as unavailable while the real cause is a hostname.
    """
    env = {
        "RAG_LLM_API_BASE": "http://localhost:1234/v1",
        "QDRANT_URL": "http://localhost:6333",
        # PATH matters as much as the endpoints: attachment OCR shells out to
        # `tesseract`, which lives in a Homebrew prefix that a scheduled job does
        # NOT have on its path. Without this, scanned attachments extract as
        # `ocr_unavailable` — and that status is cached, so a later supervised run
        # does not heal them (GH #37's failure shape, re-triggered by launchd).
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }
    env.update(overrides or {})
    return env


def sync_command(
    *,
    repo_root: Optional[str] = None,
    conda_env: Optional[str] = None,
    account: Optional[str] = None,
    model: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the fully-qualified argv a scheduler should run.

    Absolute everything: a scheduler inherits almost none of an interactive
    shell's environment, so a relative path or a bare ``python`` is the classic
    way to get a job that has been failing silently since it was installed.
    """
    repo_root = os.path.abspath(repo_root or os.getcwd())
    argv: List[str] = []
    if conda_env:
        conda = shutil.which("conda") or "/opt/homebrew/bin/conda"
        argv += [conda, "run", "--no-capture-output", "-n", conda_env]
        argv += ["python", "-m", "src.cli", "sync"]
    else:
        argv += [os.path.abspath(sys.executable), "-m", "src.cli", "sync"]
    if account:
        argv += ["--account", account]
    if model:
        argv += ["--model", model]
    argv += list(extra_args or [])
    return argv


def render_launchd_plist(
    *,
    interval_seconds: int,
    repo_root: str,
    log_path: str,
    conda_env: Optional[str] = None,
    account: Optional[str] = None,
    model: Optional[str] = None,
    label: str = LAUNCHD_LABEL,
    environment: Optional[dict] = None,
) -> str:
    """Render a LaunchAgent plist for ``~/Library/LaunchAgents/<label>.plist``.

    Two settings carry the whole "survives a laptop" claim, and they cover
    different gaps:

    * ``StartInterval`` (not ``StartCalendarInterval``) so a window missed while
      asleep is coalesced into a single run on wake instead of being lost.
    * ``RunAtLoad`` for the case coalescing does NOT cover — a restart. A
      LaunchAgent loads at login, so without this the first sync after every boot
      is one whole interval away. Measured on 2026-08-17: a reboot at 11:09 left
      the agent with zero runs at 11:35, and the next tick would not have landed
      until 15:09. The systemd sibling below never had this gap, because
      ``OnBootSec=5min`` catches up after boot — macOS was the only platform
      losing a window. A run at load costs one idle tick, which is nothing:
      ``sync`` is one-shot and idempotent over a resumable queue.

    Sleeping with the lid closed and unplugged is deliberately NOT addressed:
    macOS parks timers in standby, and waking a laptop on battery to do GPU work
    is not worth a few hours of freshness.
    """
    args = sync_command(repo_root=repo_root, conda_env=conda_env, account=account, model=model)
    program_args = "\n".join(f"        <string>{_xml_escape(a)}</string>" for a in args)
    env = scheduler_environment(environment)
    env_block = "\n".join(
        f"        <key>{_xml_escape(k)}</key>\n        <string>{_xml_escape(str(v))}</string>"
        for k, v in sorted(env.items())
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_xml_escape(label)}</string>
    <key>ProgramArguments</key>
    <array>
{program_args}
    </array>
    <key>WorkingDirectory</key>
    <string>{_xml_escape(os.path.abspath(repo_root))}</string>
    <key>EnvironmentVariables</key>
    <dict>
{env_block}
    </dict>
    <key>StartInterval</key>
    <integer>{int(interval_seconds)}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{_xml_escape(log_path)}</string>
    <key>StandardErrorPath</key>
    <string>{_xml_escape(log_path)}</string>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
"""


def render_systemd_units(
    *,
    interval_seconds: int,
    repo_root: str,
    log_path: str,
    conda_env: Optional[str] = None,
    account: Optional[str] = None,
    model: Optional[str] = None,
    unit: str = SYSTEMD_UNIT,
    environment: Optional[dict] = None,
) -> tuple[str, str]:
    """Render ``(service, timer)`` for ``~/.config/systemd/user/``.

    ``Persistent=true`` is the systemd equivalent of launchd's coalescing: a timer
    whose window passed while the machine was off fires once on boot.
    """
    args = sync_command(repo_root=repo_root, conda_env=conda_env, account=account, model=model)
    exec_start = " ".join(args)
    env_lines = "\n".join(
        f"Environment={k}={v}" for k, v in sorted(scheduler_environment(environment).items())
    )
    service = f"""[Unit]
Description=mailrag incremental mail sync
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory={os.path.abspath(repo_root)}
{env_lines}
ExecStart={exec_start}
StandardOutput=append:{log_path}
StandardError=append:{log_path}
"""
    timer = f"""[Unit]
Description=Run {unit} every {int(interval_seconds)}s

[Timer]
OnBootSec=5min
OnUnitActiveSec={int(interval_seconds)}s
Persistent=true

[Install]
WantedBy=timers.target
"""
    return service, timer


def install_hint(platform: str, label: str = LAUNCHD_LABEL, unit: str = SYSTEMD_UNIT) -> str:
    """The commands to actually activate a written unit, per platform."""
    if platform == "darwin":
        plist = f"~/Library/LaunchAgents/{label}.plist"
        return (
            f"launchctl unload {plist} 2>/dev/null; launchctl load {plist}\n"
            f"# verify:  launchctl list | grep {label}"
        )
    return (
        "systemctl --user daemon-reload\n"
        f"systemctl --user enable --now {unit}.timer\n"
        f"# verify:  systemctl --user list-timers | grep {unit}"
    )
