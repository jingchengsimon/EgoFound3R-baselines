#!/usr/bin/env python3
"""Launch and control registered process groups by exact handle paths."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def proc_identity(pid: int) -> dict[str, object] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = text.rsplit(")", 1)[1].strip().split()
        return {
            "pid": pid,
            "state": tail[0],
            "pgid": int(tail[2]),
            "start_ticks": int(tail[19]),
        }
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def checked_handle(path: Path) -> tuple[dict[str, object], dict[str, object] | None]:
    registered = json.loads(path.read_text(encoding="utf-8"))
    current = proc_identity(int(registered["pid"]))
    if current is None:
        return registered, None
    if current["pgid"] != registered["pgid"] or current["start_ticks"] != registered["start_ticks"]:
        raise RuntimeError(f"STALE_HANDLE:{path}")
    return registered, current


def launch(args: argparse.Namespace) -> dict[str, object]:
    path = Path(args.handle)
    if path.exists():
        _, current = checked_handle(path)
        if current is not None:
            raise RuntimeError(f"HANDLE_ALREADY_LIVE:{path}")
        raise RuntimeError(f"HANDLE_ALREADY_EXISTS:{path}")
    command = base64.b64decode(args.shell_b64).decode("utf-8")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    identity = proc_identity(process.pid)
    if identity is None:
        raise RuntimeError("LAUNCH_EXITED_BEFORE_REGISTRATION")
    payload = {**identity, "handle": str(path), "log": str(log_path), "command_sha256": args.command_sha256}
    atomic_json(path, payload)
    return payload


def statuses(paths: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in paths:
        path = Path(value)
        try:
            registered, current = checked_handle(path)
            result[value] = {
                "status": "exited" if current is None else "paused" if current["state"] == "T" else "running",
                **({"pid": registered["pid"], "pgid": registered["pgid"]} if current is not None else {}),
            }
        except FileNotFoundError:
            result[value] = {"status": "missing_handle"}
        except RuntimeError as error:
            result[value] = {"status": "stale_handle", "error": str(error)}
    return {"handles": result}


def send_signal(paths: list[str], signal_name: str, drain_relay: bool = False) -> dict[str, object]:
    number = {"STOP": signal.SIGSTOP, "CONT": signal.SIGCONT}[signal_name]
    result: dict[str, object] = {}
    for value in paths:
        path = Path(value)
        registered, current = checked_handle(path)
        if current is None:
            result[value] = {"status": "exited"}
            continue
        if drain_relay:
            if signal_name != "STOP":
                raise ValueError("drain relay only supports STOP")
            # Stop only the registered relay coordinator; its current inference finishes.
            frontier = [int(registered["pid"])]
            producers = []
            for _ in range(8):
                children = []
                for pid in frontier:
                    proc = Path(f"/proc/{pid}")
                    try:
                        argv = proc.joinpath("cmdline").read_bytes().decode().split("\0")
                        identity = proc_identity(pid)
                        if identity is None or identity['pgid'] != registered['pgid']:
                            continue
                        if any(arg.endswith('/run_egofound3r_relay.py') for arg in argv) and '--role' in argv and argv[argv.index('--role') + 1] == 'producer':
                            producers.append(pid)
                        children.extend(int(v) for v in proc.joinpath(f'task/{pid}/children').read_text().split())
                    except FileNotFoundError:
                        continue
                frontier = children
                if not frontier:
                    break
            if len(producers) != 1:
                raise RuntimeError(f'EXPECTED_ONE_REGISTERED_RELAY_PRODUCER:{producers}')
            os.kill(producers[0], number)
            if producers[0] != registered['pid']:
                os.kill(int(registered['pid']), number)
        else:
            os.killpg(int(registered["pgid"]), number)
        after = proc_identity(int(registered["pid"]))
        result[value] = {
            "status": "signal_sent",
            "signal": signal_name,
            "drain_relay": drain_relay,
            "observed_state": None if after is None else after["state"],
        }
    return {"handles": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--handle", required=True)
    launch_parser.add_argument("--log", required=True)
    launch_parser.add_argument("--shell-b64", required=True)
    launch_parser.add_argument("--command-sha256", required=True)
    for action in ("status", "signal"):
        child = subparsers.add_parser(action)
        child.add_argument("--handle", action="append", required=True)
        if action == "signal":
            child.add_argument("--signal", choices=("STOP", "CONT"), required=True)
            child.add_argument("--drain-relay", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "launch":
            result = launch(args)
        elif args.action == "status":
            result = statuses(args.handle)
        else:
            result = send_signal(args.handle, args.signal, args.drain_relay)
        print(json.dumps({"ok": True, **result}, sort_keys=True))
    except Exception as error:  # one bounded machine-readable failure
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
