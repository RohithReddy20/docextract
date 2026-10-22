"""Run a probe on a Colab GPU from this machine, with Google's Colab CLI.

The same requests, engine and runner as the notebook (runtimes/colab/probe.ipynb), without a browser:
the CLI rents a VM, this source tree is uploaded (uncommitted edits included), setup and the runner
work in the background on the VM, answers are copied back every few minutes, and the VM is released
when the run ends, fails or is interrupted. Answers land in artifacts/probes/<name>/ for
tools/score_probe.py; `--resume` sends back the answers already copied, so only the rest is asked.

The CLI (`uv tool install google-colab-cli`) keeps its Google sign-in under $HOME/.config/colab-cli;
set COLAB_HOME to keep this project's account apart (`HOME=$COLAB_HOME colab usage` signs in once).

    uv run python tools/colab_probe.py manifests/probes/<name>.json --gpu A100 --confirm-units <estimate>
    uv run python tools/colab_probe.py manifests/probes/<name>.json --dry-run   # CPU VM: setup only
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtimes" / "colab"))

from probe import FP8_GPUS, NO_FP8, collect, units_per_hour  # noqa: E402  (the runner on the VM)

# Installing vLLM and downloading and loading the model, before the probe's own time limit starts.
SETUP_MINUTES = 15
POLL_SECONDS = 60
COPY_EVERY = 5  # polls between copies of the answers
REMOTE = "/content/docextract"
MARK = "@@result "


def colab_env() -> dict:
    """The environment for the CLI: signed in with the account under COLAB_HOME when it is set."""
    env = dict(os.environ)
    if os.environ.get("COLAB_HOME"):
        env["HOME"] = os.environ["COLAB_HOME"]
    # Opening the kernel connection through Colab's proxy took up to 40 seconds; the kernel client
    # waits 10 by default, then reports a lost connection and hangs.
    env.setdefault("REQUEST_TIMEOUT", "120")
    return env


def colab(
    *args: str, stdin: str | None = None, capture: bool = False, check: bool = True, timeout: float = 300
) -> str:
    """One Colab CLI command.

    A command still running after `timeout` seconds is killed: when the connection to the VM drops,
    the CLI can print the error and then hang instead of exiting.
    """
    try:
        result = subprocess.run(
            ["colab", *args],
            input=stdin,
            text=True,
            env=colab_env(),
            check=check,
            stdout=subprocess.PIPE if capture else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        if check:
            raise subprocess.CalledProcessError(-9, error.cmd) from error
        print(f"no reply from `colab {args[0]}` in {timeout:.0f} s", flush=True)
        return ""
    return result.stdout or ""


def remote(session: str, code: str) -> dict:
    """Run Python in the VM's kernel; the code prints its result as one line starting with MARK."""
    out = colab("exec", "-s", session, stdin=code, capture=True, timeout=120)
    line = next((line for line in reversed(out.splitlines()) if line.startswith(MARK)), None)
    if line is None:
        raise RuntimeError(f"no result from the VM:\n{out[-2000:]}")
    return json.loads(line.removeprefix(MARK))


def usage_rate() -> float | None:
    """Compute units per hour of one VM now: the account's rate (all its VMs, as `colab usage`
    reports it) divided by its active VMs, which runs of one configuration rent alike."""
    reply = colab("usage", capture=True, check=False)
    rate = re.search(r"Usage rate: ([\d.]+)", reply)
    vms = re.search(r"Active assignments: (\d+)", reply)
    return float(rate.group(1)) / max(1, int(vms.group(1)) if vms else 1) if rate else None


def source_archive() -> bytes:
    """Every tracked or new, not ignored file of this tree as it is now (uncommitted edits included)."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in filter(None, listed.split("\0")):
            if (ROOT / name).is_file():  # deleted files may still be listed
                archive.add(ROOT / name, arcname=name)
    return buffer.getvalue()


def revision() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return head.strip() + ("+uncommitted" if dirty.strip() else "")


def unpack_code(out: str) -> str:
    """Unpack the uploaded source on the VM and make the answer folders; an earlier job's log on the
    same VM is kept under another name, so the new job's progress is read from a fresh one."""
    return f"""
import json, os, tarfile, time
if os.path.exists({out!r} + "/run.log"):
    os.rename({out!r} + "/run.log", {out!r} + f"/run-{{int(time.time())}}.log")
os.makedirs({REMOTE!r}, exist_ok=True)
with tarfile.open("/content/source.tar.gz") as archive:
    archive.extractall({REMOTE!r}, filter="data")
os.makedirs({out!r} + "/run", exist_ok=True)
print({MARK!r} + json.dumps({{"files": len(os.listdir({REMOTE!r}))}}))
"""


def unpack_store_code(path: str) -> str:
    """Unpack an answer store uploaded compressed (`path` + ".gz") in place."""
    return f"""
import gzip, json, os, shutil
with gzip.open({path!r} + ".gz", "rb") as source, open({path!r}, "wb") as target:
    shutil.copyfileobj(source, target)
os.remove({path!r} + ".gz")
print({MARK!r} + json.dumps({{"bytes": os.path.getsize({path!r})}}))
"""


def job_code(command: str, out: str) -> str:
    """Setup plus the runner, logging to run.log, in a kernel cell that waits for them to end.

    Colab keeps a VM only while its kernel is active: a job started in the background with the
    kernel left idle lost its VM after about half an hour. The job runs in its own process group,
    so it would outlive the kernel.
    """
    return f"""
import subprocess
with open({out!r} + "/run.log", "a") as log:
    subprocess.Popen(
        ["bash", "-c", {command!r}], cwd={REMOTE!r}, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    ).wait()
"""


def fetch(session: str, out: str, local: Path, names: list[str]) -> list[str]:
    """Copy files of the VM's answer folder here; returns the names copied."""
    copied = []
    for name in names:
        target = local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        colab("download", "-s", session, f"{out}/{name}", str(partial), capture=True, check=False)
        if partial.exists():
            partial.replace(target)
            copied.append(name)
    return copied


def vm_status(session: str, out: str, local: Path) -> dict | None:
    """Progress read from the VM's log, copied with the CLI's file transfer.

    `colab exec` goes through the Python kernel, whose connection dropped again and again while the
    GPU was busy; file transfer does not. None when the log could not be copied.
    """
    if not fetch(session, out, local, ["run.log"]):
        return None
    log = (local / "run.log").read_text(errors="replace").replace("\r", "\n")
    lines = [line[:220] for line in log.splitlines() if line.strip()]
    ended = next((line for line in reversed(lines) if line.startswith(("== exit ", "== done"))), None)
    code = None if ended is None else 0 if ended == "== done" else int(ended.split()[2])
    return {"exit": code, "tail": lines[-3:]}


def interrupt(*_):
    raise KeyboardInterrupt


def launch(
    args, session: str, command: str, out: str, local: Path, hours: float, store: str
) -> subprocess.Popen:
    """Check the live rate, upload the source (and earlier answers) and start the job."""
    rate = usage_rate()
    print(f"VM ready; it uses {rate:.2f} compute units per hour" if rate else "VM ready")
    if not args.dry_run and rate and rate * hours > max(args.confirm_units, 0.1) * 1.5:
        raise SystemExit("The live rate is well above the estimate: stopped before any work")
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "source.tar.gz"
        archive.write_bytes(source_archive())
        colab("upload", "-s", session, str(archive), "/content/source.tar.gz", capture=True)
    retry(lambda: remote(session, unpack_code(out)))
    if args.resume and (local / store).exists():
        # Sent compressed and unpacked there: a 39 MB store (4 MB compressed) was still uploading
        # after five minutes.
        with tempfile.TemporaryDirectory() as scratch:
            packed = Path(scratch) / "store.gz"
            with (local / store).open("rb") as source, gzip.open(packed, "wb") as target:
                shutil.copyfileobj(source, target)
            colab("upload", "-s", session, str(packed), f"{out}/{store}.gz", capture=True, timeout=1200)
        retry(lambda: remote(session, unpack_store_code(f"{out}/{store}")))
    (local / "colab-run.json").write_text(
        json.dumps({"revision": revision(), "gpu": None if args.dry_run else args.gpu, "rate": rate})
    )
    for _ in range(3):
        # The CLI call that holds the cell open may lose its connection and hang; the cell runs on.
        holder = subprocess.Popen(
            ["colab", "exec", "-s", session, "--timeout", str(int(hours * 3600) + 1800)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=colab_env(),
        )
        holder.stdin.write(job_code(command, out))
        holder.stdin.close()
        # The cell creates run.log first thing; without it the code never reached the kernel.
        for _ in range(12):
            time.sleep(20)
            if fetch(session, out, local, ["run.log"]):
                print(f"started; log: {out}/run.log", flush=True)
                return holder
            if holder.poll() is not None:
                break
        holder.kill()
        print("the job did not start: trying again", flush=True)
    raise RuntimeError("the job did not start")


def retry(step, attempts: int = 3):
    """A step that may meet a dropped connection, tried again (it must be safe to repeat)."""
    for attempt in range(attempts):
        try:
            return step()
        except (RuntimeError, subprocess.CalledProcessError) as error:
            if attempt == attempts - 1:
                raise
            print(f"retrying after: {str(error)[:200]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", type=Path, help="a probe configuration (manifests/probes/)")
    parser.add_argument("--gpu", default="L4", help="T4, L4, G4, A100 or H100")
    parser.add_argument("--confirm-units", type=float, default=0.0, help="at least the stated estimate")
    parser.add_argument("--minutes", type=float, help="the probe's time limit (default: its own)")
    parser.add_argument("--engine", default="{}", help="engine options to override, as JSON")
    parser.add_argument("--resume", action="store_true", help="continue from the answers copied here")
    parser.add_argument("--dry-run", action="store_true", help="CPU VM: setup and requests, no model")
    parser.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    parser.add_argument("--attach", action="store_true", help="watch a run already started on its VM")
    parser.add_argument("--shard", default="0/1", help="i/n: a run configuration's share for this VM")
    parser.add_argument("--score", action="store_true", help="score a product run on its VM at the end")
    parser.add_argument("--score-only", action="store_true", help="only score the results on the VM")
    parser.add_argument(
        "--replay", action="store_true", help="re-assemble a product run's stored answers (no model)"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="start on this run's VM from an earlier job (packages, model)"
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    index, count = (int(part) for part in args.shard.split("/"))
    # Each share of a run has its own VM, answer folder and session.
    name = config["name"] if count == 1 else f"{config['name']}-{index}"
    # A probe configuration (manifests/probes/) compares prompt methods; a run configuration (with
    # "variants") runs the product itself, whose answer store lets a rerun continue.
    product = "variants" in config
    store = "run/generations.jsonl" if product else "answers.jsonl"
    minutes = args.minutes or config.get("limit_minutes") or config["gpu_hours"] * 60
    local = ROOT / "artifacts" / "probes" / name
    local.mkdir(parents=True, exist_ok=True)
    engine = {**config.get("engine", config), **json.loads(args.engine)}
    if not args.dry_run and engine.get("quantization") == "fp8" and args.gpu not in FP8_GPUS:
        raise SystemExit(NO_FP8.format(args.gpu))
    # A reused VM keeps its own (newer) copy of the answers, which nothing here overwrites.
    if args.reuse and args.resume:
        raise SystemExit("--reuse continues from the answers on the VM: leave out --resume")
    if (local / store).exists() and not (args.resume or args.attach or args.reuse):
        raise SystemExit(f"{local / store} exists: pass --resume to continue it, or move it away")
    if args.dry_run:
        # A CPU VM costs a small fraction of a GPU; the setup takes about ten minutes.
        hours, gpu_flag = 0.25, []
        print("dry run on a CPU VM: setup and requests only, about 0.1-0.3 compute units")
    else:
        hours, gpu_flag = (minutes + SETUP_MINUTES) / 60, ["--gpu", args.gpu]
        estimate = math.ceil(hours * units_per_hour(args.gpu) * 10) / 10  # as printed
        print(
            f"{args.gpu}: at most {minutes:.0f} minutes of probe plus about {SETUP_MINUTES} of setup, "
            f"about {estimate:.1f} compute units at the expected rate"
        )
        if args.confirm_units < estimate:
            raise SystemExit("Not launched: pass --confirm-units at least equal to the stated estimate")

    out = f"/content/out/{name}"
    runner = [
        "python",
        "-u",
        f"runtimes/colab/{'extract' if product else 'probe'}.py",
        "--config",
        str(args.config.resolve().relative_to(ROOT)),
        "--out",
        out,
        "--minutes",
        str(minutes),
        "--engine",
        args.engine,
        *(["--dry-run"] if args.dry_run else []),
        *(["--shard", args.shard] if count > 1 else []),
        *(["--score"] if args.score else []),
        *(["--score-only"] if args.score_only else []),
        *(["--replay"] if args.replay else []),
    ]
    # Setup and runner in a subshell that stops at the first failure; its exit code ends the log.
    command = (
        "( set -e; echo '== setup'; date; pip install -q uv; "
        "uv pip install --system -q -r runtimes/colab/requirements.txt; "
        f"echo '== runner'; date; {shlex.join(runner)} ); echo \"== exit $?\"; date"
    )
    session = f"docextract-{name}"
    started = time.time()
    # The driver's own SIGTERM releases the VM like Ctrl-C does (SIGKILL leaves it running).
    signal.signal(signal.SIGTERM, interrupt)
    if not (args.attach or args.reuse):
        colab("new", "-s", session, *gpu_flag, timeout=900)
    holder = None
    try:
        if not args.attach:
            holder = launch(args, session, command, out, local, hours, store)
        polls = failures = answers = 0
        limit = started + (minutes + (0 if args.attach else SETUP_MINUTES) + 10) * 60
        while True:
            time.sleep(POLL_SECONDS)
            polls += 1
            status = vm_status(session, out, local)
            if status is None:
                failures += 1
                print(f"could not copy the log ({failures} in a row)", flush=True)
                if failures >= 10:
                    raise RuntimeError("the VM stopped answering")
                continue
            failures = 0
            if polls % COPY_EVERY == 0 and fetch(session, out, local, [store]):
                answers = sum(1 for _ in (local / store).open())
            last = status["tail"][-1] if status["tail"] else ""
            elapsed = (time.time() - started) / 60
            print(f"{elapsed:5.1f} min  {answers:4d} answers  | {last}", flush=True)
            if status["exit"] is not None:
                print(f"runner exited with code {status['exit']}")
                break
            if time.time() > limit:
                print("past the time limit with setup: stopping")
                break
    finally:
        copied = fetch(
            session, out, local, ["answers.json", store, "run.log", *(["score.json"] if product else [])]
        )
        if holder and holder.poll() is None:
            holder.kill()
        if "answers.json" not in copied and not product and (local / store).exists():
            # The runner did not finish its own summary (stopped or failed): build it from the copy.
            (local / "answers.json").write_text(json.dumps({"outputs": collect(local / "answers.jsonl")}))
        if args.keep:
            print(f"VM left running: stop it with `colab stop -s {session}`")
        else:
            for _ in range(2):  # the VM bills until it is released
                reply = colab("stop", "-s", session, capture=True, check=False, timeout=180)
                if "terminated" in reply or "not found" in reply:
                    break
            else:
                print(f"VM release not confirmed: check `colab sessions`, `colab stop -s {session}`")
        hours_used = (time.time() - started) / 3600
        print(f"{hours_used * 60:.0f} minutes on the VM; files in {local}: {', '.join(copied) or 'none'}")


if __name__ == "__main__":
    main()
