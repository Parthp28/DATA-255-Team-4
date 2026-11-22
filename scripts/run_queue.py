"""Run one GPU lane of the project queue until no runnable job remains.

Reads ops/run_queue.csv, picks READY_FOR_GPU jobs for one lane whose dependencies
are complete, runs each launch command unchanged, and records the outcome in one
status file per job so two lanes never write the same file.
"""
import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "queue_id", "experiment_id", "seed", "technical_owner", "gpu_executor", "backup_executor", "lane",
    "config_path", "config_sha256", "git_commit", "data_version", "render_version", "launch_command",
    "dependency", "priority", "estimated_minutes", "status", "start_time", "end_time", "output_path",
    "validation_status",
]
DONE = {"COMPLETE", "RESULT_VALIDATED"}
DEAD = {"FAILED", "SUPERSEDED"}
OWNER_FINAL = {"RESULT_VALIDATED", "SUPERSEDED"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_queue(path):
    """Load the queue CSV and check that every schema column is present."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"run_queue.csv is missing columns: {', '.join(missing)}")
        return [dict(r) for r in reader]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_status(status_dir, qid):
    p = Path(status_dir) / f"{qid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_status(status_dir, qid, data):
    Path(status_dir).mkdir(parents=True, exist_ok=True)
    final = Path(status_dir) / f"{qid}.json"
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    # Why: an atomic rename means a sync or a reader never sees a half-written status file.
    os.replace(tmp, final)


def effective_status(row, status):
    # Why: owner signatures (validated, superseded) live in the CSV and outrank the runner;
    # otherwise the runner's status file is the newest fact about the job.
    if row["status"] in OWNER_FINAL:
        return row["status"]
    if status:
        return status["status"]
    return row["status"]


def deps_of(row):
    return [d.strip() for d in row["dependency"].split(";") if d.strip()]


def is_stale(status, stale_after):
    if not status or status.get("status") != "RUNNING":
        return False
    try:
        beat = datetime.fromisoformat(status["heartbeat"])
    except (KeyError, ValueError):
        return True
    return (datetime.now(timezone.utc) - beat).total_seconds() > stale_after


def classify(rows, lane, status_dir, stale_after):
    """Split this lane's jobs into runnable, waiting on dependencies, and blocked."""
    by_id = {r["queue_id"]: r for r in rows}
    statuses = {r["queue_id"]: read_status(status_dir, r["queue_id"]) for r in rows}
    eff = {qid: effective_status(by_id[qid], statuses[qid]) for qid in by_id}
    runnable, waiting, blocked = [], [], []
    for r in rows:
        if r["lane"] != lane:
            continue
        st = eff[r["queue_id"]]
        if st == "RUNNING" and is_stale(statuses[r["queue_id"]], stale_after):
            st = "READY_FOR_GPU"
        if st != "READY_FOR_GPU":
            continue
        dep_states = [eff.get(d, "UNKNOWN") for d in deps_of(r)]
        if any(s in DEAD or s == "UNKNOWN" for s in dep_states):
            blocked.append(r)
        elif all(s in DONE for s in dep_states):
            runnable.append(r)
        else:
            waiting.append(r)
    runnable.sort(key=lambda r: (int(r["priority"] or 9), r["queue_id"]))
    return runnable, waiting, blocked


def git_head(workdir):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workdir, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def integrity_problem(row, workdir):
    """Return a reason string if the job differs from what its owner signed, else an empty string."""
    if row["config_sha256"]:
        cfg = Path(workdir) / row["config_path"]
        if not cfg.exists():
            return f"config not found: {row['config_path']}"
        if sha256_of(cfg) != row["config_sha256"]:
            return "config changed after READY_FOR_GPU (sha256 mismatch)"
    if row["git_commit"]:
        head = git_head(workdir)
        if head and not head.startswith(row["git_commit"]):
            return f"code commit {head[:10]} differs from signed commit {row['git_commit']}"
    return ""


def run_hook(cmd, workdir, log):
    if not cmd:
        return
    rc = subprocess.run(cmd, shell=True, cwd=workdir, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"[{now_iso()}] sync hook exited {rc}: {cmd}", file=log, flush=True)


def run_job(row, args, log_dir):
    qid = row["queue_id"]
    base = {"queue_id": qid, "lane": row["lane"], "host": socket.gethostname(), "launch_command": row["launch_command"]}
    problem = integrity_problem(row, args.workdir)
    if problem:
        write_status(args.status_dir, qid, {**base, "status": "FAILED", "reason": problem, "end_time": now_iso(), "attempts": 0})
        print(f"[{now_iso()}] {qid} refused: {problem}", flush=True)
        return False
    log_path = Path(log_dir) / f"{qid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = now_iso()
    for attempt in range(1, args.max_attempts + 1):
        with open(log_path, "a") as log:
            print(f"[{now_iso()}] attempt {attempt}: {row['launch_command']}", file=log, flush=True)
            write_status(args.status_dir, qid, {**base, "status": "RUNNING", "start_time": start, "heartbeat": now_iso(), "attempts": attempt})
            proc = subprocess.Popen(row["launch_command"], shell=True, cwd=args.workdir, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(args.heartbeat)
                write_status(args.status_dir, qid, {**base, "status": "RUNNING", "start_time": start, "heartbeat": now_iso(), "attempts": attempt})
            rc = proc.returncode
        if rc == 0:
            write_status(args.status_dir, qid, {**base, "status": "COMPLETE", "start_time": start, "end_time": now_iso(),
                                                "attempts": attempt, "exit_code": 0, "log": str(log_path)})
            print(f"[{now_iso()}] {qid} COMPLETE", flush=True)
            return True
        # Why: one retry covers a dropped session or a transient error; the trainer resumes from last.pt.
        print(f"[{now_iso()}] {qid} exited {rc} on attempt {attempt}", flush=True)
    write_status(args.status_dir, qid, {**base, "status": "FAILED", "start_time": start, "end_time": now_iso(),
                                        "attempts": args.max_attempts, "exit_code": rc, "log": str(log_path)})
    print(f"[{now_iso()}] {qid} FAILED, continuing with independent jobs", flush=True)
    return False


def merge(args):
    """Write runner outcomes from the status files back into the CSV for the team's view."""
    rows = load_queue(args.queue)
    for r in rows:
        st = read_status(args.status_dir, r["queue_id"])
        if not st:
            continue
        if r["status"] not in OWNER_FINAL:
            r["status"] = st["status"]
        r["start_time"] = st.get("start_time", r["start_time"])
        r["end_time"] = st.get("end_time", r["end_time"])
    tmp = Path(args.queue).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.queue)
    return rows


def report(args):
    rows = load_queue(args.queue)
    print(f"{'queue_id':10} {'lane':4} {'status':16} {'owner':10} experiment")
    for r in rows:
        st = effective_status(r, read_status(args.status_dir, r["queue_id"]))
        print(f"{r['queue_id']:10} {r['lane']:4} {st:16} {r['technical_owner']:10} {r['experiment_id']} s{r['seed']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", choices=["P", "H"], help="GPU lane to execute")
    ap.add_argument("--queue", default="ops/run_queue.csv")
    ap.add_argument("--status-dir", default="ops/status")
    ap.add_argument("--log-dir", default="ops/logs")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks while waiting")
    ap.add_argument("--heartbeat", type=float, default=60.0)
    ap.add_argument("--stale-after", type=float, default=900.0, help="seconds before a RUNNING job counts as dead")
    ap.add_argument("--max-idle", type=float, default=7200.0, help="seconds to wait on cross-lane dependencies")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--sync-pull", default="", help="command run before each pick, e.g. an rclone copy")
    ap.add_argument("--sync-push", default="", help="command run after each job")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        report(args); return 0
    if args.merge:
        merge(args); return 0
    if not args.lane:
        ap.error("--lane is required unless --report or --merge is given")
    sys_log = sys.stdout
    if args.dry_run:
        runnable, waiting, blocked = classify(load_queue(args.queue), args.lane, args.status_dir, args.stale_after)
        print("runnable now:", [r["queue_id"] for r in runnable])
        print("waiting on dependencies:", [r["queue_id"] for r in waiting])
        print("blocked by failed or unknown dependencies:", [r["queue_id"] for r in blocked])
        return 0
    idle_since = None
    while True:
        run_hook(args.sync_pull, args.workdir, sys_log)
        runnable, waiting, blocked = classify(load_queue(args.queue), args.lane, args.status_dir, args.stale_after)
        if runnable:
            run_job(runnable[0], args, args.log_dir)
            run_hook(args.sync_push, args.workdir, sys_log)
            idle_since = None
            continue
        if waiting:
            idle_since = idle_since or time.time()
            if time.time() - idle_since < args.max_idle:
                time.sleep(args.poll)
                continue
            print(f"[{now_iso()}] gave up waiting on: {[r['queue_id'] for r in waiting]}", flush=True)
        if blocked:
            print(f"[{now_iso()}] blocked by failed dependencies: {[r['queue_id'] for r in blocked]}", flush=True)
        print(f"[{now_iso()}] lane {args.lane}: nothing runnable remains", flush=True)
        run_hook(args.sync_push, args.workdir, sys_log)
        return 0


if __name__ == "__main__":
    sys.exit(main())
