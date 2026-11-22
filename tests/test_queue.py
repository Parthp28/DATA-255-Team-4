"""Tests T-Q1 to T-Q5 for the GPU lane runner, using tiny shell commands instead of training jobs."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_queue as rq  # noqa: E402

PY = sys.executable


def make_queue(tmp, rows):
    path = tmp / "run_queue.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rq.FIELDS)
        w.writeheader()
        for r in rows:
            full = {f: "" for f in rq.FIELDS}
            full.update({"lane": "P", "status": "READY_FOR_GPU", "priority": "3", "technical_owner": "owner"})
            full.update(r)
            w.writerow(full)
    return path


def run_lane(tmp, queue, lane="P", **extra):
    argv = ["--lane", lane, "--queue", str(queue), "--status-dir", str(tmp / "status"), "--log-dir", str(tmp / "logs"),
            "--workdir", str(tmp), "--poll", "0.05", "--heartbeat", "0.05", "--max-idle", "0.3"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return rq.main(argv)


def touch(name):
    return f'{PY} -c "open(\'{name}\', \'a\').write(\'x\')"'


def status(tmp, qid):
    return json.loads((tmp / "status" / f"{qid}.json").read_text())["status"]


def test_q1_dependency_and_priority_order(tmp_path):
    order = tmp_path / "order.txt"
    cmd = lambda q: f'{PY} -c "open(r\'{order}\', \'a\').write(\'{q} \')"'
    q = make_queue(tmp_path, [
        {"queue_id": "A", "launch_command": cmd("A"), "priority": "3"},
        {"queue_id": "B", "launch_command": cmd("B"), "priority": "1", "dependency": "C"},
        {"queue_id": "C", "launch_command": cmd("C"), "priority": "2"},
    ])
    run_lane(tmp_path, q)
    # C has the best priority among jobs with no unmet dependency, then B (priority 1) unlocks, then A.
    assert order.read_text().split() == ["C", "B", "A"]


def test_q2_failure_retried_then_independent_jobs_continue(tmp_path):
    q = make_queue(tmp_path, [
        {"queue_id": "F", "launch_command": f'{PY} -c "import sys; sys.exit(3)"', "priority": "1"},
        {"queue_id": "G", "launch_command": touch("g.txt"), "priority": "2"},
    ])
    run_lane(tmp_path, q)
    assert status(tmp_path, "F") == "FAILED"
    assert json.loads((tmp_path / "status" / "F.json").read_text())["attempts"] == 2
    assert status(tmp_path, "G") == "COMPLETE"


def test_q3_changed_config_is_refused(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("lr: 0.0003\n")
    signed = hashlib.sha256(cfg.read_bytes()).hexdigest()
    cfg.write_text("lr: 0.001\n")
    q = make_queue(tmp_path, [{"queue_id": "H", "launch_command": touch("h.txt"), "config_path": "cfg.yaml", "config_sha256": signed}])
    run_lane(tmp_path, q)
    assert status(tmp_path, "H") == "FAILED"
    assert not (tmp_path / "h.txt").exists()


def test_q4_failed_dependency_blocks_job(tmp_path):
    q = make_queue(tmp_path, [
        {"queue_id": "X", "launch_command": f'{PY} -c "import sys; sys.exit(1)"'},
        {"queue_id": "Y", "launch_command": touch("y.txt"), "dependency": "X"},
    ])
    run_lane(tmp_path, q, max_attempts=1)
    assert status(tmp_path, "X") == "FAILED"
    assert not (tmp_path / "status" / "Y.json").exists()
    assert not (tmp_path / "y.txt").exists()


def test_q5_stale_running_job_is_rerun(tmp_path):
    q = make_queue(tmp_path, [{"queue_id": "S", "launch_command": touch("s.txt")}])
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    rq.write_status(tmp_path / "status", "S", {"queue_id": "S", "status": "RUNNING", "heartbeat": old})
    run_lane(tmp_path, q)
    assert status(tmp_path, "S") == "COMPLETE"
    assert (tmp_path / "s.txt").exists()


def test_other_lane_and_unsigned_jobs_are_ignored(tmp_path):
    q = make_queue(tmp_path, [
        {"queue_id": "L1", "launch_command": touch("l1.txt"), "lane": "H"},
        {"queue_id": "L2", "launch_command": touch("l2.txt"), "status": "PREPARING"},
    ])
    run_lane(tmp_path, q, lane="P")
    assert not (tmp_path / "l1.txt").exists() and not (tmp_path / "l2.txt").exists()


def test_cross_lane_dependency_completes_from_status_file(tmp_path):
    q = make_queue(tmp_path, [
        {"queue_id": "H02", "launch_command": touch("h02.txt"), "lane": "H", "priority": "1"},
        {"queue_id": "P05", "launch_command": touch("p05.txt"), "dependency": "H02", "priority": "1"},
    ])
    run_lane(tmp_path, q, lane="H")
    run_lane(tmp_path, q, lane="P")
    assert status(tmp_path, "P05") == "COMPLETE"
