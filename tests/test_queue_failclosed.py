"""Queue runner fails closed for real jobs (review: Kaushik). Dummy jobs only (they touch a file); no GPU.

Outside --test-mode a job is refused when git is unavailable, the signed commit is missing or differs, or
the config hash is unsigned or differs. The only unsigned config allowed is a configs/frozen/ file that a
selection step wrote after signing, and it must verify its own freeze_record."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_config  # noqa: E402
import run_queue as rq  # noqa: E402
from test_queue import make_queue, run_lane, status, touch  # noqa: E402


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


def repo(tmp):
    git(tmp, "init", "-q")
    (tmp / "cfg.yaml").write_text("lr: 0.0003\n")
    git(tmp, "add", "cfg.yaml")
    git(tmp, "commit", "-qm", "init")
    return git(tmp, "rev-parse", "HEAD")


def reason(tmp, qid):
    return json.loads((tmp / "status" / f"{qid}.json").read_text()).get("reason", "")


def test_unsigned_job_is_refused(tmp_path):
    (tmp_path / "cfg.yaml").write_text("x: 1\n")
    q = make_queue(tmp_path, [{"queue_id": "U", "launch_command": touch("u.txt"), "config_path": "cfg.yaml",
                               "config_sha256": rq.sha256_of(tmp_path / "cfg.yaml")}])
    run_lane(tmp_path, q, test_mode=False)
    assert status(tmp_path, "U") == "FAILED" and "unsigned job" in reason(tmp_path, "U")
    assert not (tmp_path / "u.txt").exists()


def test_signed_commit_is_refused_when_git_is_unavailable(tmp_path):
    (tmp_path / "cfg.yaml").write_text("x: 1\n")
    q = make_queue(tmp_path, [{"queue_id": "N", "launch_command": touch("n.txt"), "git_commit": "abc123",
                               "config_path": "cfg.yaml", "config_sha256": rq.sha256_of(tmp_path / "cfg.yaml")}])
    for test_mode in (False, True):                       # even test mode never passes an unverifiable commit
        run_lane(tmp_path, q, test_mode=test_mode)
        assert status(tmp_path, "N") == "FAILED" and "cannot verify the code commit" in reason(tmp_path, "N")
    assert not (tmp_path / "n.txt").exists()


def test_commit_mismatch_and_config_change_are_refused(tmp_path):
    head = repo(tmp_path)
    sha = rq.sha256_of(tmp_path / "cfg.yaml")
    q = make_queue(tmp_path, [
        {"queue_id": "C", "launch_command": touch("c.txt"), "git_commit": "deadbeef", "config_path": "cfg.yaml",
         "config_sha256": sha},
        {"queue_id": "H", "launch_command": touch("h.txt"), "git_commit": head, "config_path": "cfg.yaml",
         "config_sha256": "0" * 64}])
    run_lane(tmp_path, q, test_mode=False)
    assert "differs from signed commit" in reason(tmp_path, "C") and "sha256 mismatch" in reason(tmp_path, "H")
    assert not (tmp_path / "c.txt").exists() and not (tmp_path / "h.txt").exists()


def test_signed_job_runs_in_real_mode(tmp_path):
    head = repo(tmp_path)
    q = make_queue(tmp_path, [{"queue_id": "OK", "launch_command": touch("ok.txt"), "git_commit": head[:10],
                               "config_path": "cfg.yaml", "config_sha256": rq.sha256_of(tmp_path / "cfg.yaml")}])
    run_lane(tmp_path, q, test_mode=False)
    assert status(tmp_path, "OK") == "COMPLETE" and (tmp_path / "ok.txt").exists()


def test_only_a_verified_selection_frozen_config_may_be_unsigned(tmp_path):
    head = repo(tmp_path)
    frozen = tmp_path / "configs" / "frozen"
    freeze_config.main(["--config", str(ROOT / "experiments" / "E1.yaml"), "--rule", "test",
                        "--out", str(frozen / "E1.yaml")])
    (frozen / "E3.yaml").write_text((ROOT / "experiments" / "E3.yaml").read_text())      # no freeze_record
    q = make_queue(tmp_path, [
        {"queue_id": "A", "launch_command": touch("a.txt"), "git_commit": head, "config_path": "cfg.yaml"},
        {"queue_id": "B", "launch_command": touch("b.txt"), "git_commit": head, "config_path": "configs/frozen/E3.yaml"},
        {"queue_id": "F", "launch_command": touch("f.txt"), "git_commit": head, "config_path": "configs/frozen/E1.yaml"},
        {"queue_id": "M", "launch_command": touch("m.txt"), "git_commit": head, "config_path": "configs/frozen/E2.yaml"}])
    run_lane(tmp_path, q, test_mode=False)
    assert "unsigned config" in reason(tmp_path, "A")
    assert "does not verify" in reason(tmp_path, "B")
    assert "not produced yet" in reason(tmp_path, "M")
    assert status(tmp_path, "F") == "COMPLETE" and (tmp_path / "f.txt").exists()
