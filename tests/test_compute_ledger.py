"""compute_ledger.py (review: Kaushik): GPU-hours and kWh from run records only, categories, the logistics firewall,
and the synthetic guard. Hand-made run records; not results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import math  # noqa: E402

import compute_ledger as cl  # noqa: E402


def write(runs, rid, **rec):
    d = runs / rid
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"run_id": rid, "status": "complete", **rec}))


def make_runs(tmp):
    runs = tmp / "runs"
    t4 = "Tesla T4"
    write(runs, "E1_s0_a", experiment_id="E1", gpu_model=t4, gpu_minutes=60.0, wall_minutes=61.0)
    write(runs, "E1_s0_b", experiment_id="E1", gpu_model=t4, gpu_minutes=30.0, wall_minutes=31.0)
    write(runs, "E1_s1_c", experiment_id="E1", gpu_model=t4, gpu_minutes=120.0, wall_minutes=121.0,
          frozen_config_path="configs/frozen/E1.yaml", lane="P", gpu_executor="someone")
    write(runs, "E2_s0_d", experiment_id="E2", gpu_model=t4, gpu_minutes=6.0, wall_minutes=6.0, max_steps=5)
    write(runs, "E2_s0_e", experiment_id="E2", gpu_model=t4, gpu_minutes=12.0, wall_minutes=12.0, status="superseded")
    write(runs, "E5_s0_f", experiment_id="E5", gpu_model="cpu", gpu_minutes=0.0, wall_minutes=3.0)
    return runs


def test_gpu_hours_and_energy_come_from_the_records(tmp_path):
    led = cl.ledger(cl.load_records(make_runs(tmp_path))).set_index(["experiment_id", "category"])
    e1 = led.loc[("E1", "tuning/dev")]
    assert e1["runs"] == 2 and math.isclose(e1["gpu_hours"], 1.5) and math.isclose(e1["energy_kwh"], 1.5 * 0.070)
    assert math.isclose(led.loc[("E1", "final"), "energy_kwh"], 2.0 * 0.070)
    assert led.loc[("E2", "debug"), "runs"] == 1 and led.loc[("E2", "superseded"), "runs"] == 1
    cpu = led.loc[("E5", "tuning/dev")]
    assert cpu["gpu_hours"] == 0 and cpu["energy_kwh"] == 0 and math.isclose(cpu["wall_hours"], 0.05)
    total = led.loc[("TOTAL", "all")]
    assert total["runs"] == 6 and math.isclose(total["gpu_hours"], (60 + 30 + 120 + 6 + 12) / 60)


def test_logistics_fields_never_reach_the_ledger(tmp_path):
    runs = make_runs(tmp_path)
    out, tex = tmp_path / "ledger.csv", tmp_path / "ledger.tex"
    assert cl.main(["--runs-dir", str(runs), "--out", str(out), "--tex", str(tex)]) == 0
    text = out.read_text() + tex.read_text()
    assert "someone" not in text and "lane" not in text.lower() and "executor" not in text.lower()
    assert "70 W" in tex.read_text() and tex.read_text().count("\\begin{") == tex.read_text().count("\\end{")


def test_unknown_gpu_energy_is_not_claimed(tmp_path):
    runs = tmp_path / "runs"
    write(runs, "E1_s0_x", experiment_id="E1", gpu_model="Some Other GPU", gpu_minutes=60.0, wall_minutes=60.0)
    led = cl.ledger(cl.load_records(runs))
    assert led["energy_kwh"].isna().all()


def test_synthetic_records_cannot_feed_the_report(tmp_path):
    runs = tmp_path / "runs"
    write(runs, "E1_s0_y", experiment_id="E1", gpu_model="cpu", gpu_minutes=0, wall_minutes=1, data_version="synthetic_v0")
    tex = tmp_path / "report" / "generated" / "tables" / "table_ledger.tex"
    assert cl.main(["--runs-dir", str(runs), "--out", str(tmp_path / "l.csv"), "--tex", str(tex)]) == 2
    assert not tex.exists()
