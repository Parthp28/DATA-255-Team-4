"""make_report_assets.py (review: Kaushik; Parth integrates the report): numbers copied from evaluate.py's tables,
source hashes in every fragment, synthetic outputs refused or stamped and kept out of report/, and well-formed
LaTeX. CPU only; synthetic fixtures; not results."""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import evaluate as ev  # noqa: E402
import make_report_assets as mra  # noqa: E402
from make_synthetic_predictions import fixture_dataset, make_all  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_eval(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("assets")
    windows, registry = fixture_dataset()
    data = tmp / "data"
    data.mkdir()
    windows.to_parquet(data / "windows.parquet", index=False)
    registry.to_parquet(data / "render_registry.parquet", index=False)
    files = make_all(windows, tmp / "predictions")
    ev.evaluate(files, data, "test", tmp / "eval", allow_synthetic=True, n_boot=20)
    return tmp / "eval"


def fake_masking(tmp):
    rng = np.random.default_rng(0)
    d = tmp / "masking"
    d.mkdir()
    for e in mra.IMAGE_EXPS:
        rows = [{"run_id": f"{e}_s0_00000000", "window_id": f"w{i}", "render_id": "R0", "region": g, "fill_mode": m,
                 "area_frac": 0.01, "pred_class_orig": 1, "p_pred_orig": 0.5,
                 "p_pred_masked": 0.5 - (0.2 if g == "candles" and m == "background" else 0.01) * rng.random(),
                 "delta_p": 0.0, "flipped": False}
                for i in range(10) for g in mra.REGION_ORDER for m in ("background", "random_area_matched")]
        df = pd.DataFrame(rows)
        df["delta_p"] = df["p_pred_orig"] - df["p_pred_masked"]
        df.to_parquet(d / f"{e}_s0_00000000.parquet", index=False)
    return d


def fake_attribution(tmp):
    from src.interpretability.regions import REGIONS
    d = tmp / "attribution"
    d.mkdir()
    for e in mra.IMAGE_EXPS:
        method = "rollout" if e in ("E2", "E4b", "E4") else "gradcam"
        rows = [{"run_id": f"{e}_s0_00000000", "window_id": f"w{i}", "render_id": "R0", "method": method,
                 "region": g, "attribution_mass": 1 / len(REGIONS), "area_frac": 0.1, "mass_ratio": 10 / len(REGIONS)}
                for i in range(5) for g in REGIONS]
        pd.DataFrame(rows).to_parquet(d / f"{e}_s0_00000000.parquet", index=False)
    return d


def check_latex(path: Path):
    t = path.read_text()
    assert t.count("\\begin{") == t.count("\\end{")
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", t).group(1)
    ncol = len(re.findall(r"[lcrpXY]", spec))
    body = t.split("\\midrule")[1].split("\\bottomrule")[0]
    for line in [ln for ln in body.strip().splitlines() if ln.strip()]:
        assert line.count(" & ") + 1 == ncol, (path.name, line)
    return t


def report_snapshot():
    """{relative path: sha256} of everything under report/generated (the committed frozen-result assets)."""
    import hashlib
    gen = ROOT / "report" / "generated"
    return {str(p.relative_to(gen)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(gen.rglob("*")) if p.is_file()} if gen.exists() else {}


def test_synthetic_outputs_are_refused_without_the_flag_and_never_land_in_report(synthetic_eval, tmp_path):
    # report/generated holds the committed assets of the frozen results, so the check is that a refused synthetic
    # run changes nothing there (and stamps nothing anywhere under report/), not that the folder is absent.
    before = report_snapshot()
    with pytest.raises(mra.AssetRefused, match="synthetic"):
        mra.generate(synthetic_eval, tmp_path / "out")
    with pytest.raises(mra.AssetRefused, match="under report/"):
        mra.generate(synthetic_eval, ROOT / "report" / "generated", allow_synthetic=True)
    with pytest.raises(mra.AssetRefused, match="under report/"):
        mra.generate(synthetic_eval, ROOT / "report" / "generated" / "preview", allow_synthetic=True)
    assert report_snapshot() == before
    assert not (ROOT / "report" / "generated" / "preview").exists()
    stamped = [p for p in (ROOT / "report").rglob("*.tex") if mra.STAMP in p.read_text(errors="ignore")]
    assert stamped == []
    assert mra.main(["--eval-dir", str(synthetic_eval), "--out", str(tmp_path / "x")]) == 2


def test_synthetic_preview_is_stamped_everywhere(synthetic_eval, tmp_path):
    man = mra.generate(synthetic_eval, tmp_path / "out", masking_dir=fake_masking(tmp_path),
                       attribution_dir=fake_attribution(tmp_path), allow_synthetic=True)
    assert "figures/fig6_attribution.pdf" in man["written"] and "attribution/E1_s0_00000000.parquet" in man["sources"]
    assert man["synthetic"] is True
    assert {"tables/table1_main.tex", "tables/table2_per_render.tex", "tables/table3_masking.tex",
            "tables/table4_seeds.tex", "tables/table5_comparisons.tex", "tables/table7_calibration.tex",
            "figures/fig3_flip_by_render.pdf", "figures/fig4_tradeoff.pdf", "figures/fig7_masking.pdf"} <= set(man["written"])
    for p in (tmp_path / "out" / "tables").glob("*.tex"):
        assert mra.STAMP in check_latex(p)
    for p in (tmp_path / "out" / "figures").glob("*.pdf"):
        assert p.read_bytes()[:4] == b"%PDF"


def test_numbers_are_copied_from_the_evaluation_tables_with_their_hashes(synthetic_eval, tmp_path):
    real = tmp_path / "eval"
    shutil.copytree(synthetic_eval, real)
    (real / "SYNTHETIC_NOT_RESULTS.txt").unlink()            # stands in for a real evaluation directory
    mra.generate(real, tmp_path / "out", masking_dir=fake_masking(tmp_path))
    seeds = pd.read_csv(real / "seeds.csv")
    t1 = check_latex(tmp_path / "out" / "tables" / "table1_main.tex")
    assert mra.STAMP not in t1 and f"sha256 {mra.sha(real / 'seeds.csv')}" in t1
    for e in ("E1", "E4", "E5"):
        r = seeds[(seeds["experiment_id"] == e) & (seeds["metric"] == "macro_f1")].iloc[0]
        line = next(ln for ln in t1.splitlines() if ln.startswith(f"{e} &"))
        assert line.split(" & ")[1] == mra.pm(r["mean"], r["sd"])
    best = pd.read_csv(real / "comparisons.csv").query("comparison_id == 'A8'")["exp_a"].iloc[0]
    assert f"best B0 ({best})" in t1
    t3 = check_latex(tmp_path / "out" / "tables" / "table3_masking.tex")
    assert "candles" in t3 and "masking/E1_s0_00000000.parquet" in t3
    for name in ("table2_per_render.tex", "table4_seeds.tex", "table5_comparisons.tex", "table7_calibration.tex"):
        check_latex(tmp_path / "out" / "tables" / name)


def test_missing_evaluation_tables_are_refused(tmp_path):
    (tmp_path / "eval").mkdir()
    with pytest.raises(mra.AssetRefused, match="run scripts/evaluate.py"):
        mra.generate(tmp_path / "eval", tmp_path / "out")


def test_reliability_and_confusion_figures_come_from_the_frozen_predictions(synthetic_eval, tmp_path):
    import hashlib
    import json
    preds = sorted((synthetic_eval.parent / "predictions").glob("*.parquet"))
    man = {"runs": [{"run_id": p.stem, "prediction_path": str(p),
                     "prediction_sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in preds]}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(man))
    out = mra.generate(synthetic_eval, tmp_path / "out", freeze=freeze, allow_synthetic=True)
    assert {"figures/fig5_reliability.pdf", "figures/fig8_confusion.pdf"} <= set(out["written"])
    assert "freeze manifest" in out["sources"]
    for f in ("fig5_reliability.pdf", "fig8_confusion.pdf"):
        assert (tmp_path / "out" / "figures" / f).read_bytes()[:4] == b"%PDF"
