"""GPU session launchers (review: Kaushik): the D16 T4 pre-flight, the session script's argument handling and its
stop before any data step when no T4 is visible, and the two one-cell notebooks. Nothing is launched; no GPU."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

import check_gpu  # noqa: E402

MEMBER_NAMES = ("parth", "khushi", "kaushik", "farzaan")
REPO_URL = "https://github.com/Parthp28/DATA-255-Team-4.git"


@pytest.mark.parametrize("names,code", [(["Tesla T4"], 0), (["Tesla T4", "Tesla T4"], 0),
                                        (["NVIDIA A100-SXM4-40GB"], 3), (["Tesla T4", "NVIDIA L4"], 3), ([], 4)])
def test_d16_preflight_accepts_only_t4(names, code):
    assert check_gpu.check(names)[0] == code
    assert check_gpu.main(names=names) == code


bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


@bash
def test_session_script_parses_and_requires_its_arguments():
    script = ROOT / "scripts" / "gpu_session.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert subprocess.run(["bash", str(script), "--help"], capture_output=True).returncode == 0
    assert subprocess.run(["bash", str(script), "--lane", "X"], capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(script), "--bogus"], capture_output=True).returncode == 2


@bash
@pytest.mark.skipif(check_gpu.check(check_gpu.gpu_names())[0] == 0, reason="a T4 is visible here")
def test_session_stops_at_the_gpu_check_before_touching_data(tmp_path):
    r = subprocess.run(["bash", str(ROOT / "scripts" / "gpu_session.sh"), "--lane", "X", "--data-src",
                        str(tmp_path / "absent"), "--no-install", "--check-only"], capture_output=True, text=True,
                       cwd=ROOT)
    out = r.stdout + r.stderr
    assert r.returncode in (3, 4) and ("T4" in out or "no CUDA GPU" in out)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("name", ["kaggle_launch.ipynb", "colab_launch.ipynb"])
def test_notebooks_are_one_launch_cell_with_no_logistics(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert nb["nbformat"] == 4 and len(code) == 1 and not code[0]["outputs"]
    src = "".join(code[0]["source"])
    assert "scripts/gpu_session.sh" in src and "CHECK_ONLY = True" in src
    text = json.dumps(nb).lower().replace(REPO_URL.lower(), "")      # the repository address is not logistics
    assert not any(n in text for n in MEMBER_NAMES) and "5090" not in text and "rtx" not in text
