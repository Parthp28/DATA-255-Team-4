"""report/main.tex against the generators (review: Parth, report integration): every generated fragment the report
pulls in is one a generator writes, every generated asset is used, and the LaTeX is balanced. No LaTeX needed."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_report_assets as mra  # noqa: E402

TEX = (ROOT / "report" / "main.tex").read_text(encoding="utf-8")
LEDGER = "tables/table_ledger.tex"          # written by scripts/compute_ledger.py --tex
BS = "\\"


def referenced():
    body = re.sub(r"(?<!\\)%.*", "", TEX)                       # comments may show usage examples
    uses = re.findall(r"\\gen\{([^}#]+)\}", body)
    figs = {f"figures/{f}" for f in re.findall(r"\\genfig\{([^}#]+)\}", body)}
    figs |= {m for m in re.findall(r"IfFileExists\{generated/(figures/[^}#]+)\}", body)}
    return set(uses) | figs


def test_every_reference_has_a_generator_and_every_asset_is_used():
    refs = referenced()
    assert refs - set(mra.ASSETS) - {LEDGER} == set(), "report pulls in a fragment no generator writes"
    assert set(mra.ASSETS) - refs == set(), "a generated asset is never used in the report"
    assert LEDGER in refs


def test_generator_writes_only_declared_assets():
    written = set(re.findall(r'w\.(?:table|figure)\("([^"]+)"', Path(mra.__file__).read_text()))
    assert written == {a.split("/", 1)[1] for a in mra.ASSETS}


def test_latex_is_balanced():
    body = re.sub(r"(?<!\\)%.*", "", TEX)
    escaped_open, escaped_close = body.count(BS + "{"), body.count(BS + "}")
    assert body.count("{") - escaped_open == body.count("}") - escaped_close
    for env in set(re.findall(r"\\begin\{([^}#]+)\}", body)):
        assert body.count(BS + "begin{" + env + "}") == body.count(BS + "end{" + env + "}"), env
    assert BS + "newcommand{" + BS + "genfig}[4]" in TEX
