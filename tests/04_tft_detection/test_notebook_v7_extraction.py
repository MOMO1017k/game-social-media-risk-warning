import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    REPO_ROOT
    / "notebooks"
    / "04_tft_detection"
    / "legacy"
    / "TFT_main_v5-9-pc.ipynb"
)
FINAL_DIR = REPO_ROOT / "scripts" / "04_tft_detection" / "final"


def _cell_source(index: int) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(notebook["cells"][index]["source"])


def test_v7_detector_is_exact_notebook_cell_204():
    extracted = (FINAL_DIR / "notebook_v7_detector.py").read_text(encoding="utf-8")
    assert extracted == _cell_source(204)


def test_v7_original_run_cell_is_preserved():
    extracted = (FINAL_DIR / "notebook_v7_run_cell.py").read_text(encoding="utf-8")
    assert extracted.rstrip("\n") == _cell_source(206).rstrip("\n")
