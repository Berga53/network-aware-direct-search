"""Path setup shared by the experiment scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA = ROOT / "data"
# Published results live in results/; scripts write elsewhere by default so a
# re-run never overwrites them.
OUTPUTS = ROOT / "outputs"
