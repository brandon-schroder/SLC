"""Ensure the repo root is importable regardless of pytest invocation dir."""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)