#!/usr/bin/env python3
"""AD-6 / smoothness lint (Architecture Spec AD-6, ARCH-4.2, ARCH-7).

Greps for patterns that violate the numerical-contract rules. Deliberately
simple and conservative: it flags *tokens*, humans adjudicate; a false
positive is silenced by an inline ``# ad6: allow`` with a justification.

Rules enforced
--------------
R1 (closures/ except smoothmath): no direct hard-kink numerics -- these must
   come from smoothmath so section 7.3 C1 continuity is auditable:
   ``.clip(``, ``np.clip``, ``np.maximum``, ``np.minimum``, ``np.abs(``,
   ``math.``, ``import math``, ``if`` chains are not detectable textually --
   reviewed by humans per ARCH-4.2, but ``np.where(`` is flagged for review.
R2 (residual-path packages: assembly/, transport/, grid/): no in-place
   array-mutating operators on state-derived arrays -- flags ``+=``, ``-=``,
   ``*=``, ``/=`` and ``[...] =`` slice-assignment. (Scalar counters trip this
   too; silence legit uses with ``# ad6: allow``.)
R3 (everywhere in the kernel): no ``import numpy`` outside the modules
   allowed to bind it directly (namespace injection instead).

Run:  python tools/check_ad6.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = "slcflow"
ROOT = Path(__file__).resolve().parents[1]

ALLOW_TAG = "ad6: allow"

# Modules allowed to import numpy directly (leaves + validation helpers).
NUMPY_ALLOWED = {
    f"{PKG}/_namespace.py",
    f"{PKG}/closures/smoothmath.py",  # validation helpers only
    f"{PKG}/fluid/base.py",           # typing only
}

R1_PATTERNS = [
    r"\.clip\(", r"np\.clip", r"np\.maximum", r"np\.minimum",
    r"np\.abs\(", r"\bimport math\b", r"\bmath\.", r"np\.where\(",
]
R2_PATTERNS = [r"\+=", r"-=", r"\*=", r"/=", r"\]\s*="]
R2_DIRS = ("assembly", "transport", "grid")


def lines(path: Path):
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if ALLOW_TAG in line:
            continue
        yield i, line


def main() -> int:
    violations = []
    for path in sorted((ROOT / PKG).rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()

        # R3
        if rel not in NUMPY_ALLOWED:
            for i, line in lines(path):
                if re.search(r"^\s*import numpy|^\s*from numpy", line):
                    violations.append((rel, i, "R3: direct numpy import (use xp injection)"))

        # R1
        if rel.startswith(f"{PKG}/closures/") and not rel.endswith(
            ("smoothmath.py", "__init__.py")
        ):
            for i, line in lines(path):
                for pat in R1_PATTERNS:
                    if re.search(pat, line):
                        violations.append((rel, i, f"R1: hard-kink numeric {pat!r} -- use smoothmath"))

        # R2
        if any(rel.startswith(f"{PKG}/{d}/") for d in R2_DIRS):
            for i, line in lines(path):
                if line.lstrip().startswith("#"):
                    continue
                for pat in R2_PATTERNS:
                    if re.search(pat, line):
                        violations.append((rel, i, f"R2: in-place mutation {pat!r} on residual path"))

    for rel, i, msg in violations:
        print(f"{rel}:{i}: {msg}")
    if not violations:
        print("AD-6 lint: OK")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())