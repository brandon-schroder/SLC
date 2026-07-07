#!/usr/bin/env python3
"""AD-6 / smoothness lint (Architecture Spec AD-6, ARCH-4.2, ARCH-7).

Greps for patterns that violate the numerical-contract rules. Deliberately
simple and conservative: it flags *tokens*, humans adjudicate; a false
positive is silenced by an inline ``# ad6: allow`` **with a justification**
-- a bare tag is itself a violation (R0), so every waiver stays auditable.

Rules enforced
--------------
R0 (everywhere): an ``ad6: allow`` waiver must carry a justification: the
   line's comment text, with the tag itself removed, must still say
   something (>= 10 non-filler characters).
R1 (flow-array code per the section 7.3 C1 discipline: closures/ except
   smoothmath, AND transport/): no direct hard-kink numerics -- these must
   come from smoothmath so C1 continuity is auditable. Matched in BOTH the
   ``np.`` and the injected ``xp.`` spelling (2026-07 audit: the original
   ``np.``-only patterns never matched the namespace the kernel actually
   routes through): ``.clip(``, ``maximum``, ``minimum``, ``abs(``,
   ``where(``, plus ``import math`` / ``math.``. ``if`` chains are not
   detectable textually -- reviewed by humans per ARCH-4.2.
R2 (residual-path packages: assembly/, transport/, grid/): no in-place
   array-mutating operators on state-derived arrays -- flags ``+=``, ``-=``,
   ``*=``, ``/=`` and ``[...] =`` slice-assignment. (Scalar counters trip
   this too; silence legit uses with a justified ``# ad6: allow``.)
R3 (everywhere in the kernel): no ``import numpy`` outside the modules
   allowed to bind it directly (namespace injection instead).

The per-file scan is a pure function (``scan_file``) so the checker itself
carries negative controls -- ``tests/test_lint_tools.py`` proves each rule
rejects a known violation, per the CLAUDE.md checker discipline.

Run:  python tools/check_ad6.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PKG = "slcflow"
ROOT = Path(__file__).resolve().parents[1]

ALLOW_TAG = "ad6: allow"
_MIN_JUSTIFICATION = 10   # non-filler comment characters besides the tag

# Modules allowed to import numpy directly (leaves + validation helpers).
NUMPY_ALLOWED = {
    f"{PKG}/_namespace.py",
    f"{PKG}/closures/smoothmath.py",  # validation helpers only
    f"{PKG}/fluid/base.py",           # typing only
}

R1_PATTERNS = [
    r"\.clip\(",                       # method form + np.clip( + xp.clip(
    r"np\.maximum", r"xp\.maximum",
    r"np\.minimum", r"xp\.minimum",
    r"np\.abs\(", r"xp\.abs\(",
    r"np\.where\(", r"xp\.where\(",
    r"\bimport math\b", r"\bmath\.",
]
R1_DIRS = ("closures", "transport")    # section 7.3 flow-array code
R1_EXEMPT_SUFFIXES = ("smoothmath.py", "__init__.py")
R2_PATTERNS = [r"\+=", r"-=", r"\*=", r"/=", r"\]\s*="]
R2_DIRS = ("assembly", "transport", "grid")


def has_justification(line: str) -> bool:
    """R0: the waiver line's comment, minus the tag and filler characters
    (whitespace, ``#``, dashes), must still contain a real justification."""
    comment = line[line.index("#"):] if "#" in line else ""
    rest = re.sub(r"[#\s\-]+", "", comment.replace(ALLOW_TAG, ""))
    return len(rest) >= _MIN_JUSTIFICATION


def scan_file(rel: str, text: str):
    """All violations for one file, as ``(rel, lineno, message)`` tuples.

    ``rel`` is the repo-relative posix path (rule scoping keys off it);
    pure over its arguments so the negative controls in
    tests/test_lint_tools.py can feed synthetic content.
    """
    violations = []
    r1 = any(rel.startswith(f"{PKG}/{d}/") for d in R1_DIRS) \
        and not rel.endswith(R1_EXEMPT_SUFFIXES)
    r2 = any(rel.startswith(f"{PKG}/{d}/") for d in R2_DIRS)
    r3 = rel not in NUMPY_ALLOWED

    for i, line in enumerate(text.splitlines(), start=1):
        if ALLOW_TAG in line:
            if not has_justification(line):
                violations.append((rel, i, "R0: 'ad6: allow' without a "
                                           "justification comment"))
            continue
        if r3 and re.search(r"^\s*import numpy|^\s*from numpy", line):
            violations.append((rel, i, "R3: direct numpy import "
                                       "(use xp injection)"))
        if r1:
            for pat in R1_PATTERNS:
                if re.search(pat, line):
                    violations.append((rel, i, f"R1: hard-kink numeric "
                                               f"{pat!r} -- use smoothmath"))
        if r2 and not line.lstrip().startswith("#"):
            for pat in R2_PATTERNS:
                if re.search(pat, line):
                    violations.append((rel, i, f"R2: in-place mutation "
                                               f"{pat!r} on residual path"))
    return violations


def main() -> int:
    violations = []
    for path in sorted((ROOT / PKG).rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        violations.extend(scan_file(rel, path.read_text()))
    for rel, i, msg in violations:
        print(f"{rel}:{i}: {msg}")
    if not violations:
        print("AD-6 lint: OK")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
