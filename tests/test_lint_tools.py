"""Negative + positive controls for the two CI lint gates
(``tools/check_imports.py``, ``tools/check_ad6.py``; AD-5, AD-6, ARCH-7).

Provenance: 2026-07 audit follow-up. The audit found both checkers weaker
than the decisions they claim to enforce: check_ad6's R1 patterns matched
only the ``np.`` spelling and never the injected ``xp`` namespace the
kernel actually routes through; ``# ad6: allow`` suppressed a line with no
justification required; ``transport/`` (flow-array schedules/mixing)
escaped the C1 token rule entirely; and check_imports enforced layer
DIRECTION only, so ``assembly`` importing Lieblein correlation constants
would have passed the "AD-5 enforcement". No live violations existed --
the discipline held by convention -- but a gate that has never rejected
anything is unproven. These tests apply the CLAUDE.md checker rule (every
checker ships a negative control demonstrating it rejects a known
violation) to the lint gates themselves, plus positive controls that the
current tree and the shipped waivers stay clean.

The tools are scripts, not package modules; they are loaded here by file
path. Their scan cores (``scan_file``, ``violation_for``) are pure
functions over synthetic input, so no temp files are involved.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ad6 = _load_tool("check_ad6")
imp = _load_tool("check_imports")


def _msgs(violations):
    return [msg for _, _, msg in violations]


# --------------------------------------------------------------------------
# Positive control: the current tree passes both gates end-to-end
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tool", ["check_imports.py", "check_ad6.py"])
def test_repo_is_clean_end_to_end(tool):
    res = subprocess.run([sys.executable, str(ROOT / "tools" / tool)],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


# --------------------------------------------------------------------------
# AD-6 R1: hard-kink numerics in flow-array code -- BOTH np. and xp. forms
# --------------------------------------------------------------------------
@pytest.mark.parametrize("snippet", [
    "y = xp.abs(x - x0)",          # the audit's headline blind spot
    "y = xp.maximum(a, b)",
    "y = xp.minimum(a, b)",
    "y = xp.where(mask, a, b)",
    "y = np.maximum(a, b)",
    "y = arr.clip(0.0, 1.0)",
    "import math",
    "y = math.copysign(1.0, x)",
])
def test_ad6_r1_rejects_hard_kinks_in_closures(snippet):
    v = ad6.scan_file(f"{ad6.PKG}/closures/somefamily/loss.py", snippet + "\n")
    assert any("R1" in m for m in _msgs(v)), snippet


def test_ad6_r1_now_covers_transport():
    # transport/ carries flow-array schedules and the mixing operator
    # (section 7.3 C1 discipline applies); it escaped R1 before the audit.
    v = ad6.scan_file(f"{ad6.PKG}/transport/newschedule.py",
                      "ramp = xp.minimum(t, 1.0)\n")
    assert any("R1" in m for m in _msgs(v))


def test_ad6_r1_exemptions_hold():
    # smoothmath is the C1 toolbox (its interior clip is C2-proven) and
    # __init__ re-exports carry no numerics.
    assert ad6.scan_file(f"{ad6.PKG}/closures/smoothmath.py",
                         "t = xp.clip(t, 0.0, 1.0)\n") == []
    assert ad6.scan_file(f"{ad6.PKG}/closures/__init__.py",
                         "t = xp.clip(t, 0.0, 1.0)\n") == []
    # ...and R1 does not reach non-flow-array kernel packages (the
    # assembler's capacity scan legitimately uses np.maximum).
    v = ad6.scan_file(f"{ad6.PKG}/assembly/other.py", "y = np.maximum(a, b)\n")
    assert not any("R1" in m for m in _msgs(v))


# --------------------------------------------------------------------------
# AD-6 R0: waivers must be justified
# --------------------------------------------------------------------------
def test_ad6_bare_waiver_is_itself_a_violation():
    v = ad6.scan_file(f"{ad6.PKG}/closures/x/loss.py",
                      "y = xp.abs(x)  # ad6: allow\n")
    assert any("R0" in m for m in _msgs(v))


def test_ad6_justified_waiver_silences_the_line():
    line = ("y = xp.abs(x)  # ad6: allow -- C1-proved in test_x, "
            "kink unreachable in-domain\n")
    assert ad6.scan_file(f"{ad6.PKG}/closures/x/loss.py", line) == []


def test_ad6_existing_waiver_style_still_accepted():
    # The repo's dominant style puts the justification BEFORE the tag.
    line = ("import numpy as np  # assembly is numpy/scipy-bound via "
            "PCHIP (5.3)  # ad6: allow\n")
    assert ad6.scan_file(f"{ad6.PKG}/assembly/assembler.py", line) == []


# --------------------------------------------------------------------------
# AD-6 R2 / R3 negative controls
# --------------------------------------------------------------------------
def test_ad6_r2_rejects_inplace_mutation_on_residual_path():
    v = ad6.scan_file(f"{ad6.PKG}/assembly/new.py", "acc += flux\n")
    assert any("R2" in m for m in _msgs(v))
    v = ad6.scan_file(f"{ad6.PKG}/grid/new.py", "q[1:-1] = q_new\n")
    assert any("R2" in m for m in _msgs(v))


def test_ad6_r3_rejects_direct_numpy_import():
    v = ad6.scan_file(f"{ad6.PKG}/drivers/new.py", "import numpy as np\n")
    assert any("R3" in m for m in _msgs(v))
    assert ad6.scan_file(f"{ad6.PKG}/_namespace.py",
                         "import numpy as np\n") == []


# --------------------------------------------------------------------------
# check_imports: AD-5 firewall (the audit's direction-only gap)
# --------------------------------------------------------------------------
def test_ad5_rejects_kernel_importing_closure_implementations():
    # Layer direction alone permits all of these (closures is layer 2);
    # AD-5 must reject them: machine-type knowledge stays in closures/.
    for importer, imported in [
        ("slcflow.assembly.assembler",
         "slcflow.closures.axial_compressor.lieblein"),
        ("slcflow.drivers.classical", "slcflow.closures.centrifugal.wiesner"),
        ("slcflow.machine", "slcflow.closures.axial_turbine.loss"),
        ("slcflow.transport.mixing", "slcflow.closures.simple"),
        ("slcflow.drivers.newton", "slcflow.closures.conversions"),
        ("slcflow.drivers.classical", "slcflow.closures"),  # opaque names
    ]:
        msg = imp.violation_for(importer, imported)
        assert msg is not None and "AD-5" in msg, (importer, imported)


def test_ad5_allows_the_interface_grade_modules():
    assert imp.violation_for("slcflow.drivers.classical",
                             "slcflow.closures.interfaces") is None
    assert imp.violation_for("slcflow.transport.schedules",
                             "slcflow.closures.smoothmath") is None


def test_ad5_exempts_closures_internals_and_facade():
    # Correlation families import their own package freely...
    assert imp.violation_for("slcflow.closures.axial_turbine.loss",
                             "slcflow.closures.conversions") is None
    # ...and the bare-package facade (layer 8) may re-export anything.
    assert imp.violation_for("slcflow",
                             "slcflow.closures.axial_compressor") is None


def test_direction_rule_still_enforced():
    # The pre-existing rule the firewall was added alongside.
    msg = imp.violation_for("slcflow.transport.mixing",
                            "slcflow.drivers.classical")
    assert msg is not None and "wrong direction" in msg
    assert imp.violation_for("slcflow.drivers.classical",
                             "slcflow.assembly.assembler") is None
