#!/usr/bin/env python3
"""Dependency-direction + AD-5 firewall check (Architecture Spec ARCH-2, AD-5).

Enforces the layered import order:

    _namespace/errors (0) -> fluid / geometry / closures.smoothmath (1)
    -> closures / grid (2) -> transport (3) -> assembly (4) -> drivers (5)
    -> machine (6) -> io (7)

A module may import only from layers <= its own. ``verification`` and tests
are exempt (may import anything). Exit code 1 on violation; prints offenders.

Additionally the **AD-5 firewall** (2026-07 audit): the layer order alone
would let kernel modules import closure IMPLEMENTATIONS -- ``closures``
sits below ``assembly``/``drivers``, so e.g. the assembler importing
Lieblein constants passed the direction rule. Machine-type knowledge must
not leak (AD-5: "the kernel imports interfaces, never implementations"),
so from outside ``closures/`` only the interface-grade closure modules in
``AD5_ALLOWED`` are importable. The per-edge verdict is a pure function
(``violation_for``) with negative controls in tests/test_lint_tools.py.

Run:  python tools/check_imports.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PKG = "slcflow"
ROOT = Path(__file__).resolve().parents[1]

# Longest-prefix match wins.
LAYERS = {
    f"{PKG}._namespace": 0,
    f"{PKG}.errors": 0,
    f"{PKG}.types": 0,
    f"{PKG}.fluid": 1,
    f"{PKG}.geometry": 1,
    f"{PKG}.closures.smoothmath": 1,
    f"{PKG}.closures": 2,
    f"{PKG}.grid": 2,
    f"{PKG}.transport": 3,
    f"{PKG}.assembly": 4,
    f"{PKG}.drivers": 5,
    f"{PKG}.machine": 6,
    f"{PKG}.io": 7,
    f"{PKG}.diagnostics": 1,  # leaf-ish: importable by all layers >= 2
    f"{PKG}": 8,  # bare package __init__ may import anything (facade)
}
EXEMPT_PREFIXES = (f"{PKG}.verification",)

# AD-5 firewall: closure modules importable from OUTSIDE closures/. Interfaces
# and the C1 toolbox are contract-grade; everything else in closures/ is a
# machine-type implementation (correlation constants) the kernel must not see.
AD5_ALLOWED = {
    f"{PKG}.closures.interfaces",
    f"{PKG}.closures.smoothmath",
}


def layer_of(modname: str) -> int | None:
    best, best_len = None, -1
    for prefix, layer in LAYERS.items():
        if (modname == prefix or modname.startswith(prefix + ".")) and len(
            prefix
        ) > best_len:
            best, best_len = layer, len(prefix)
    return best


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative(importer: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Resolve a relative import to an absolute module name.

    From a *package* (``__init__.py``), ``level=1`` refers to the package
    itself; from a plain module, ``level=1`` refers to its parent package.
    """
    base = importer.split(".")
    strip = node.level - 1 if is_package else node.level
    if strip:
        base = base[: len(base) - strip]
    if node.module:
        base.append(node.module)
    return ".".join(base)


def imported_modules(path: Path, importer: str):
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PKG):
                    yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                yield resolve_relative(importer, is_package, node), node.lineno
            elif node.module and node.module.startswith(PKG):
                yield node.module, node.lineno


def ad5_violation(importer: str, imported: str) -> str | None:
    """AD-5 firewall verdict for one import edge, or None if clean.

    Closures internals (and the bare-package facade, layer 8) are exempt;
    any other module importing from ``slcflow.closures`` must hit the
    ``AD5_ALLOWED`` interface set exactly. A bare ``slcflow.closures``
    import is rejected too -- the checker cannot see which names it binds.
    """
    closures = f"{PKG}.closures"
    if importer == PKG or importer == closures \
            or importer.startswith(closures + "."):
        return None
    if imported != closures and not imported.startswith(closures + "."):
        return None
    if imported in AD5_ALLOWED:
        return None
    return (f"{importer} imports {imported} -- AD-5 firewall: outside "
            f"closures/ only {sorted(AD5_ALLOWED)} are importable "
            "(interfaces, never implementations)")


def violation_for(importer: str, imported: str) -> str | None:
    """Combined direction + AD-5 verdict for one import edge (the pure,
    unit-testable core; negative controls in tests/test_lint_tools.py)."""
    my_layer = layer_of(importer)
    their_layer = layer_of(imported)
    if my_layer is None:
        return f"unlayered module {importer!r}: add to LAYERS"
    if their_layer is None:
        return f"import of unlayered {imported!r}"
    if their_layer > my_layer:
        return (f"{importer} (layer {my_layer}) imports {imported} "
                f"(layer {their_layer}) -- wrong direction")
    return ad5_violation(importer, imported)


def main() -> int:
    violations = []
    for path in sorted((ROOT / PKG).rglob("*.py")):
        importer = module_name(path)
        if importer.startswith(EXEMPT_PREFIXES):
            continue
        if layer_of(importer) is None:
            violations.append((path, 0, f"unlayered module {importer!r}: add to LAYERS"))
            continue
        for imported, lineno in imported_modules(path, importer):
            msg = violation_for(importer, imported)
            if msg:
                violations.append((path, lineno, msg))
    for path, lineno, msg in violations:
        print(f"{path}:{lineno}: {msg}")
    if not violations:
        print("import-direction check: OK")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())