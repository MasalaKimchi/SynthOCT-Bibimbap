"""Make the `synthoct` package importable from the flat baseline-format scripts.

The official SynthOCT baseline is a set of flat scripts you run directly
(``python Orchestrator.py``).  This folder mirrors that shape, but the real
logic lives in the installable ``synthoct`` package under ``../src``.  Importing
this module first (``import _bootstrap``) reproduces the baseline's
"just run the script" ergonomics:

* If ``synthoct`` is already importable (e.g. ``pip install -e .`` was run, or
  the active environment is the pinned ``synthoct-py312`` conda env), this is a
  no-op and the installed package is used unchanged.
* Otherwise the repository's ``src`` directory is prepended to ``sys.path`` so
  the scripts work in a fresh checkout with no install step.

This module is intentionally dependency-free (standard library only) so it can
run before any third-party import.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root is the parent of this ``baseline_format`` directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"


def ensure_synthoct_importable() -> Path | None:
    """Ensure ``import synthoct`` works; return the path added, if any."""
    try:
        import synthoct  # noqa: F401  (import for its side effect: availability)

        return None
    except ModuleNotFoundError:
        pass

    if _SRC.is_dir():
        src = str(_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        return _SRC

    raise ModuleNotFoundError(
        "Could not import 'synthoct'. Expected the package under "
        f"{_SRC!s} or an installed 'synthoct' distribution. Run "
        "`python -m pip install -e .` from the repository root, or run these "
        "scripts from inside a checkout that contains ./src/synthoct."
    )


# Run on import so a bare `import _bootstrap` is enough to wire the path.
ADDED_PATH = ensure_synthoct_importable()
