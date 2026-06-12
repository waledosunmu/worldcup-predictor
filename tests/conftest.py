"""Guard: ensure the test suite imports THIS worktree's src/wcpred, not an
editable install or a sibling checkout that parallel agents may be editing.

pyproject's [tool.pytest.ini_options] pythonpath=["src"] puts the worktree's
src on sys.path. This assertion fails loudly if anything else shadows it.
"""

from pathlib import Path

import wcpred

_HERE = Path(__file__).resolve().parents[1]


def test_imports_worktree_src():
    resolved = Path(wcpred.__file__).resolve()
    assert _HERE in resolved.parents, (
        f"wcpred resolved to {resolved}, not this worktree ({_HERE}). "
        "An editable install or sibling checkout is shadowing src/."
    )
