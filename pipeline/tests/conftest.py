import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DDL_BUILD = _REPO_ROOT / "ddl" / "build"

for p in (str(_REPO_ROOT), str(_DDL_BUILD)):
    if p not in sys.path:
        sys.path.insert(0, p)
