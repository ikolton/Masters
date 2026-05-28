import sys
from pathlib import Path


def bootstrap() -> Path:
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return project_root
