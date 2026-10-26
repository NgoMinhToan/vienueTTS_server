import sys
from pathlib import Path

# Ensure the parent directory of vienueTTS_server is on sys.path
# so that imports like `from vienueTTS_server...` resolve whether
# tests are run from the monorepo root or directly within vienueTTS_server.
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
