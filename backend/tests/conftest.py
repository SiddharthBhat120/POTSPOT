import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("POTSPOT_DATA", tempfile.mkdtemp(prefix="potspot-test-"))
os.environ["POTSPOT_OFFLINE"] = "1"  # never call Nominatim or tile servers from tests
