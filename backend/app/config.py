"""Settings from environment variables or a .env file in the project root."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("POTSPOT_DATA", str(ROOT / "data" / "app")))
    merge_radius_m: float = float(os.getenv("MERGE_RADIUS_M", "8"))
    sample_fps: float = float(os.getenv("SAMPLE_FPS", "15"))
    camera_height_m: float = float(os.getenv("CAMERA_HEIGHT_M", "1.2"))
    horizon_frac: float = float(os.getenv("HORIZON_FRAC", "0.45"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "1500"))
    offline: bool = os.getenv("POTSPOT_OFFLINE", "0") == "1"   # no Nominatim or map tiles
    contact: str = os.getenv("NOMINATIM_CONTACT", "https://github.com/sidxcodes/POTSPOT")

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'potspot.db').as_posix()}"

    @property
    def trips_dir(self) -> Path:
        return self.data_dir / "trips"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


settings = Settings()
