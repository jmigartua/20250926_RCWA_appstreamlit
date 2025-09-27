from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rcwa_app.domain.models import ModelConfig


def _slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip())
    safe = "-".join(filter(None, safe.split("-")))
    return safe.lower() or "preset"


class LocalPresetStore:
    """Filesystem-based preset storage (JSON), schema-version aware.

    Presets are stored in ``<base_dir>/<slug>.json``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.cwd() / "presets").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.json"))

    def path_for(self, name: str) -> Path:
        return self.base_dir / f"{_slugify(name)}.json"

    def save(self, name: str, cfg: ModelConfig) -> None:
        path = self.path_for(name)
        data = cfg.model_dump()
        data["schema_version"] = cfg.version
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, name: str) -> ModelConfig:
        path = self.path_for(name)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Future: migrate by data['schema_version'] if needed
        if "schema_version" in data:
            data.pop("schema_version")
        return ModelConfig.model_validate(data)

    def remove(self, name: str) -> None:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
