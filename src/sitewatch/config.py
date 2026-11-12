from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "defaults.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["_path"] = str(config_path)
    data["_root"] = str(ROOT)
    return data


def label_of(config: dict, class_name: str) -> str:
    key = class_name.strip().lower()
    return config.get("labels", {}).get(key, class_name)


def level_label(config: dict, level: str) -> str:
    return config.get("level_labels", {}).get(level, level)
