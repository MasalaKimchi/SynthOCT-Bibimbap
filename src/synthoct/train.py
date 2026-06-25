from __future__ import annotations

from pathlib import Path


def train_hybrid(config_path: str | Path) -> str:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return (
        "Hybrid training scaffold is configured. "
        "Next step: generate scanner-supervised synthetic pairs on Windows, then train the "
        "pretrained-CNN parameter head from this config."
    )
