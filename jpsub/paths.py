# -*- coding: utf-8 -*-
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("JPSUB_MODEL_DIR", str(ROOT / "models")))
ASSETS = Path(__file__).resolve().parent / "assets"
BACKGROUNDS = ASSETS / "backgrounds"
USER_DATA = Path(os.environ.get("JPSUB_DATA_DIR", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "YishengTranslator" / "UserData")))
CACHE = USER_DATA / "cache"
HISTORY = USER_DATA / "history"
CONFIG_PATH = USER_DATA / "config.json"
LOG_PATH = USER_DATA / "jpsub.log"


def ensure_dirs() -> None:
    # Packaged assets can live under a read-only installation. A missing model
    # is a readiness error, not a reason to create an empty installation folder.
    for d in (USER_DATA, CACHE, HISTORY):
        d.mkdir(parents=True, exist_ok=True)
