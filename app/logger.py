from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import get_settings


def append_route_log(entry: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    log_path = Path(get_settings().log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")