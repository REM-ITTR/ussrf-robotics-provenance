import json
from datetime import datetime
from typing import Any, Dict

def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["created_at_utc"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
