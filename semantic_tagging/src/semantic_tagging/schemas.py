import json
from pathlib import Path
from typing import Any

import jsonschema


def load_json_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(str(exc)) from exc
