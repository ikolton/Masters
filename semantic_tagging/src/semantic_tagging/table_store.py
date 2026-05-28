import json
from pathlib import Path
from typing import Any, Iterable


class TableStore:
    def write_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def read_records(self, path: Path) -> list[dict[str, Any]]:
        raise NotImplementedError

    def exists(self, path: Path) -> bool:
        raise NotImplementedError


class ParquetTableStore(TableStore):
    def write_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        writer = _resolve_parquet_writer()
        writer(path, records)

    def read_records(self, path: Path) -> list[dict[str, Any]]:
        reader = _resolve_parquet_reader()
        return reader(path)

    def exists(self, path: Path) -> bool:
        return path.exists()


class MemoryTableStore(TableStore):
    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}

    def write_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        self._tables[str(path)] = [dict(record) for record in records]

    def read_records(self, path: Path) -> list[dict[str, Any]]:
        return [dict(record) for record in self._tables[str(path)]]

    def exists(self, path: Path) -> bool:
        return str(path) in self._tables


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_parquet_writer():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet writing requires pyarrow. Install the parquet dependency in the semantic_tagging environment."
        ) from exc

    def _writer(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            table = pa.table({"_empty": pa.array([], type=pa.string())})
        else:
            normalized = _normalize_records(records)
            table = pa.Table.from_pylist(normalized)
        pq.write_table(table, path)

    return _writer


def _resolve_parquet_reader():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet reading requires pyarrow. Install the parquet dependency in the semantic_tagging environment."
        ) from exc

    def _reader(path: Path) -> list[dict[str, Any]]:
        table = pq.read_table(path)
        rows = table.to_pylist()
        if len(rows) == 1 and set(rows[0].keys()) == {"_empty"} and rows[0]["_empty"] is None:
            return []
        return rows

    return _reader


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append(_normalize_value(record))
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value
