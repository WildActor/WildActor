from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import ActorRecord


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_actor_records(path: str | Path) -> Iterator[ActorRecord]:
    for item in read_jsonl(path):
        yield ActorRecord.from_json(item)


def write_actor_records(path: str | Path, rows: Iterable[ActorRecord]) -> None:
    write_jsonl(path, (row.to_json() for row in rows))

