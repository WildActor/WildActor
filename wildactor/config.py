from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") or value.startswith(("'", '"')):
        return ast.literal_eval(value.replace("null", "None"))
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rows.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(rows):
            return {}, index

        is_list = rows[index][1].startswith("- ")
        if is_list:
            items: list[Any] = []
            while index < len(rows):
                current_indent, text = rows[index]
                if current_indent < indent or not text.startswith("- "):
                    break
                if current_indent > indent:
                    child, index = parse_block(index, current_indent)
                    if items:
                        items[-1] = child
                    continue
                value = text[2:].strip()
                if value == "":
                    child, index = parse_block(index + 1, indent + 2)
                    items.append(child)
                else:
                    items.append(_parse_scalar(value))
                    index += 1
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(rows):
            current_indent, text = rows[index]
            if current_indent < indent or text.startswith("- "):
                break
            if current_indent > indent:
                index += 1
                continue
            if ":" not in text:
                index += 1
                continue
            key, value = text.split(":", 1)
            if value.strip() == "":
                if index + 1 < len(rows) and rows[index + 1][0] > current_indent:
                    child, index = parse_block(index + 1, rows[index + 1][0])
                    mapping[key.strip()] = child
                else:
                    mapping[key.strip()] = {}
                    index += 1
            elif value.strip() in {"|", ">"}:
                style = value.strip()
                block_index = index + 1
                if block_index >= len(rows) or rows[block_index][0] <= current_indent:
                    mapping[key.strip()] = ""
                    index += 1
                    continue
                block_indent = rows[block_index][0]
                lines: list[str] = []
                while block_index < len(rows) and rows[block_index][0] > current_indent:
                    line_indent, line_text = rows[block_index]
                    lines.append(" " * max(line_indent - block_indent, 0) + line_text)
                    block_index += 1
                separator = "\n" if style == "|" else " "
                trailing = "\n" if style == "|" and lines else ""
                mapping[key.strip()] = separator.join(lines) + trailing
                index = block_index
            else:
                mapping[key.strip()] = _parse_scalar(value)
                index += 1
        return mapping, index

    parsed, _ = parse_block(0, rows[0][0] if rows else 0)
    return parsed if isinstance(parsed, dict) else {}


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expr, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "include" in cfg and len(cfg) == 1:
            return load_config(path.parent / cfg["include"])
        return _expand_env(cfg)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        cfg = _load_simple_yaml(path)
        if "include" in cfg and len(cfg) == 1:
            return load_config(path.parent / cfg["include"])
        return _expand_env(cfg)

    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if isinstance(cfg, dict) and "include" in cfg and len(cfg) == 1:
        return load_config(path.parent / cfg["include"])
    return _expand_env(cfg)
