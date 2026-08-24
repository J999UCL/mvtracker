"""Small manifest readers for the source-specific data launchers."""

from __future__ import annotations

import json
from pathlib import Path


def load_syn4d_manifest(path: str | Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in payload.get("sequences", []):
        if not isinstance(item, dict):
            continue
        environment = str(item.get("environment", ""))
        sequence = str(item.get("sequence", ""))
        split = str(item.get("split", "train"))
        key = environment, sequence, split
        if not environment or not sequence or split not in {"train", "validation"} or key in seen:
            continue
        seen.add(key)
        rows.append(
            {"environment": environment, "sequence": sequence, "split": split}
        )
    return payload, rows


def load_mvkubric_manifest(path: str | Path) -> tuple[dict[str, object], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scene_ids: list[str] = []
    seen: set[str] = set()
    for scene in payload.get("train_scene_ids", []):
        scene_id = str(scene)
        if not scene_id.isdigit() or scene_id in seen:
            continue
        seen.add(scene_id)
        scene_ids.append(scene_id)
    for item in payload.get("train_ranges", []):
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        excluded = {str(value) for value in item.get("exclude", [])}
        for value in range(start, end + 1):
            scene_id = str(value)
            if scene_id not in excluded and scene_id not in seen:
                seen.add(scene_id)
                scene_ids.append(scene_id)
    return payload, sorted(scene_ids, key=int)


__all__ = ["load_mvkubric_manifest", "load_syn4d_manifest"]
