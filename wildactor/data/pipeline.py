from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from tqdm import tqdm

from wildactor.config import load_config
from .io import read_actor_records, write_actor_records
from .schema import ActorRecord, ReferenceImage


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return basic video metadata via ffprobe when available."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    values = result.stdout.strip().splitlines()
    if len(values) < 3:
        return {}
    width, height, duration = values[:3]
    return {
        "width": int(float(width)),
        "height": int(float(height)),
        "duration_sec": float(duration) if duration != "N/A" else None,
    }


def coarse_filter(record: ActorRecord, cfg: dict[str, Any]) -> tuple[bool, str]:
    filters = cfg.get("filters", {})
    if record.duration_sec is not None and record.duration_sec < filters.get("min_duration_sec", 0):
        return False, "short_duration"
    if record.width is not None and record.width < filters.get("min_width", 0):
        return False, "small_width"
    if record.height is not None and record.height < filters.get("min_height", 0):
        return False, "small_height"
    if filters.get("require_single_person", False) and record.metadata.get("num_persons") not in (None, 1):
        return False, "not_single_person"

    # Sparse 1 fps identity stability can be supplied as a precomputed score.
    face_similarity = record.metadata.get("face_similarity")
    if face_similarity is not None and face_similarity < filters.get("face_similarity_threshold", 0.4):
        return False, "low_face_similarity"
    return True, "ok"


def fine_filter(record: ActorRecord, cfg: dict[str, Any]) -> tuple[bool, str]:
    filters = cfg.get("filters", {})

    # Fine filtering can consume track stability and frame-level consistency scores.
    track_ratio = record.metadata.get("track_ratio")
    if track_ratio is not None and track_ratio < filters.get("min_track_ratio", 0.6):
        return False, "low_track_ratio"

    clip_similarity = record.metadata.get("clip_similarity")
    if clip_similarity is not None and clip_similarity < filters.get("clip_similarity_threshold", 0.45):
        return False, "low_clip_similarity"

    return True, "ok"


def attach_basic_metadata(record: ActorRecord, cfg: dict[str, Any]) -> ActorRecord:
    if record.width and record.height and record.duration_sec:
        return record
    meta = probe_video(record.video)
    record.width = record.width or meta.get("width")
    record.height = record.height or meta.get("height")
    record.duration_sec = record.duration_sec or meta.get("duration_sec")
    return record


def normalize_existing_refs(record: ActorRecord) -> ActorRecord:
    """Keep caller-provided references and normalize common top-level fields."""

    if record.refs:
        return record

    top_level_refs = []
    for key, region in (
        ("face_img", "face"),
        ("face_closeup_img", "face"),
        ("body_img", "body"),
        ("3views_img", "three_view"),
        ("three_views_img", "three_view"),
    ):
        value = record.metadata.get(key)
        if isinstance(value, str):
            top_level_refs.append(ReferenceImage(path=value, region=region))
    record.refs = top_level_refs
    return record


def reference_source(ref: ReferenceImage) -> str:
    return str(ref.metadata.get("source") or ref.metadata.get("stage") or "")


def build_subset_records(records: list[ActorRecord]) -> dict[str, list[ActorRecord]]:
    any_view = []
    attr_aug = []
    canonical = []

    for record in records:
        normalized = normalize_existing_refs(record)
        ref_sources = {reference_source(ref) for ref in normalized.refs}
        subset = normalized.metadata.get("subset")
        if subset in {"A", "actor18m_a"} or any(src in {"view_aug", "self_crop", "subset_a"} for src in ref_sources) or not ref_sources:
            any_view.append(normalized)
        if subset in {"B", "actor18m_b"} or any(src in {"attr_aug", "subset_b"} for src in ref_sources):
            attr_aug.append(normalized)
        if subset in {"C", "actor18m_c"} or any(ref.region == "three_view" or ref.view in {"front_side_back", "canonical"} or reference_source(ref) in {"canonical", "subset_c"} for ref in normalized.refs):
            canonical.append(normalized)

    return {
        "actor18m_a_any_views.jsonl": any_view,
        "actor18m_b_attribute_aug.jsonl": attr_aug,
        "actor18m_c_canonical_3views.jsonl": canonical,
        "actor18m_all.jsonl": records,
    }


def run_pipeline(config_path: str | Path, dry_run: bool = False) -> dict[str, int]:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output_dir"])
    accepted: list[ActorRecord] = []
    rejected: dict[str, int] = {}

    input_jsonl = cfg.get("input_jsonl")
    if not input_jsonl:
        raise KeyError("input_jsonl is required")
    rows = list(read_actor_records(input_jsonl))
    for record in tqdm(rows, desc="Actor-18M pipeline"):
        record = attach_basic_metadata(record, cfg)
        ok, reason = coarse_filter(record, cfg)
        if ok:
            ok, reason = fine_filter(record, cfg)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        record.metadata.setdefault(
            "pipeline",
            {
                "coarse_filter": "ArcFace identity stability at 1 fps when face_similarity is provided",
                "fine_filter": "CoTracker track stability and CLIP consistency at 8 fps when scores are provided",
                "references": "RetinaFace+BiSeNet for face and YOLO-World+SAM2 for body when external annotations are provided",
            },
        )
        accepted.append(record)

    if not dry_run:
        for filename, subset in build_subset_records(accepted).items():
            write_actor_records(output_dir / filename, subset)

    return {"accepted": len(accepted), **{f"rejected_{k}": v for k, v in rejected.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public Actor-18M JSONL files.")
    parser.add_argument("--config", default="configs/actor18m_pipeline.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args()

    stats = run_pipeline(args.config, dry_run=args.dry_run)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
