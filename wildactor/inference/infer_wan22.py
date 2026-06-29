from __future__ import annotations

import argparse
import inspect
import importlib
import json
import time
from pathlib import Path
from typing import Any

from wildactor.config import load_config
from wildactor.model import (
    adapter_paths,
    adapter_status,
    apply_wildactor_i_rope,
    inspect_runtime,
    load_wan22_pipeline,
    model_identifier,
    reference_type_sequence,
    validate_adapter_config,
)


def load_request(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_optional_image(path: str | None):
    if not path:
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def open_image_list(paths: list[str]) -> list[Any]:
    return [image for image in (open_optional_image(path) for path in paths) if image is not None]


def resolve_path(path: str | None, base_dir: Path) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def resolve_path_list(paths: Any, base_dir: Path) -> list[str]:
    if paths in (None, ""):
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [resolved for path in paths if (resolved := resolve_path(str(path), base_dir))]


def normalize_request_paths(request: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized = dict(request)
    normalized["face_refs"] = resolve_path_list(request.get("face_refs"), base_dir)
    normalized["body_refs"] = resolve_path_list(request.get("body_refs"), base_dir)
    normalized["three_view_ref"] = resolve_path(request.get("three_view_ref"), base_dir)
    return normalized


def build_multi_view_inputs(request: dict[str, Any]) -> dict[str, Any | None]:
    face_images = open_image_list(request.get("face_refs") or [])
    body_images = open_image_list(request.get("body_refs") or [])
    three_view_image = open_optional_image(request.get("three_view_ref"))
    reference_types = reference_type_sequence(len(face_images), len(body_images), three_view_image is not None)
    body_condition_images = list(body_images)
    if three_view_image is not None:
        body_condition_images.append(three_view_image)

    return {
        "face_refs": face_images,
        "body_refs": body_images,
        "reference_types": reference_types,
        "face_img": face_images,
        "body_img": body_images,
        "face_closeup_img": face_images,
        "3views_img": three_view_image,
        "three_views_img": body_condition_images,
    }


def first_reference_image(multi_view_inputs: dict[str, Any | None]):
    for value in (
        multi_view_inputs.get("3views_img"),
        multi_view_inputs.get("body_img"),
        multi_view_inputs.get("face_closeup_img"),
    ):
        if isinstance(value, list):
            if value:
                return value[0]
        elif value is not None:
            return value
    return None


def call_pipeline(pipe: Any, request: dict[str, Any], gen: dict[str, Any], multi_view_inputs: dict[str, Any | None]):
    kwargs = {
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt", gen.get("negative_prompt", "")),
        "num_frames": _int_value(request.get("num_frames", gen.get("num_frames")), 81),
        "seed": _int_value(request.get("seed", gen.get("seed")), 42),
        "tiled": _bool_value(request.get("tiled", gen.get("tiled")), True),
        "height": _int_value(request.get("height", gen.get("height")), 720),
        "width": _int_value(request.get("width", gen.get("width")), 1280),
    }
    params = set(inspect.signature(pipe.__call__).parameters)
    if hasattr(pipe, "dit"):
        pipe.dit.wildactor_reference_types = multi_view_inputs.get("reference_types")
    if hasattr(pipe, "dit2") and pipe.dit2 is not None:
        pipe.dit2.wildactor_reference_types = multi_view_inputs.get("reference_types")
    if "multi_views_image" in params:
        kwargs["multi_views_image"] = multi_view_inputs
    elif "reference_image" in params:
        kwargs["reference_image"] = first_reference_image(multi_view_inputs)
    elif "input_image" in params:
        kwargs["input_image"] = first_reference_image(multi_view_inputs)
    return pipe(**kwargs)


def _int_value(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _bool_value(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}

def validate_request(config: dict[str, Any], request: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not request.get("prompt"):
        errors.append("request.prompt is required")
    if not request.get("face_refs"):
        errors.append("request.face_refs must contain at least one image")
    if not request.get("body_refs") and not request.get("three_view_ref"):
        errors.append("request.body_refs or request.three_view_ref must contain identity references")

    for key in ("face_refs", "body_refs"):
        for image_path in request.get(key, []):
            if image_path and not Path(image_path).exists():
                errors.append(f"{key} image does not exist: {image_path}")
    three_view = request.get("three_view_ref")
    if three_view and not Path(three_view).exists():
        errors.append(f"three_view_ref image does not exist: {three_view}")

    errors.extend(validate_adapter_config(config))
    return errors


def build_run_summary(config: dict[str, Any], request: dict[str, Any], output: Path) -> dict[str, Any]:
    gen = config["generation"]
    return {
        "backend": config.get("backend", {}).get("name", "diffsynth"),
        "model": model_identifier(config),
        "adapter_status": adapter_status(config),
        "adapter_paths": adapter_paths(config),
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt", gen.get("negative_prompt", "")),
        "num_frames": _int_value(request.get("num_frames", gen.get("num_frames")), 81),
        "height": _int_value(request.get("height", gen.get("height")), 720),
        "width": _int_value(request.get("width", gen.get("width")), 1280),
        "fps": _int_value(gen.get("fps"), 15),
        "seed": _int_value(request.get("seed", gen.get("seed")), 42),
        "output": str(output),
        "references": {
            "face_refs": request.get("face_refs", []),
            "body_refs": request.get("body_refs", []),
            "three_view_ref": request.get("three_view_ref"),
        },
        "wildactor_adapter": config.get("wildactor_adapter", {}),
    }


def run_inference(config_path: str | Path, request_path: str | Path, output_path: str | None = None, dry_run: bool = False) -> Path:
    config = load_config(config_path)
    request = normalize_request_paths(load_request(request_path), Path.cwd())
    errors = validate_request(config, request)
    if errors:
        raise ValueError("Invalid inference request:\n- " + "\n- ".join(errors))
    gen = config["generation"]
    output = Path(output_path or request.get("output") or "outputs/wildactor.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = build_run_summary(config, request, output)
    sidecar = output.with_suffix(output.suffix + ".json")

    if dry_run:
        summary["status"] = "dry_run"
        sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return sidecar

    started = time.time()
    pipe = load_wan22_pipeline(config)
    multi_view_inputs = build_multi_view_inputs(request)
    apply_wildactor_i_rope(pipe, config, multi_view_inputs.get("reference_types"))
    video = call_pipeline(pipe, request, gen, multi_view_inputs)
    save_module = importlib.import_module(config.get("backend", {}).get("save_video_module", "diffsynth"))
    save_module.save_video(video, str(output), fps=_int_value(gen.get("fps"), 15), quality=_int_value(gen.get("quality"), 5))
    summary["status"] = "generated"
    summary["elapsed_sec"] = round(time.time() - started, 3)
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WildActor inference with a Wan2.2-5B/DiffSynth backend.")
    parser.add_argument("--config", default="configs/inference_wan22.yaml")
    parser.add_argument("--request", default="examples/inference_request.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate-only", action="store_true", help="Validate config/request/images without loading the video model.")
    parser.add_argument("--inspect", action="store_true", help="Print runtime/backend availability as JSON.")
    parser.add_argument("--check-backend", action="store_true", help="Import the configured backend during --inspect.")
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved run summary without loading the video model.")
    args = parser.parse_args()

    config = load_config(args.config)
    request = normalize_request_paths(load_request(args.request), Path.cwd())
    errors = validate_request(config, request)
    if errors:
        raise SystemExit("Invalid inference request:\n- " + "\n- ".join(errors))

    if args.inspect:
        print(json.dumps(inspect_runtime(config, check_backend=args.check_backend), ensure_ascii=False, indent=2))
        return
    if args.validate_only:
        print(f"inference_request: ok ({adapter_status(config)})")
        return

    output = run_inference(args.config, args.request, args.output, dry_run=args.dry_run)
    print(f"Saved {'summary' if args.dry_run else 'video'} to {output}")


if __name__ == "__main__":
    main()
