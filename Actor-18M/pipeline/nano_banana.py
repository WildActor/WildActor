from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


DEFAULT_OFFICIAL_MODEL = "gemini-3.1-flash-image"
OFFICIAL_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


THREE_VIEW_PROMPT = """Based on the provided reference images, generate a standard three-view character sheet of the person.

Requirements:
1. Composition: a single image showing the person's full-body front, side, and back views arranged side-by-side.
2. Background: a completely plain, solid white background with no shadows or textures.
3. Pose and expression: neutral standing pose with hands naturally at the sides, holding no objects, neutral facial expression.
4. Fidelity: accurately preserve body shape, hairstyle, skin tone, and clothing from the reference images.
5. Clean output: no watermark, text, logo, or unrelated artifacts."""


FACE_CLOSEUP_PROMPT = """Using the provided reference images, generate a single high-fidelity photorealistic close-up portrait of the person.

Requirements:
1. Preserve the person's exact facial features, face shape, hairstyle, hair color, and skin tone.
2. Professional headshot composition, focused on face and shoulders, looking directly at camera when possible.
3. Neutral expression.
4. Clean studio background with soft professional lighting.
5. No watermark, text, logo, or unrelated objects."""


@dataclass(frozen=True)
class NanoBananaConfig:
    model: str = DEFAULT_OFFICIAL_MODEL
    api_key: str | None = None
    retries: int = 2
    retry_sleep_sec: float = 5.0


def _guess_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def _load_pil_image() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for Nano-Banana image generation.") from exc
    return Image


def _is_pil_image(image: Any) -> bool:
    return image.__class__.__module__.startswith("PIL.")


def _pil_to_jpeg_bytes(image: Any) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def image_to_part(image: str | Path | Any) -> dict[str, str]:
    if _is_pil_image(image):
        data = base64.b64encode(_pil_to_jpeg_bytes(image)).decode("utf-8")
        return {"type": "image", "mime_type": "image/jpeg", "data": data}

    path = Path(image)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "mime_type": _guess_mime(path), "data": data}


def extract_frame(video_path: str | Path, frame_index: int) -> Any | None:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required when Nano-Banana references must be extracted from videos.") from exc
    Image = _load_pil_image()

    video_path = str(video_path)
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _best_frame_index(items: list[dict[str, Any]] | None, view: str) -> int | None:
    if not items:
        return None
    best = max(items, key=lambda item: item.get("scores", {}).get(view, 0))
    value = best.get("frame_index")
    return int(value) if value is not None else None


def build_reference_images(row: dict[str, Any], target: str) -> list[str | Any]:
    explicit = row.get(f"{target}_reference_images") or row.get("reference_images") or row.get("source_images")
    if explicit:
        return list(explicit)

    if target == "face":
        refs = [row.get("face_closeup_img"), row.get("face_img")]
        return [ref for ref in refs if ref]

    orientation_map = row.get("orientation_map")
    video = row.get("video")
    if not orientation_map or not video:
        return []

    body_map = orientation_map.get("body", {})
    head_map = orientation_map.get("head", {})
    if target == "three_views":
        views = ("front", "side", "back")
        indices = [
            _best_frame_index(body_map.get(view), view) or _best_frame_index(head_map.get(view), view)
            for view in views
        ]
    else:
        indices = [
            _best_frame_index(head_map.get("front"), "front") or _best_frame_index(body_map.get("front"), "front"),
            _best_frame_index(head_map.get("side"), "side"),
            _best_frame_index(head_map.get("back"), "back"),
        ]
        if all(index is None for index in indices):
            indices = [_best_frame_index(body_map.get("side"), "side") or _best_frame_index(body_map.get("back"), "back")]

    images: list[str | Any] = []
    for index in sorted({idx for idx in indices if idx is not None}):
        image = extract_frame(video, index)
        if image is not None:
            images.append(image)
    return images


class NanoBananaClient:
    def __init__(self, config: NanoBananaConfig):
        self.config = config

    def generate(self, prompt: str, images: list[str | Path | Any]) -> Any:
        return self._generate_official(prompt, images)

    def _generate_official(self, prompt: str, images: list[str | Path | Any]) -> Any:
        Image = _load_pil_image()
        api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the official Gemini Nano-Banana API.")

        payload = {
            "model": self.config.model,
            "input": [{"type": "text", "text": prompt}] + [image_to_part(image) for image in images],
        }
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        response = self._post_json(OFFICIAL_INTERACTIONS_URL, payload, headers)
        image_block = response.get("output_image")
        if not image_block or not image_block.get("data"):
            raise RuntimeError(f"Official Nano-Banana response did not contain output_image.data: {response.keys()}")
        return Image.open(BytesIO(base64.b64decode(image_block["data"]))).convert("RGB")

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_sleep_sec)
        raise RuntimeError(f"Nano-Banana request failed after retries: {last_error}")


def _output_path(output_root: Path, subdir: str, video_or_id: str, suffix: str = ".jpg") -> Path:
    stem = Path(video_or_id).stem or "sample"
    return output_root / subdir / f"{stem}{suffix}"


def process_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    output_root: str | Path,
    client: NanoBananaClient,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    output_root = Path(output_root)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    (output_root / "3views").mkdir(parents=True, exist_ok=True)
    (output_root / "face").mkdir(parents=True, exist_ok=True)

    stats = {"rows": 0, "three_views": 0, "face": 0, "skipped": 0}
    with input_jsonl.open("r", encoding="utf-8") as fin, output_jsonl.open("w", encoding="utf-8") as fout:
        iterator = tqdm(fin, desc="Nano-Banana")
        for line in iterator:
            if limit is not None and stats["rows"] >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1

            row_id = row.get("video") or row.get("id") or f"row_{stats['rows']:08d}"
            for target, prompt, subdir, field in (
                ("three_views", THREE_VIEW_PROMPT, "3views", "three_views_img"),
                ("face", FACE_CLOSEUP_PROMPT, "face", "face_closeup_img"),
            ):
                if row.get(field) and os.path.exists(row[field]):
                    continue
                refs = build_reference_images(row, target)
                if not refs:
                    stats["skipped"] += 1
                    continue
                save_path = _output_path(output_root, subdir, row_id)
                if not dry_run:
                    generated = client.generate(prompt, refs)
                    generated.save(save_path, quality=95)
                row[field] = str(save_path)
                stats[target] += 1

            row.setdefault("metadata", {})
            if isinstance(row["metadata"], dict):
                row["metadata"].setdefault("nano_banana", {"provider": "official", "model": client.config.model})
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Actor-18M-C references with the official Gemini image API.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", default=os.environ.get("NANO_BANANA_MODEL", DEFAULT_OFFICIAL_MODEL))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = NanoBananaClient(
        NanoBananaConfig(
            model=args.model,
        )
    )
    stats = process_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        output_root=args.output_root,
        client=client,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
