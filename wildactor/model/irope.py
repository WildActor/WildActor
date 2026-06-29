from __future__ import annotations

import inspect
from typing import Any


FACE_REFERENCE_TYPES = {"face", "face_ref", "face_img", "face_closeup", "face_closeup_img"}
BODY_REFERENCE_TYPES = {"body", "body_ref", "body_img", "three_view", "three_views", "3views", "3views_img", "canonical"}


def i_rope_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = (config or {}).get("wildactor_adapter", {}).get("i_rope", {})
    face_temporal_offset = int(raw.get("face_temporal_offset", 4))
    return {
        "face_temporal_offset": face_temporal_offset,
        "body_temporal_offset": int(raw.get("body_temporal_offset", 128)),
        "reference_frame_stride": int(raw.get("reference_frame_stride", face_temporal_offset)),
        "spatial_start": raw.get("spatial_start", "max_video_hw"),
    }


def reference_type_sequence(face_count: int, body_count: int, has_three_view: bool) -> list[str]:
    sequence = ["face"] * max(int(face_count), 0)
    sequence.extend(["body"] * max(int(body_count), 0))
    if has_three_view:
        sequence.append("body")
    return sequence


def normalize_reference_types(reference_types: Any, num_reference_frames: int) -> list[str]:
    if isinstance(reference_types, str):
        sequence = [item.strip() for item in reference_types.split(",") if item.strip()]
    elif reference_types is None:
        sequence = []
    else:
        sequence = [str(item) for item in reference_types if str(item)]

    if not sequence:
        sequence = ["face"] + ["body"] * max(num_reference_frames - 1, 0)
    if len(sequence) < num_reference_frames:
        sequence.extend(["body"] * (num_reference_frames - len(sequence)))
    normalized = [_normalize_reference_type(item) for item in sequence]
    if len(normalized) > num_reference_frames:
        has_face = "face" in normalized
        has_body = "body" in normalized
        compressed = []
        if has_face and num_reference_frames > 0:
            compressed.append("face")
        if has_body:
            compressed.extend(["body"] * (num_reference_frames - len(compressed)))
        if len(compressed) < num_reference_frames:
            compressed.extend(normalized[: num_reference_frames - len(compressed)])
        return compressed[:num_reference_frames]
    return normalized[:num_reference_frames]


def temporal_indices(video_frames: int, num_reference_frames: int, reference_types: Any, config: dict[str, Any] | None = None) -> tuple[list[int], list[str]]:
    cfg = i_rope_config(config)
    normalized_types = normalize_reference_types(reference_types, num_reference_frames)
    indices = []
    face_index = 0
    body_index = 0
    stride = int(cfg["reference_frame_stride"])
    for ref_type in normalized_types:
        if ref_type == "face":
            indices.append(int(video_frames) + int(cfg["face_temporal_offset"]) + face_index * stride)
            face_index += 1
        else:
            indices.append(int(video_frames) + int(cfg["body_temporal_offset"]) + body_index * stride)
            body_index += 1
    return indices, normalized_types


def build_identity_rope_freqs(
    dit: Any,
    f: int,
    h: int,
    w: int,
    f_views: int,
    h_views: int,
    w_views: int,
    device: Any,
    reference_types: Any = None,
    config: dict[str, Any] | None = None,
):
    import torch

    cfg = i_rope_config(config or {"wildactor_adapter": {"i_rope": getattr(dit, "wildactor_i_rope", {})}})
    ref_indices, normalized_types = temporal_indices(f, f_views, reference_types, {"wildactor_adapter": {"i_rope": cfg}})

    h_start = h if cfg.get("spatial_start", "max_video_hw") == "max_video_hw" else int(cfg["spatial_start"])
    w_start = w if cfg.get("spatial_start", "max_video_hw") == "max_video_hw" else int(cfg["spatial_start"])

    _ensure_freq_capacity(dit.freqs[0], max(ref_indices), "temporal")
    _ensure_freq_capacity(dit.freqs[1], h_start + h_views - 1, "height")
    _ensure_freq_capacity(dit.freqs[2], w_start + w_views - 1, "width")

    temporal_index_tensor = torch.tensor(ref_indices, dtype=torch.long)
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(device)

    freqs_views = torch.cat(
        [
            dit.freqs[0][temporal_index_tensor].view(f_views, 1, 1, -1).expand(f_views, h_views, w_views, -1),
            dit.freqs[1][h_start : h_start + h_views].view(1, h_views, 1, -1).expand(f_views, h_views, w_views, -1),
            dit.freqs[2][w_start : w_start + w_views].view(1, 1, w_views, -1).expand(f_views, h_views, w_views, -1),
        ],
        dim=-1,
    ).reshape(f_views * h_views * w_views, 1, -1).to(device)

    meta = {
        "reference_types": normalized_types,
        "temporal_indices": ref_indices,
        "face_temporal_offset": int(cfg["face_temporal_offset"]),
        "body_temporal_offset": int(cfg["body_temporal_offset"]),
        "reference_frame_stride": int(cfg["reference_frame_stride"]),
        "spatial_h_start": int(h_start),
        "spatial_w_start": int(w_start),
        "spatial_start": cfg.get("spatial_start", "max_video_hw"),
    }
    return torch.cat([freqs, freqs_views], dim=0), freqs_views, meta


def apply_wildactor_i_rope(pipe: Any, config: dict[str, Any], reference_types: Any = None) -> None:
    cfg = i_rope_config(config)
    for module_name in ("dit", "dit2"):
        dit = getattr(pipe, module_name, None)
        if dit is None:
            continue
        dit.wildactor_i_rope = cfg
        if reference_types is not None:
            dit.wildactor_reference_types = reference_types


def patch_backend_i_rope(module: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    fn = getattr(module, "model_fn_wan_video", None)
    if fn is None:
        return {"available": False, "reason": "model_fn_wan_video_not_found"}
    if getattr(fn, "_wildactor_irope_patched", False):
        return {"available": True, "patched": True, "already_patched": True}

    try:
        source = inspect.getsource(fn)
    except OSError as exc:
        return {"available": False, "reason": f"source_unavailable: {exc}"}

    start = source.find("    shift_views = 4\n")
    end_marker = "    freqs = torch.cat([freqs, freqs_views], dim=0)\n"
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        return {"available": False, "reason": "known_i_rope_block_not_found"}
    end += len(end_marker)

    replacement = """    from wildactor.model.irope import build_identity_rope_freqs

    reference_types = getattr(dit, "wildactor_reference_types", None)
    if reference_types is None:
        reference_types = kwargs.get("wildactor_reference_types")
    freqs, freqs_views, wildactor_irope_meta = build_identity_rope_freqs(
        dit=dit,
        f=f,
        h=h,
        w=w,
        f_views=f_views,
        h_views=h_views,
        w_views=w_views,
        device=x.device,
        reference_types=reference_types,
        config={"wildactor_adapter": {"i_rope": getattr(dit, "wildactor_i_rope", {})}},
    )
    dit.wildactor_irope_last_meta = wildactor_irope_meta
"""
    namespace = module.__dict__
    exec(source[:start] + replacement + source[end:], namespace)
    patched = namespace["model_fn_wan_video"]
    patched._wildactor_irope_patched = True
    patched._wildactor_irope_config = i_rope_config(config)
    module.model_fn_wan_video = patched
    return {
        "available": True,
        "patched": True,
        "face_temporal_offset": patched._wildactor_irope_config["face_temporal_offset"],
        "body_temporal_offset": patched._wildactor_irope_config["body_temporal_offset"],
        "reference_frame_stride": patched._wildactor_irope_config["reference_frame_stride"],
        "spatial_start": patched._wildactor_irope_config["spatial_start"],
    }


def patch_backend_reference_embedder(module: Any) -> dict[str, Any]:
    cls = getattr(module, "WanVideoUnit_ImageEmbedderFused_tail", None)
    if cls is None:
        return {"available": False, "reason": "reference_embedder_not_found"}
    if getattr(cls, "_wildactor_reference_embedder_patched", False):
        return {"available": True, "patched": True, "already_patched": True}

    def process(
        self,
        pipe,
        use_orientation_sequence,
        latents,
        face_closeup_img,
        three_views_img,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        import torch

        if not use_orientation_sequence:
            return {}

        pipe.load_models_to_device(self.onload_model_names)
        orientation_frames = _image_sequence(face_closeup_img) + _image_sequence(three_views_img)
        encoded_latents = []
        for frame_image in orientation_frames:
            image = pipe.preprocess_image(frame_image.resize((width, height))).transpose(0, 1)
            latent = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            encoded_latents.append(latent)

        if not encoded_latents:
            return {}

        latents_views = torch.cat(encoded_latents, dim=2)
        return {"latents": latents, "latents_views": latents_views}

    cls.process = process
    cls._wildactor_reference_embedder_patched = True
    return {"available": True, "patched": True, "list_inputs": True}


def patch_backend_for_wildactor(module: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    i_rope = patch_backend_i_rope(module, config)
    reference_embedder = patch_backend_reference_embedder(module)
    return {"i_rope": i_rope, "reference_embedder": reference_embedder}


def _normalize_reference_type(ref_type: str) -> str:
    value = ref_type.strip().lower()
    if value in FACE_REFERENCE_TYPES:
        return "face"
    if value in BODY_REFERENCE_TYPES:
        return "body"
    return "body"


def _ensure_freq_capacity(freqs: Any, index: int, axis: str) -> None:
    if index >= freqs.shape[0]:
        raise ValueError(f"I-RoPE {axis} index {index} exceeds precomputed RoPE table length {freqs.shape[0]}.")


def _image_sequence(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        result = []
        for image in images:
            result.extend(_image_sequence(image))
        return result
    return [images]
