from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from .adapter import adapter_paths, adapter_status, apply_wildactor_adapter
from .irope import apply_wildactor_i_rope, patch_backend_for_wildactor


DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def model_identifier(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    return model_cfg.get("model_root") or model_cfg.get("model_id") or DEFAULT_MODEL_ID


def model_config_kwargs(config: dict[str, Any], pattern: str, offload_device: str) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    model_root = model_cfg.get("model_root")
    if model_root:
        root = Path(model_root)
        if root.exists() and root.parent.name:
            local_model_path = str(root.parent.parent)
            model_id = f"{root.parent.name}/{root.name}"
            return {
                "model_id": model_id,
                "origin_file_pattern": pattern,
                "local_model_path": local_model_path,
                "skip_download": True,
                "offload_device": offload_device,
            }
    return {
        "model_id": model_cfg.get("model_id", DEFAULT_MODEL_ID),
        "origin_file_pattern": pattern,
        "offload_device": offload_device,
    }


def prepare_backend_paths(config: dict[str, Any]) -> list[str]:
    added = []
    for raw_path in _as_list(config.get("backend", {}).get("extra_python_paths")):
        if not raw_path:
            continue
        path = str(Path(raw_path).expanduser().resolve())
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added


def safe_find_spec(module_name: str) -> tuple[bool, str | None]:
    try:
        return importlib.util.find_spec(module_name) is not None, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def inspect_runtime(config: dict[str, Any], check_backend: bool = False) -> dict[str, Any]:
    added_paths = prepare_backend_paths(config)
    backend = config.get("backend", {})
    pipeline_module = backend.get("pipeline_module", "diffsynth.pipelines.wan_video_new")
    save_module = backend.get("save_video_module", "diffsynth")
    pipeline_available, pipeline_error = safe_find_spec(pipeline_module)
    save_available, save_error = safe_find_spec(save_module)
    result: dict[str, Any] = {
        "model": model_identifier(config),
        "adapter_status": adapter_status(config),
        "adapter_paths": adapter_paths(config),
        "backend_extra_python_paths_added": added_paths,
        "pipeline_module": pipeline_module,
        "save_video_module": save_module,
        "pipeline_module_available": pipeline_available,
        "save_video_module_available": save_available,
    }
    if pipeline_error:
        result["pipeline_module_error"] = pipeline_error
    if save_error:
        result["save_video_module_error"] = save_error
    if check_backend:
        try:
            module = importlib.import_module(pipeline_module)
            importlib.import_module(save_module)
            pipeline_class = getattr(module, backend.get("pipeline_class", "WanVideoPipeline"))
            model_config_class = getattr(module, backend.get("model_config_class", "ModelConfig"))
            call_signature = inspect.signature(pipeline_class.__call__)
            call_params = set(call_signature.parameters)
            result["backend_import"] = "ok"
            result["pipeline_class"] = pipeline_class.__name__
            result["model_config_class"] = model_config_class.__name__
            result["pipeline_call_support"] = {
                "multi_views_image": "multi_views_image" in call_params,
                "reference_image": "reference_image" in call_params,
                "input_image": "input_image" in call_params,
            }
            if hasattr(pipeline_class, "load_lora"):
                result["load_lora_signature"] = str(inspect.signature(pipeline_class.load_lora))
            result["wildactor_backend"] = patch_backend_for_wildactor(module, config)
        except Exception as exc:
            result["backend_import"] = "failed"
            result["backend_error"] = f"{type(exc).__name__}: {exc}"
    return result


def dtype_from_name(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_wan22_pipeline(config: dict[str, Any]):
    prepare_backend_paths(config)
    backend_cfg = config.get("backend", {})
    model_cfg = config["model"]
    module = importlib.import_module(backend_cfg.get("pipeline_module", "diffsynth.pipelines.wan_video_new"))
    patch_result = patch_backend_for_wildactor(module, config)
    i_rope_result = patch_result.get("i_rope", {})
    embedder_result = patch_result.get("reference_embedder", {})
    if not i_rope_result.get("patched"):
        raise RuntimeError(f"Configured backend does not expose a patchable WildActor I-RoPE path: {i_rope_result.get('reason')}")
    if not embedder_result.get("patched"):
        raise RuntimeError(f"Configured backend does not expose a patchable WildActor reference embedder: {embedder_result.get('reason')}")
    model_config_class = getattr(module, backend_cfg.get("model_config_class", "ModelConfig"))
    pipeline_class = getattr(module, backend_cfg.get("pipeline_class", "WanVideoPipeline"))
    offload_device = model_cfg.get("offload_device", "cpu")
    patterns = model_cfg.get("component_patterns", {})
    if not patterns:
        patterns = {
            "diffusion": "diffusion_pytorch_model*.safetensors",
            "text_encoder": "models_t5_umt5-xxl-enc-bf16.pth",
            "vae": "Wan2.1_VAE.pth",
        }
    pipe = pipeline_class.from_pretrained(
        torch_dtype=dtype_from_name(model_cfg.get("torch_dtype", "bfloat16")),
        device=model_cfg.get("device", "cuda"),
        model_configs=[model_config_class(**model_config_kwargs(config, pattern, offload_device)) for pattern in patterns.values()],
    )

    if hasattr(pipe, "enable_vram_management") and model_cfg.get("enable_vram_management", True):
        pipe.enable_vram_management()
    apply_wildactor_i_rope(pipe, config)
    apply_wildactor_adapter(pipe, config)
    return pipe
