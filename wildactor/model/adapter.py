from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


REQUIRED_LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _float_value(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def adapter_paths(config: dict[str, Any]) -> dict[str, str]:
    adapter = config.get("adapter", {})
    paths = {"lora": adapter.get("lora") or config.get("model", {}).get("lora")}
    return {key: str(value) for key, value in paths.items() if value not in (None, "", "null")}


def adapter_status(config: dict[str, Any]) -> str:
    configured = adapter_paths(config)
    if configured:
        missing = [path for path in configured.values() if not Path(path).exists()]
        return "missing_files" if missing else "ready"
    return "not_configured"


def validate_adapter_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    adapter_cfg = config.get("wildactor_adapter", {})
    i_rope = adapter_cfg.get("i_rope", {})
    if i_rope.get("face_temporal_offset") != 4:
        errors.append("wildactor_adapter.i_rope.face_temporal_offset should be 4")
    if i_rope.get("body_temporal_offset") != 128:
        errors.append("wildactor_adapter.i_rope.body_temporal_offset should be 128")
    if i_rope.get("reference_frame_stride", 4) != 4:
        errors.append("wildactor_adapter.i_rope.reference_frame_stride should be 4")

    target_modules = ",".join(
        _as_list(config.get("adapter", {}).get("lora_target_modules") or config.get("model", {}).get("lora_target_modules"))
    )
    for module in REQUIRED_LORA_TARGETS:
        if module not in target_modules:
            errors.append(f"adapter.lora_target_modules is missing {module}")
    if bool(adapter_cfg.get("reference_only_lora", True)) and "self_attn.o" in target_modules:
        errors.append("reference-only LoRA should target self_attn.q/k/v, not self_attn.o")

    for name, path in adapter_paths(config).items():
        if not Path(path).exists():
            errors.append(f"adapter {name} does not exist: {path}")
    return errors


def _load_lora_state_dict(path: str) -> dict[str, Any]:
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to load .safetensors WildActor adapter weights.") from exc
        return dict(load_file(path, device="cpu"))

    import torch

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "module", "model"):
            value = state.get(key)
            if isinstance(value, dict):
                return value
        return state
    raise RuntimeError(f"Unsupported LoRA checkpoint format: {path}")


def _lora_name_dict(state_dict: dict[str, Any]) -> dict[str, tuple[str, str]]:
    name_dict = {}
    for key in state_dict:
        if ".lora_B." not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        if keys and keys[0] == "diffusion_model":
            keys.pop(0)
        keys.pop(-1)
        target_name = ".".join(keys)
        lora_a = key.replace(".lora_B.", ".lora_A.")
        if lora_a in state_dict:
            name_dict[target_name] = (lora_a, key)
    return name_dict


def _get_parent_module(root: Any, module_name: str) -> tuple[Any, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _make_reference_only_lora_linear(base_layer: Any, lora_a: Any, lora_b: Any, scale: float) -> Any:
    import torch

    class _ReferenceOnlyLoRALinear(torch.nn.Module):
        def __init__(self, base_layer: torch.nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float):
            super().__init__()
            self.base_layer = base_layer
            self.scale = float(scale)
            if lora_a.ndim == 4:
                lora_a = lora_a.squeeze(3).squeeze(2)
            if lora_b.ndim == 4:
                lora_b = lora_b.squeeze(3).squeeze(2)
            self.register_buffer("lora_A", lora_a.detach().cpu())
            self.register_buffer("lora_B", lora_b.detach().cpu())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base = self.base_layer(x)
            lora_a = self.lora_A.to(device=x.device, dtype=x.dtype)
            lora_b = self.lora_B.to(device=x.device, dtype=x.dtype)
            return base + self.scale * torch.nn.functional.linear(torch.nn.functional.linear(x, lora_a), lora_b)

    return _ReferenceOnlyLoRALinear(base_layer, lora_a, lora_b, scale)


def _install_reference_only_lora(module: Any, path: str, target_modules: list[str], scale: float) -> int:
    import torch

    state_dict = _load_lora_state_dict(path)
    lora_map = _lora_name_dict(state_dict)
    installed = 0
    for name, child in list(module.named_modules()):
        if not any(name.endswith(target) for target in target_modules):
            continue
        if name not in lora_map:
            continue
        lora_a_key, lora_b_key = lora_map[name]
        base_layer = child.base_layer if hasattr(child, "base_layer") else child
        if not isinstance(base_layer, torch.nn.Linear):
            continue
        parent, child_name = _get_parent_module(module, name)
        wrapped = _make_reference_only_lora_linear(base_layer, state_dict[lora_a_key], state_dict[lora_b_key], scale)
        setattr(parent, child_name, wrapped)
        installed += 1
    return installed


def _load_reference_only_lora(pipe: Any, config: dict[str, Any], path: str, scale: float) -> int:
    module_name = config.get("adapter", {}).get("lora_module", "dit")
    if not hasattr(pipe, module_name):
        raise RuntimeError(f"Pipeline does not expose adapter.lora_module '{module_name}'.")
    target_modules = _as_list(config.get("adapter", {}).get("lora_target_modules")) or list(REQUIRED_LORA_TARGETS)
    return _install_reference_only_lora(getattr(pipe, module_name), path, target_modules, scale)


def apply_wildactor_adapter(pipe: Any, config: dict[str, Any]) -> None:
    paths = adapter_paths(config)
    if not paths:
        return
    scale = _float_value(config.get("adapter", {}).get("lora_scale"), 1.0)
    reference_only = bool(config.get("wildactor_adapter", {}).get("reference_only_lora", True))
    loaded = False
    for path in paths.values():
        if reference_only:
            installed = _load_reference_only_lora(pipe, config, path, scale)
            if installed <= 0:
                raise RuntimeError(
                    "No WildActor reference-only LoRA tensors matched adapter.lora_target_modules. "
                    "Check that the adapter keys target self_attn.q/k/v/o."
                )
            loaded = True
            continue
        if hasattr(pipe, "load_lora"):
            signature = inspect.signature(pipe.load_lora)
            if "module" in signature.parameters:
                module_name = config.get("adapter", {}).get("lora_module", "dit")
                if not hasattr(pipe, module_name):
                    raise RuntimeError(f"Pipeline does not expose adapter.lora_module '{module_name}'.")
                pipe.load_lora(getattr(pipe, module_name), path, alpha=scale)
            elif "lora_alpha" in signature.parameters:
                pipe.load_lora(path, lora_alpha=scale)
            elif "alpha" in signature.parameters:
                pipe.load_lora(path, alpha=scale)
            else:
                pipe.load_lora(path)
            loaded = True
        elif hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(path)
            loaded = True
        elif hasattr(pipe, "load_adapter"):
            pipe.load_adapter(path)
            loaded = True
        else:
            raise RuntimeError("Pipeline does not expose load_lora/load_lora_weights/load_adapter for WildActor adapter weights.")
    if loaded and hasattr(pipe, "set_lora_scale"):
        pipe.set_lora_scale(scale)
