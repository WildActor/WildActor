from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildactor.config import load_config


@dataclass(frozen=True)
class StageSpec:
    name: str
    pipeline_step: str
    script: str
    workdir: str
    env: str
    command: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    notes: str = ""


DEFAULT_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        name="opens2v_extract_human",
        pipeline_step="single-person self crop from OpenS2V masks",
        script="data/ProcessOpenS2V/step_1_extract_human.py",
        workdir="data/ProcessOpenS2V",
        env="base",
        command="{python} step_1_extract_human.py --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.opens2v_json_dir", "paths.opens2v_video_root"),
        outputs=("paths.extract_human_root",),
        notes="Uses OpenS2V mask/bbox JSON, keeps single person subjects, writes mask_person/face_img JSONL.",
    ),
    StageSpec(
        name="qwen_orientation",
        pipeline_step="Qwen3-VL orientation and invalid-subject filtering",
        script="data/ProcessOpenS2V/step_2_anno_qwenvl_direction.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_2_anno_qwenvl_direction.py --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.extract_human_root",),
        outputs=("paths.orientation_root",),
        models=("models.qwen3_vl_32b",),
        notes="Batched Qwen3-VL image-text inference: orientation, sub_angle, facing_direction, confidence.",
    ),
    StageSpec(
        name="face_crop_bisenet",
        pipeline_step="RetinaFace/InsightFace + BiSeNet face crop",
        script="data/ProcessOpenS2V/step_3_crop_face_for_opens2v.py",
        workdir="data/ProcessOpenS2V",
        env="stand_in",
        command="{python} step_3_crop_face_for_opens2v.py --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.extract_human_root",),
        outputs=("paths.face_crop_root",),
        models=("models.antelopev2", "models.bisenet"),
        notes="Uses InsightFace antelopev2 for face boxes and facexlib BiSeNet parsing for white-background closeups.",
    ),
    StageSpec(
        name="qwen_body_shot",
        pipeline_step="body visibility and framing annotation",
        script="data/ProcessOpenS2V/step_4_anno_qwenvl_full_body.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_4_anno_qwenvl_full_body.py --image_root {paths.extract_human_root} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.extract_human_root",),
        outputs=("paths.body_shot_root",),
        models=("models.qwen3_vl_32b",),
        notes="Classifies full_body, upper_body, head_shoulders, head_only, headless_body, invalid.",
    ),
    StageSpec(
        name="qwen_face_attributes",
        pipeline_step="face quality, facing direction, and composition annotation",
        script="data/ProcessOpenS2V/step_6_anno_face_dir_and_cloth.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_6_anno_face_dir_and_cloth.py --image_root {paths.face_crop_root} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.face_crop_root",),
        outputs=("paths.face_attr_root",),
        models=("models.qwen3_vl_32b",),
        notes="Labels high/low/invalid face quality, facing direction, and pure_face vs face_with_clothes.",
    ),
    StageSpec(
        name="merge_filter_opens2v",
        pipeline_step="merge self-crop annotations and filter invalid rows",
        script="data/ProcessOpenS2V/step_5_merge_fliter_jsonl_opens2v.py",
        workdir="data/ProcessOpenS2V",
        env="base",
        command="{python} step_5_merge_fliter_jsonl_opens2v.py",
        inputs=("paths.extract_human_root", "paths.orientation_root", "paths.body_shot_root", "paths.face_crop_root", "paths.face_attr_root"),
        outputs=("paths.opens2v_merged_jsonl",),
        notes="This step uses the configured crop and annotation locations.",
    ),
    StageSpec(
        name="pt2v_pose_face",
        pipeline_step="YOLO pose/face detection for PT2V and orientation evidence",
        script="data/process/1.pose_face.py",
        workdir="data/process",
        env="yolo",
        command="{python} 1.pose_face.py --root_dir {paths.pt2v_jsonl_root} --output_dir {paths.pt2v_pose_face_root} --pose_model {models.yolo_pose} --face_model {models.yolo_face} --workers {workers} --total_parts {total_machines} --current_part {machine_id}",
        inputs=("paths.pt2v_jsonl_root",),
        outputs=("paths.pt2v_pose_face_root",),
        models=("models.yolo_pose", "models.yolo_face"),
        notes="Samples frames at roughly 1 fps and stores YOLO pose_17 plus matched face boxes.",
    ),
    StageSpec(
        name="pt2v_pose_orientation",
        pipeline_step="DWPose-style head/body front-side-back scoring",
        script="data/process/2.anno_pose.py",
        workdir="data/process",
        env="base",
        command="{python} 2.anno_pose.py --root_dir {paths.pt2v_pose_face_root}",
        inputs=("paths.pt2v_pose_face_root",),
        outputs=("paths.pt2v_pose_direction_root",),
        notes="Computes head/body front, side, back scores from 17 keypoints and filters small people.",
    ),
    StageSpec(
        name="identity_cluster_sweep",
        pipeline_step="ArcFace/InsightFace identity consistency sweep",
        script="data/ProcessOpenS2V/step_10_face_id_cluster.py",
        workdir="data/ProcessOpenS2V",
        env="stand_in",
        command="{python} step_10_face_id_cluster.py",
        inputs=("paths.identity_cluster_input",),
        outputs=("paths.identity_cluster_cache",),
        models=("models.insightface_buffalo_l",),
        notes="Runs FAISS range-search over normalized InsightFace embeddings and sweeps 0.35-0.75 thresholds.",
    ),
    StageSpec(
        name="qwen_edit_prompt",
        pipeline_step="Actor-18M-B Qwen3-VL instruction generation",
        script="data/ProcessOpenS2V/step_11_step_2_gen_edit_prompt_for_flitered_data.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_11_step_2_gen_edit_prompt_for_flitered_data.py --input_jsonl {paths.qwen_edit_seed_jsonl} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.qwen_edit_seed_jsonl",),
        outputs=("paths.qwen_edit_enriched_prefix",),
        models=("models.qwen3_vl_32b",),
        notes="Generates edit_body/edit_face prompt lists for scene, outfit, style, camera, and lighting diversity.",
    ),
    StageSpec(
        name="qwen_edit_generate",
        pipeline_step="Actor-18M-B Qwen-Image-Edit generation",
        script="data/ProcessOpenS2V/step_7_qwen_edit_multi_angle/Qwen-Image-Edit-Angles/step_11_step_3_get_real_edit_pairs.py",
        workdir="data/ProcessOpenS2V/step_7_qwen_edit_multi_angle/Qwen-Image-Edit-Angles",
        env="qwen_edit",
        command="{python} step_11_step_3_get_real_edit_pairs.py --input_base {paths.qwen_edit_enriched_prefix} --machine_id {machine_id} {rapid_flag}",
        inputs=("paths.qwen_edit_enriched_prefix",),
        outputs=("paths.qwen_edit_results_root",),
        models=("models.qwen_image_edit", "models.qwen_image_edit_rapid", "models.qwen_image_edit_angles_lora"),
        notes="Uses QwenImageEditPlusPipeline, optional Rapid transformer, and writes generated body/face edit logs.",
    ),
    StageSpec(
        name="qwen_edit_evaluate",
        pipeline_step="Qwen3-VL image quality and harmony audit",
        script="data/ProcessOpenS2V/step_11_step_6_evaluate_edit_data.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_11_step_6_evaluate_edit_data.py --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.qwen_edit_results_root",),
        outputs=("paths.qwen_edit_eval_root",),
        models=("models.qwen3_vl_32b",),
        notes="Evaluates generated body/face edits for quality, harmony, proportions, visibility, and orientation.",
    ),
    StageSpec(
        name="nano_banana_generate",
        pipeline_step="Actor-18M-C official Nano-Banana character sheet and face generation",
        script="local:Actor-18M/pipeline/nano_banana.py",
        workdir="local:.",
        env="nano",
        command="{python} Actor-18M/pipeline/nano_banana.py --input-jsonl {paths.nano_seed_jsonl} --output-jsonl {paths.nano_output_jsonl} --output-root {paths.nano_output_root} --model {nano_banana_model}",
        inputs=("paths.nano_seed_jsonl",),
        outputs=("paths.nano_output_root", "paths.nano_output_jsonl"),
        models=("endpoints.nano_banana",),
        notes="Open-source path uses the official Gemini Interactions API by default. Set GEMINI_API_KEY or GOOGLE_API_KEY before --execute.",
    ),
    StageSpec(
        name="multi_angle_qwen_edit",
        pipeline_step="Actor-18M-A Qwen-Image-Edit-Multiple-Angles",
        script="data/ProcessOpenS2V/step_7_qwen_edit_multi_angle/Qwen-Image-Edit-Angles/process_nano_step_6.py",
        workdir="data/ProcessOpenS2V/step_7_qwen_edit_multi_angle/Qwen-Image-Edit-Angles",
        env="qwen_edit",
        command="{python} process_nano_step_6.py --input_jsonl {paths.multi_angle_seed_jsonl} --img_save_root {paths.multi_angle_output_root} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.multi_angle_seed_jsonl",),
        outputs=("paths.multi_angle_output_root",),
        models=("models.qwen_image_edit", "models.qwen_image_edit_angles_lora"),
        notes="Generates camera-rotated views with the 'Qwen-Edit-2509-Multiple-angles' LoRA.",
    ),
    StageSpec(
        name="consistency_audit",
        pipeline_step="Qwen3-VL + InsightFace consistency filtering",
        script="data/ProcessOpenS2V/step_15_step_2_improve_consistency_anno.py",
        workdir="data/ProcessOpenS2V",
        env="qwen",
        command="{python} step_15_step_2_improve_consistency_anno.py --dataset {dataset} --target {target} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.unified_clean_root",),
        outputs=("paths.consistency_eval_root",),
        models=("models.qwen3_vl_32b", "models.insightface_buffalo_l"),
        notes="Audits Nano, AnyViews, and QwenEdit rows for face/body consistency; run per dataset/target pair.",
    ),
    StageSpec(
        name="face_sim_for_qwenedit_body",
        pipeline_step="identity preservation score for edited body refs",
        script="data/ProcessOpenS2V/step_15_step_4_get_sim_for_cropped_qwenedit_body.py",
        workdir="data/ProcessOpenS2V",
        env="stand_in",
        command="{python} step_15_step_4_get_sim_for_cropped_qwenedit_body.py --version {filter_version} --machine_id {machine_id} --total_machines {total_machines}",
        inputs=("paths.unified_clean_root",),
        outputs=("paths.consistency_eval_root",),
        models=("models.insightface_buffalo_l",),
        notes="Computes source-vs-edited face cosine similarity after target face crop mapping.",
    ),
    StageSpec(
        name="export_tfrecord",
        pipeline_step="final clean, split, stats, and TFRecord export",
        script="data/ProcessOpenS2V/step_13_step_5_jsonl_to_tfrecord.py",
        workdir="data/ProcessOpenS2V",
        env="base",
        command="{python} step_13_step_5_jsonl_to_tfrecord.py",
        inputs=("paths.final_jsonl",),
        outputs=("paths.tfrecord_root",),
        notes="Final exporter for dataset shards.",
    ),
)


def _stage_map() -> dict[str, StageSpec]:
    return {stage.name: stage for stage in DEFAULT_STAGES}


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _format(template: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(key)
        return str(values[key])

    return re.sub(r"\{([^{}]+)\}", replace, template)


def _strip_local_prefix(value: str) -> str:
    return value[len("local:") :] if value.startswith("local:") else value


def _config_dir(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent


def _resolve_existing_root(cfg: dict[str, Any], config_path: str | Path, prefer_local: bool) -> Path:
    roots = cfg.get("source_roots", {})
    config_dir = _config_dir(config_path)
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    if prefer_local and roots.get("local_reference"):
        candidates.append((config_dir / roots["local_reference"]).resolve())
        candidates.append((repo_root.parent / "remote_reference" / "code").resolve())
    if roots.get("remote_vlogger"):
        candidates.append(Path(roots["remote_vlogger"]).expanduser())
    if roots.get("local_reference"):
        candidates.append((config_dir / roots["local_reference"]).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    raise ValueError("source_roots.remote_vlogger or source_roots.local_reference must be configured")


def _selected_stage_names(cfg: dict[str, Any], requested: list[str] | None) -> list[str]:
    available = _stage_map()
    names = requested or cfg.get("stage_order") or list(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    return list(names)


def _runtime_values(cfg: dict[str, Any], machine_id: int, total_machines: int) -> dict[str, Any]:
    values = _flatten(cfg)
    values.update(
        {
            "machine_id": machine_id,
            "total_machines": total_machines,
            "python": cfg.get("runtime", {}).get("python", "python"),
            "workers": cfg.get("runtime", {}).get("workers", 8),
            "rapid_flag": "--use_rapid" if cfg.get("runtime", {}).get("use_rapid", False) else "",
            "dataset": cfg.get("runtime", {}).get("dataset", "QwenEdit"),
            "target": cfg.get("runtime", {}).get("target", "body"),
            "filter_version": cfg.get("runtime", {}).get("filter_version", "f93"),
            "nano_banana_provider": cfg.get("runtime", {}).get("nano_banana_provider", "official"),
            "nano_banana_model": cfg.get("runtime", {}).get("nano_banana_model", "gemini-3.1-flash-image"),
        }
    )
    return values


def build_stage_plan(
    config_path: str | Path,
    stages: list[str] | None = None,
    machine_id: int = 0,
    total_machines: int = 1,
    prefer_local_reference: bool = False,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    root = _resolve_existing_root(cfg, config_path, prefer_local=prefer_local_reference)
    values = _runtime_values(cfg, machine_id, total_machines)
    stage_specs = _stage_map()

    plan: list[dict[str, Any]] = []
    for name in _selected_stage_names(cfg, stages):
        spec = stage_specs[name]
        repo_root = Path(__file__).resolve().parents[2]
        if spec.workdir.startswith("local:"):
            workdir = (repo_root / _strip_local_prefix(spec.workdir)).resolve()
        else:
            workdir = root / spec.workdir
        if spec.script.startswith("local:"):
            script = (repo_root / _strip_local_prefix(spec.script)).resolve()
        else:
            script = root / spec.script
        command = _format(spec.command, values)
        shell_command = f"cd {shlex.quote(str(workdir))} && {command}"

        env_name = cfg.get("runtime", {}).get("envs", {}).get(spec.env, spec.env)
        conda_setup = cfg.get("runtime", {}).get("conda_setup")
        if env_name and env_name != "base":
            if conda_setup:
                shell_command = f"source {shlex.quote(str(conda_setup))} && conda activate {shlex.quote(str(env_name))} && {shell_command}"
            else:
                shell_command = f"conda run -n {shlex.quote(str(env_name))} bash -lc {shlex.quote(shell_command)}"

        plan.append(
            {
                "name": spec.name,
                "pipeline_step": spec.pipeline_step,
                "script": str(script),
                "script_exists": script.exists(),
                "workdir": str(workdir),
                "workdir_exists": workdir.exists(),
                "env": env_name,
                "command": shell_command,
                "inputs": list(spec.inputs),
                "outputs": list(spec.outputs),
                "models": list(spec.models),
                "notes": spec.notes,
            }
        )
    return plan


def validate_stage_plan(plan: list[dict[str, Any]], strict: bool = True) -> None:
    missing = []
    for stage in plan:
        if not stage["script_exists"]:
            missing.append(f"{stage['name']}: script not found at {stage['script']}")
        if not stage["workdir_exists"]:
            missing.append(f"{stage['name']}: workdir not found at {stage['workdir']}")
    if missing and strict:
        raise FileNotFoundError("\n".join(missing))


def run_full_pipeline(
    config_path: str | Path,
    stages: list[str] | None = None,
    machine_id: int = 0,
    total_machines: int = 1,
    dry_run: bool = True,
    prefer_local_reference: bool = False,
    strict: bool = True,
) -> list[dict[str, Any]]:
    plan = build_stage_plan(
        config_path=config_path,
        stages=stages,
        machine_id=machine_id,
        total_machines=total_machines,
        prefer_local_reference=prefer_local_reference,
    )
    validate_stage_plan(plan, strict=strict)

    if dry_run:
        return plan

    env = os.environ.copy()
    for stage in plan:
        print(f"[wildactor-data-full] running {stage['name']}")
        subprocess.run(stage["command"], shell=True, executable="/bin/bash", check=True, env=env)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Actor-18M data pipeline registry.")
    parser.add_argument("--config", default="configs/actor18m_full_pipeline.yaml")
    parser.add_argument("--stage", action="append", help="Stage to run. Repeat to run multiple stages in order.")
    parser.add_argument("--machine_id", type=int, default=0)
    parser.add_argument("--total_machines", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Actually execute commands. Default is dry-run.")
    parser.add_argument("--prefer-local-reference", action="store_true", help="Use local remote_reference/code when available.")
    parser.add_argument("--no-strict", action="store_true", help="Print commands even if referenced scripts are missing.")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON.")
    args = parser.parse_args()

    plan = run_full_pipeline(
        config_path=args.config,
        stages=args.stage,
        machine_id=args.machine_id,
        total_machines=args.total_machines,
        dry_run=not args.execute,
        prefer_local_reference=args.prefer_local_reference or not args.execute,
        strict=not args.no_strict,
    )

    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        for stage in plan:
            exists = "ok" if stage["script_exists"] else "missing"
            print(f"[{exists}] {stage['name']}: {stage['command']}")


if __name__ == "__main__":
    main()
