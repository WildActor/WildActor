# Actor-18M Configs

`full_pipeline.yaml` is the public configuration for the Actor-18M construction
pipeline. It intentionally contains no user-specific absolute paths. Export the
environment variables referenced in the YAML before running on your data.

For example:

```bash
export VLOGGER_ROOT=/path/to/vlogger
export CONDA_SETUP=/path/to/miniconda3/bin/activate
export QWEN3_VL_32B=/path/to/Qwen3-VL-32B-Instruct
export OPENS2V_JSON_DIR=/path/to/OpenS2V/Jsons/mask_and_bbox
export OPENS2V_VIDEO_ROOT=/path/to/OpenS2V/Videos
```

The same config is used for local dry-runs and remote execution. The pipeline
does not read model or data locations from Python source files.

`model_assets.yaml` lists the public model repositories, checkpoint URLs, and
environment variables used by the data pipeline. Use
`Actor-18M/scripts/download_models.sh` to fetch the public Qwen, CoTracker, and
SAM2 assets. The downloader honors `HTTP_PROXY`, `HTTPS_PROXY`, and
`HF_ENDPOINT`.

`augmentation_recipes.yaml` stores prompt templates and attribute pools for
Actor-18M-A/B/C construction: Qwen3-VL annotation prompts,
Qwen-Image-Edit instruction generation prompts, multi-angle camera prompts, and
Nano-Banana character-sheet prompts. The text pools referenced by that file live
under `configs/attributes/`.

The data-construction model contract includes both downloadable checkpoints and
externally supplied assets/precomputed outputs. In addition to the Qwen,
CoTracker, SAM2, and Nano-Banana stages, provide or precompute ArcFace,
CLIP/CLIP-I, RetinaFace, YOLO-World, DWPose, InsightFace, BiSeNet, YOLO pose,
and YOLO face components described in `model_assets.yaml`.
