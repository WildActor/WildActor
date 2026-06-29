# Actor-18M

Actor-18M is a large-scale human video dataset for identity-preserving video generation. It contains identity-consistent videos and reference images across face, body, and canonical three-view conditions.

The full dataset is under filtering and safety review. This folder releases the public construction pipeline and JSONL data schema first.

## Pipeline

Actor-18M construction includes:

1. collecting candidate single-person videos,
2. filtering identity and motion consistency,
3. extracting face and body references,
4. generating view-augmented references for Actor-18M-A,
5. generating attribute-diverse references for Actor-18M-B,
6. generating canonical front/side/back anchors for Actor-18M-C,
7. exporting JSONL files for downstream use.

## Quick Start

```bash
cd ..
python -m wildactor.data.pipeline --config configs/actor18m_pipeline.yaml
```

Set `input_jsonl` in `configs/actor18m_pipeline.yaml` to your licensed input data.

For the full construction stage registry:

```bash
python Actor-18M/pipeline/run_full_pipeline.py \
  --config Actor-18M/configs/full_pipeline.yaml \
  --json
```

For Actor-18M-C generation with the official Gemini image API:

```bash
GEMINI_API_KEY=... python Actor-18M/pipeline/run_full_pipeline.py \
  --config Actor-18M/configs/full_pipeline.yaml \
  --stage nano_banana_generate \
  --execute
```

## Models

Model and data paths are supplied through YAML files and environment variables. The release tree does not contain machine-specific absolute paths.

Optional public assets can be downloaded with:

```bash
cd ..
MODEL_ROOT=weights THIRDPARTY_ROOT=third_party \
MODELS="qwen3_vl_32b qwen_image_edit qwen_image_edit_rapid qwen_image_edit_angles_lora cotracker sam2_tiny" \
bash Actor-18M/scripts/download_models.sh
```

The downloader supports `HTTP_PROXY`, `HTTPS_PROXY`, and `HF_ENDPOINT`.

Some assets must be supplied by users according to their own licenses, including ArcFace/InsightFace, CLIP/CLIP-I, RetinaFace, YOLO-World, DWPose, YOLO pose, and YOLO face checkpoints. Users may also provide precomputed scores and masks in the input JSONL files.

## Files

* [pipeline/SCHEMA.md](pipeline/SCHEMA.md): JSONL data schema.
* [pipeline/FULL_PIPELINE.md](pipeline/FULL_PIPELINE.md): full construction stages.
* [configs/full_pipeline.yaml](configs/full_pipeline.yaml): stage configuration.
* [configs/model_assets.yaml](configs/model_assets.yaml): public model asset registry.
* [configs/augmentation_recipes.yaml](configs/augmentation_recipes.yaml): prompt templates and attribute pools.
* [scripts/download_models.sh](scripts/download_models.sh): optional public model downloader.
