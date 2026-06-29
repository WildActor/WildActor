# Full Actor-18M Construction Pipeline

This page describes the full Actor-18M construction stages exposed by the release. Paths to models, raw data, and optional external tools are configured through `Actor-18M/configs/full_pipeline.yaml`.

Preview the stage registry:

```bash
python Actor-18M/pipeline/run_full_pipeline.py \
  --config Actor-18M/configs/full_pipeline.yaml \
  --json
```

Run one stage:

```bash
python Actor-18M/pipeline/run_full_pipeline.py \
  --config Actor-18M/configs/full_pipeline.yaml \
  --stage opens2v_extract_human \
  --execute
```

## Stages

The released registry covers the following construction steps:

* video crop and single-person extraction,
* Qwen3-VL orientation and body-shot annotation,
* face cropping with face detection and parsing,
* pose and front/side/back evidence extraction,
* identity clustering and consistency filtering,
* Qwen-Image-Edit prompt generation and image editing,
* Nano-Banana / Gemini character-sheet generation,
* multi-angle reference generation,
* final JSONL or TFRecord export.

## Model Components

The pipeline can use the following external components when available:

* ArcFace or InsightFace-compatible identity embeddings,
* CoTracker for dense motion consistency,
* CLIP or CLIP-I for appearance and image consistency,
* RetinaFace, BiSeNet, YOLO-World, SAM2, DWPose, YOLO pose, and YOLO face models,
* Qwen3-VL and Qwen-Image-Edit for annotation and image editing,
* the official Gemini image API for Actor-18M-C canonical references.

Users can provide model paths directly through YAML/environment variables, or provide precomputed scores and masks in the input JSONL files.
