<h1 align="center">WildActor: Unconstrained Identity-Preserving Video Generation</h1>

<p align="center"><b>ICML 2026</b></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.00586"><img src="https://img.shields.io/badge/arXiv-2603.00586-b31b1b.svg" alt="arXiv"></a>
  <a href="https://wildactor.github.io/"><img src="https://img.shields.io/badge/Project%20Page-WildActor-blue" alt="Project Page"></a>
  <a href="https://github.com/WildActor/WildActor/tree/main/Actor-18M"><img src="https://img.shields.io/badge/Dataset-Actor--18M-green?logo=github" alt="Dataset"></a>
</p>

## News

* **[2026.06]** We release the Actor-18M construction pipeline and Wan2.2-5B-compatible inference code.
* **[2026.03]** WildActor is accepted to **ICML 2026**.
* **[2026.03]** The paper is available on [arXiv](https://arxiv.org/abs/2603.00586).

## Overview

WildActor targets identity-preserving human video generation under unconstrained viewpoints, compositions, and motions. The project includes:

* **Actor-18M**: a human video data construction pipeline with face, body, and canonical three-view references.
* **WildActor model code**: a Wan2.2-5B/DiffSynth-compatible inference entrypoint for multi-reference identity conditioning.

The released model code uses Wan2.2-5B as the public video backbone.

## Installation

```bash
conda create -n wildactor python=3.10 -y
conda activate wildactor
pip install -e ".[inference]"
```

For data utilities only:

```bash
pip install -e .
```

## Inference

Prepare a request JSON with a text prompt and identity references. See [examples/inference_request.json](examples/inference_request.json).

Set the WildActor-compatible DiffSynth backend and adapter weight path:

```bash
export DIFFSYNTH_ROOT=/path/to/DiffSynth-Studio
export WILDACTOR_LORA=/path/to/wildactor_lora.safetensors
```

```bash
python -m wildactor.inference.infer_wan22 \
  --config configs/inference_wan22.yaml \
  --request examples/inference_request.json \
  --output outputs/wildactor_demo.mp4
```

## Actor-18M

Actor-18M construction code is provided under [Actor-18M](Actor-18M). It supports:

* filtering identity-consistent single-person videos,
* extracting face/body references,
* generating view-augmented and attribute-diverse references,
* producing canonical front/side/back identity anchors,
* writing JSONL files for downstream use.

Run the lightweight pipeline:

```bash
python -m wildactor.data.pipeline \
  --config configs/actor18m_pipeline.yaml
```

Set `input_jsonl` in [configs/actor18m_pipeline.yaml](configs/actor18m_pipeline.yaml) to your licensed input data.

## TODO

- [ ] Release WildActor adapter weights on Wan2.2-5B.

## Citation

```bibtex
@inproceedings{guo2026wildactor,
  title={WildActor: Unconstrained Identity-Preserving Video Generation},
  author={Qin Guo and Tianyu Yang and Xuanhua He and Fei Shen and Yong Zhang and Zhuoliang Kang and Xiaoming Wei and Dan Xu},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=wXkCkP8TtK}
}
```
