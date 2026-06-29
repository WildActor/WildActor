#!/usr/bin/env bash
set -euo pipefail

# Release-safe downloader for Hugging Face hosted Actor-18M data-pipeline models.
#
# Optional environment variables:
#   MODEL_ROOT      Destination root. Defaults to ./weights.
#   THIRDPARTY_ROOT Third-party source checkout root. Defaults to ./third_party.
#   HF_ENDPOINT     Hugging Face mirror endpoint, e.g. https://hf-mirror.com.
#   HF_TOKEN        Hugging Face token for gated or authenticated mirrors.
#   HTTP_PROXY      Proxy URL.
#   HTTPS_PROXY     Proxy URL.
#   MODELS          Space-separated logical names to download.

MODEL_ROOT="${MODEL_ROOT:-weights}"
THIRDPARTY_ROOT="${THIRDPARTY_ROOT:-third_party}"
MODELS="${MODELS:-qwen3_vl_32b qwen_image_edit qwen_image_edit_rapid qwen_image_edit_angles_lora}"

if [[ -n "${HTTP_PROXY:-}" ]]; then export http_proxy="$HTTP_PROXY"; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export https_proxy="$HTTPS_PROXY"; fi
if [[ -n "${HF_ENDPOINT:-}" ]]; then export HF_ENDPOINT; fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli was not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

repo_for() {
  case "$1" in
    qwen3_vl_32b) echo "Qwen/Qwen3-VL-32B-Instruct" ;;
    qwen_image_edit) echo "Qwen/Qwen-Image-Edit-2509" ;;
    qwen_image_edit_rapid) echo "linoyts/Qwen-Image-Edit-Rapid-AIO" ;;
    qwen_image_edit_angles_lora) echo "dx8152/Qwen-Edit-2509-Multiple-angles" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

dir_for() {
  case "$1" in
    qwen3_vl_32b) echo "Qwen3-VL-32B-Instruct" ;;
    qwen_image_edit) echo "Qwen-Image-Edit-2509" ;;
    qwen_image_edit_rapid) echo "Qwen-Image-Edit-Rapid-AIO" ;;
    qwen_image_edit_angles_lora) echo "Qwen-Edit-2509-Multiple-angles" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

mkdir -p "$MODEL_ROOT"
mkdir -p "$THIRDPARTY_ROOT"

echo "Model root: $MODEL_ROOT"
echo "HF endpoint: ${HF_ENDPOINT:-default}"
echo "Models: $MODELS"

download_hf_file() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    echo "    exists: $target"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$target" "$url"
  else
    curl -L --continue-at - -o "$target" "$url"
  fi
}

clone_or_update() {
  local repo="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" pull --ff-only
  else
    git clone --depth 1 "$repo" "$target"
  fi
}

for key in $MODELS; do
  if [[ "$key" == "cotracker" ]]; then
    target="$THIRDPARTY_ROOT/co-tracker"
    clone_or_update "https://github.com/facebookresearch/co-tracker.git" "$target"
    download_hf_file "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" "$MODEL_ROOT/cotracker/scaled_offline.pth"
    continue
  fi
  if [[ "$key" == "sam2_tiny" || "$key" == "sam2" ]]; then
    target="$THIRDPARTY_ROOT/segment-anything-2"
    clone_or_update "https://github.com/facebookresearch/sam2.git" "$target"
    download_hf_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" "$MODEL_ROOT/sam2/sam2.1_hiera_tiny.pt"
    continue
  fi
  if [[ "$key" == "sam2_all" ]]; then
    target="$THIRDPARTY_ROOT/segment-anything-2"
    clone_or_update "https://github.com/facebookresearch/sam2.git" "$target"
    download_hf_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" "$MODEL_ROOT/sam2/sam2.1_hiera_tiny.pt"
    download_hf_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" "$MODEL_ROOT/sam2/sam2.1_hiera_small.pt"
    download_hf_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" "$MODEL_ROOT/sam2/sam2.1_hiera_base_plus.pt"
    download_hf_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" "$MODEL_ROOT/sam2/sam2.1_hiera_large.pt"
    continue
  fi
  repo="$(repo_for "$key")"
  target="$MODEL_ROOT/$(dir_for "$key")"
  echo
  echo "==> Downloading $key"
  echo "    repo:   $repo"
  echo "    target: $target"

  args=(download --resume-download "$repo" --local-dir "$target" --local-dir-use-symlinks False)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    args+=(--token "$HF_TOKEN")
  fi
  huggingface-cli "${args[@]}"
done

cat <<EOF

Downloads finished.

Export these variables before running the full data pipeline:
  export QWEN3_VL_32B="$MODEL_ROOT/Qwen3-VL-32B-Instruct"
  export QWEN_IMAGE_EDIT_ROOT="$MODEL_ROOT/Qwen-Image-Edit-2509"
  export QWEN_IMAGE_EDIT_RAPID_ROOT="$MODEL_ROOT/Qwen-Image-Edit-Rapid-AIO"
  export QWEN_IMAGE_EDIT_ANGLES_LORA="$MODEL_ROOT/Qwen-Edit-2509-Multiple-angles"
  export COTRACKER_REPO="$THIRDPARTY_ROOT/co-tracker"
  export COTRACKER_MODEL="$MODEL_ROOT/cotracker/scaled_offline.pth"
  export SAM2_REPO="$THIRDPARTY_ROOT/segment-anything-2"
  export SAM2_MODEL="$MODEL_ROOT/sam2/sam2.1_hiera_tiny.pt"
  export SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"

External assets still need to be provided separately:
  ANTELOPEV2_ROOT, INSIGHTFACE_BUFFALO_L, YOLO_POSE_MODEL, YOLO_FACE_MODEL
EOF
