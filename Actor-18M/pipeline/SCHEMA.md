# Actor-18M JSONL Schema

Each row is a JSON object. The public pipeline accepts both raw video rows and
processed rows with reference images.

```json
{
  "video": "path/or/url/to/video.mp4",
  "prompt": "caption or generation prompt",
  "actor_id": "optional stable subject id",
  "source": "dataset/source name",
  "width": 1280,
  "height": 720,
  "duration_sec": 5.4,
  "refs": [
    {
      "path": "path/to/reference.png",
      "region": "face | body | three_view",
      "view": "front | left | right | side | back | up | down | canonical",
      "angle": 0.0,
      "visibility": 0.91,
      "source_frame": 16,
      "metadata": {"source": "self_crop | view_aug | attr_aug | canonical"}
    }
  ]
}
```

The released data is still under filtering and safety review. These utilities
therefore release the construction procedure and JSONL schema first.
