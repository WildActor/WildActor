from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReferenceImage:
    path: str
    region: str
    view: str = "unknown"
    angle: float | None = None
    visibility: float | None = None
    source_frame: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "region": self.region,
            "view": self.view,
        }
        if self.angle is not None:
            data["angle"] = self.angle
        if self.visibility is not None:
            data["visibility"] = self.visibility
        if self.source_frame is not None:
            data["source_frame"] = self.source_frame
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class ActorRecord:
    video: str
    prompt: str = ""
    actor_id: str | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    refs: list[ReferenceImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "ActorRecord":
        video = item.get("video") or item.get("filename")
        if not video:
            raise ValueError("Manifest row must contain `video` or `filename`.")

        refs = []
        for ref in item.get("refs", []):
            refs.append(
                ReferenceImage(
                    path=ref["path"],
                    region=ref.get("region", "unknown"),
                    view=ref.get("view", "unknown"),
                    angle=ref.get("angle"),
                    visibility=ref.get("visibility"),
                    source_frame=ref.get("source_frame"),
                    metadata=ref.get("metadata", {}),
                )
            )

        known_keys = {
            "video",
            "filename",
            "prompt",
            "caption",
            "actor_id",
            "id",
            "source",
            "width",
            "height",
            "duration_sec",
            "refs",
            "metadata",
        }
        metadata = dict(item.get("metadata", {}))
        for key, value in item.items():
            if key not in known_keys:
                metadata[key] = value

        return cls(
            video=video,
            prompt=item.get("prompt") or item.get("caption") or "",
            actor_id=item.get("actor_id") or item.get("id"),
            source=item.get("source"),
            width=item.get("width"),
            height=item.get("height"),
            duration_sec=item.get("duration_sec"),
            refs=refs,
            metadata=metadata,
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "video": self.video,
            "prompt": self.prompt,
            "refs": [ref.to_json() for ref in self.refs],
        }
        for key in ("actor_id", "source", "width", "height", "duration_sec"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.metadata:
            data["metadata"] = self.metadata
        return data
