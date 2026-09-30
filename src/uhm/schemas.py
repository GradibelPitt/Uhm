"""Canonical contracts. JSON Schema and TypeScript are generated from these models."""

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ontology import BODY_REGIONS, DIRECTIONS, EVENT_CLASSES, RECIPIENTS, ROOT

SCHEMA_VERSION = "0.1"
BodyRegion = Enum("BodyRegion", {x: x for x in BODY_REGIONS}, type=str)
EventClass = Enum("EventClass", {x: x for x in EVENT_CLASSES}, type=str)
PhysicalEventClass = Enum(
    "PhysicalEventClass", {x: x for x in EVENT_CLASSES if x != "no_event"}, type=str
)
Recipient = Enum("Recipient", {x: x for x in RECIPIENTS}, type=str)
Direction = Enum("Direction", {x: x for x in DIRECTIONS}, type=str)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Confidence(Contract):
    event: float = Field(ge=0, le=1)
    recipient: float = Field(ge=0, le=1)
    region: float = Field(ge=0, le=1)
    # Zero explicitly means no calibrated intensity confidence is available in V0.
    intensity: float = Field(ge=0, le=1)


class BodyForceEvent(Contract):
    event_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0, description="Source video presentation time, not wall clock")
    event_class: PhysicalEventClass
    recipient: Recipient
    body_region: BodyRegion
    intensity: float = Field(ge=0, le=1, description="Perceptual salience, not physical force")
    duration_ms: int = Field(gt=0, le=5000)
    direction: Direction = Direction.unknown
    confidence: Confidence


class ClipAnnotation(Contract):
    video_id: str = Field(min_length=1)
    video_path: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    split: Literal["train", "val", "test"]
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    event_class: EventClass
    recipient: Recipient
    body_region: BodyRegion
    intensity: float = Field(ge=0, le=1)
    observable_region: bool
    ambiguous: bool = False
    synthetic: bool = False
    viewpoint: Literal["first_person", "third_person", "unknown"] = Field(
        default="unknown", description="Source camera viewpoint; unknown for unlabeled legacy data",
    )

    @model_validator(mode="after")
    def check_evidence(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be after start_ms")
        if (not self.observable_region or self.ambiguous) and self.body_region.value != "unknown":
            raise ValueError("Unobservable or ambiguous location must be unknown")
        if self.observable_region and self.body_region.value == "unknown":
            raise ValueError("An observable annotation needs a known region")
        if self.event_class.value == "no_event" and (
            self.recipient.value != "unknown" or self.body_region.value != "unknown"
            or self.intensity != 0 or self.observable_region
        ):
            raise ValueError("no_event must have unknown recipient/region and zero intensity")
        return self


class Prediction(Contract):
    """A model window prediction, before temporal aggregation. Not a rendered event."""

    timestamp_ms: int = Field(ge=0)
    event_class: EventClass
    recipient: Recipient
    body_region: BodyRegion
    intensity: float = Field(ge=0, le=1)
    confidence: Confidence
    observable_probability: float = Field(ge=0, le=1)


def export_schemas(destination: Path | None = None) -> Path:
    destination = destination or ROOT / "schemas"
    destination.mkdir(parents=True, exist_ok=True)
    for name, model in (("body_force_event", BodyForceEvent), ("clip_annotation", ClipAnnotation)):
        schema = model.model_json_schema()
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["$id"] = f"https://universal-haptic-model.local/schemas/{name}.schema.json"
        (destination / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )
    return destination
