"""The ontology is shared by datasets, checkpoints, runtime and the SVG renderer."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BODY_ONTOLOGY = yaml.safe_load((ROOT / "ontology/body_regions.yaml").read_text("utf-8"))
EVENT_ONTOLOGY = yaml.safe_load((ROOT / "ontology/events.yaml").read_text("utf-8"))
BODY_REGIONS = [r["id"] for r in BODY_ONTOLOGY["regions"]] + BODY_ONTOLOGY["special_regions"]
EVENT_CLASSES = EVENT_ONTOLOGY["event_classes"]
RECIPIENTS = EVENT_ONTOLOGY["recipients"]
DIRECTIONS = EVENT_ONTOLOGY["directions"]
ONTOLOGY = {"body": BODY_ONTOLOGY, "events": EVENT_ONTOLOGY}
ONTOLOGY_HASH = hashlib.sha256(json.dumps(ONTOLOGY, sort_keys=True).encode()).hexdigest()

for labels in (BODY_REGIONS, EVENT_CLASSES, RECIPIENTS, DIRECTIONS):
    if len(labels) != len(set(labels)):
        raise ValueError("Ontology labels must be unique")
