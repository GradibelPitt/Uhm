"""Real checkpoint inference. This module never loads ground-truth annotations."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from uhm.data.video import preprocess_frames, sample_clip
from uhm.models.checkpoint import load_checkpoint
from uhm.ontology import BODY_REGIONS, EVENT_CLASSES, RECIPIENTS
from uhm.schemas import Confidence, Prediction


class ModelPredictor:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.model, self.metadata = load_checkpoint(checkpoint, device)
        self.device = device
        self.checkpoint = Path(checkpoint).resolve()
        self.config = self.metadata["config"]
        self.data_config = self.config["data"]
        torch.set_num_threads(self.config["training"].get("num_threads", 4))

    def _synchronize(self):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_tensor(self, frames: torch.Tensor, timestamp_ms: int) -> tuple[Prediction, dict]:
        """Consume normalized [T,3,H,W] pixels, retaining no hidden state across requests."""
        started = perf_counter()
        output = self.model.predict(frames.unsqueeze(0).to(self.device))
        self._synchronize()
        model_ms = (perf_counter() - started) * 1000
        confidences, labels = {}, {}
        for name, vocabulary in (("event", EVENT_CLASSES), ("recipient", RECIPIENTS),
                                 ("region", BODY_REGIONS)):
            score, index = output[f"{name}_probs"][0].max(dim=0)
            confidences[name] = float(score)
            labels[name] = vocabulary[int(index)]
        prediction = Prediction(
            timestamp_ms=timestamp_ms, event_class=labels["event"],
            recipient=labels["recipient"], body_region=labels["region"],
            intensity=float(output["intensity"][0]),
            observable_probability=float(output["observable_prob"][0]),
            confidence=Confidence(**confidences, intensity=0.0),
        )
        return prediction, {"model_ms": model_ms}

    def predict_frames(self, frames: list[np.ndarray], timestamp_ms: int) -> tuple[Prediction, dict]:
        """RGB frame adapter for future capture sources. Caller supplies causal timestamps.

        The caller samples at window_ms/(num_frames-1); short buffers pad their
        first available frame. No game-specific event source is consulted.
        """
        if not frames:
            raise ValueError("At least one frame is required")
        length = self.data_config["num_frames"]
        frames = ([frames[0]] * max(0, length - len(frames)) + frames)[-length:]
        started = perf_counter()
        tensor = preprocess_frames(frames, self.data_config["image_size"])
        sampled = perf_counter()
        prediction, timing = self.predict_tensor(tensor, timestamp_ms)
        return prediction, timing | {"sampling_ms": (sampled - started) * 1000,
                                    "latency_ms": (perf_counter() - started) * 1000}

    def predict_video(self, video: Path, timestamp_ms: int) -> tuple[Prediction, dict]:
        started = perf_counter()
        frames = sample_clip(
            video, max(0, timestamp_ms - self.data_config["window_ms"]), timestamp_ms,
            self.data_config["num_frames"], self.data_config["image_size"],
        )
        sampled = perf_counter()
        prediction, timing = self.predict_tensor(frames, timestamp_ms)
        return prediction, timing | {"sampling_ms": (sampled - started) * 1000,
                                    "latency_ms": (perf_counter() - started) * 1000}
