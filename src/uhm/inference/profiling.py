"""Measurements separate compute/decoding from unmeasured display latency."""

from collections import deque
from time import perf_counter

import numpy as np
import psutil
import torch


class Profiler:
    def __init__(self, device: str = "cpu", frames_per_window: int = 6):
        self.device = device
        self.frames_per_window = frames_per_window
        self.timings = deque(maxlen=10000)
        self.started = perf_counter()
        self.samples = 0
        self.compute_seconds = 0.0
        self.peak_rss_mb = 0.0
        self.last_region = None
        self.region_pairs = 0
        self.region_changes = 0

    def record(self, timing: dict, prediction=None):
        self.timings.append(timing)
        self.samples += 1
        self.compute_seconds += timing["latency_ms"] / 1000
        self.peak_rss_mb = max(self.peak_rss_mb, psutil.Process().memory_info().rss / 2**20)
        region = None
        if prediction is not None and prediction.event_class.value != "no_event" and prediction.recipient.value == "player":
            region = prediction.body_region.value
        if region is not None and self.last_region is not None:
            self.region_pairs += 1
            self.region_changes += int(region != self.last_region)
        self.last_region = region

    def summary(self) -> dict:
        def quantiles(key):
            values = [row[key] for row in self.timings]
            return {"p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95))} if values else {"p50": None, "p95": None}

        windows_per_second = self.samples / self.compute_seconds if self.compute_seconds else None
        return {
            "device": self.device, "sampled_windows": self.samples,
            "percentile_window_limit": 10000,
            "model_ms": quantiles("model_ms"), "decode_preprocess_ms": quantiles("sampling_ms"),
            "pipeline_ms": quantiles("latency_ms"),
            "sampled_windows_per_compute_second": windows_per_second,
            "processed_frames_per_compute_second": windows_per_second * self.frames_per_window if windows_per_second else None,
            "wall_elapsed_seconds": perf_counter() - self.started,
            "peak_process_rss_mb": self.peak_rss_mb,
            "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2**20 if torch.device(self.device).type == "cuda" else None,
            "gpu_utilization_percent": None,
            "raw_region_transition_rate": self.region_changes / self.region_pairs if self.region_pairs else None,
            "raw_region_comparable_pairs": self.region_pairs,
            "end_to_end_display_latency_ms": None,
            "notes": ["Compute throughput excludes playback waits and includes repeated context frames.",
                      "Pipeline latency includes decoding, preprocessing and model; excludes sampling/confirmation wait, transport and rendering.",
                      "Raw region transitions compare consecutive player-positive windows before gating.",
                      "GPU utilization and browser display latency are not measured. CPU VRAM is unavailable.",
                      "80ms p50 / 150ms p95 are future end-to-end targets, not achieved claims."],
        }
