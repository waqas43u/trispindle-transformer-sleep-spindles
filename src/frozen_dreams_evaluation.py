#!/usr/bin/env python3
"""Frozen MASS-SS2 Transformer-only evaluation on the DREAMS spindle excerpts.

This is deliberately a standalone inference application.  It imports model,
preprocessing, calibration, decoding, and metric definitions from
``transformer_mass_pipeline.py`` but never dispatches that module's training
entry point.  DREAMS annotations are loaded only after probability timelines
and decoded events have been produced.

The primary unit of evaluation is one MASS fold-specific two-member ensemble.
The five folds are never pooled into a single model and their decoder settings
are never averaged.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import platform
import re
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "1.0.0"
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "global": {
        "directory": "results_Global_mass_xai_5fold",
        "block_type": "transformer",
        "parameter_count": 2_776_800,
        "label": "Global Transformer",
    },
    "sparse_local": {
        "directory": "results_sparse_local_mass_xai_5fold",
        "block_type": "local_transformer",
        "parameter_count": 2_776_800,
        "label": "Sparse-Local Transformer",
    },
    "conformer": {
        "directory": "results_conformer_mass_xai_5fold",
        "block_type": "conformer",
        "parameter_count": 4_829_280,
        "label": "Conformer",
    },
}
EXPECTED_EXCERPT_CHANNELS = {
    1: "C3-A1", 2: "CZ-A1", 3: "C3-A1", 4: "CZ-A1",
    5: "CZ-A1", 6: "CZ-A1", 7: "CZ-A1", 8: "CZ-A1",
}
EXPERT2_EXCERPTS = frozenset(range(1, 7))
REQUIRED_CONFIG = {
    "model_mode": "segmentation_transformer_only",
    "resample_hz": 256.0,
    "bandpass_low_hz": 0.3,
    "bandpass_high_hz": 35.0,
    "notch_enabled": False,
    "sigma_low_hz": 12.0,
    "sigma_high_hz": 16.0,
    "segmentation_use_sigma_channel": True,
    "segmentation_use_slow_sigma_channel": True,
    "segmentation_slow_sigma_low_hz": 10.0,
    "segmentation_slow_sigma_high_hz": 12.0,
    "segmentation_use_teo_channel": False,
    "segmentation_use_envelope_channel": True,
    "segmentation_use_secondary_eeg_channel": False,
    "segmentation_normalize_inputs": False,
    "segmentation_clip_uv": 150.0,
    "segmentation_window_sec": 20.0,
    "segmentation_overlap": 0.5,
    "segmentation_structured_decoder_enabled": False,
    "segmentation_density_adaptive_threshold_enabled": False,
    "segmentation_candidate_consensus_enabled": False,
    "transformer_ssl_external_data_dir": "",
    "transformer_external_supervised_pretraining_enabled": False,
    "external_dataset": "none",
    "external_data_dir": "",
}


@dataclass(frozen=True)
class EdfSignalHeader:
    path: str
    labels: Tuple[str, ...]
    physical_units: Tuple[str, ...]
    physical_min: Tuple[float, ...]
    physical_max: Tuple[float, ...]
    digital_min: Tuple[float, ...]
    digital_max: Tuple[float, ...]
    samples_per_record: Tuple[int, ...]
    n_records: int
    record_duration_sec: float
    header_bytes: int


@dataclass(frozen=True)
class ReferenceSpec:
    name: str
    events: Tuple[Tuple[float, float], ...]
    coverage_start_sec: float
    coverage_end_sec: float
    provenance: str
    scoring_channel: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)


def canonical_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("frozen_transformer_dreams")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    ensure_dir(output_dir)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def import_mass_pipeline(path: Path) -> Any:
    """Import definitions without invoking the source module's guarded main()."""
    if not path.is_file():
        raise FileNotFoundError(f"MASS pipeline module does not exist: {path}")
    name = "_authoritative_transformer_mass_pipeline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # SHAP is optional in the authoritative source and imports numba/LLVM.  It
    # is irrelevant to frozen inference and can make a clean server process
    # spend minutes compiling during import.  Make that optional import fail in
    # the source's existing try/except, then restore the caller's module table.
    sentinel = object()
    optional_previous = {key: sys.modules.get(key, sentinel) for key in ("mne", "matplotlib", "shap")}
    # EDF reading/resampling in this evaluator is explicit and does not call
    # MNE. A lightweight placeholder prevents importing MNE's optional stack.
    sys.modules["mne"] = types.ModuleType("mne")
    sys.modules["matplotlib"] = None
    sys.modules["shap"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        for key, previous in optional_previous.items():
            if previous is sentinel:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
    return module


def parse_int_selection(value: str, minimum: int, maximum: int, label: str) -> List[int]:
    if value.strip().lower() in {"all", "*"}:
        return list(range(minimum, maximum + 1))
    selected: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.extend(range(int(left), int(right) + 1))
        else:
            selected.append(int(item))
    selected = sorted(set(selected))
    if not selected or any(x < minimum or x > maximum for x in selected):
        raise ValueError(f"{label} must select values from {minimum} through {maximum}.")
    return selected


def parse_architectures(value: str) -> List[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(ARCHITECTURES)
    aliases = {
        "global_transformer": "global", "transformer": "global",
        "local": "sparse_local", "local_transformer": "sparse_local",
        "sparse-local": "sparse_local",
    }
    out: List[str] = []
    for token in value.split(","):
        key = aliases.get(token.strip().lower(), token.strip().lower())
        if key not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {token!r}; choose {sorted(ARCHITECTURES)}.")
        if key not in out:
            out.append(key)
    return out


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _ascii_float(raw: bytes, field: str) -> float:
    try:
        return float(raw.decode("ascii", errors="strict").strip())
    except Exception as exc:
        raise ValueError(f"Invalid EDF numeric field {field}: {raw!r}") from exc


def _ascii_int(raw: bytes, field: str) -> int:
    return int(_ascii_float(raw, field))


def read_edf_header(path: Path) -> EdfSignalHeader:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Truncated EDF fixed header: {path}")
        header_bytes = _ascii_int(fixed[184:192], "header_bytes")
        n_records = _ascii_int(fixed[236:244], "n_records")
        record_duration = _ascii_float(fixed[244:252], "record_duration")
        n_signals = _ascii_int(fixed[252:256], "n_signals")
        signal_header = handle.read(header_bytes - 256)
    if n_signals < 1 or record_duration <= 0 or n_records < 1:
        raise ValueError(f"Unsupported EDF dimensions in {path}")
    offset = 0

    def strings(width: int) -> Tuple[str, ...]:
        nonlocal offset
        size = width * n_signals
        block = signal_header[offset:offset + size]
        if len(block) != size:
            raise ValueError(f"Truncated EDF signal header in {path}")
        offset += size
        return tuple(
            block[i * width:(i + 1) * width].decode("latin-1").strip()
            for i in range(n_signals)
        )

    labels = strings(16)
    strings(80)  # transducer
    units = strings(8)
    pmin = tuple(float(x) for x in strings(8))
    pmax = tuple(float(x) for x in strings(8))
    dmin = tuple(float(x) for x in strings(8))
    dmax = tuple(float(x) for x in strings(8))
    strings(80)  # prefiltering
    samples = tuple(int(float(x)) for x in strings(8))
    strings(32)  # reserved
    return EdfSignalHeader(
        path=str(path), labels=labels, physical_units=units,
        physical_min=pmin, physical_max=pmax, digital_min=dmin,
        digital_max=dmax, samples_per_record=samples,
        n_records=n_records, record_duration_sec=record_duration,
        header_bytes=header_bytes,
    )


def normalized_channel_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def read_edf_channel_volts(path: Path, expected_channel: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read one EDF channel and explicitly convert the physical unit to volts.

    DREAMS uses the non-canonical string ``UV``.  It is interpreted explicitly
    as microvolts, converted to volts once, and then passed to the authoritative
    MASS scaling function.  This avoids depending on MNE's unit recognition.
    """
    header = read_edf_header(path)
    expected = normalized_channel_name(expected_channel)
    candidates = [i for i, name in enumerate(header.labels) if normalized_channel_name(name) == expected]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {expected_channel} channel in {path.name}; "
            f"found {[header.labels[i] for i in candidates]} among {header.labels}."
        )
    idx = candidates[0]
    samples_per_record = header.samples_per_record[idx]
    record_samples_total = sum(header.samples_per_record)
    record_bytes = 2 * record_samples_total
    channel_offset = 2 * sum(header.samples_per_record[:idx])
    pieces: List[np.ndarray] = []
    with path.open("rb") as handle:
        for record_idx in range(header.n_records):
            handle.seek(header.header_bytes + record_idx * record_bytes + channel_offset)
            raw = handle.read(2 * samples_per_record)
            if len(raw) != 2 * samples_per_record:
                raise ValueError(f"Truncated EDF data record {record_idx} in {path}")
            pieces.append(np.frombuffer(raw, dtype="<i2").astype(np.float64))
    digital = np.concatenate(pieces)
    dmin, dmax = header.digital_min[idx], header.digital_max[idx]
    pmin, pmax = header.physical_min[idx], header.physical_max[idx]
    if not dmax > dmin or not pmax > pmin:
        raise ValueError(f"Invalid EDF calibration range for {header.labels[idx]} in {path}")
    physical = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    unit = header.physical_units[idx].strip().replace("µ", "u").lower()
    factors = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}
    if unit not in factors:
        raise ValueError(f"Unsupported EDF physical unit {header.physical_units[idx]!r} in {path}")
    volts = (physical * factors[unit]).astype(np.float32)
    sfreq = samples_per_record / header.record_duration_sec
    duration = len(volts) / sfreq
    expected_size = header.header_bytes + header.n_records * record_bytes
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f"EDF file size is smaller than its declared record layout: {path}")
    metadata = {
        "edf_path": str(path.resolve()), "selected_channel": header.labels[idx],
        "edf_unit": header.physical_units[idx], "unit_policy": f"{unit}->volts",
        "original_sfreq_hz": sfreq, "original_samples": len(volts),
        "duration_sec": duration, "n_records": header.n_records,
        "record_duration_sec": header.record_duration_sec,
        "physical_min": pmin, "physical_max": pmax,
        "digital_min": dmin, "digital_max": dmax,
        "p95_abs_volts": float(np.percentile(np.abs(volts), 95)),
    }
    if not np.all(np.isfinite(volts)) or not 0.0 < metadata["p95_abs_volts"] < 0.1:
        raise ValueError(f"Implausible or non-finite EEG amplitude after EDF conversion: {metadata}")
    return volts, metadata


def _mne_reflect_limited_pad(x: np.ndarray, left: int, right: int) -> np.ndarray:
    left_zero = np.zeros(max(left - len(x) + 1, 0), dtype=x.dtype)
    right_zero = np.zeros(max(right - len(x) + 1, 0), dtype=x.dtype)
    return np.concatenate([
        left_zero, 2 * x[0] - x[left:0:-1], x,
        2 * x[-1] - x[-2:-right - 2:-1], right_zero,
    ])


def resample_mne_fft_compatible(x: np.ndarray, original_hz: float, target_hz: float) -> np.ndarray:
    """Reproduce MNE's FFT resampling defaults used by MASS ``Raw.resample``."""
    x = np.asarray(x, dtype=np.float64)
    if math.isclose(original_hz, target_hz, rel_tol=0.0, abs_tol=1e-9):
        return x.astype(np.float32)
    ratio = float(target_hz) / float(original_hz)
    final_len = max(int(round(ratio * len(x))), 1)
    min_add = min(len(x) // 8, 100) * 2
    total_pad = 2 ** int(np.ceil(np.log2(len(x) + min_add))) - len(x)
    left, extra = divmod(total_pad, 2)
    pads = (left, left + extra)
    padded = _mne_reflect_limited_pad(x, *pads)
    original_padded_len = len(padded)
    new_len = max(int(round(ratio * original_padded_len)), 1)
    remove_left = int(round(ratio * pads[0]))
    remove_right = new_len - final_len - remove_left
    spectrum = np.fft.rfft(padded)
    use_len = min(new_len, original_padded_len)
    if use_len % 2 == 0:
        nyquist = use_len // 2
        spectrum[nyquist:nyquist + 1] *= 2.0 if new_len < original_padded_len else 0.5
    spectrum *= float(new_len) / float(original_padded_len)
    y = np.fft.irfft(spectrum, new_len)
    if remove_right > 0:
        y = y[remove_left:-remove_right]
    else:
        y = y[remove_left:]
    if len(y) != final_len:
        raise AssertionError(f"Resampled length {len(y)} does not equal expected length {final_len}.")
    return y.astype(np.float32)


def build_inference_record(
    excerpt_idx: int,
    dreams_dir: Path,
    cfg: Any,
    mass: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = dreams_dir / f"excerpt{excerpt_idx}.edf"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DREAMS EDF: {path}")
    raw_volts, metadata = read_edf_channel_volts(path, EXPECTED_EXCERPT_CHANNELS[excerpt_idx])
    original_hz = float(metadata["original_sfreq_hz"])
    raw_volts = resample_mne_fft_compatible(raw_volts, original_hz, float(cfg.resample_hz))
    sfreq = float(cfg.resample_hz)
    stage1 = mass.preprocess_candidate_signal(raw_volts, sfreq, cfg)
    sigma = mass.compute_sigma_band(stage1, sfreq, cfg.sigma_low_hz, cfg.sigma_high_hz)
    slow_sigma = mass.compute_sigma_band(
        stage1, sfreq, cfg.segmentation_slow_sigma_low_hz,
        cfg.segmentation_slow_sigma_high_hz,
    )
    envelope = mass.compute_envelope(sigma)
    n_samples = len(raw_volts)
    record = {
        "subject_id": f"dreams_excerpt{excerpt_idx}", "excerpt": excerpt_idx,
        "sfreq": sfreq,
        "segmentation_raw": mass.prepare_segmentation_channel(raw_volts, cfg),
        "segmentation_sigma": mass.prepare_segmentation_channel(sigma, cfg),
        "segmentation_slow_sigma": mass.prepare_segmentation_channel(slow_sigma, cfg),
        "segmentation_envelope": mass.prepare_segmentation_channel(envelope, cfg),
        # Compatibility only. Values are never read for prediction or decoding.
        "truth_binary_stage": np.zeros(n_samples, dtype=np.uint8),
        "stage_segment_bounds": [(0, n_samples)],
        "retained_len_samples": n_samples,
    }
    feature_order = ["raw", "sigma", "slow_sigma", "envelope"]
    probe = mass.build_segmentation_input_window(
        record, 0, min(n_samples, int(round(20 * sfreq))), int(round(20 * sfreq)), cfg,
    )
    if probe.shape != (4, 5120) or probe.dtype != np.float32 or not np.all(np.isfinite(probe)):
        raise ValueError(f"Invalid transformer input for excerpt {excerpt_idx}: {probe.shape}, {probe.dtype}")
    metadata.update({
        "excerpt": excerpt_idx, "record_id": record["subject_id"],
        "resampled_sfreq_hz": sfreq, "resampled_samples": n_samples,
        "resampled_duration_sec": n_samples / sfreq,
        "feature_order": feature_order, "input_probe_shape": list(probe.shape),
        "normalization_enabled": bool(cfg.segmentation_normalize_inputs),
        "clip_uv": float(cfg.segmentation_clip_uv),
    })
    return record, metadata


def window_starts(n_samples: int, window_len: int, overlap: float) -> List[int]:
    hop = max(1, int(round(window_len * (1.0 - overlap))))
    if n_samples <= window_len:
        return [0]
    starts = list(range(0, n_samples - window_len + 1, hop))
    final = n_samples - window_len
    if starts[-1] != final:
        starts.append(final)
    return starts


def predict_member_timeline(
    model: torch.nn.Module,
    record: Mapping[str, Any],
    cfg: Any,
    mass: Any,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Authoritative center-retained overlap reconstruction with coverage checks."""
    model.eval()
    n_samples = len(record["segmentation_raw"])
    window_len = max(8, int(round(cfg.segmentation_window_sec * record["sfreq"])))
    starts = window_starts(n_samples, window_len, cfg.segmentation_overlap)
    sums = np.zeros(n_samples, dtype=np.float64)
    coverage = np.zeros(n_samples, dtype=np.int32)
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start:batch_start + batch_size]
            windows = []
            valid_lengths = []
            for start in batch_starts:
                stop = min(n_samples, start + window_len)
                valid_lengths.append(stop - start)
                windows.append(mass.build_segmentation_input_window(record, start, stop, window_len, cfg))
            x = torch.from_numpy(np.ascontiguousarray(np.stack(windows), dtype=np.float32)).to(device)
            output = model(x)
            logits = mass.primary_segmentation_logits(output)
            if logits.ndim != 3 or logits.shape[0] != len(batch_starts) or logits.shape[1] != 1:
                raise RuntimeError(f"Unexpected model output shape {tuple(logits.shape)}")
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            if not np.all(np.isfinite(probabilities)) or np.min(probabilities) < 0 or np.max(probabilities) > 1:
                raise RuntimeError("Model produced invalid probabilities.")
            for row, (start, valid_len) in enumerate(zip(batch_starts, valid_lengths)):
                stop = min(n_samples, start + window_len)
                if n_samples <= window_len:
                    keep_s, keep_e = 0, valid_len
                else:
                    keep_s = window_len // 4
                    keep_e = min(valid_len, 3 * window_len // 4)
                    if start == 0:
                        keep_s = 0
                    if stop >= n_samples:
                        keep_e = valid_len
                absolute_start, absolute_end = start + keep_s, start + keep_e
                if absolute_end > absolute_start:
                    sums[absolute_start:absolute_end] += probabilities[row, keep_s:keep_e]
                    coverage[absolute_start:absolute_end] += 1
    if np.any(coverage == 0):
        missing = np.flatnonzero(coverage == 0)
        raise RuntimeError(
            f"Sliding-window reconstruction left {len(missing)} samples uncovered; "
            f"first missing sample={int(missing[0])}."
        )
    timeline = (sums / coverage).astype(np.float32)
    if len(timeline) != n_samples or not np.all(np.isfinite(timeline)):
        raise RuntimeError("Invalid reconstructed probability timeline.")
    return timeline, coverage


def combine_member_probabilities(
    member_timelines: Sequence[np.ndarray],
    member_weights: Sequence[float],
) -> np.ndarray:
    if not member_timelines:
        raise ValueError("At least one member probability timeline is required.")
    lengths = {len(np.asarray(item)) for item in member_timelines}
    if len(lengths) != 1:
        raise ValueError("Ensemble member timelines must have identical lengths.")
    weights = np.asarray(member_weights, dtype=np.float64)
    if len(weights) != len(member_timelines) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Ensemble weights must be finite, non-negative, and match member count.")
    if float(weights.sum()) <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    weights /= weights.sum()
    combined = np.average(np.stack(member_timelines), axis=0, weights=weights).astype(np.float32)
    if not np.all(np.isfinite(combined)) or np.any((combined < 0) | (combined > 1)):
        raise ValueError("Weighted member probabilities are invalid.")
    return combined


def frozen_postprocess(
    raw_probability: np.ndarray,
    calibrator: Optional[Dict[str, Any]],
    params: Any,
    cfg: Any,
    mass: Any,
    sfreq: float,
) -> Tuple[np.ndarray, float, List[Tuple[float, float]], np.ndarray]:
    """Apply the saved MASS calibration, adaptive rule, and decoder verbatim."""
    probability = mass.apply_probability_calibrator(calibrator, raw_probability)
    threshold = mass.resolve_segmentation_probability_threshold(
        probability, params.probability_threshold,
        params.adaptive_threshold_quantile,
        cfg.segmentation_adaptive_threshold_max_adjustment,
    )
    events, binary = mass.decode_segmentation_events(
        probability, sfreq, threshold, params.final_min_sec,
        params.final_merge_gap_sec, params.boundary_gap,
        params.boundary_max_extend_sec, params.boundary_trim_sec,
        params.event_min_mean_prob, params.event_min_peak_prob,
        params.final_max_sec,
    )
    return np.asarray(probability, dtype=np.float32), float(threshold), events, binary


def validate_saved_config(cfg: Any, architecture: str, mass: Any) -> None:
    expected = ARCHITECTURES[architecture]
    errors: List[str] = []
    for key, value in REQUIRED_CONFIG.items():
        actual = getattr(cfg, key, None)
        equal = math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-9) if isinstance(value, float) else actual == value
        if not equal:
            errors.append(f"{key}={actual!r}, expected {value!r}")
    if cfg.segmentation_transformer_only_block_type != expected["block_type"]:
        errors.append(
            f"block_type={cfg.segmentation_transformer_only_block_type!r}, "
            f"expected {expected['block_type']!r}"
        )
    if tuple(cfg.segmentation_ensemble_seed_offsets) != (0, 2):
        errors.append(f"seed offsets are {cfg.segmentation_ensemble_seed_offsets}, expected (0, 2)")
    if mass.segmentation_input_channels(cfg) != 4:
        errors.append(f"input channel count is {mass.segmentation_input_channels(cfg)}, expected 4")
    if errors:
        raise ValueError(f"Incompatible saved {architecture} configuration: " + "; ".join(errors))


def _checkpoint_state(path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Mapping[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"Non-tensor state entry {name!r} in {path}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite checkpoint tensor {name!r} in {path}")
    return dict(state), payload if isinstance(payload, Mapping) else {}


def load_fold_bundle(
    architecture: str,
    fold: int,
    checkpoint_base_dir: Path,
    mass: Any,
    device: torch.device,
    logger: logging.Logger,
    validate_forward: bool = False,
) -> Dict[str, Any]:
    root = checkpoint_base_dir / ARCHITECTURES[architecture]["directory"]
    fold_dir = root / f"fold_{fold:02d}"
    required = [
        root / "config.json", fold_dir / "config.json",
        fold_dir / "best_segmentation_model_seed0.pt",
        fold_dir / "best_segmentation_model_seed2.pt",
        fold_dir / "segmentation_ensemble_weights.json",
        fold_dir / "probability_calibrator.json",
        fold_dir / "segmentation_params.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required frozen artifacts:\n" + "\n".join(missing))
    root_config = read_json(root / "config.json")
    fold_config = read_json(fold_dir / "config.json")
    root_differences = {
        key: {"root": root_config.get(key), "fold": fold_config.get(key)}
        for key in sorted(set(root_config) | set(fold_config))
        if root_config.get(key) != fold_config.get(key)
    }
    # A resumed run can leave the root start_fold at the last invocation while
    # the fold snapshots retain the original value. This is execution metadata,
    # not architecture or preprocessing state. Any other difference is rejected.
    unexpected_root_differences = set(root_differences) - {"start_fold"}
    if unexpected_root_differences:
        raise ValueError(
            f"Root and fold configuration differ in inference-relevant or unapproved fields "
            f"for {architecture} fold {fold}: {sorted(unexpected_root_differences)}"
        )
    cfg = mass.Config(**fold_config)
    validate_saved_config(cfg, architecture, mass)
    models: List[torch.nn.Module] = []
    checkpoint_rows = []
    for seed_offset in (0, 2):
        checkpoint = fold_dir / f"best_segmentation_model_seed{seed_offset}.pt"
        model = mass.build_segmentation_model(cfg, device)
        count = sum(parameter.numel() for parameter in model.parameters())
        expected_count = ARCHITECTURES[architecture]["parameter_count"]
        if count != expected_count:
            raise ValueError(
                f"Parameter count mismatch for {architecture}: reconstructed {count:,}, "
                f"expected {expected_count:,}."
            )
        state, payload = _checkpoint_state(checkpoint, device)
        embedded = payload.get("config") if isinstance(payload, Mapping) else None
        if isinstance(embedded, Mapping):
            model_keys = set(REQUIRED_CONFIG) | {
                key for key in fold_config
                if key.startswith("segmentation_transformer_only_")
                or key in {
                    "segmentation_auxiliary_heads_enabled", "segmentation_multi_annotator_enabled",
                    "segmentation_supervised_contrastive_enabled",
                    "segmentation_supervised_contrastive_projection_dim",
                    "segmentation_event_geometry_enabled", "segmentation_secondary_gate_enabled",
                    "segmentation_secondary_gate_initial", "model_kernel_size", "model_dropout",
                }
            }
            mismatched = [
                key for key in sorted(model_keys)
                if canonical_value(embedded.get(key)) != canonical_value(fold_config.get(key))
            ]
            if mismatched:
                raise ValueError(
                    f"Checkpoint-embedded architecture differs from fold config in {mismatched}: {checkpoint}"
                )
        model.load_state_dict(state, strict=True)
        model.eval()
        if validate_forward:
            with torch.no_grad():
                probe = torch.zeros((1, 4, 5120), dtype=torch.float32, device=device)
                output = mass.primary_segmentation_logits(model(probe))
                probabilities = torch.sigmoid(output)
                if tuple(output.shape) != (1, 1, 5120):
                    raise RuntimeError(f"Unexpected forward shape {tuple(output.shape)} for {checkpoint}")
                if not bool(torch.isfinite(probabilities).all()) or not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
                    raise RuntimeError(f"Invalid forward probabilities for {checkpoint}")
        models.append(model)
        checkpoint_rows.append({
            "seed_offset": seed_offset, "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size, "parameter_count": count,
            "strict_load": True, "finite_state": True,
            "embedded_config_present": bool(isinstance(payload, Mapping) and "config" in payload),
        })
    weight_state = read_json(fold_dir / "segmentation_ensemble_weights.json")
    if list(weight_state.get("seed_offsets", [])) != [0, 2]:
        raise ValueError(f"Saved seed order is not [0, 2] in {fold_dir}")
    weights = mass.normalized_segmentation_ensemble_weights(2, weight_state.get("member_weights"))
    calibrator_state = read_json(fold_dir / "probability_calibrator.json")
    calibrator = mass.restore_calibrator(calibrator_state)
    calibrator_probe = mass.apply_probability_calibrator(calibrator, np.array([0.1, 0.5, 0.9], dtype=np.float32))
    if not np.all(np.isfinite(calibrator_probe)) or np.any((calibrator_probe < 0) | (calibrator_probe > 1)):
        raise ValueError(f"Invalid restored probability calibrator in {fold_dir}")
    params_state = read_json(fold_dir / "segmentation_params.json")
    params = mass.SegmentationParams(**params_state)
    manifest = {
        "architecture": architecture, "architecture_label": ARCHITECTURES[architecture]["label"],
        "fold": fold, "source_root": str(root.resolve()),
        "block_type": cfg.segmentation_transformer_only_block_type,
        "input_channels": mass.segmentation_input_channels(cfg),
        "checkpoints": checkpoint_rows, "member_weights": weights.tolist(),
        "calibrator": calibrator_state, "decoder": params_state,
        "root_fold_config_differences": root_differences,
        "frozen_evaluation": True,
    }
    logger.info(
        "Validated %s fold %d | parameters=%d | weights=%s | threshold=%.3f",
        architecture, fold, ARCHITECTURES[architecture]["parameter_count"],
        weights.tolist(), params.probability_threshold,
    )
    return {
        "cfg": cfg, "models": models, "weights": weights,
        "calibrator": calibrator, "calibrator_state": calibrator_state,
        "params": params, "params_state": params_state,
        "manifest": manifest, "fold_dir": fold_dir,
    }


def parse_scoring_file(path: Path, mass: Any) -> Tuple[List[Tuple[float, float]], str]:
    if not path.is_file():
        return [], ""
    events: List[Tuple[float, float]] = []
    channel = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            match = re.search(r"/([^\]]+)\]", line)
            if match:
                channel = match.group(1).replace("_", "-")
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            onset, duration = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if duration > 0:
            events.append((onset, onset + duration))
    return mass.merge_intervals(events), channel


def clip_events(events: Iterable[Tuple[float, float]], start: float, end: float) -> List[Tuple[float, float]]:
    clipped = []
    for left, right in events:
        a, b = max(float(left), start), min(float(right), end)
        if b > a:
            clipped.append((a, b))
    return clipped


def load_references_after_prediction(
    excerpt: int,
    dreams_dir: Path,
    duration_sec: float,
    mass: Any,
    expert1_coverage_sec: float,
    expert2_coverage_sec: float,
) -> List[ReferenceSpec]:
    e1_path = dreams_dir / f"Visual_scoring1_excerpt{excerpt}.txt"
    e2_path = dreams_dir / f"Visual_scoring2_excerpt{excerpt}.txt"
    e1, e1_channel = parse_scoring_file(e1_path, mass)
    if not e1_path.is_file():
        raise FileNotFoundError(f"Expert 1 annotation is missing: {e1_path}")
    e1_end = min(duration_sec, expert1_coverage_sec)
    references = [ReferenceSpec(
        "expert1", tuple(clip_events(e1, 0.0, e1_end)), 0.0, e1_end,
        str(e1_path.resolve()), e1_channel,
    )]
    if excerpt in EXPERT2_EXCERPTS:
        if not e2_path.is_file():
            raise FileNotFoundError(f"Expert 2 annotation is missing: {e2_path}")
        e2, e2_channel = parse_scoring_file(e2_path, mass)
        e2_end = min(duration_sec, expert2_coverage_sec)
        references.append(ReferenceSpec(
            "expert2", tuple(clip_events(e2, 0.0, e2_end)), 0.0, e2_end,
            str(e2_path.resolve()), e2_channel,
        ))
        union_end = min(e1_end, e2_end, 1000.0)
        union = mass.merge_intervals(
            clip_events(e1, 0.0, union_end) + clip_events(e2, 0.0, union_end)
        )
        references.append(ReferenceSpec(
            "union", tuple(union), 0.0, union_end,
            f"overlap-only union of {e1_path.name} and {e2_path.name}",
            f"{e1_channel}|{e2_channel}",
        ))
    return references


def binary_from_events(events: Sequence[Tuple[float, float]], n: int, sfreq: float) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint8)
    for start, stop in events:
        left = max(0, min(n, int(round(start * sfreq))))
        right = max(left, min(n, int(round(stop * sfreq))))
        out[left:right] = 1
    return out


def metric_row(
    architecture: str,
    fold: int,
    excerpt: int,
    channel: str,
    reference: ReferenceSpec,
    full_predicted_events: Sequence[Tuple[float, float]],
    full_probability: np.ndarray,
    sfreq: float,
    resolved_threshold: float,
    mass: Any,
    cfg: Any,
) -> Dict[str, Any]:
    start, end = reference.coverage_start_sec, reference.coverage_end_sec
    predicted = clip_events(full_predicted_events, start, end)
    truth = list(reference.events)
    n_samples = len(full_probability)
    left = max(0, min(n_samples, int(round(start * sfreq))))
    right = max(left, min(n_samples, int(round(end * sfreq))))
    shifted_truth = [(a - start, b - start) for a, b in truth]
    shifted_pred = [(a - start, b - start) for a, b in predicted]
    local_n = right - left
    true_binary = binary_from_events(shifted_truth, local_n, sfreq)
    pred_binary = binary_from_events(shifted_pred, local_n, sfreq)
    probability = full_probability[left:right]
    sample = mass.sample_level_metrics(true_binary, pred_binary, probability)
    strict_threshold = float(cfg.event_iou_thresholds[0])
    strict = mass.event_metrics(truth, predicted, strict_threshold)
    tolerant = mass.tolerant_event_metrics(truth, predicted, float(cfg.tolerant_match_sec))
    row: Dict[str, Any] = {
        "architecture": architecture, "architecture_label": ARCHITECTURES[architecture]["label"],
        "fold": fold, "excerpt": excerpt, "record_id": f"dreams_excerpt{excerpt}",
        "channel": channel, "reference": reference.name,
        "coverage_start_sec": start, "coverage_end_sec": end,
        "scored_duration_min": (end - start) / 60.0,
        "reference_provenance": reference.provenance,
        "scoring_channel": reference.scoring_channel,
        "resolved_probability_threshold": resolved_threshold,
        "strict_iou_threshold": strict_threshold,
        "tolerant_match_sec": float(cfg.tolerant_match_sec),
        "n_reference_events": len(truth), "n_predicted_events": len(predicted),
        "reference_events_per_min": len(truth) / max((end - start) / 60.0, 1e-12),
        "predicted_events_per_min": len(predicted) / max((end - start) / 60.0, 1e-12),
        "strict_event_precision": strict["precision"],
        "strict_event_recall": strict["recall"], "strict_event_f1": strict["f1"],
        "strict_tp": strict["tp"], "strict_fp": strict["fp"], "strict_fn": strict["fn"],
        "tolerant_event_precision": tolerant["precision"],
        "tolerant_event_recall": tolerant["recall"], "tolerant_event_f1": tolerant["f1"],
        "tolerant_tp": tolerant["tp"], "tolerant_fp": tolerant["fp"], "tolerant_fn": tolerant["fn"],
        "sample_precision": sample["sample_precision"], "sample_recall": sample["sample_recall"],
        "sample_f1": sample["sample_f1"], "sample_accuracy": sample["ACC"],
        "sample_specificity": sample["SPE"], "sample_kappa": sample["KAPPA"],
        "sample_roc_auc": sample.get("sample_auc", float("nan")),
        "sample_pr_auc": sample.get("sample_pr_auc", float("nan")),
    }
    for threshold in (0.2, 0.3, 0.5):
        metric = mass.event_metrics(truth, predicted, threshold)
        suffix = str(threshold).replace(".", "_")
        row[f"event_iou_precision_{suffix}"] = metric["precision"]
        row[f"event_iou_recall_{suffix}"] = metric["recall"]
        row[f"event_iou_f1_{suffix}"] = metric["f1"]
        row[f"event_iou_tp_{suffix}"] = metric["tp"]
        row[f"event_iou_fp_{suffix}"] = metric["fp"]
        row[f"event_iou_fn_{suffix}"] = metric["fn"]
    row["predicted_events"] = predicted
    row["reference_events"] = truth
    return row


def _f1_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate_fold_reference(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (architecture, fold, reference), group in rows.groupby(["architecture", "fold", "reference"], sort=True):
        item: Dict[str, Any] = {
            "architecture": architecture, "fold": int(fold), "reference": reference,
            "n_excerpts": int(len(group)),
            "scored_duration_min": float(group["scored_duration_min"].sum()),
            "macro_strict_event_f1": float(group["strict_event_f1"].mean()),
            "macro_tolerant_event_f1": float(group["tolerant_event_f1"].mean()),
            "macro_event_iou_f1_0_5": float(group["event_iou_f1_0_5"].mean()),
            "macro_sample_pr_auc": float(group["sample_pr_auc"].mean()),
            "macro_sample_roc_auc": float(group["sample_roc_auc"].mean()),
        }
        for prefix in ("strict", "tolerant", "event_iou"):
            suffix = "" if prefix in {"strict", "tolerant"} else "_0_5"
            tp = float(group[f"{prefix}_tp{suffix}"].sum())
            fp = float(group[f"{prefix}_fp{suffix}"].sum())
            fn = float(group[f"{prefix}_fn{suffix}"].sum())
            precision, recall, f1 = _f1_from_counts(tp, fp, fn)
            name = prefix if prefix != "event_iou" else "event_iou_0_5"
            item[f"micro_{name}_precision"] = precision
            item[f"micro_{name}_recall"] = recall
            item[f"micro_{name}_f1"] = f1
            item[f"micro_{name}_tp"] = int(tp)
            item[f"micro_{name}_fp"] = int(fp)
            item[f"micro_{name}_fn"] = int(fn)
        output.append(item)
    return pd.DataFrame(output)


def architecture_summary(fold_reference: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_strict_event_f1", "micro_strict_f1", "macro_tolerant_event_f1",
        "micro_tolerant_f1", "macro_event_iou_f1_0_5", "micro_event_iou_0_5_f1",
        "macro_sample_pr_auc", "macro_sample_roc_auc",
    ]
    output = []
    for (architecture, reference), group in fold_reference.groupby(["architecture", "reference"], sort=True):
        row: Dict[str, Any] = {
            "architecture": architecture, "architecture_label": ARCHITECTURES[architecture]["label"],
            "reference": reference, "n_fold_ensembles": int(group["fold"].nunique()),
            "dependency_note": "Same DREAMS excerpts; MASS folds share training subjects. Descriptive fold dispersion, not independent replication.",
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        output.append(row)
    return pd.DataFrame(output)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def audit_artifacts(
    architectures: Sequence[str],
    folds: Sequence[int],
    checkpoint_base_dir: Path,
    mass: Any,
    device: torch.device,
    output_dir: Path,
    logger: logging.Logger,
    validate_forward: bool,
) -> List[Dict[str, Any]]:
    manifests = []
    forward_done: set[str] = set()
    for architecture in architectures:
        for fold in folds:
            bundle = load_fold_bundle(
                architecture, fold, checkpoint_base_dir, mass, device, logger,
                validate_forward=validate_forward and architecture not in forward_done,
            )
            manifests.append(bundle["manifest"])
            forward_done.add(architecture)
            del bundle["models"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_json(manifests, output_dir / "validated_checkpoint_manifest.json")
    return manifests


def run_evaluation(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    logger = configure_logging(output_dir, args.verbose)
    mass_path = Path(args.mass_pipeline).resolve()
    logger.info("Importing authoritative MASS inference implementation from %s", mass_path)
    mass = import_mass_pipeline(mass_path)
    architectures = parse_architectures(args.architectures)
    folds = parse_int_selection(args.folds, 1, 5, "folds")
    excerpts = parse_int_selection(args.excerpts, 1, 8, "excerpts")
    checkpoint_base = Path(args.checkpoint_base_dir).resolve()
    dreams_dir = Path(args.dreams_dir).resolve()
    device = resolve_device(args.device)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not checkpoint_base.is_dir():
        raise FileNotFoundError(f"Checkpoint base directory does not exist: {checkpoint_base}")
    if not dreams_dir.is_dir():
        raise FileNotFoundError(f"DREAMS directory does not exist: {dreams_dir}")
    if not args.dry_run and not args.accept_planned_coverage_policy:
        raise RuntimeError(
            "Inference is blocked because the repository does not independently document the Expert 2 "
            "scoring interval. Review README_Transformer_Only_DREAMS.md and explicitly pass "
            "--accept-planned-coverage-policy to use Expert 1=0-1000 s, Expert 2=0-1800 s, "
            "and union=0-1000 s for excerpts 1-6."
        )
    run_manifest = {
        "script_version": SCRIPT_VERSION, "timestamp_unix": time.time(),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "mass_pipeline": str(mass_path), "checkpoint_base_dir": str(checkpoint_base),
        "dreams_dir": str(dreams_dir), "output_dir": str(output_dir),
        "architectures": architectures, "folds": folds, "excerpts": excerpts,
        "device": str(device), "batch_size": args.batch_size,
        "save_probabilities": args.save_probabilities, "dry_run": args.dry_run,
        "frozen_evaluation": True, "dreams_label_access_during_prediction": False,
        "external_training": False, "external_calibration": False,
        "external_decoder_tuning": False, "external_model_selection": False,
        "coverage_policy": {
            "expert1": [0.0, args.expert1_coverage_sec],
            "expert2": [0.0, args.expert2_coverage_sec],
            "union": [0.0, min(args.expert1_coverage_sec, 1000.0)],
            "expert2_missing_excerpts": [7, 8],
            "explicitly_acknowledged": bool(args.accept_planned_coverage_policy),
        },
    }
    write_json(run_manifest, output_dir / "run_manifest.json")
    logger.info("Device=%s | architectures=%s | folds=%s | excerpts=%s", device, architectures, folds, excerpts)
    if args.dry_run:
        audit_artifacts(
            architectures, folds, checkpoint_base, mass, device, output_dir,
            logger, validate_forward=not args.skip_forward_validation,
        )
        first_root = checkpoint_base / ARCHITECTURES[architectures[0]]["directory"]
        preprocessing_cfg = mass.Config(**read_json(first_root / f"fold_{folds[0]:02d}" / "config.json"))
        metadata = []
        for excerpt in excerpts:
            _, signal_metadata = build_inference_record(
                excerpt, dreams_dir, preprocessing_cfg, mass,
            )
            metadata.append({
                "excerpt": excerpt, "channel": signal_metadata["selected_channel"],
                "unit": signal_metadata["edf_unit"],
                "unit_policy": signal_metadata["unit_policy"],
                "original_sfreq_hz": signal_metadata["original_sfreq_hz"],
                "original_samples": signal_metadata["original_samples"],
                "duration_sec": signal_metadata["duration_sec"],
                "p95_abs_volts": signal_metadata["p95_abs_volts"],
                "resampled_sfreq_hz": signal_metadata["resampled_sfreq_hz"],
                "resampled_samples": signal_metadata["resampled_samples"],
                "feature_order": ",".join(signal_metadata["feature_order"]),
                "input_probe_shape": "x".join(map(str, signal_metadata["input_probe_shape"])),
                "expert1_available": (dreams_dir / f"Visual_scoring1_excerpt{excerpt}.txt").is_file(),
                "expert2_available": (dreams_dir / f"Visual_scoring2_excerpt{excerpt}.txt").is_file(),
            })
        save_dataframe(pd.DataFrame(metadata), output_dir / "dreams_header_audit.csv")
        write_json(metadata, output_dir / "dreams_header_audit.json")
        logger.info("Dry-run validation completed. No DREAMS inference was performed.")
        return 0

    all_metric_rows: List[Dict[str, Any]] = []
    metadata_by_architecture: Dict[str, List[Dict[str, Any]]] = {}
    for architecture in architectures:
        architecture_dir = ensure_dir(output_dir / architecture)
        architecture_metrics: List[Dict[str, Any]] = []
        metadata_by_architecture[architecture] = []
        for fold in folds:
            fold_output = ensure_dir(architecture_dir / f"fold_{fold:02d}")
            bundle = load_fold_bundle(architecture, fold, checkpoint_base, mass, device, logger, validate_forward=True)
            write_json(bundle["manifest"], fold_output / "validated_checkpoint_manifest.json")
            write_json(asdict(bundle["cfg"]), fold_output / "resolved_config.json")
            write_json(bundle["params_state"], fold_output / "segmentation_params.json")
            write_json(bundle["calibrator_state"], fold_output / "probability_calibrator.json")
            write_json({"member_weights": bundle["weights"].tolist(), "seed_offsets": [0, 2]}, fold_output / "ensemble_weights.json")
            predicted_events_payload: Dict[str, Any] = {}
            threshold_rows = []
            excerpt_metadata = []
            for excerpt in excerpts:
                # This phase intentionally has no annotation arguments or annotation objects.
                record, metadata = build_inference_record(excerpt, dreams_dir, bundle["cfg"], mass)
                excerpt_metadata.append(metadata)
                member_timelines = []
                coverage_arrays = []
                for model in bundle["models"]:
                    timeline, coverage = predict_member_timeline(
                        model, record, bundle["cfg"], mass, device, args.batch_size,
                    )
                    member_timelines.append(timeline)
                    coverage_arrays.append(coverage)
                raw_probability = combine_member_probabilities(member_timelines, bundle["weights"])
                probability, threshold, predicted_events, _ = frozen_postprocess(
                    raw_probability, bundle["calibrator"], bundle["params"],
                    bundle["cfg"], mass, record["sfreq"],
                )
                if len(probability) != len(record["segmentation_raw"]):
                    raise RuntimeError("Calibrated timeline length changed unexpectedly.")
                if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
                    raise RuntimeError("Invalid calibrated probability timeline.")
                params = bundle["params"]
                threshold_rows.append({
                    "architecture": architecture, "fold": fold, "excerpt": excerpt,
                    "base_threshold": params.probability_threshold,
                    "adaptive_quantile": params.adaptive_threshold_quantile,
                    "max_adjustment": bundle["cfg"].segmentation_adaptive_threshold_max_adjustment,
                    "resolved_threshold": threshold,
                })
                predicted_events_payload[str(excerpt)] = {
                    "events": predicted_events, "resolved_threshold": threshold,
                    "timeline_samples": len(probability), "sfreq": record["sfreq"],
                    "annotation_access_during_prediction": False,
                }
                if args.save_probabilities:
                    np.savez_compressed(
                        fold_output / f"excerpt_{excerpt:02d}_probabilities.npz",
                        member_seed0=member_timelines[0], member_seed2=member_timelines[1],
                        ensemble_raw=raw_probability, ensemble_calibrated=probability,
                    )
                # References are first opened here, after frozen predictions exist.
                references = load_references_after_prediction(
                    excerpt, dreams_dir, metadata["resampled_duration_sec"], mass,
                    args.expert1_coverage_sec, args.expert2_coverage_sec,
                )
                for reference in references:
                    row = metric_row(
                        architecture, fold, excerpt, metadata["selected_channel"],
                        reference, predicted_events, probability, record["sfreq"],
                        threshold, mass, bundle["cfg"],
                    )
                    architecture_metrics.append(row)
                    all_metric_rows.append(row)
                logger.info(
                    "%s fold %d excerpt %d | channel=%s | threshold=%.4f | predictions=%d",
                    architecture, fold, excerpt, metadata["selected_channel"], threshold,
                    len(predicted_events),
                )
            write_json(predicted_events_payload, fold_output / "predicted_events.json")
            save_dataframe(pd.DataFrame(threshold_rows), fold_output / "resolved_thresholds.csv")
            write_json(threshold_rows, fold_output / "resolved_thresholds.json")
            save_dataframe(pd.DataFrame(excerpt_metadata), fold_output / "dreams_dataset_metadata.csv")
            write_json(excerpt_metadata, fold_output / "dreams_dataset_metadata.json")
            metadata_by_architecture[architecture].extend(excerpt_metadata)
            fold_rows = [row.copy() for row in architecture_metrics if row["fold"] == fold]
            for row in fold_rows:
                row.pop("predicted_events", None)
                row.pop("reference_events", None)
            save_dataframe(pd.DataFrame(fold_rows), fold_output / "excerpt_reference_metrics.csv")
            del bundle["models"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        clean_arch_rows = []
        for row in architecture_metrics:
            clean = row.copy()
            clean.pop("predicted_events", None)
            clean.pop("reference_events", None)
            clean_arch_rows.append(clean)
        arch_df = pd.DataFrame(clean_arch_rows)
        save_dataframe(arch_df, architecture_dir / "fold_by_excerpt_reference_metrics.csv")
        fold_reference = aggregate_fold_reference(arch_df)
        save_dataframe(fold_reference, architecture_dir / "fold_by_reference_metrics.csv")
        summary = architecture_summary(fold_reference)
        save_dataframe(summary, architecture_dir / "architecture_summary.csv")
        write_json(summary.to_dict("records"), architecture_dir / "architecture_summary.json")
        subgroup = arch_df.groupby(["fold", "reference", "channel"], as_index=False).agg(
            n_excerpts=("excerpt", "count"),
            strict_event_f1=("strict_event_f1", "mean"),
            tolerant_event_f1=("tolerant_event_f1", "mean"),
            event_iou_f1_0_5=("event_iou_f1_0_5", "mean"),
            sample_pr_auc=("sample_pr_auc", "mean"),
        )
        save_dataframe(subgroup, architecture_dir / "channel_derivation_subgroups.csv")

    clean_rows = []
    for row in all_metric_rows:
        clean = row.copy()
        clean.pop("predicted_events", None)
        clean.pop("reference_events", None)
        clean_rows.append(clean)
    all_df = pd.DataFrame(clean_rows)
    save_dataframe(all_df, output_dir / "comparison_fold_excerpt_reference_metrics.csv")
    all_fold_reference = aggregate_fold_reference(all_df)
    save_dataframe(all_fold_reference, output_dir / "comparison_fold_reference_metrics.csv")
    comparison = architecture_summary(all_fold_reference)
    save_dataframe(comparison, output_dir / "comparison_summary.csv")
    write_json(comparison.to_dict("records"), output_dir / "comparison_summary.json")
    matched = all_df.pivot_table(
        index=["fold", "excerpt", "reference", "channel"], columns="architecture",
        values=["strict_event_f1", "tolerant_event_f1", "event_iou_f1_0_5", "sample_pr_auc"],
        aggfunc="first",
    )
    matched.columns = [f"{metric}__{architecture}" for metric, architecture in matched.columns]
    save_dataframe(matched.reset_index(), output_dir / "matched_architecture_comparison.csv")
    coverage_audit = all_df[[
        "excerpt", "reference", "coverage_start_sec", "coverage_end_sec",
        "scored_duration_min", "reference_provenance", "scoring_channel",
    ]].drop_duplicates().sort_values(["excerpt", "reference"])
    save_dataframe(coverage_audit, output_dir / "scoring_coverage_audit.csv")
    write_json(coverage_audit.to_dict("records"), output_dir / "scoring_coverage_audit.json")
    limitations = {
        "dataset_shift": "MASS training retained N2; DREAMS excerpts are evaluated as mixed-stage because no verified stage mask is applied.",
        "channel_shift": "MASS used C3; DREAMS uses C3-A1 for excerpts 1/3 and CZ-A1 for the other excerpts.",
        "coverage": "Expert 1 is scored over 0-1000 s; Expert 2 uses the explicitly acknowledged planned 0-1800 s policy; union is restricted to 0-1000 s and excerpts 1-6.",
        "dependency": "Five MASS fold ensembles evaluate the same DREAMS records and have overlapping MASS training sets; they are not independent replicates.",
        "adaptation": "No DREAMS training, calibration, threshold tuning, decoder tuning, or label-informed selection was performed.",
        "exposure": "Saved MASS-only manifests and empty external SSL paths should be cited before calling the evaluation fully unseen.",
    }
    write_json(limitations, output_dir / "dataset_shift_and_limitations.json")
    logger.info("Frozen DREAMS evaluation completed. Summary: %s", output_dir / "comparison_summary.csv")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen external DREAMS evaluation of MASS-trained Transformer-only spindle detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dreams-dir", default="DREAMS Dataset", help="Directory containing excerpt1.edf..excerpt8.edf and expert scoring files.")
    parser.add_argument("--mass-pipeline", default="transformer_mass_pipeline.py", help="Authoritative MASS Transformer pipeline module.")
    parser.add_argument("--checkpoint-base-dir", default="Updated results", help="Directory containing the three architecture result roots.")
    parser.add_argument("--output-dir", default="results_transformer_only_dreams_frozen", help="New external-evaluation output directory.")
    parser.add_argument("--architectures", default="all", help="Comma-separated: global,sparse_local,conformer, or all.")
    parser.add_argument("--folds", default="all", help="Fold selection such as all, 1, or 1-3.")
    parser.add_argument("--excerpts", default="all", help="Excerpt selection such as all, 1, or 1,3,6.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of 20-s windows per inference batch.")
    parser.add_argument("--save-probabilities", action="store_true", help="Save compressed member, ensemble, and calibrated probability timelines.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all selected artifacts and DREAMS headers without inference.")
    parser.add_argument("--skip-forward-validation", action="store_true", help="Dry-run only: skip one synthetic forward pass per architecture.")
    parser.add_argument("--accept-planned-coverage-policy", action="store_true", help="Acknowledge the planned expert scoring intervals described in the audit report.")
    parser.add_argument("--expert1-coverage-sec", type=float, default=1000.0)
    parser.add_argument("--expert2-coverage-sec", type=float, default=1800.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prohibited = [name for name in vars(args) if any(word in name for word in ("finetune", "tune_decoding", "loo_calibration"))]
    if prohibited:
        raise RuntimeError(f"External adaptation options are prohibited: {prohibited}")
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
