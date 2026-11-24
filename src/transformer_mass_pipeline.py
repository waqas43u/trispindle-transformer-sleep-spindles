#!/usr/bin/env python3
"""MASS-SS2-only research pipeline (Python 3.10+, PyTorch, MNE, SciPy,
NumPy, pandas, scikit-learn, matplotlib; SHAP optional).

Linux examples (all experimental logic is in this file):
  python transformer_mass_pipeline.py --data_dir /data/MASS/SS2 --output_dir /runs/new_mass --mode all
  python transformer_mass_pipeline.py --data_dir /data/MASS/SS2 --output_dir /runs/new_xai --mode xai --checkpoint_dir /runs/new_mass
  python transformer_mass_pipeline.py --data_dir /data/MASS/SS2 --validate_only
Use --segmentation_transformer_only_block_type transformer, local_transformer,
or conformer with the original architecture/hyperparameter CLI options.
Use --segmentation_transformer_only_local_global_stride with local_transformer
for the original sparse-global-key setting; zero retains purely local attention.
Use --paper_comparison_dirs with comma-separated completed run directories to
export architecture comparisons. Existing scientific CLI settings remain available.
Defaults select four EEG representations, fold-local SSL and saved checkpoints;
pass the same architecture, loss and ensemble options as the comparison experiment.
XAI requires the saved decoder, ensemble weights, calibrator and subject splits;
legacy checkpoints lacking these artifacts are rejected rather than recalibrated.
Outputs are created exclusively; existing experiment directories are never reused.
XAI paper images: at most six JPGs per run in explainability_paper/
(three TP/FP/FN composites, SHAP and SmoothGrad summaries, ablation effects).
Per-subject numerical outputs are retained.
XAI explains frozen weighted probabilities before calibration; ablation uses the
frozen calibrated decoder. Attribution is descriptive, not a causal effect.
"""
import argparse
import gc
import itertools
import json
import logging
import math
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import mne
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from sklearn import metrics as sk_metrics
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    matplotlib = None
    plt = None

try:
    import shap
except Exception:  # pragma: no cover - SHAP is optional at runtime
    shap = None


ANNOTATION_MODES = ("expert1", "expert2", "union", "intersection")
STAGE_MODES = ("n2_only", "n2_n3", "nrem", "all_sleep", "none")
CANDIDATE_MODES = ("jiang_teo", "sigma_envelope", "hybrid_union")
LOSS_TYPES = ("weighted_bce", "focal", "bce_dice")
CV_MODES = ("paper_exact_e12", "random_fivefold_e12", "fivefold", "loso")
NORMALIZATION_MODES = ("dataset_train", "subject_zscore", "subject_robust", "none")
CALIBRATION_METHODS = ("none", "platt", "isotonic")
MODEL_MODES = ("candidate_cnn", "segmentation_unet", "segmentation_unet_transformer", "segmentation_transformer_only")
SEGMENTATION_LOSS_TYPES = ("bce_tversky", "bce_fpfn", "fpfn", "bce_tversky_fpfn")

PAPER_TEST_ID = np.array([
    [2, 5, 3],
    [4, 14, 8],
    [7, 10, 13],
    [1, 6, 9],
    [11, 15, 12],
], dtype=int)
PAPER_TRAIN_ID = np.array([
    [6, 13, 7, 10, 11, 9, 14, 15, 8],
    [3, 1, 7, 2, 15, 12, 10, 11, 5],
    [3, 1, 14, 9, 2, 8, 4, 12, 5],
    [14, 3, 12, 5, 7, 15, 13, 10, 11],
    [1, 6, 8, 3, 2, 10, 13, 9, 7],
], dtype=int)
PAPER_VAL_ID = np.array([
    [12, 1, 4],
    [13, 9, 6],
    [6, 15, 11],
    [2, 8, 4],
    [14, 5, 4],
], dtype=int)


@dataclass
class Config:
    data_dir: str
    output_dir: str = "results_mass_ss2_only_transformer"
    mode: str = "all"
    checkpoint_dir: str = ""
    annotation_mode: str = "union"
    cv_mode: str = "paper_exact_e12"
    stage_mode: str = "n2_only"
    candidate_mode: str = "sigma_envelope"
    model_mode: str = "segmentation_transformer_only"
    channel_target: str = "C3"
    segmentation_use_secondary_eeg_channel: bool = False
    segmentation_secondary_channel_target: str = "C4"
    segmentation_secondary_gate_enabled: bool = False
    segmentation_secondary_gate_initial: float = 0.10
    resample_hz: float = 256.0
    max_folds: Optional[int] = None
    start_fold: int = 1
    debug: bool = False

    # Compact publication mode by default. Dense per-sample timelines/vectors and
    # model checkpoints can be enabled from the CLI when needed for debugging.
    save_subject_level_outputs: bool = True
    save_probabilities: bool = False
    save_candidates: bool = False
    save_checkpoints: bool = True
    save_training_history: bool = True
    save_candidate_diagnostics: bool = False
    allow_implicit_spindle_labels: bool = False
    require_dual_expert_files: bool = True

    remove_anomaly: bool = True
    anomaly_range_divisor: float = 530.0
    anomaly_jump_factor: float = 200.0
    anomaly_recovery_factor: float = 50.0
    bandpass_low_hz: float = 0.3
    bandpass_high_hz: float = 35.0
    notch_enabled: bool = False
    notch_hz: float = 50.0
    sigma_low_hz: float = 12.0
    sigma_high_hz: float = 16.0
    normalization_mode: str = "subject_robust"

    use_raw_channel: bool = True
    use_sigma_channel: bool = True
    use_teo_channel: bool = True
    use_envelope_channel: bool = True
    use_candidate_feature_channels: bool = True

    candidate_window_sec: float = 4.0
    candidate_min_sec: float = 0.5
    candidate_max_sec: float = 2.0
    candidate_merge_gap_sec: float = 0.10
    candidate_threshold_B: float = 1.5
    candidate_threshold_T: float = 4.0
    tune_candidate_params: bool = True
    resume_tuning: bool = True
    candidate_B_grid: Tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0)
    candidate_T_grid: Tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    envelope_mad_multiplier: float = 2.5
    candidate_recall_boost_enabled: bool = True
    candidate_recall_boost_B_offset: float = 0.25
    candidate_recall_boost_T_offset: float = 1.0
    candidate_recall_boost_min_sec: float = 0.4
    candidate_recall_boost_merge_gap_sec: float = 0.15
    candidate_peak_proposals_enabled: bool = True
    candidate_peak_z_threshold: float = 1.5
    candidate_peak_duration_sec: float = 1.0
    candidate_peak_min_distance_sec: float = 0.35
    candidate_peak_max_per_min: float = 30.0
    candidate_tuning_mean_f1_weight: float = 0.55
    candidate_tuning_min_f1_weight: float = 0.25
    candidate_tuning_mean_recall_weight: float = 0.20
    soft_candidate_labels_enabled: bool = True
    soft_candidate_label_lower_scale: float = 0.5
    soft_candidate_label_gamma: float = 1.0

    # Keep label matching moderately permissive, but avoid the 0.05 setting that
    # previously increased precision at the expense of strict event recall/F1.
    lambda_grid: Tuple[float, ...] = (0.10, 0.12, 0.15, 0.18)
    threshold_grid: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    # Defaults are shifted toward recall recovery; the validation search below
    # tunes these explicitly per fold.
    hysteresis_gap: float = 0.10
    hysteresis_gap_grid: Tuple[float, ...] = (0.05, 0.10, 0.15)
    final_min_sec: float = 0.4
    final_min_sec_grid: Tuple[float, ...] = (0.4, 0.5)
    final_max_sec: float = 2.0
    final_max_sec_grid: Tuple[float, ...] = (1.4, 1.6, 2.0)
    final_merge_gap_sec: float = 0.15
    final_merge_gap_grid: Tuple[float, ...] = (0.10, 0.15)
    adaptive_threshold_quantile_grid: Tuple[float, ...] = (1.0, 0.9, 0.8)
    adaptive_threshold_max_adjustment: float = 0.10
    stage3_selection_mean_f1_weight: float = 0.65
    stage3_selection_mean_recall_weight: float = 0.10
    stage3_selection_min_f1_weight: float = 0.20
    stage3_selection_min_recall_weight: float = 0.05
    stage3_recall_floor: float = 0.50
    stage3_recall_floor_penalty: float = 0.20
    stage3_prob_threshold_floor: float = 0.35
    stage3_prob_threshold_cap: float = 0.80
    stage3_support_smooth_sec: float = 0.10
    stage3_support_relative_threshold: float = 0.35
    event_iou_thresholds: Tuple[float, ...] = (0.2, 0.3, 0.5)
    tolerant_match_sec: float = 0.5

    batch_size: int = 512
    
    epochs: int = 40
    tuning_epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip: float = 5.0
    mixed_precision: bool = True
    
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    window_extraction_batch_size: int = 4096
    cache_candidate_windows: bool = True
    enable_tf32: bool = True
    cudnn_benchmark: bool = True
    
    
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    dice_weight: float = 0.4
    positive_weight_max: float = 12.0

    model_channels: Tuple[int, ...] = (32, 64, 96, 128, 256)
    model_kernel_size: int = 11
    model_dropout: float = 0.10
    model_use_residual: bool = True
    model_use_temporal_attention_gate: bool = True

    segmentation_window_sec: float = 20.0
    segmentation_overlap: float = 0.50
    segmentation_batch_size: int = 32
    segmentation_epochs: int = 100
    segmentation_learning_rate: float = 1e-4
    segmentation_weight_decay: float = 1e-8
    segmentation_optimizer: str = "adam"
    segmentation_early_stopping_patience: int = 15
    segmentation_scheduler_patience: int = 10
    segmentation_scheduler_metric: str = "val_loss"
    # Window-level event reconstruction is useful as an optional diagnostic but
    # was both slow and poorly correlated with full-record subject performance.
    segmentation_checkpoint_metric: str = "sample_pr_auc"
    segmentation_checkpoint_threshold_grid: Tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70)
    # The two-seed baseline generalized best in this range. Add only one lower
    # recall-oriented endpoint instead of changing the learned representation.
    segmentation_threshold_grid: Tuple[float, ...] = (0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75)
    segmentation_min_sec: float = 0.5
    segmentation_min_sec_grid: Tuple[float, ...] = (0.5, 0.6)
    # A value of zero disables cropping. Positive values cap overly broad
    # detections by retaining the highest-probability contiguous sub-event.
    segmentation_max_sec_grid: Tuple[float, ...] = (0.0,)
    segmentation_merge_gap_sec: float = 0.0
    segmentation_merge_gap_grid: Tuple[float, ...] = (0.05, 0.10, 0.15)
    segmentation_boundary_gap_grid: Tuple[float, ...] = (0.0,)
    segmentation_boundary_max_extend_sec_grid: Tuple[float, ...] = (0.0,)
    # Optional symmetric contraction after probability decoding. This is tuned
    # on validation subjects and is useful when soft boundary targets produce
    # systematically overlong spindle masks.
    segmentation_boundary_trim_sec_grid: Tuple[float, ...] = (0.0,)
    segmentation_adaptive_threshold_quantile_grid: Tuple[float, ...] = (0.90, 0.95, 1.0)
    segmentation_adaptive_threshold_max_adjustment: float = 0.15
    segmentation_density_penalty_enabled: bool = True
    segmentation_density_target_low: float = 0.80
    segmentation_density_target_high: float = 1.15
    segmentation_density_over_penalty: float = 0.15
    segmentation_density_under_penalty: float = 0.10
    segmentation_duration_penalty_enabled: bool = False
    segmentation_duration_ratio_low: float = 0.75
    segmentation_duration_ratio_high: float = 1.50
    segmentation_duration_over_penalty: float = 0.10
    segmentation_duration_under_penalty: float = 0.05
    segmentation_selection_mean_f1_weight: float = 0.75
    segmentation_selection_min_f1_weight: float = 0.20
    segmentation_selection_mean_precision_weight: float = 0.0
    segmentation_selection_mean_recall_weight: float = 0.05
    segmentation_selection_tolerant_f1_weight: float = 0.0
    segmentation_precision_floor: float = 0.72
    segmentation_min_precision_floor: float = 0.55
    segmentation_precision_floor_penalty: float = 0.15
    segmentation_min_precision_floor_penalty: float = 0.05
    segmentation_recall_floor: float = 0.68
    segmentation_min_recall_floor: float = 0.50
    segmentation_recall_floor_penalty: float = 0.10
    segmentation_min_recall_floor_penalty: float = 0.05
    segmentation_f1_fallback_enabled: bool = True
    segmentation_f1_fallback_margin: float = 0.002
    segmentation_f1_fallback_min_precision: float = 0.68
    segmentation_clip_uv: float = 150.0
    segmentation_normalize_inputs: bool = False
    segmentation_use_attention_gates: bool = True
    segmentation_use_sigma_channel: bool = True
    segmentation_use_slow_sigma_channel: bool = True
    segmentation_slow_sigma_low_hz: float = 10.0
    segmentation_slow_sigma_high_hz: float = 12.0
    segmentation_use_teo_channel: bool = False
    segmentation_use_envelope_channel: bool = True
    segmentation_event_mean_prob_grid: Tuple[float, ...] = (0.0,)
    segmentation_event_peak_prob_grid: Tuple[float, ...] = (0.0,)
    # Two members recover most ensemble stability at two-thirds the cost of the
    # previous three-member experiment.
    segmentation_ensemble_seed_offsets: Tuple[int, ...] = (0, 2)
    # With a heterogeneous SSL/scratch ensemble, select the blend on validation
    # sample PR-AUC before calibration and event-level decoding. The restricted
    # grid avoids unstable single-model selection from only three validators.
    segmentation_tune_ensemble_weights_enabled: bool = False
    segmentation_ensemble_first_weight_grid: Tuple[float, ...] = (0.25, 0.50, 0.75)
    segmentation_ensemble_weight_selection_metric: str = "sample_pr_auc"
    segmentation_probability_calibration_enabled: bool = True
    segmentation_probability_calibration_method: str = "platt"
    segmentation_calibration_max_samples: int = 200000
    segmentation_base_channels: int = 32
    segmentation_depth: int = 4
    segmentation_tversky_weight: float = 0.5
    segmentation_tversky_alpha: float = 0.5
    segmentation_tversky_beta: float = 0.5
    segmentation_loss_type: str = "bce_fpfn"
    segmentation_bce_weight: float = 1.0
    segmentation_fpfn_weight: float = 1.0
    segmentation_positive_window_sampling_enabled: bool = True
    segmentation_positive_window_weight: float = 3.0
    segmentation_subject_balanced_sampling_enabled: bool = False
    segmentation_soft_labels_enabled: bool = True
    segmentation_soft_label_shoulder_sec: float = 0.15
    segmentation_soft_label_edge_value: float = 0.50
    # Retain the union reference for evaluation while reducing the supervised
    # confidence of events supported by only one expert.
    segmentation_annotator_confidence_labels_enabled: bool = False
    segmentation_single_expert_target_weight: float = 0.75
    segmentation_annotator_agreement_tolerance_sec: float = 0.30
    segmentation_temporal_bottleneck_enabled: bool = False
    segmentation_temporal_bottleneck_layers: int = 1
    segmentation_temporal_bottleneck_dropout: float = 0.15
    segmentation_transformer_heads: int = 2
    segmentation_transformer_ff_multiplier: int = 2
    segmentation_transformer_only_embed_dim: int = 192
    segmentation_transformer_only_layers: int = 5
    segmentation_transformer_only_heads: int = 6
    segmentation_transformer_only_ff_multiplier: int = 4
    segmentation_transformer_only_dropout: float = 0.20
    segmentation_transformer_only_patch_size: int = 32
    segmentation_transformer_only_patch_stride: int = 16
    segmentation_transformer_only_max_patches: int = 2048
    segmentation_transformer_only_multiscale_enabled: bool = False
    segmentation_transformer_only_long_patch_size: int = 128
    segmentation_transformer_only_block_type: str = "conformer"
    segmentation_transformer_only_local_attention_window: int = 9
    # Optional sparse global keys complement the local attention band. A value
    # of zero gives a strictly local Transformer; e.g. 16 lets every query
    # attend to every 16th token while retaining its dense local neighborhood.
    segmentation_transformer_only_local_global_stride: int = 0
    segmentation_transformer_only_conv_kernel: int = 9
    segmentation_transformer_only_local_mixing: bool = True
    segmentation_transformer_only_local_mixing_dropout: float = 0.10
    segmentation_transformer_only_patch_stats: bool = True
    # Sinusoidal initialization supplies temporal order before supervised
    # learning; the embedding remains trainable during fine-tuning.
    segmentation_transformer_only_positional_init: str = "zeros"
    # Optional dual-domain tokenization. A centered long-context FFT token is
    # fused with each short temporal patch through a learnable residual gate.
    # This remains a Transformer-only architecture (no convolutional frontend).
    segmentation_transformer_only_spectral_tokens_enabled: bool = False
    segmentation_transformer_only_spectral_patch_size: int = 256
    segmentation_transformer_only_spectral_low_hz: float = 8.0
    segmentation_transformer_only_spectral_high_hz: float = 20.0
    segmentation_transformer_only_spectral_input_channels: int = 1
    segmentation_transformer_only_spectral_gate_initial: float = 0.10
    # Event-aware supervised contrastive regularization acts on Transformer
    # patch embeddings during fine-tuning. Unlike generic EEG SSL, it directly
    # separates spindle and background tokens and has no inference-time cost.
    segmentation_supervised_contrastive_enabled: bool = False
    segmentation_supervised_contrastive_weight: float = 0.05
    segmentation_supervised_contrastive_temperature: float = 0.10
    segmentation_supervised_contrastive_projection_dim: int = 64
    segmentation_supervised_contrastive_max_tokens: int = 512
    segmentation_supervised_contrastive_positive_threshold: float = 0.25
    # Anchor-free instance supervision: each Transformer patch predicts whether
    # it owns a spindle plus its center offset and log-duration. The auxiliary
    # heads are discarded at inference, so dense decoding remains comparable.
    segmentation_event_geometry_enabled: bool = False
    segmentation_event_geometry_objectness_weight: float = 0.10
    segmentation_event_geometry_center_weight: float = 0.05
    segmentation_event_geometry_duration_weight: float = 0.05
    segmentation_event_geometry_focal_gamma: float = 2.0
    transformer_ssl_pretraining_enabled: bool = True
    transformer_ssl_pretraining_epochs: int = 8
    transformer_ssl_objective: str = "masked_reconstruction"
    transformer_ssl_mask_ratio: float = 0.40
    transformer_ssl_learning_rate: float = 1e-4
    transformer_ssl_temperature: float = 0.10
    transformer_ssl_external_data_dir: str = ""
    # Optionally pretrain only the first ensemble member. This creates a
    # transformer-only heterogeneous ensemble: SSL supplies recall while a
    # supervised-from-scratch member can recover precision.
    transformer_ssl_ensemble_mode: str = "all"
    # Optional labelled-domain transfer. DREAMS annotations supervise a short
    # intermediate segmentation stage after SSL and before MASS fine-tuning.
    transformer_external_supervised_pretraining_enabled: bool = False
    transformer_external_supervised_pretraining_epochs: int = 8
    transformer_external_supervised_learning_rate: float = 5e-5
    transformer_external_supervised_validation_records: int = 2
    transformer_external_supervised_ensemble_mode: str = "first_only"
    transformer_warmstart_cnn_dir: str = ""
    transformer_warmstart_enabled: bool = False
    segmentation_auxiliary_heads_enabled: bool = True
    segmentation_center_loss_weight: float = 0.25
    segmentation_boundary_loss_weight: float = 0.15
    segmentation_edge_consistency_weight: float = 0.0
    segmentation_multi_annotator_enabled: bool = False
    segmentation_multi_annotator_loss_weight: float = 0.15
    segmentation_auxiliary_focal_gamma: float = 0.0
    segmentation_auxiliary_pos_weight: float = 1.0
    segmentation_center_label_width_sec: float = 0.25
    segmentation_boundary_label_width_sec: float = 0.08
    segmentation_structured_decoder_enabled: bool = False
    segmentation_center_threshold: float = 0.35
    segmentation_boundary_threshold: float = 0.35
    segmentation_center_min_distance_sec: float = 0.30
    segmentation_boundary_max_shift_sec: float = 0.30
    segmentation_density_adaptive_threshold_enabled: bool = False
    segmentation_density_adaptive_low_multiplier: float = 0.85
    segmentation_density_adaptive_high_multiplier: float = 1.15
    segmentation_density_adaptive_max_adjustment: float = 0.10
    segmentation_density_adaptive_steps: int = 21
    segmentation_candidate_consensus_enabled: bool = False
    segmentation_candidate_consensus_B: float = 0.75
    segmentation_candidate_consensus_T: float = 8.0
    segmentation_candidate_consensus_overlap_sec: float = 0.05
    segmentation_candidate_consensus_keep_peak_prob: float = 0.85

    augment_noise_std: float = 0.01
    augment_scale_std: float = 0.01
    augment_time_jitter_sec: float = 0.05
    augment_invert_probability: float = 0.5
    augment_enabled: bool = True
    balanced_sampling: bool = True
    hard_negative_mining_enabled: bool = True
    hard_negative_fraction: float = 0.25
    hard_negative_boost: float = 3.0
    hard_negative_finetune_epochs: int = 5
    hard_negative_accept_only_if_improved: bool = True
    hard_negative_accept_metric: str = "val_candidate_pr_auc"
    hard_negative_accept_margin: float = 0.0
    # Window-level hard-negative fine-tuning hurt most transformer-only members
    # in the reference run. Keep it available for candidate models, but opt the
    # segmentation branch out unless explicitly requested.
    segmentation_hard_negative_mining_enabled: bool = False
    segmentation_abort_below_fold_f1: float = 0.0
    segmentation_abort_check_fold: int = 1
    subject_failure_report_top_k: int = 5
    probability_calibration_enabled: bool = True
    probability_calibration_method: str = "platt"
    generate_paper_outputs: bool = True
    paper_figure_dpi: int = 180
    paper_example_window_sec: float = 60.0
    paper_comparison_dirs: Tuple[str, ...] = ()
    paper_comparison_only: bool = False
    paper_data_folder_name: str = "Paper data"
    run_explainability: bool = False
    explainability_figure_dpi: int = 300
    explainability_background_windows: int = 16
    explainability_shap_windows: int = 12
    explainability_smoothgrad_samples: int = 24
    explainability_smoothgrad_noise_std: float = 0.08
    explainability_examples_per_type: int = 1
    explainability_window_sec: float = 8.0
    external_dataset: str = "none"
    external_data_dir: str = ""
    external_checkpoint_dir: str = ""
    external_max_models: int = 25
    external_use_mass_fold_params: bool = True
    external_default_threshold: float = 0.675
    external_default_merge_gap_sec: float = 0.09
    external_default_min_sec: float = 0.6
    external_default_adaptive_quantile: float = 1.0
    external_tune_decoding: bool = False
    external_calibration_ids: str = ""
    external_test_ids: str = ""
    external_probability_calibration_enabled: bool = False

    statistical_baseline_dir: str = ""
    independent_run_seeds: Tuple[int, ...] = ()
    seed: int = 42
    device: str = "auto"


def is_segmentation_mode(cfg: Config) -> bool:
    return cfg.model_mode in ("segmentation_unet", "segmentation_unet_transformer", "segmentation_transformer_only")


def is_transformer_segmentation_mode(cfg: Config) -> bool:
    return cfg.model_mode == "segmentation_unet_transformer"


def is_transformer_only_segmentation_mode(cfg: Config) -> bool:
    return cfg.model_mode == "segmentation_transformer_only"


@dataclass
class SubjectFiles:
    subject_id: str
    psg_path: Path
    base_path: Optional[Path]
    spindle_e1_path: Optional[Path]
    spindle_e2_path: Optional[Path]


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class Stage2Params:
    lambda_thr: float
    probability_threshold: float
    hysteresis_gap: float
    final_merge_gap_sec: float
    final_min_sec: float
    final_max_sec: float
    adaptive_threshold_quantile: float


@dataclass
class SegmentationParams:
    probability_threshold: float
    final_merge_gap_sec: float
    final_min_sec: float
    adaptive_threshold_quantile: float
    boundary_gap: float = 0.0
    boundary_max_extend_sec: float = 0.0
    boundary_trim_sec: float = 0.0
    event_min_mean_prob: float = 0.0
    event_min_peak_prob: float = 0.0
    final_max_sec: float = 0.0


def parse_float_tuple_arg(value: Optional[str], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if value is None:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return tuple(float(x) for x in items) if items else default


def parse_int_tuple_arg(value: Optional[str], default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return tuple(int(x) for x in items) if items else default


def parse_str_tuple_arg(value: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return tuple(items) if items else default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} Â± {std:.{decimals}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def plotting_enabled() -> bool:
    return plt is not None


def save_fold_training_plot(fold_dir: Path, history: Dict[str, List[float]], cfg: Config, logger: logging.Logger) -> None:
    if not plotting_enabled():
        logger.warning("matplotlib not available; skipping training plot for %s", fold_dir.name)
        return
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_f1 = history.get("val_candidate_f1", history.get("val_sample_f1", []))
    val_pr_auc = history.get("val_candidate_pr_auc", history.get("val_sample_pr_auc", []))
    if not train_loss:
        return
    epochs = np.arange(1, len(train_loss) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2)
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{fold_dir.name} Training Curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    metric_epochs = np.arange(1, len(val_f1) + 1)
    if val_f1:
        axes[1].plot(metric_epochs, val_f1, label="val_f1", linewidth=2)
    if val_pr_auc:
        axes[1].plot(np.arange(1, len(val_pr_auc) + 1), val_pr_auc, label="val_pr_auc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fold_dir / "paper_training_curves.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def save_fold_subject_plot(fold_dir: Path, subject_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    if not plotting_enabled():
        return
    if subject_df.empty:
        return
    df = subject_df.sort_values("strict_event_f1", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df["strict_event_f1"], width=width, label="strict_event_f1")
    if is_segmentation_mode(cfg):
        ax.bar(x, df["tolerant_event_f1"], width=width, label="tolerant_event_f1")
        ax.bar(x + width, df["sample_f1"], width=width, label="sample_f1")
    else:
        ax.bar(x, df["candidate_f1"], width=width, label="candidate_classifier_f1")
        ax.bar(x + width, df["candidate_recall"], width=width, label="candidate_classifier_recall")
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject_id"], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{fold_dir.name} Subject-Level Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fold_dir / "paper_subject_performance.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    cfg: Config,
    title: str = "Precision-Recall Curve",
) -> None:
    if not plotting_enabled():
        return
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
    ap = float(sk_metrics.average_precision_score(y_true, y_prob))
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.plot(recall, precision, color="#1f77b4", linewidth=1.8, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(Path(path).with_suffix(".jpg"), dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_fold_example_plot(fold_dir: Path, subject_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    if not plotting_enabled():
        return
    if subject_df.empty:
        return
    subject_id = str(subject_df.sort_values("strict_event_f1", ascending=False).iloc[0]["subject_id"])
    vectors_path = fold_dir / f"{subject_id}_vectors.csv"
    if not vectors_path.exists():
        return
    vec = pd.read_csv(vectors_path)
    if vec.empty or "truth" not in vec:
        return
    truth = vec["truth"].to_numpy()
    n = len(truth)
    positive_idx = np.flatnonzero(truth > 0)
    center = int(positive_idx[0]) if len(positive_idx) else n // 2
    half_window = int(round(cfg.paper_example_window_sec * cfg.resample_hz / 2.0))
    start = max(0, center - half_window)
    stop = min(n, center + half_window)
    t = np.arange(start, stop) / cfg.resample_hz
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, vec["probability"].to_numpy()[start:stop], color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("Prob.")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(f"{fold_dir.name} qualitative example: {subject_id}")
    axes[1].plot(t, vec["truth"].to_numpy()[start:stop], label="truth", linewidth=1.5)
    axes[1].plot(t, vec["pred_binary"].to_numpy()[start:stop], label="pred", linewidth=1.2)
    axes[1].set_ylabel("Truth/Pred")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[2].plot(t, vec["stage1_candidate_binary"].to_numpy()[start:stop], color="tab:orange", linewidth=1.2)
    axes[2].set_ylabel("Stage1")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fold_dir / f"paper_qualitative_example_{subject_id}.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def generate_fold_paper_outputs(fold_dir: Path, cfg: Config, logger: logging.Logger) -> None:
    if not cfg.generate_paper_outputs:
        return
    history_path = fold_dir / "training_history.json"
    subject_metrics_path = fold_dir / "subject_metrics.csv"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        save_fold_training_plot(fold_dir, history, cfg, logger)
    if subject_metrics_path.exists():
        subject_df = pd.read_csv(subject_metrics_path)
        save_fold_subject_plot(fold_dir, subject_df, cfg, logger)
        save_fold_example_plot(fold_dir, subject_df, cfg, logger)


def save_reference_result_tables(outdir: Path, logger: logging.Logger) -> None:
    subject_path = outdir / "subject_metrics_by_reference.csv"
    fold_path = outdir / "fold_summary_by_reference.csv"
    if subject_path.exists():
        subject_df = pd.read_csv(subject_path)
        metric_map = {
            "strict_event_precision": "Precision",
            "strict_event_recall": "Recall",
            "strict_event_f1": "F1",
        }
        pivot = subject_df.pivot_table(
            index="subject_id",
            columns="reference",
            values=list(metric_map.keys()),
            aggfunc="first",
        )
        rows = []
        dataset_path = outdir / "dataset_subject_summary.csv"
        all_subjects = []
        if dataset_path.exists():
            all_subjects = pd.read_csv(dataset_path)["subject_id"].astype(str).tolist()
        for sid in all_subjects or sorted(subject_df["subject_id"].astype(str).unique()):
            row: Dict[str, Any] = {"Subject": sid}
            for ref_label, ref_key in (("E1", "expert1"), ("E2", "expert2"), ("Union", "union")):
                for metric_key, metric_label in metric_map.items():
                    value = float("nan")
                    try:
                        value = float(pivot.loc[sid, (metric_key, ref_key)])
                    except Exception:
                        pass
                    row[f"{ref_label} {metric_label}"] = value
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(outdir / "paper_subject_reference_results_table.csv", index=False)
        (outdir / "paper_subject_reference_results_table.md").write_text(
            dataframe_to_markdown(table.round(4)),
            encoding="utf-8",
        )
    if fold_path.exists():
        fold_df = pd.read_csv(fold_path)
        metric_map = {
            "strict_event_precision": "Precision",
            "strict_event_recall": "Recall",
            "strict_event_f1": "F1",
        }
        pivot = fold_df.pivot_table(
            index="fold",
            columns="reference",
            values=list(metric_map.keys()),
            aggfunc="first",
        )
        rows = []
        for fold in sorted(fold_df["fold"].unique()):
            row = {"Fold": int(fold)}
            for ref_label, ref_key in (("E1", "expert1"), ("E2", "expert2"), ("Union", "union")):
                for metric_key, metric_label in metric_map.items():
                    value = float("nan")
                    try:
                        value = float(pivot.loc[fold, (metric_key, ref_key)])
                    except Exception:
                        pass
                    row[f"{ref_label} {metric_label}"] = value
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(outdir / "paper_fold_reference_results_table.csv", index=False)
        (outdir / "paper_fold_reference_results_table.md").write_text(
            dataframe_to_markdown(table.round(4)),
            encoding="utf-8",
        )


def save_fold_metric_reference_plots(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    fold_path = outdir / "fold_summary_by_reference.csv"
    if not plotting_enabled() or not fold_path.exists():
        return
    df = pd.read_csv(fold_path)
    ref_order = [("expert1", "E1"), ("expert2", "E2"), ("union", "Union")]
    metric_sets = [
        (
            "strict",
            {
                "strict_event_precision": "Precision",
                "strict_event_recall": "Recall",
                "strict_event_f1": "F1",
            },
        ),
        (
            "tolerant",
            {
                "tolerant_event_precision": "Precision",
                "tolerant_event_recall": "Recall",
                "tolerant_event_f1": "F1",
            },
        ),
    ]
    for prefix, metrics_dict in metric_sets:
        fig, axes = plt.subplots(len(ref_order), 1, figsize=(9, 9), sharex=True)
        if len(ref_order) == 1:
            axes = [axes]
        for ax, (ref_key, ref_label) in zip(axes, ref_order):
            ref_df = df[df["reference"] == ref_key].sort_values("fold")
            for metric, label in metrics_dict.items():
                if metric in ref_df:
                    ax.plot(ref_df["fold"], ref_df[metric], marker="o", linewidth=2, label=label)
            ax.set_title(f"{ref_label} {prefix.capitalize()} Event Metrics")
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Score")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right")
        axes[-1].set_xlabel("Fold")
        fig.tight_layout()
        fig.savefig(outdir / f"paper_fold_{prefix}_metrics_by_reference.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_model_parameter_table(outdir: Path, cfg: Config) -> None:
    rows = [
        ("Model mode", cfg.model_mode),
        ("Input channel", cfg.channel_target),
        (
            "Secondary EEG channel",
            cfg.segmentation_secondary_channel_target
            if cfg.segmentation_use_secondary_eeg_channel else "disabled",
        ),
        (
            "Secondary EEG learnable gate",
            f"enabled (initial={cfg.segmentation_secondary_gate_initial:g})"
            if cfg.segmentation_secondary_gate_enabled else "disabled",
        ),
        ("Annotation mode", cfg.annotation_mode),
        ("Stage mode", cfg.stage_mode),
        ("Sampling frequency", f"{cfg.resample_hz:g} Hz"),
        ("Segmentation window", f"{cfg.segmentation_window_sec:g} s"),
        ("Window overlap", f"{cfg.segmentation_overlap:.2f}"),
        ("Input channels", ", ".join(segmentation_channel_names(cfg)) if is_segmentation_mode(cfg) else "candidate channels"),
        (
            "Slow sigma band",
            f"{cfg.segmentation_slow_sigma_low_hz:g}-{cfg.segmentation_slow_sigma_high_hz:g} Hz"
            if cfg.segmentation_use_slow_sigma_channel else "disabled",
        ),
        ("Subject-balanced window sampling", cfg.segmentation_subject_balanced_sampling_enabled),
        (
            "Annotator-confidence supervision",
            (
                f"single-expert={cfg.segmentation_single_expert_target_weight:g}, "
                f"tolerance={cfg.segmentation_annotator_agreement_tolerance_sec:g} s"
            )
            if cfg.segmentation_annotator_confidence_labels_enabled
            else "disabled",
        ),
        ("Segmentation input normalization", cfg.normalization_mode if cfg.segmentation_normalize_inputs else "fixed physical scaling only"),
        ("U-Net base channels", "not used" if is_transformer_only_segmentation_mode(cfg) else cfg.segmentation_base_channels),
        ("U-Net depth", "not used" if is_transformer_only_segmentation_mode(cfg) else cfg.segmentation_depth),
        ("Kernel size", "not used" if is_transformer_only_segmentation_mode(cfg) else cfg.model_kernel_size),
        ("Attention gates", "not used" if is_transformer_only_segmentation_mode(cfg) else cfg.segmentation_use_attention_gates),
        ("Temporal bottleneck", "Transformer" if is_transformer_segmentation_mode(cfg) else ("not used" if is_transformer_only_segmentation_mode(cfg) else cfg.segmentation_temporal_bottleneck_enabled)),
        ("Transformer heads", cfg.segmentation_transformer_heads if is_transformer_segmentation_mode(cfg) else ""),
        ("Transformer layers", cfg.segmentation_temporal_bottleneck_layers if is_transformer_segmentation_mode(cfg) else ""),
        ("Transformer dropout", cfg.segmentation_temporal_bottleneck_dropout if is_transformer_segmentation_mode(cfg) else ""),
        ("Transformer-only embed dim", cfg.segmentation_transformer_only_embed_dim if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only layers", cfg.segmentation_transformer_only_layers if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only heads", cfg.segmentation_transformer_only_heads if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only block", cfg.segmentation_transformer_only_block_type if is_transformer_only_segmentation_mode(cfg) else ""),
        (
            "Local attention window",
            cfg.segmentation_transformer_only_local_attention_window
            if is_transformer_only_segmentation_mode(cfg)
            and cfg.segmentation_transformer_only_block_type == "local_transformer"
            else "",
        ),
        (
            "Local attention global stride",
            cfg.segmentation_transformer_only_local_global_stride
            if is_transformer_only_segmentation_mode(cfg)
            and cfg.segmentation_transformer_only_block_type == "local_transformer"
            else "",
        ),
        (
            "Transformer positional initialization",
            cfg.segmentation_transformer_only_positional_init
            if is_transformer_only_segmentation_mode(cfg) else "",
        ),
        ("Transformer-only conv kernel", cfg.segmentation_transformer_only_conv_kernel if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only patch size", cfg.segmentation_transformer_only_patch_size if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only patch stride", cfg.segmentation_transformer_only_patch_stride if is_transformer_only_segmentation_mode(cfg) else ""),
        (
            "Transformer-only multiscale patches",
            (
                f"{cfg.segmentation_transformer_only_patch_size}+"
                f"{cfg.segmentation_transformer_only_long_patch_size} samples"
            )
            if is_transformer_only_segmentation_mode(cfg)
            and cfg.segmentation_transformer_only_multiscale_enabled
            else "disabled",
        ),
        ("Transformer-only local mixing", cfg.segmentation_transformer_only_local_mixing if is_transformer_only_segmentation_mode(cfg) else ""),
        ("Transformer-only patch statistics", cfg.segmentation_transformer_only_patch_stats if is_transformer_only_segmentation_mode(cfg) else ""),
        (
            "Transformer spectral tokens",
            (
                f"{cfg.segmentation_transformer_only_spectral_low_hz:g}-"
                f"{cfg.segmentation_transformer_only_spectral_high_hz:g} Hz, "
                f"context={cfg.segmentation_transformer_only_spectral_patch_size / cfg.resample_hz:g} s, "
                f"initial gate={cfg.segmentation_transformer_only_spectral_gate_initial:g}"
            )
            if cfg.segmentation_transformer_only_spectral_tokens_enabled
            else "disabled",
        ),
        (
            "Spindle-aware supervised token contrastive loss",
            (
                f"weight={cfg.segmentation_supervised_contrastive_weight:g}, "
                f"temperature={cfg.segmentation_supervised_contrastive_temperature:g}, "
                f"projection={cfg.segmentation_supervised_contrastive_projection_dim}"
            )
            if cfg.segmentation_supervised_contrastive_enabled
            else "disabled",
        ),
        (
            "Anchor-free event geometry supervision",
            (
                f"objectness={cfg.segmentation_event_geometry_objectness_weight:g}, "
                f"center={cfg.segmentation_event_geometry_center_weight:g}, "
                f"duration={cfg.segmentation_event_geometry_duration_weight:g}"
            )
            if cfg.segmentation_event_geometry_enabled
            else "disabled",
        ),
        (
            "Transformer self-supervised pretraining",
            (
                f"{cfg.transformer_ssl_objective}, "
                f"{cfg.transformer_ssl_pretraining_epochs} epochs"
                + (", external EEG included" if cfg.transformer_ssl_external_data_dir else "")
                + f", ensemble mode={cfg.transformer_ssl_ensemble_mode}"
            )
            if is_transformer_only_segmentation_mode(cfg) and cfg.transformer_ssl_pretraining_enabled
            else "disabled",
        ),
        (
            "External supervised pretraining",
            (
                f"DREAMS, {cfg.transformer_external_supervised_pretraining_epochs} epochs, "
                f"lr={cfg.transformer_external_supervised_learning_rate:g}, "
                f"ensemble mode={cfg.transformer_external_supervised_ensemble_mode}"
            )
            if cfg.transformer_external_supervised_pretraining_enabled
            else "disabled",
        ),
        (
            "Warm-start checkpoint directory",
            cfg.transformer_warmstart_cnn_dir
            if (is_transformer_segmentation_mode(cfg) or is_transformer_only_segmentation_mode(cfg))
            and cfg.transformer_warmstart_enabled
            and cfg.transformer_warmstart_cnn_dir else "disabled",
        ),
        ("Auxiliary heads", cfg.segmentation_auxiliary_heads_enabled),
        ("Center/boundary loss weights", f"{cfg.segmentation_center_loss_weight:g}/{cfg.segmentation_boundary_loss_weight:g}" if cfg.segmentation_auxiliary_heads_enabled else "disabled"),
        ("Main-output edge consistency weight", cfg.segmentation_edge_consistency_weight),
        (
            "Multi-annotator auxiliary supervision",
            f"E1 + E2, weight {cfg.segmentation_multi_annotator_loss_weight:g}"
            if cfg.segmentation_multi_annotator_enabled
            else "disabled",
        ),
        ("Structured event decoder", cfg.segmentation_structured_decoder_enabled),
        ("Loss", cfg.segmentation_loss_type if is_segmentation_mode(cfg) else cfg.loss_type),
        ("Learning rate", cfg.segmentation_learning_rate if is_segmentation_mode(cfg) else cfg.learning_rate),
        ("Optimizer", cfg.segmentation_optimizer if is_segmentation_mode(cfg) else "AdamW"),
        ("Weight decay", cfg.segmentation_weight_decay if is_segmentation_mode(cfg) else cfg.weight_decay),
        ("Scheduler target", cfg.segmentation_scheduler_metric if is_segmentation_mode(cfg) else "validation loss"),
        ("Batch size", cfg.segmentation_batch_size if is_segmentation_mode(cfg) else cfg.batch_size),
        ("Max epochs", cfg.segmentation_epochs if is_segmentation_mode(cfg) else cfg.epochs),
        ("Early stopping patience", cfg.segmentation_early_stopping_patience if is_segmentation_mode(cfg) else cfg.early_stopping_patience),
        ("Checkpoint metric", cfg.segmentation_checkpoint_metric if is_segmentation_mode(cfg) else "validation loss"),
        ("Ensemble seed offsets", ",".join(str(x) for x in cfg.segmentation_ensemble_seed_offsets)),
        (
            "Validation-tuned ensemble weights",
            ",".join(f"{x:g}" for x in cfg.segmentation_ensemble_first_weight_grid)
            if cfg.segmentation_tune_ensemble_weights_enabled
            else "disabled",
        ),
        (
            "Ensemble weight selection metric",
            cfg.segmentation_ensemble_weight_selection_metric
            if cfg.segmentation_tune_ensemble_weights_enabled
            else "disabled",
        ),
        ("Hard-negative fine-tune", f"{cfg.hard_negative_finetune_epochs} epochs, top {cfg.hard_negative_fraction:g}, boost {cfg.hard_negative_boost:g}" if cfg.segmentation_hard_negative_mining_enabled else "disabled"),
        (
            "Fold regression guard",
            f"fold {cfg.segmentation_abort_check_fold}: F1 < {cfg.segmentation_abort_below_fold_f1:g}"
            if is_segmentation_mode(cfg) and cfg.segmentation_abort_below_fold_f1 > 0 else "disabled",
        ),
        ("Threshold grid", ",".join(f"{x:g}" for x in cfg.segmentation_threshold_grid)),
        ("Merge gap grid", ",".join(f"{x:g}" for x in cfg.segmentation_merge_gap_grid)),
        ("Minimum duration grid", ",".join(f"{x:g}" for x in cfg.segmentation_min_sec_grid)),
        ("Peak-preserving maximum duration grid", ",".join(f"{x:g}" for x in cfg.segmentation_max_sec_grid)),
        (
            "Validation selection weights",
            (
                f"strictF1={cfg.segmentation_selection_mean_f1_weight:g}, "
                f"minF1={cfg.segmentation_selection_min_f1_weight:g}, "
                f"precision={cfg.segmentation_selection_mean_precision_weight:g}, "
                f"recall={cfg.segmentation_selection_mean_recall_weight:g}, "
                f"tolerantF1={cfg.segmentation_selection_tolerant_f1_weight:g}"
            ),
        ),
        (
            "Validation floors",
            (
                f"precision={cfg.segmentation_precision_floor:g}/min={cfg.segmentation_min_precision_floor:g}, "
                f"recall={cfg.segmentation_recall_floor:g}/min={cfg.segmentation_min_recall_floor:g}"
            ),
        ),
        (
            "Validation F1 fallback",
            (
                f"enabled, margin={cfg.segmentation_f1_fallback_margin:g}, "
                f"min_precision={cfg.segmentation_f1_fallback_min_precision:g}"
                if cfg.segmentation_f1_fallback_enabled
                else "disabled"
            ),
        ),
        ("Density adaptive threshold", cfg.segmentation_density_adaptive_threshold_enabled),
        (
            "Validation duration-ratio penalty",
            f"{cfg.segmentation_duration_ratio_low:g}-{cfg.segmentation_duration_ratio_high:g}"
            if cfg.segmentation_duration_penalty_enabled else "disabled",
        ),
        ("Candidate consensus", cfg.segmentation_candidate_consensus_enabled),
        ("Explainability", "SHAP + SmoothGrad" if cfg.run_explainability else "disabled"),
    ]
    table = pd.DataFrame(rows, columns=["Parameter", "Value"])
    table.to_csv(outdir / "paper_model_parameters_table.csv", index=False)
    (outdir / "paper_model_parameters_table.md").write_text(dataframe_to_markdown(table), encoding="utf-8")


def save_dataset_description_outputs(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    path = outdir / "dataset_subject_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    table_cols = [
        "subject_id", "retained", "n_e1", "n_e2", "n_union",
        "e1_median_duration_sec", "e2_median_duration_sec", "union_median_duration_sec",
    ]
    table = df[[c for c in table_cols if c in df.columns]].copy()
    table.to_csv(outdir / "paper_dataset_description_table.csv", index=False)
    (outdir / "paper_dataset_description_table.md").write_text(
        dataframe_to_markdown(table.round(4)),
        encoding="utf-8",
    )
    if not plotting_enabled():
        return
    plot_df = df[df["retained"].astype(bool)].copy() if "retained" in df else df.copy()
    if plot_df.empty:
        return
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, plot_df["n_e1"], width=width, label="Expert 1")
    ax.bar(x, plot_df["n_e2"], width=width, label="Expert 2")
    ax.bar(x + width, plot_df["n_union"], width=width, label="Union")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subject_id"], rotation=35, ha="right")
    ax.set_ylabel("Spindle count")
    ax.set_title("Expert Spindle Count Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "paper_dataset_spindle_counts.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, plot_df["e1_median_duration_sec"], width=width, label="Expert 1")
    ax.bar(x, plot_df["e2_median_duration_sec"], width=width, label="Expert 2")
    ax.bar(x + width, plot_df["union_median_duration_sec"], width=width, label="Union")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subject_id"], rotation=35, ha="right")
    ax.set_ylabel("Median duration (s)")
    ax.set_title("Expert Spindle Duration Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "paper_dataset_spindle_durations.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def save_run_feature_importance_outputs(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    frames = []
    for p in sorted(outdir.glob("fold_*/feature_importance.csv")):
        df = pd.read_csv(p)
        mean_df = df[df["model"].astype(str) == "mean"].copy()
        if mean_df.empty:
            continue
        mean_df.insert(0, "fold", int(p.parent.name.split("_")[-1]))
        frames.append(mean_df)
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(outdir / "paper_feature_importance_by_fold.csv", index=False)
    channel_cols = [c for c in segmentation_channel_names(cfg) if c in all_df.columns]
    rows = [{"channel": c, "importance": float(pd.to_numeric(all_df[c], errors="coerce").mean())} for c in channel_cols]
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "paper_feature_importance.csv", index=False)
    if not plotting_enabled() or summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["channel"], summary["importance"], color=["tab:blue", "tab:orange", "tab:green"][:len(summary)])
    ax.set_ylim(0.0, max(1.0, float(summary["importance"].max()) * 1.15))
    ax.set_ylabel("Mean normalized first-layer weight magnitude")
    ax.set_title("Input Channel Importance Across Folds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "paper_feature_importance.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def draw_labeled_box(
    ax: Any,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 8.5,
) -> None:
    rect = plt.Rectangle(xy, width, height, facecolor=color, edgecolor="#1f2933", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2.0, xy[1] + height / 2.0, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax: Any, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#1f2933"})


def save_pipeline_overview_figure(outdir: Path, cfg: Config) -> None:
    if not plotting_enabled():
        return
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.set_axis_off()
        model_label = "Transformer-only\npatch segmentation" if is_transformer_only_segmentation_mode(cfg) else "CNN-Transformer\nU-Net segmentation"
        labels = [
            "MASS SS2 PSG\nEDF + expert labels",
            "Preprocessing\nC3, N2, robust norm",
            "Input channels\nraw + sigma",
            model_label,
            "Event decoding\nthreshold + duration",
            "Evaluation\nE1 / E2 / union",
            "Explainability\nSHAP + SmoothGrad",
        ]
        colors = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7", "#fee2e2", "#ede9fe", "#fce7f3"]
        x0, y0, w, h, gap = 0.02, 0.38, 0.125, 0.32, 0.018
        for i, (label, color) in enumerate(zip(labels, colors)):
            x = x0 + i * (w + gap)
            draw_labeled_box(ax, (x, y0), w, h, label, color)
            if i < len(labels) - 1:
                draw_arrow(ax, (x + w, y0 + h / 2), (x + w + gap * 0.85, y0 + h / 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Pipeline Overview", pad=8)
        fig.tight_layout()
        fig.savefig(outdir / "paper_pipeline_overview.jpg", dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_architecture_figure(outdir: Path, cfg: Config) -> None:
    if not plotting_enabled():
        return
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.set_axis_off()
        if is_transformer_only_segmentation_mode(cfg):
            token_label = (
                "Multiscale tokens\n"
                f"{cfg.segmentation_transformer_only_patch_size} / "
                f"{cfg.segmentation_transformer_only_long_patch_size} samples"
                if cfg.segmentation_transformer_only_multiscale_enabled
                else f"Patch tokens\n{cfg.segmentation_transformer_only_patch_size} samples"
            )
            encoder_label = (
                f"{cfg.segmentation_transformer_only_block_type.replace('_', ' ').title()}\n"
                f"{cfg.segmentation_transformer_only_layers} encoder blocks"
            )
            labels = [
                "C3 features\nraw | sigma\nslow sigma | envelope",
                token_label,
                encoder_label,
                "Overlap-add\ndecoder",
                "Center / boundary\nauxiliary heads",
                "Calibrated\nprobability",
            ]
        else:
            labels = [
                "Input\nraw + sigma",
                "CNN encoder\nmulti-scale features",
                "Transformer\ncontext bottleneck",
                "Attention skips\nU-Net decoder",
                "Auxiliary heads\ncenter + boundary",
                "Output\nsample probability",
            ]
        colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#ede9fe", "#fce7f3", "#fee2e2"]
        x0, y0, w, h, gap = 0.04, 0.42, 0.13, 0.30, 0.035
        centers = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            x = x0 + i * (w + gap)
            draw_labeled_box(ax, (x, y0), w, h, label, color, fontsize=6.6)
            centers.append((x + w / 2, y0 + h / 2))
            if i < len(labels) - 1:
                draw_arrow(ax, (x + w, y0 + h / 2), (x + w + gap * 0.85, y0 + h / 2))
        if not is_transformer_only_segmentation_mode(cfg):
            ax.plot([centers[1][0], centers[3][0]], [0.25, 0.25], color="#64748b", lw=1.0)
            draw_arrow(ax, (centers[1][0], y0), (centers[1][0], 0.25))
            draw_arrow(ax, (centers[1][0], 0.25), (centers[3][0], y0))
            ax.text((centers[1][0] + centers[3][0]) / 2, 0.16, "skip connections", ha="center", va="center", fontsize=8, color="#475569")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Transformer-Only Architecture" if is_transformer_only_segmentation_mode(cfg) else "CNN-Transformer Architecture", pad=8)
        fig.tight_layout()
        architecture_name = (
            "paper_transformer_architecture.jpg"
            if is_transformer_only_segmentation_mode(cfg)
            else "paper_cnn_transformer_architecture.jpg"
        )
        fig.savefig(outdir / architecture_name, dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_protocol_figure(outdir: Path, cfg: Config) -> None:
    if not plotting_enabled():
        return
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.set_axis_off()
        row_y = [0.68, 0.38]
        labels_top = ["19 MASS SS2 subjects", "5 subject-level folds", "Train / validation / test", "No test leakage"]
        labels_bottom = ["Train\n9 subjects", "Validation\n3 subjects", "Test\n3 subjects", "Report\nE1, E2, union"]
        for i, label in enumerate(labels_top):
            draw_labeled_box(ax, (0.05 + i * 0.23, row_y[0]), 0.18, 0.18, label, "#dbeafe")
            if i < len(labels_top) - 1:
                draw_arrow(ax, (0.23 + i * 0.23, row_y[0] + 0.09), (0.28 + i * 0.23, row_y[0] + 0.09))
        for i, label in enumerate(labels_bottom):
            draw_labeled_box(ax, (0.05 + i * 0.23, row_y[1]), 0.18, 0.18, label, ["#dcfce7", "#fef3c7", "#fee2e2", "#ede9fe"][i])
            if i < len(labels_bottom) - 1:
                draw_arrow(ax, (0.23 + i * 0.23, row_y[1] + 0.09), (0.28 + i * 0.23, row_y[1] + 0.09))
        ax.text(0.5, 0.18, "Training uses N2 windows only; validation tunes threshold/merge/min-duration; test is held out.", ha="center", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Training, Validation, and Testing Protocol", pad=8)
        fig.tight_layout()
        fig.savefig(outdir / "paper_training_validation_testing_protocol.jpg", dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_subject_wise_performance_figure(outdir: Path, cfg: Config) -> None:
    if not plotting_enabled():
        return
    path = outdir / "subject_metrics_by_reference.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    union = df[df["reference"].astype(str) == "union"].copy()
    if union.empty:
        return
    union = union.sort_values("strict_event_f1", ascending=False)
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(union["subject_id"].astype(str), union["strict_event_f1"].astype(float), color="#1f77b4", label="Strict F1")
        ax.plot(union["subject_id"].astype(str), union["strict_event_precision"].astype(float), color="#ff7f0e", marker="o", linewidth=1.0, label="Precision")
        ax.plot(union["subject_id"].astype(str), union["strict_event_recall"].astype(float), color="#2ca02c", marker="s", linewidth=1.0, label="Recall")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_xlabel("Subject")
        ax.set_title("Subject-wise Event Detection Performance")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / "paper_subject_wise_performance.jpg", dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def publication_experiment_label(path: Path) -> str:
    label = path.name
    for prefix in ("results_conformer_", "results_c3_", "results_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_2seed_5fold", "").replace("_seed42_5fold", "").replace("_", " ")


def save_transformer_comparison_outputs(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    run_dirs = [Path(path) for path in cfg.paper_comparison_dirs] + [outdir]
    unique_dirs: List[Path] = []
    seen: Set[str] = set()
    for run_dir in run_dirs:
        key = str(run_dir.resolve()) if run_dir.exists() else str(run_dir)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(run_dir)
    rows: List[Dict[str, Any]] = []
    fold_frames: Dict[str, pd.DataFrame] = {}
    metric_names = [
        "strict_event_precision", "strict_event_recall", "strict_event_f1",
        "tolerant_event_f1", "event_iou_f1_0_5", "sample_pr_auc", "ACC", "KAPPA",
    ]
    for run_dir in unique_dirs:
        summary_path = run_dir / "overall_summary.csv"
        fold_path = run_dir / "fold_summary.csv"
        if not summary_path.exists() or not fold_path.exists():
            logger.warning("Skipping paper comparison directory without complete summaries: %s", run_dir)
            continue
        summary = pd.read_csv(summary_path).set_index("metric")
        folds = pd.read_csv(fold_path)
        architecture = "unknown"
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                architecture = str(load_json(config_path).get("segmentation_transformer_only_block_type", "unknown"))
            except Exception:
                architecture = "unknown"
        experiment_label = publication_experiment_label(run_dir)
        row: Dict[str, Any] = {
            "experiment": experiment_label,
            "architecture": architecture,
            "directory": str(run_dir),
            "n_folds": int(len(folds)),
            "complete_5fold": bool(len(folds) == 5),
        }
        for metric in metric_names:
            row[metric] = (
                float(summary.loc[metric, "mean"])
                if metric in summary.index
                else float("nan")
            )
        values = pd.to_numeric(folds.get("strict_event_f1"), errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(values) > 1:
            sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            margin = float(scipy.stats.t.ppf(0.975, len(values) - 1) * sem)
        else:
            margin = float("nan")
        row["strict_event_f1_ci95_low"] = float(np.mean(values) - margin) if len(values) else float("nan")
        row["strict_event_f1_ci95_high"] = float(np.mean(values) + margin) if len(values) else float("nan")
        rows.append(row)
        fold_copy = folds.copy()
        fold_copy["experiment"] = experiment_label
        fold_copy["architecture"] = architecture
        fold_frames[experiment_label] = fold_copy
    if not rows:
        return
    table = pd.DataFrame(rows).sort_values("strict_event_f1", ascending=False).reset_index(drop=True)
    table.to_csv(outdir / "paper_transformer_experiment_comparison.csv", index=False)
    (outdir / "paper_transformer_experiment_comparison.md").write_text(
        dataframe_to_markdown(table.round(4)),
        encoding="utf-8",
    )
    combined_folds = pd.concat(fold_frames.values(), ignore_index=True) if fold_frames else pd.DataFrame()
    if not combined_folds.empty:
        combined_folds.to_csv(outdir / "paper_transformer_foldwise_comparison.csv", index=False)

    pairwise_rows: List[Dict[str, Any]] = []
    for experiment_a, experiment_b in itertools.combinations(sorted(fold_frames), 2):
        a = fold_frames[experiment_a][["fold", "strict_event_f1"]].rename(
            columns={"strict_event_f1": "f1_a"}
        )
        b = fold_frames[experiment_b][["fold", "strict_event_f1"]].rename(
            columns={"strict_event_f1": "f1_b"}
        )
        paired = a.merge(b, on="fold", how="inner").dropna()
        if len(paired) < 2:
            continue
        differences = paired["f1_a"].to_numpy(dtype=float) - paired["f1_b"].to_numpy(dtype=float)
        paired_t_p = float(scipy.stats.ttest_rel(paired["f1_a"], paired["f1_b"]).pvalue)
        try:
            wilcoxon_p = float(scipy.stats.wilcoxon(paired["f1_a"], paired["f1_b"]).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        difference_sd = float(np.std(differences, ddof=1))
        mean_a = float(np.mean(paired["f1_a"]))
        mean_b = float(np.mean(paired["f1_b"]))
        pairwise_rows.append({
            "experiment_a": experiment_a,
            "experiment_b": experiment_b,
            "n_paired_folds": int(len(paired)),
            "mean_f1_a": mean_a,
            "mean_f1_b": mean_b,
            "mean_difference_a_minus_b": float(np.mean(differences)),
            "percentage_improvement_a_over_b": float(100.0 * (mean_a - mean_b) / max(abs(mean_b), 1e-12)),
            "cohen_dz": float(np.mean(differences) / difference_sd) if difference_sd > 0 else float("nan"),
            "paired_t_p": paired_t_p,
            "wilcoxon_p": wilcoxon_p,
        })
    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        # Holm correction is applied separately to each family of pairwise tests.
        for source, destination in (
            ("paired_t_p", "paired_t_p_holm"),
            ("wilcoxon_p", "wilcoxon_p_holm"),
        ):
            values = pairwise[source].to_numpy(dtype=float)
            order = np.argsort(values)
            adjusted = np.empty_like(values)
            running = 0.0
            count = len(values)
            for rank, index in enumerate(order):
                running = max(running, min(1.0, (count - rank) * values[index]))
                adjusted[index] = running
            pairwise[destination] = adjusted
        pairwise.to_csv(outdir / "paper_transformer_pairwise_statistics.csv", index=False)
        (outdir / "paper_transformer_pairwise_statistics.md").write_text(
            dataframe_to_markdown(pairwise.round(5)),
            encoding="utf-8",
        )
    if not plotting_enabled():
        return
    plot_table = table.sort_values("strict_event_f1", ascending=True)
    y = np.arange(len(plot_table))
    means = plot_table["strict_event_f1"].to_numpy(dtype=float)
    lower = means - plot_table["strict_event_f1_ci95_low"].to_numpy(dtype=float)
    upper = plot_table["strict_event_f1_ci95_high"].to_numpy(dtype=float) - means
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.45 * len(plot_table) + 1.4)))
    ax.errorbar(means, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=4, color="#1d4ed8")
    ax.axvline(0.86, color="#dc2626", linestyle="--", linewidth=1.5, label="Target F1 = 0.86")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_table["experiment"])
    ax.set_xlabel("Mean strict event F1 (95% CI)")
    ax.set_title("Transformer Experiment Progression")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "paper_transformer_experiment_forest.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)

    transfer = table.dropna(subset=["sample_pr_auc", "strict_event_f1"])
    if not transfer.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        scatter = ax.scatter(
            transfer["sample_pr_auc"],
            transfer["strict_event_f1"],
            c=transfer["strict_event_recall"],
            cmap="viridis",
            s=75,
            edgecolor="black",
            linewidth=0.5,
        )
        for _, row in transfer.iterrows():
            ax.annotate(
                str(row["experiment"]),
                (row["sample_pr_auc"], row["strict_event_f1"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.axhline(0.86, color="#dc2626", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Sample PR-AUC (representation discrimination)")
        ax.set_ylabel("Strict event F1 (event reconstruction)")
        ax.set_title("Representation-to-Event Transfer")
        ax.grid(alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="Strict event recall")
        fig.tight_layout()
        fig.savefig(outdir / "paper_representation_event_transfer.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
        plt.close(fig)

    if not combined_folds.empty and "strict_event_f1" in combined_folds:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        architecture_labels = {
            "conformer": "Conformer",
            "transformer": "Global\nTransformer",
            "local_transformer": "Sparse-Local\nTransformer",
        }
        group_column = "architecture" if "architecture" in combined_folds.columns else "experiment"
        for experiment, frame in combined_folds.groupby(group_column):
            ordered = frame.sort_values("fold")
            ax.plot(
                ordered["fold"],
                ordered["strict_event_f1"],
                marker="o",
                linewidth=1.8,
                label=architecture_labels.get(str(experiment), str(experiment)),
            )
        ax.set_xticks(sorted(combined_folds["fold"].dropna().astype(int).unique()))
        ax.set_xlabel("Cross-validation fold")
        ax.set_ylabel("Strict event F1")
        ax.set_title("Fold-wise Transformer Architecture Generalization")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "paper_transformer_fold_profile.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
        plt.close(fig)

    heatmap_metrics = [
        "strict_event_precision", "strict_event_recall", "strict_event_f1",
        "tolerant_event_f1", "event_iou_f1_0_5", "sample_pr_auc",
    ]
    available_heatmap_metrics = [m for m in heatmap_metrics if m in table]
    if "architecture" in table:
        architecture_order = ["conformer", "transformer", "local_transformer"]
        architecture_labels = {
            "conformer": "Conformer",
            "transformer": "Global Transformer",
            "local_transformer": "Sparse-Local Transformer",
        }
        heatmap_table = (
            table.set_index("architecture")[available_heatmap_metrics]
            .reindex(architecture_order)
            .dropna(how="all")
            .rename(index=architecture_labels)
            .astype(float)
        )
    else:
        heatmap_table = table.set_index("experiment")[available_heatmap_metrics].astype(float)
    if not heatmap_table.empty:
        # Portrait-oriented dimensions remain legible when reduced to one IEEE column.
        fig, ax = plt.subplots(figsize=(4.2, max(4.8, 0.85 * len(heatmap_table) + 2.4)))
        image_data = heatmap_table.to_numpy(dtype=float)
        image = ax.imshow(image_data, aspect="auto", cmap="YlGnBu", vmin=0.45, vmax=0.90)
        ax.set_xticks(np.arange(len(heatmap_table.columns)))
        metric_labels = {
            "strict_event_precision": "Strict precision",
            "strict_event_recall": "Strict recall",
            "strict_event_f1": "Strict F1",
            "tolerant_event_f1": "Tolerant F1",
            "event_iou_f1_0_5": "IoU F1@0.5",
            "sample_pr_auc": "PR-AUC",
        }
        ax.set_xticklabels(
            [metric_labels.get(name, name.replace("_", " ")) for name in heatmap_table.columns],
            rotation=42,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(np.arange(len(heatmap_table.index)))
        ax.set_yticklabels(heatmap_table.index, fontsize=8)
        for row_idx in range(image_data.shape[0]):
            for col_idx in range(image_data.shape[1]):
                value = image_data[row_idx, col_idx]
                if math.isfinite(value):
                    text_color = "white" if value >= 0.74 else "black"
                    ax.text(
                        col_idx, row_idx, f"{value:.3f}",
                        ha="center", va="center", fontsize=6, color=text_color,
                    )
        ax.set_title("Transformer Architecture Metric Profile", fontsize=10)
        colorbar = fig.colorbar(image, ax=ax, label="Score", fraction=0.06, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "paper_transformer_metric_heatmap.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_ensemble_diagnostic_outputs(outdir: Path, fold_df: pd.DataFrame, cfg: Config) -> None:
    rows: List[Dict[str, Any]] = []
    for _, fold_row in fold_df.sort_values("fold").iterrows():
        fold = int(fold_row["fold"])
        fold_dir = outdir / f"fold_{fold:02d}"
        weights_path = fold_dir / "segmentation_ensemble_weights.json"
        tuning_path = fold_dir / "segmentation_ensemble_weight_tuning.csv"
        if not weights_path.exists():
            continue
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = payload.get("member_weights", [])
        row: Dict[str, Any] = {
            "fold": fold,
            "first_member_weight": float(weights[0]) if len(weights) > 0 else float("nan"),
            "second_member_weight": float(weights[1]) if len(weights) > 1 else float("nan"),
            "selection_metric": payload.get("selection_metric", ""),
        }
        if tuning_path.exists():
            tuning = pd.read_csv(tuning_path)
            if not tuning.empty and math.isfinite(row["first_member_weight"]):
                selected = tuning.iloc[
                    (tuning["first_member_weight"].astype(float) - row["first_member_weight"]).abs().argmin()
                ]
                for column in (
                    "validation_sample_pr_auc", "validation_event_f1_proxy",
                    "event_f1_proxy_threshold", "selection_score",
                ):
                    if column in selected:
                        row[column] = float(selected[column])
        for metric in (
            "strict_event_precision", "strict_event_recall", "strict_event_f1",
            "tolerant_event_f1", "event_iou_f1_0_5", "sample_pr_auc",
        ):
            if metric in fold_row:
                row[f"test_{metric}"] = float(fold_row[metric])
        rows.append(row)
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "paper_ensemble_weight_diagnostics.csv", index=False)
    (outdir / "paper_ensemble_weight_diagnostics.md").write_text(
        dataframe_to_markdown(table.round(4)),
        encoding="utf-8",
    )
    if not plotting_enabled():
        return
    fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(table))
    ax1.bar(x, table["first_member_weight"], color="#8b5cf6", alpha=0.75, label="SSL member weight")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Selected SSL-member weight")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Fold {int(v)}" for v in table["fold"]])
    ax2 = ax1.twinx()
    for metric, label, color, marker in (
        ("test_strict_event_precision", "Precision", "#f97316", "o"),
        ("test_strict_event_recall", "Recall", "#16a34a", "s"),
        ("test_strict_event_f1", "Strict F1", "#1d4ed8", "^"),
    ):
        if metric in table:
            ax2.plot(x, table[metric], color=color, marker=marker, linewidth=1.8, label=label)
    ax2.set_ylim(0.65, 0.90)
    ax2.set_ylabel("Held-out event score")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="lower center", ncol=4, fontsize=8)
    ax1.set_title("Fold-wise SSL/Scratch Contribution and Generalization")
    ax1.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "paper_ensemble_weight_generalization.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def save_subject_error_phenotype_outputs(outdir: Path, cfg: Config) -> None:
    path = outdir / "subject_metrics_by_reference.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    union = df[df["reference"].astype(str) == "union"].copy()
    if union.empty:
        return
    union["false_discovery_rate"] = 1.0 - union["strict_event_precision"].astype(float)
    union["miss_rate"] = 1.0 - union["strict_event_recall"].astype(float)
    union["tolerance_gain"] = union["tolerant_event_f1"].astype(float) - union["strict_event_f1"].astype(float)
    union["density_deviation"] = (union["n_pred_events"].astype(float) / union["n_true_events"].clip(lower=1).astype(float) - 1.0).abs()
    union["ranking_error"] = 1.0 - union["sample_pr_auc"].astype(float)
    columns = [
        "fold", "subject_id", "strict_event_precision", "strict_event_recall",
        "strict_event_f1", "tolerant_event_f1", "sample_pr_auc",
        "false_discovery_rate", "miss_rate", "tolerance_gain",
        "density_deviation", "ranking_error",
    ]
    table = union[[c for c in columns if c in union]].sort_values(["fold", "subject_id"])
    table.to_csv(outdir / "paper_subject_error_phenotypes.csv", index=False)
    (outdir / "paper_subject_error_phenotypes.md").write_text(
        dataframe_to_markdown(table.round(4)),
        encoding="utf-8",
    )
    if not plotting_enabled():
        return
    heatmap_rows = [
        ("false_discovery_rate", "False discovery"),
        ("miss_rate", "Miss rate"),
        ("tolerance_gain", "Boundary/tolerance gap"),
        ("density_deviation", "Event-density deviation"),
        ("ranking_error", "1 - PR-AUC"),
    ]
    matrix = np.vstack([table[key].to_numpy(dtype=float) for key, _ in heatmap_rows])
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    image_obj = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.35, float(np.nanpercentile(matrix, 95))))
    ax.set_yticks(np.arange(len(heatmap_rows)))
    ax.set_yticklabels([label for _, label in heatmap_rows])
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table["subject_id"].astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_title("Subject-specific Transformer Error Phenotypes")
    fig.colorbar(image_obj, ax=ax, label="Error magnitude")
    fig.tight_layout()
    fig.savefig(outdir / "paper_subject_error_phenotype_heatmap.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def save_annotator_disagreement_outputs(outdir: Path, cfg: Config) -> None:
    dataset_path = outdir / "dataset_subject_summary.csv"
    metrics_path = outdir / "subject_metrics_by_reference.csv"
    if not dataset_path.exists() or not metrics_path.exists():
        return
    dataset = pd.read_csv(dataset_path)
    metrics = pd.read_csv(metrics_path)
    union = metrics[metrics["reference"].astype(str) == "union"].copy()
    if union.empty:
        return
    table = dataset.merge(union, on="subject_id", how="inner")
    table["expert_count_disagreement"] = (
        (table["n_e1"].astype(float) - table["n_e2"].astype(float)).abs()
        / table["n_union"].clip(lower=1).astype(float)
    )
    if {"e1_median_duration_sec", "e2_median_duration_sec"}.issubset(table.columns):
        table["expert_duration_disagreement_sec"] = (
            table["e1_median_duration_sec"].astype(float) - table["e2_median_duration_sec"].astype(float)
        ).abs()
    rho, p_value = scipy.stats.spearmanr(
        table["expert_count_disagreement"].astype(float),
        table["strict_event_f1"].astype(float),
        nan_policy="omit",
    )
    table["spearman_rho_count_disagreement_vs_f1"] = float(rho)
    table["spearman_p_value"] = float(p_value)
    export_cols = [
        "subject_id", "n_e1", "n_e2", "n_union", "expert_count_disagreement",
        "expert_duration_disagreement_sec", "strict_event_precision",
        "strict_event_recall", "strict_event_f1",
        "spearman_rho_count_disagreement_vs_f1", "spearman_p_value",
    ]
    export = table[[c for c in export_cols if c in table]]
    export.to_csv(outdir / "paper_annotator_disagreement_performance.csv", index=False)
    (outdir / "paper_annotator_disagreement_performance.md").write_text(
        dataframe_to_markdown(export.round(4)),
        encoding="utf-8",
    )
    if not plotting_enabled():
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    scatter = ax.scatter(
        table["expert_count_disagreement"],
        table["strict_event_f1"],
        c=table["strict_event_recall"],
        cmap="plasma",
        s=65,
        edgecolor="black",
        linewidth=0.5,
    )
    for _, row in table.iterrows():
        ax.annotate(str(row["subject_id"]), (row["expert_count_disagreement"], row["strict_event_f1"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Normalized E1/E2 spindle-count disagreement")
    ax.set_ylabel("Union-reference strict F1")
    ax.set_title(f"Annotation Disagreement vs Detection (Spearman ρ={rho:.2f}, p={p_value:.3f})")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Strict recall")
    fig.tight_layout()
    fig.savefig(outdir / "paper_annotator_disagreement_vs_performance.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


def collect_paper_artifacts(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    paper_dir = ensure_dir(outdir / cfg.paper_data_folder_name)
    figure_dir = ensure_dir(paper_dir / "Figures")
    table_dir = ensure_dir(paper_dir / "Tables")
    rows: List[Dict[str, str]] = []
    candidates = sorted(outdir.glob("paper_*")) + [
        path for path in (outdir / "statistical_analysis.csv", outdir / "statistical_analysis.md") if path.exists()
    ]
    for source in candidates:
        if not source.is_file():
            continue
        destination_dir = figure_dir if source.suffix.lower() in {".jpg", ".jpeg"} else table_dir
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        rows.append({
            "artifact": source.name,
            "type": "figure" if destination_dir == figure_dir else "table",
            "relative_path": str(destination.relative_to(outdir)),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(paper_dir / "paper_artifact_manifest.csv", index=False)
    guide = [
        "# Paper data",
        "",
        "This folder is generated automatically after cross-fold evaluation.",
        "",
        "## Recommended transformer-specific main-paper items",
        "",
        "1. `Figures/paper_transformer_experiment_forest.jpg` - progression and 95% confidence intervals.",
        "2. `Figures/paper_representation_event_transfer.jpg` - PR-AUC versus strict event reconstruction.",
        "3. `Figures/paper_ensemble_weight_generalization.jpg` - fold-specific SSL/scratch contribution.",
        "4. `Figures/paper_subject_error_phenotype_heatmap.jpg` - subject-level failure mechanisms.",
        "5. `Figures/paper_annotator_disagreement_vs_performance.jpg` - annotation uncertainty and performance.",
        "6. `Tables/paper_transformer_experiment_comparison.csv` - transformer ablation/comparison table.",
        "7. `Figures/paper_transformer_fold_profile.jpg` - paired fold-wise architecture behavior.",
        "8. `Figures/paper_transformer_metric_heatmap.jpg` - precision, recall, overlap, and PR-AUC profile.",
        "9. `Tables/paper_transformer_pairwise_statistics.csv` - paired tests, Holm correction, and effect sizes.",
        "10. `Tables/paper_ensemble_weight_diagnostics.csv` - validation-selected weights and held-out results.",
        "",
        "Use the architecture and protocol diagrams once; move conventional per-fold plots to supplementary material.",
    ]
    (paper_dir / "README.md").write_text("\n".join(guide), encoding="utf-8")
    logger.info("Collected %d paper artifacts under %s", len(rows), paper_dir)


def save_ieee_publication_figures(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    if not cfg.generate_paper_outputs:
        return
    save_pipeline_overview_figure(outdir, cfg)
    save_architecture_figure(outdir, cfg)
    save_protocol_figure(outdir, cfg)
    save_subject_wise_performance_figure(outdir, cfg)
    logger.info("Saved IEEE publication summary figures to %s", outdir)


def generate_run_paper_outputs(outdir: Path, fold_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    if not cfg.generate_paper_outputs:
        return
    save_reference_result_tables(outdir, logger)
    save_model_parameter_table(outdir, cfg)
    save_dataset_description_outputs(outdir, cfg, logger)
    save_run_feature_importance_outputs(outdir, cfg, logger)
    key_rows = [
        ("strict_event_f1", "Strict event F1"),
        ("strict_event_precision", "Strict event precision"),
        ("strict_event_recall", "Strict event recall"),
        ("tolerant_event_f1", "Tolerant event F1"),
        ("sample_f1", "Sample F1"),
        ("KAPPA", "Kappa"),
    ]
    if not is_segmentation_mode(cfg):
        key_rows.insert(4, ("candidate_f1", "Candidate classifier F1"))
        key_rows.insert(5, ("candidate_pr_auc", "Candidate classifier PR-AUC"))
    summary_df = pd.read_csv(outdir / "overall_summary.csv")
    summary_lookup = summary_df.set_index("metric")
    table_rows: List[Dict[str, Any]] = []
    for metric, label in key_rows:
        if metric in summary_lookup.index:
            mean = float(summary_lookup.loc[metric, "mean"])
            std = float(summary_lookup.loc[metric, "std"])
            table_rows.append({"Metric": label, "Mean Â± SD": format_mean_std(mean, std)})
    paper_table = pd.DataFrame(table_rows)
    paper_table.to_csv(outdir / "paper_main_results_table.csv", index=False)
    (outdir / "paper_main_results_table.md").write_text(dataframe_to_markdown(paper_table), encoding="utf-8")

    if is_segmentation_mode(cfg):
        fold_export_cols = [
            "fold", "prob_thr", "final_merge_gap_sec", "final_min_sec", "adaptive_threshold_quantile",
            "strict_event_f1", "strict_event_precision", "strict_event_recall",
            "tolerant_event_f1", "sample_f1", "sample_pr_auc", "KAPPA",
        ]
    else:
        fold_export_cols = [
            "fold", "B", "T", "lambda_thr", "prob_thr", "hysteresis_gap", "final_merge_gap_sec",
            "final_min_sec", "final_max_sec", "adaptive_threshold_quantile", "strict_event_f1", "strict_event_precision",
            "strict_event_recall", "candidate_f1", "candidate_pr_auc",
        ]
    fold_table = fold_df[[c for c in fold_export_cols if c in fold_df.columns]].copy()
    fold_table.to_csv(outdir / "paper_fold_results_table.csv", index=False)
    (outdir / "paper_fold_results_table.md").write_text(dataframe_to_markdown(fold_table.round(4)), encoding="utf-8")

    save_transformer_comparison_outputs(outdir, cfg, logger)
    save_ensemble_diagnostic_outputs(outdir, fold_df, cfg)
    save_subject_error_phenotype_outputs(outdir, cfg)
    save_annotator_disagreement_outputs(outdir, cfg)

    if not plotting_enabled():
        logger.warning("matplotlib not available; skipping run-level paper plots.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fold_df))
    width = 0.25
    third_metric = "sample_f1" if is_segmentation_mode(cfg) else "candidate_f1"
    third_label = "sample_f1" if is_segmentation_mode(cfg) else "candidate_f1"
    ax.bar(x - width, fold_df["strict_event_f1"], width=width, label="strict_event_f1")
    ax.bar(x, fold_df["tolerant_event_f1"], width=width, label="tolerant_event_f1")
    if third_metric in fold_df.columns:
        ax.bar(x + width, fold_df[third_metric], width=width, label=third_label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(v)}" for v in fold_df["fold"]])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Fold Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "paper_cross_fold_performance.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fold_df["fold"], fold_df["strict_event_precision"], marker="o", label="strict_precision")
    ax.plot(fold_df["fold"], fold_df["strict_event_recall"], marker="o", label="strict_recall")
    ax.plot(fold_df["fold"], fold_df["strict_event_f1"], marker="o", label="strict_f1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Strict Event Metrics Across Folds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "paper_strict_event_metrics.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)
    save_fold_metric_reference_plots(outdir, cfg, logger)
    save_ieee_publication_figures(outdir, cfg, logger)

def package_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "mne": mne.__version__,
        "sklearn": __import__("sklearn").__version__,
        "torch": torch.__version__,
    }
    versions["shap"] = getattr(shap, "__version__", "not_installed") if shap is not None else "not_installed"
    if torch.cuda.is_available():
        versions["cuda"] = torch.version.cuda or "unknown"
    return versions


def stage2_tuning_justification() -> List[str]:
    return [
        "The lambda grid now spans 0.10-0.18 because fold-1 errors showed that higher label-overlap thresholds may improve precision without collapsing recall.",
        "Completed precision-first runs already produced strong candidate PR-AUC, so stage-2 tuning now searches decoding parameters that govern event assembly rather than only candidate labels.",
        "Stage-3 model selection now gives more weight to mean and worst-subject F1 because the latest fold recovered recall but lost precision and boundary quality.",
        "Hysteresis gap, final merge gap, final minimum duration, final maximum duration, and subject-adaptive threshold quantiles are tuned jointly to recover true spindles without over-extending event boundaries.",
    ]


def init_logging(output_dir: Path, debug: bool) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger("mass_ss2_spindle_cnn")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_torch_runtime(cfg: Config, device: torch.device, logger: logging.Logger) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.enable_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(cfg.enable_tf32)
    if hasattr(torch, "set_float32_matmul_precision") and cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
    logger.info(
        "CUDA runtime tuned | tf32=%s | cudnn_benchmark=%s | num_workers=%d | batch_size=%d",
        cfg.enable_tf32,
        cfg.cudnn_benchmark,
        cfg.num_workers,
        cfg.segmentation_batch_size if is_segmentation_mode(cfg) else cfg.batch_size,
    )


def normalize_channel_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def normalize_annotation_desc(desc: str) -> str:
    return re.sub(r"\s+", " ", str(desc).strip().lower())


def is_spindle_annotation(desc: str) -> bool:
    d = normalize_annotation_desc(desc)
    return (
        "spindle" in d
        or 'groupname="spindle"' in d
        or d == "sp"
        or d.startswith("sp ")
    )


def is_non_spindle_sleep_annotation(desc: str) -> bool:
    d = normalize_annotation_desc(desc)
    markers = ("stage ", "sleep stage", "rem", "wake", "arousal", "movement", "artifact", "k-complex", "kcomplex")
    return any(token in d for token in markers)


def subject_has_required_annotation_files(subject: SubjectFiles, cfg: Config) -> bool:
    if cfg.require_dual_expert_files or cfg.annotation_mode in ("union", "intersection"):
        return subject.spindle_e1_path is not None and subject.spindle_e2_path is not None
    if cfg.annotation_mode == "expert1":
        return subject.spindle_e1_path is not None
    if cfg.annotation_mode == "expert2":
        return subject.spindle_e2_path is not None
    return True


def discover_subjects(data_dir: Path, logger: logging.Logger) -> List[SubjectFiles]:
    pattern = re.compile(r"^(?P<sid>\d{2}-\d{2}-\d{4})[ _-]+(?P<kind>.+)\.edf$", re.IGNORECASE)
    subject_map: Dict[str, Dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.edf")):
        match = pattern.match(path.name)
        if not match:
            continue
        sid = match.group("sid")
        if not sid.startswith("01-02-"):
            continue  # MASS subset SS2 only; do not load other cohorts.
        kind = match.group("kind").strip().lower()
        subject_map.setdefault(sid, {})[kind] = path

    subjects: List[SubjectFiles] = []
    for sid in sorted(subject_map):
        files = subject_map[sid]
        if "psg" not in files:
            logger.warning("Skipping %s because PSG.edf is missing.", sid)
            continue
        subjects.append(
            SubjectFiles(
                subject_id=sid,
                psg_path=files["psg"],
                base_path=files.get("base"),
                spindle_e1_path=files.get("spindles_e1"),
                spindle_e2_path=files.get("spindles_e2"),
            )
        )
    logger.info("Discovered %d MASS SS2 subjects in %s", len(subjects), data_dir)
    return subjects


def select_best_channel(ch_names: Sequence[str], target: str, logger: logging.Logger) -> str:
    normalized = {name: normalize_channel_name(name) for name in ch_names}
    preferred = [
        target,
        f"EEG {target}",
        "C3",
        "C3CLE",
        "C3LER",
        "C3A1",
        "C3A2",
        "EEGC3",
        "EEGC3CLE",
        "EEGC3LER",
    ]
    preferred_norm = [normalize_channel_name(x) for x in preferred]
    for want in preferred_norm:
        for name, clean in normalized.items():
            if clean == want:
                return name
    for want in preferred_norm:
        for name, clean in normalized.items():
            if want in clean:
                return name
    logger.warning("Target channel %s not found; falling back to first channel.", target)
    return ch_names[0]


def select_optional_channel(
    ch_names: Sequence[str],
    target: str,
    exclude: Sequence[str] = (),
) -> Optional[str]:
    """Return a matching EEG channel without falling back to an unrelated lead."""
    excluded = {str(name) for name in exclude}
    normalized = {name: normalize_channel_name(name) for name in ch_names if name not in excluded}
    candidates = [
        target,
        f"EEG {target}",
        f"{target}CLE",
        f"{target}LER",
        f"{target}A1",
        f"{target}A2",
        f"EEG{target}",
        f"EEG{target}CLE",
        f"EEG{target}LER",
    ]
    wanted = [normalize_channel_name(value) for value in candidates]
    for want in wanted:
        for name, clean in normalized.items():
            if clean == want:
                return name
    for want in wanted:
        for name, clean in normalized.items():
            if want and want in clean:
                return name
    return None


def select_best_channel_from_priority(
    ch_names: Sequence[str],
    targets: Sequence[str],
    logger: logging.Logger,
    context: str = "",
) -> str:
    normalized = {name: normalize_channel_name(name) for name in ch_names}
    seen: set[str] = set()
    for target in targets:
        if not target:
            continue
        candidates = [
            target,
            f"EEG {target}",
            target.replace("-", ""),
            target.replace("_", ""),
        ]
        for candidate in candidates:
            clean_candidate = normalize_channel_name(candidate)
            if not clean_candidate or clean_candidate in seen:
                continue
            seen.add(clean_candidate)
            for name, clean in normalized.items():
                if clean == clean_candidate:
                    logger.info("Selected channel %s for %s.", name, context or "record")
                    return name
        for candidate in candidates:
            clean_candidate = normalize_channel_name(candidate)
            if not clean_candidate:
                continue
            for name, clean in normalized.items():
                if clean_candidate in clean:
                    logger.info("Selected channel %s for %s.", name, context or "record")
                    return name
    logger.warning(
        "None of target channels %s found for %s; falling back to first channel %s.",
        list(targets),
        context or "record",
        ch_names[0],
    )
    return ch_names[0]


def read_annotations(path: Optional[Path], logger: Optional[logging.Logger], debug: bool) -> List[Tuple[float, float, str]]:
    if path is None or not path.exists():
        return []
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    out: List[Tuple[float, float, str]] = []
    descs: List[str] = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if dur <= 0:
            continue
        out.append((float(onset), float(onset + dur), str(desc)))
        descs.append(str(desc))
    if debug and logger and descs:
        logger.info("Annotation descriptions in %s: %s", path.name, sorted(set(descs))[:20])
    return out


def parse_spindle_events(
    path: Optional[Path],
    logger: logging.Logger,
    debug: bool,
    allow_implicit_labels: bool,
) -> List[Tuple[float, float]]:
    events = read_annotations(path, logger if debug else None, debug)
    if not events:
        return []

    spindle_events = [(onset, offset) for onset, offset, desc in events if is_spindle_annotation(desc)]
    if spindle_events:
        spindle_events = merge_intervals(spindle_events)
        if debug:
            logger.info("Parsed %d spindle events from %s", len(spindle_events), path.name if path else "missing")
        return spindle_events

    unique_desc = sorted({normalize_annotation_desc(desc) for _, _, desc in events if normalize_annotation_desc(desc)})
    if allow_implicit_labels and path is not None and "spindle" in path.stem.lower():
        if unique_desc and any(is_non_spindle_sleep_annotation(desc) for desc in unique_desc):
            logger.warning(
                "Spindle file %s contains only unrecognized/non-spindle descriptors %s; refusing implicit parsing.",
                path.name,
                unique_desc[:10],
            )
            return []
        logger.warning(
            "No explicit spindle labels found in %s; falling back to implicit file-based parsing. "
            "Set --no-allow_implicit_spindle_labels to disable this.",
            path.name,
        )
        return merge_intervals([(onset, offset) for onset, offset, _ in events])

    if path is not None:
        logger.warning(
            "No recognized spindle labels found in %s. Unique descriptors sample: %s",
            path.name,
            unique_desc[:10],
        )
    return []


def parse_sleep_stages(base_path: Optional[Path], logger: logging.Logger, debug: bool) -> List[Tuple[float, float, str]]:
    stage_events = read_annotations(base_path, logger if debug else None, debug)
    out: List[Tuple[float, float, str]] = []
    for onset, offset, desc in stage_events:
        d = desc.lower()
        stage: Optional[str] = None
        if "stage 2" in d or "sleep stage 2" in d or "n2" in d:
            stage = "N2"
        elif "stage 3" in d or "stage 4" in d or "sleep stage 3" in d or "sleep stage 4" in d or "n3" in d or "n4" in d:
            stage = "N3"
        elif "stage 1" in d or "sleep stage 1" in d or "n1" in d:
            stage = "N1"
        elif "stage r" in d or "rem" in d:
            stage = "REM"
        elif "stage w" in d or "wake" in d:
            stage = "W"
        if stage is not None:
            out.append((onset, offset, stage))
    return out


def merge_intervals(events: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not events:
        return []
    events = sorted(events)
    merged = [events[0]]
    for s, e in events[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def intersect_intervals(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s1, e1 in a:
        for s2, e2 in b:
            s = max(s1, s2)
            e = min(e1, e2)
            if e > s:
                out.append((s, e))
    return merge_intervals(out)


def build_true_events(
    mode: str,
    events_e1: List[Tuple[float, float]],
    events_e2: List[Tuple[float, float]],
    logger: logging.Logger,
    subject_id: str,
) -> Tuple[List[Tuple[float, float]], bool]:
    if mode == "expert1":
        return events_e1, True
    if mode == "expert2":
        return events_e2, True
    if mode == "union":
        return merge_intervals(events_e1 + events_e2), True
    if mode == "intersection":
        if not events_e1 or not events_e2:
            logger.warning("Subject %s excluded from intersection mode due to missing expert annotations.", subject_id)
            return [], False
        return intersect_intervals(events_e1, events_e2), True
    raise ValueError(f"Unsupported annotation mode: {mode}")


def binary_to_segments(binary: np.ndarray) -> List[Tuple[int, int]]:
    # Use a signed integer dtype before differencing. With uint8, the 1->0
    # transition wraps to 255 instead of -1, which silently removes all stops.
    x = np.asarray(binary, dtype=np.int8)
    starts = np.flatnonzero(np.diff(np.r_[0, x]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[x, 0]) == -1)
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def binary_to_events(binary: np.ndarray, sfreq: float) -> List[Tuple[float, float]]:
    return [(s / sfreq, (e + 1) / sfreq) for s, e in binary_to_segments(binary)]


def binary_to_events_segmented(
    binary: np.ndarray,
    sfreq: float,
    segments: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[float, float]]:
    if not segments:
        return binary_to_events(binary, sfreq)
    events: List[Tuple[float, float]] = []
    for seg_start, seg_stop in segments:
        if seg_stop <= seg_start:
            continue
        seg_events = binary_to_events(binary[seg_start:seg_stop], sfreq)
        offset_sec = seg_start / sfreq
        events.extend([(s + offset_sec, e + offset_sec) for s, e in seg_events])
    return events


def events_to_binary(events: List[Tuple[float, float]], n_samples: int, sfreq: float) -> np.ndarray:
    y = np.zeros(n_samples, dtype=np.uint8)
    for onset, offset in events:
        s = max(0, int(round(onset * sfreq)))
        e = min(n_samples, int(round(offset * sfreq)))
        if e > s:
            y[s:e] = 1
    return y


def stage_mode_to_allowed_labels(stage_mode: str) -> Tuple[str, ...]:
    if stage_mode == "n2_only":
        return ("N2",)
    if stage_mode == "n2_n3":
        return ("N2", "N3")
    if stage_mode == "nrem":
        return ("N1", "N2", "N3")
    if stage_mode == "all_sleep":
        return ("N1", "N2", "N3", "REM")
    if stage_mode == "none":
        return tuple()
    raise ValueError(stage_mode)


def build_stage_sample_intervals(stage_events: List[Tuple[float, float, str]], sfreq: float, stage_mode: str, n_samples: int) -> List[Tuple[int, int]]:
    if stage_mode == "none":
        return [(0, n_samples)]
    allowed = set(stage_mode_to_allowed_labels(stage_mode))
    intervals: List[Tuple[int, int]] = []
    for onset, offset, stage in stage_events:
        if stage not in allowed:
            continue
        s = max(0, int(round(onset * sfreq)))
        e = min(n_samples, int(round(offset * sfreq)))
        if e > s:
            intervals.append((s, e))
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def concatenate_intervals(signal_in: np.ndarray, truth_binary: np.ndarray, intervals: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    if not intervals:
        return np.zeros((0,), dtype=signal_in.dtype), np.zeros((0,), dtype=truth_binary.dtype)
    sig_parts = [signal_in[s:e] for s, e in intervals if e > s]
    truth_parts = [truth_binary[s:e] for s, e in intervals if e > s]
    if not sig_parts:
        return np.zeros((0,), dtype=signal_in.dtype), np.zeros((0,), dtype=truth_binary.dtype)
    return np.concatenate(sig_parts), np.concatenate(truth_parts)


def concatenate_parts(parts: List[np.ndarray], dtype: np.dtype) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    non_empty = [np.asarray(part) for part in parts if len(part) > 0]
    if not non_empty:
        return np.zeros((0,), dtype=dtype), []
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for part in non_empty:
        nxt = cursor + len(part)
        bounds.append((cursor, nxt))
        cursor = nxt
    return np.concatenate(non_empty).astype(dtype), bounds


def remove_stage_anomalies(
    eeg: np.ndarray,
    spindle_binary: np.ndarray,
    range_divisor: float,
    jump_factor: float,
    recovery_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(eeg) < 3:
        return eeg.astype(np.float32), spindle_binary.astype(np.uint8)
    sf_range = (float(np.max(eeg)) - float(np.min(eeg))) / (range_divisor + 1e-8)
    anomaly_eeg_th = jump_factor * sf_range
    anomaly_cali_th = recovery_factor * sf_range
    e_diff = np.diff(np.r_[eeg, 0.0])
    e_diff_abn = np.flatnonzero(np.abs(e_diff) >= anomaly_eeg_th)
    jj = 0
    begin_abn: List[int] = []
    stop_abn: List[int] = []
    while jj < len(e_diff_abn) - 1:
        s = int(e_diff_abn[jj])
        idx = np.flatnonzero(np.abs(eeg - eeg[s]) < anomaly_cali_th)
        idx = idx[idx > s]
        if len(idx) == 0:
            break
        e = int(idx[0])
        begin_abn.append(s)
        stop_abn.append(e)
        if e <= e_diff_abn[jj + 1]:
            jj += 1
        else:
            next_idx = np.flatnonzero(e_diff_abn >= e)
            if len(next_idx) == 0:
                break
            jj = int(next_idx[0])
    eeg_out = eeg.copy()
    truth_out = spindle_binary.copy()
    for k in range(len(begin_abn) - 1, -1, -1):
        s = begin_abn[k]
        e = stop_abn[k]
        if e > s + 1:
            eeg_out = np.delete(eeg_out, np.arange(s + 1, e))
            truth_out = np.delete(truth_out, np.arange(s + 1, e))
    return eeg_out.astype(np.float32), truth_out.astype(np.uint8)


def butter_sos_bandpass(low_hz: float, high_hz: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sfreq
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    return signal.butter(order, [low, high], btype="bandpass", output="sos")


def safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:
        return x.astype(np.float32)
    try:
        return signal.sosfiltfilt(sos, x).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, x).astype(np.float32)


def safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:
        return x.astype(np.float32)
    try:
        return signal.filtfilt(b, a, x).astype(np.float32)
    except ValueError:
        return signal.lfilter(b, a, x).astype(np.float32)


def preprocess_candidate_signal(x: np.ndarray, sfreq: float, cfg: Config) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    sos = butter_sos_bandpass(cfg.bandpass_low_hz, cfg.bandpass_high_hz, sfreq)
    y = safe_sosfiltfilt(sos, y)
    if cfg.notch_enabled and cfg.notch_hz > 0:
        b, a = signal.iirnotch(cfg.notch_hz, 30.0, sfreq)
        y = safe_filtfilt(b, a, y)
    return y.astype(np.float32)


def compute_sigma_band(x: np.ndarray, sfreq: float, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter_sos_bandpass(low_hz, high_hz, sfreq)
    return safe_sosfiltfilt(sos, x).astype(np.float32)


def compute_teager_energy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return np.zeros_like(x, dtype=np.float32)
    teo = np.zeros_like(x, dtype=np.float64)
    teo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    teo[0] = teo[1]
    teo[-1] = teo[-2]
    return teo.astype(np.float32)


def compute_envelope(x: np.ndarray) -> np.ndarray:
    analytic = signal.hilbert(np.asarray(x, dtype=np.float64))
    return np.abs(analytic).astype(np.float32)


def normalize_subject_signal(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none" or len(x) == 0:
        return x.astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if mode == "subject_zscore":
        mean = float(np.mean(x))
        std = float(np.std(x))
        return ((x - mean) / (std + 1e-8)).astype(np.float32)
    if mode == "subject_robust":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        return ((x - med) / (1.4826 * mad + 1e-8)).astype(np.float32)
    return x.astype(np.float32)


def spindleunet_scale_raw(x: np.ndarray, clip_uv: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    # MNE returns EEG in volts for EDF files; SpindleU-Net preprocessing is
    # specified in microvolts. If the signal already appears to be in uV, keep it.
    finite = y[np.isfinite(y)]
    if len(finite) and float(np.nanpercentile(np.abs(finite), 95)) < 1.0:
        y = y * 1e6
    clip = max(float(clip_uv), 1e-6)
    y = np.clip(y, -clip, clip)
    return (y / (2.0 * clip)).astype(np.float32)


def prepare_segmentation_channel(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply stable physical scaling followed by the configured record normalization."""
    scaled = spindleunet_scale_raw(x, cfg.segmentation_clip_uv)
    if cfg.segmentation_normalize_inputs and cfg.normalization_mode in ("subject_zscore", "subject_robust"):
        scaled = normalize_subject_signal(scaled, cfg.normalization_mode)
    return scaled.astype(np.float32)


def prepare_segmentation_teo_channel(x: np.ndarray) -> np.ndarray:
    """Scale non-negative Teager energy without treating its V^2 units as EEG voltage."""
    energy = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    finite_positive = energy[np.isfinite(energy) & (energy > 0.0)]
    if len(finite_positive) == 0:
        return np.zeros_like(energy, dtype=np.float32)
    baseline = max(float(np.median(finite_positive)), 1e-20)
    scaled = np.log1p(energy / baseline)
    finite_scaled = scaled[np.isfinite(scaled)]
    upper = max(float(np.percentile(finite_scaled, 99.5)), 1e-6)
    return np.clip(scaled / upper, 0.0, 1.0).astype(np.float32)


def detection_candidate(teo: np.ndarray, tmin: float, threshold: float, baseline: float, fs: float) -> np.ndarray:
    min_peak_distance = max(1, int(round(tmin * fs)))
    locs, _ = signal.find_peaks(teo, distance=min_peak_distance, height=threshold)
    if len(locs) == 0:
        return np.zeros(len(teo), dtype=np.uint8)
    begin_det = locs.astype(int).copy()
    stop_det = locs.astype(int).copy()
    while np.any(teo[begin_det] >= baseline) or np.any(teo[stop_det] >= baseline):
        idx1 = np.flatnonzero(teo[begin_det] >= baseline)
        idx2 = np.flatnonzero(teo[stop_det] >= baseline)
        idx1 = idx1[begin_det[idx1] > 0]
        idx2 = idx2[stop_det[idx2] < (len(teo) - 1)]
        if len(idx1) == 0 and len(idx2) == 0:
            break
        begin_det[idx1] -= 1
        stop_det[idx2] += 1
    detvec = np.zeros(len(teo), dtype=np.uint8)
    for s, e in zip(begin_det, stop_det):
        detvec[s:e + 1] = 1
    return detvec


def merge_close_binary_runs(detvec: np.ndarray, gap_sec: float, fs: float) -> np.ndarray:
    out = np.asarray(detvec, dtype=np.uint8).copy()
    segments = binary_to_segments(out)
    if len(segments) <= 1:
        return out
    max_gap = int(round(gap_sec * fs))
    for (s1, e1), (s2, _) in zip(segments[:-1], segments[1:]):
        if s2 - e1 < max_gap:
            out[e1:s2] = 1
    return out


def prune_candidate_binary(detvec: np.ndarray, min_sec: float, max_sec: float, merge_gap_sec: float, fs: float) -> np.ndarray:
    out = merge_close_binary_runs(detvec, merge_gap_sec, fs)
    segments = binary_to_segments(out)
    cleaned = np.zeros_like(out, dtype=np.uint8)
    min_len = int(round(min_sec * fs))
    max_len = int(round(max_sec * fs))
    for s, e in segments:
        dur = e - s
        if dur < min_len:
            continue
        if dur > max_len:
            center = int(round((s + e) / 2.0))
            half = max_len // 2
            s = max(0, center - half)
            e = min(len(out) - 1, s + max_len)
        cleaned[s:e + 1] = 1
    return cleaned


def compute_jiang_teo_thresholds(raw_signal: np.ndarray, sigma_signal: np.ndarray, B: float, T: float) -> Tuple[float, float]:
    sf_range = (float(np.max(raw_signal)) - float(np.min(raw_signal))) / (265.0 * 2.0 + 1e-8)
    sf_s = float(np.mean(np.abs(np.diff(sigma_signal)))) / (0.8105 + 1e-8)
    baseline = B * sf_s * sf_range
    threshold = T * sf_s * sf_range
    return float(baseline), float(threshold)


def envelope_threshold_detector(envelope: np.ndarray, fs: float, B: float, T: float, min_sec: float) -> np.ndarray:
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    baseline = median + B * mad
    threshold = median + T * mad
    return detection_candidate(envelope, min_sec, threshold, baseline, fs)


def detvec_to_events(detvec: np.ndarray, sfreq: float) -> List[Tuple[float, float]]:
    return [(s / sfreq, (e + 1) / sfreq) for s, e in binary_to_segments(detvec)]


def merge_candidate_event_lists(
    base_events: Sequence[Tuple[float, float]],
    extra_events: Sequence[Tuple[float, float]],
    overlap_thr: float = 0.5,
) -> List[Tuple[float, float]]:
    merged = list(base_events)
    for event in extra_events:
        if any(interval_iou(event, existing) >= overlap_thr for existing in merged):
            continue
        merged.append(event)
    return sorted(merged, key=lambda x: (x[0], x[1]))


def peak_proposal_events(
    envelope_signal: np.ndarray,
    fs: float,
    cfg: Config,
) -> List[Tuple[float, float]]:
    if len(envelope_signal) < 3 or not cfg.candidate_peak_proposals_enabled:
        return []
    env = np.asarray(envelope_signal, dtype=np.float32)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    scale = 1.4826 * mad + 1e-8
    env_z = (env - med) / scale
    distance = max(1, int(round(cfg.candidate_peak_min_distance_sec * fs)))
    peaks, props = signal.find_peaks(env_z, height=cfg.candidate_peak_z_threshold, distance=distance)
    if len(peaks) == 0:
        return []
    heights = props.get("peak_heights", env_z[peaks])
    max_count = max(1, int(round(cfg.candidate_peak_max_per_min * (len(env) / fs) / 60.0)))
    if len(peaks) > max_count:
        keep = np.argpartition(heights, -max_count)[-max_count:]
        peaks = peaks[keep]
        heights = heights[keep]
    order = np.argsort(peaks)
    peaks = peaks[order]
    half = 0.5 * cfg.candidate_peak_duration_sec
    events: List[Tuple[float, float]] = []
    for peak in peaks:
        center_sec = float(peak) / fs
        start = max(0.0, center_sec - half)
        stop = center_sec + half
        events.append((start, stop))
    return events


def generate_candidates_from_signals(
    raw_signal: np.ndarray,
    sigma_signal: np.ndarray,
    teo_signal: np.ndarray,
    envelope_signal: np.ndarray,
    cfg: Config,
    B: float,
    T: float,
    fs: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    detvecs: List[np.ndarray] = []
    details: Dict[str, float] = {}
    if cfg.candidate_mode in ("jiang_teo", "hybrid_union"):
        baseline, threshold = compute_jiang_teo_thresholds(raw_signal, sigma_signal, B, T)
        det_teo = detection_candidate(teo_signal, cfg.candidate_min_sec, threshold, baseline, fs)
        detvecs.append(det_teo)
        details["teo_baseline"] = baseline
        details["teo_threshold"] = threshold
    if cfg.candidate_mode in ("sigma_envelope", "hybrid_union"):
        det_env = envelope_threshold_detector(envelope_signal, fs, B, T, cfg.candidate_min_sec)
        detvecs.append(det_env)
    if not detvecs:
        raise ValueError(f"Unsupported candidate mode: {cfg.candidate_mode}")
    detvec = detvecs[0].copy()
    for extra in detvecs[1:]:
        detvec = np.maximum(detvec, extra)
    detvec = prune_candidate_binary(detvec, cfg.candidate_min_sec, cfg.candidate_max_sec, cfg.candidate_merge_gap_sec, fs)

    if cfg.candidate_recall_boost_enabled:
        relaxed_B = max(0.25, B - cfg.candidate_recall_boost_B_offset)
        relaxed_T = max(1.5, T - cfg.candidate_recall_boost_T_offset)
        recall_detvecs: List[np.ndarray] = []
        baseline, threshold = compute_jiang_teo_thresholds(raw_signal, sigma_signal, relaxed_B, relaxed_T)
        recall_detvecs.append(detection_candidate(teo_signal, cfg.candidate_recall_boost_min_sec, threshold, baseline, fs))
        recall_detvecs.append(envelope_threshold_detector(envelope_signal, fs, relaxed_B, relaxed_T, cfg.candidate_recall_boost_min_sec))
        recall_detvec = recall_detvecs[0]
        for extra in recall_detvecs[1:]:
            recall_detvec = np.maximum(recall_detvec, extra)
        recall_detvec = prune_candidate_binary(
            recall_detvec,
            cfg.candidate_recall_boost_min_sec,
            cfg.candidate_max_sec,
            cfg.candidate_recall_boost_merge_gap_sec,
            fs,
        )
        details["candidate_recall_boost_b"] = relaxed_B
        details["candidate_recall_boost_t"] = relaxed_T
        details["candidate_recall_boost_added_samples"] = float(np.sum((recall_detvec > 0) & (detvec == 0)))
        detvec = np.maximum(detvec, recall_detvec.astype(np.uint8))
    detvec = prune_candidate_binary(detvec, cfg.candidate_min_sec, cfg.candidate_max_sec, cfg.candidate_merge_gap_sec, fs)
    return detvec.astype(np.uint8), details


def generate_candidates_for_record(rec: Dict[str, Any], cfg: Config, B: float, T: float) -> Dict[str, Any]:
    fs = rec["sfreq"]
    bounds = rec.get("stage_segment_bounds") or [(0, len(rec["candidate_raw"]))]
    detvec = np.zeros(len(rec["candidate_raw"]), dtype=np.uint8)
    teo_baselines: List[float] = []
    teo_thresholds: List[float] = []
    details: Dict[str, float] = {}
    for seg_start, seg_stop in bounds:
        if seg_stop <= seg_start:
            continue
        seg_detvec, seg_details = generate_candidates_from_signals(
            rec["candidate_raw"][seg_start:seg_stop],
            rec["candidate_sigma"][seg_start:seg_stop],
            rec["candidate_teo"][seg_start:seg_stop],
            rec["candidate_envelope"][seg_start:seg_stop],
            cfg,
            B,
            T,
            fs,
        )
        detvec[seg_start:seg_stop] = np.maximum(detvec[seg_start:seg_stop], seg_detvec.astype(np.uint8))
        if "teo_baseline" in seg_details:
            teo_baselines.append(seg_details["teo_baseline"])
        if "teo_threshold" in seg_details:
            teo_thresholds.append(seg_details["teo_threshold"])
    events = binary_to_events_segmented(detvec, fs, bounds)
    if cfg.candidate_peak_proposals_enabled:
        peak_events: List[Tuple[float, float]] = []
        for seg_start, seg_stop in bounds:
            if seg_stop <= seg_start:
                continue
            seg_peak_events = peak_proposal_events(rec["candidate_envelope"][seg_start:seg_stop], fs, cfg)
            offset_sec = seg_start / fs
            peak_events.extend([(s + offset_sec, e + offset_sec) for s, e in seg_peak_events])
        details["candidate_peak_proposals"] = float(len(peak_events))
        events = merge_candidate_event_lists(events, peak_events, overlap_thr=0.35)
    if teo_baselines:
        details["teo_baseline_mean"] = float(np.mean(teo_baselines))
    if teo_thresholds:
        details["teo_threshold_mean"] = float(np.mean(teo_thresholds))
    return {
        "events": events,
        "detvec": detvec.astype(np.uint8),
        "n_candidates": len(events),
        **details,
    }


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def overlap_fraction(event: Tuple[float, float], reference_events: List[Tuple[float, float]]) -> float:
    s, e = event
    if e <= s:
        return 0.0
    total = e - s
    overlap = 0.0
    for rs, re_ in reference_events:
        overlap += max(0.0, min(e, re_) - max(s, rs))
    return overlap / total if total > 0 else 0.0


def candidate_overlap_scores(candidates: List[Tuple[float, float]], true_events: List[Tuple[float, float]]) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=np.float32)
    for i, cand in enumerate(candidates):
        scores[i] = float(overlap_fraction(cand, true_events))
    return scores


def hard_candidate_labels_from_scores(scores: np.ndarray, lam: float) -> np.ndarray:
    return (scores >= lam).astype(np.float32)


def soft_candidate_labels_from_record(
    candidates: List[Tuple[float, float]],
    rec: Dict[str, Any],
    lam: float,
    cfg: Config,
) -> np.ndarray:
    union_scores = candidate_overlap_scores(candidates, rec["true_events_stage"])
    expert_scores = [union_scores]
    e1_events = rec.get("true_events_e1_stage", [])
    e2_events = rec.get("true_events_e2_stage", [])
    if e1_events:
        expert_scores.append(candidate_overlap_scores(candidates, e1_events))
    if e2_events:
        expert_scores.append(candidate_overlap_scores(candidates, e2_events))
    score = np.mean(np.stack(expert_scores, axis=0), axis=0)
    lower = max(0.0, cfg.soft_candidate_label_lower_scale * lam)
    scaled = np.clip((score - lower) / max(lam - lower, 1e-6), 0.0, 1.0)
    if cfg.soft_candidate_label_gamma != 1.0:
        scaled = np.power(scaled, cfg.soft_candidate_label_gamma)
    return scaled.astype(np.float32)


def candidate_labels_for_record(
    candidates: List[Tuple[float, float]],
    rec: Dict[str, Any],
    lam: float,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    hard_scores = candidate_overlap_scores(candidates, rec["true_events_stage"])
    hard_labels = hard_candidate_labels_from_scores(hard_scores, lam)
    if cfg.soft_candidate_labels_enabled:
        soft_labels = soft_candidate_labels_from_record(candidates, rec, lam, cfg)
    else:
        soft_labels = hard_labels.astype(np.float32)
    return hard_labels.astype(np.float32), soft_labels.astype(np.float32)


def candidate_diagnostics(subject_id: str, candidates: List[Tuple[float, float]], true_events: List[Tuple[float, float]], retained_len_samples: int, sfreq: float) -> Dict[str, Any]:
    strict = event_metrics(true_events, candidates, 0.2)
    retained_min = retained_len_samples / sfreq / 60.0 if retained_len_samples > 0 else 0.0
    return {
        "subject_id": subject_id,
        "retained_len_samples": int(retained_len_samples),
        "retained_minutes": float(retained_min),
        "n_candidates": int(len(candidates)),
        "n_true_events": int(len(true_events)),
        "candidate_density_per_min": float(len(candidates) / max(retained_min, 1e-8)),
        "candidate_strict_precision": strict["precision"],
        "candidate_strict_recall": strict["recall"],
        "candidate_strict_f1": strict["f1"],
    }


def segment_bounds_for_index(index: int, segments: Optional[Sequence[Tuple[int, int]]], n_samples: int) -> Tuple[int, int]:
    if not segments:
        return 0, n_samples
    for seg_start, seg_stop in segments:
        if seg_start <= index < seg_stop:
            return int(seg_start), int(seg_stop)
    return 0, n_samples


def extract_window(
    signal_in: np.ndarray,
    center_idx: int,
    window_len: int,
    valid_bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    lower, upper = valid_bounds if valid_bounds is not None else (0, len(signal_in))
    lower = max(0, int(lower))
    upper = min(len(signal_in), int(upper))
    if upper <= lower:
        return np.zeros((window_len,), dtype=np.float32)
    local = signal_in[lower:upper]
    center_local = int(np.clip(center_idx - lower, 0, len(local) - 1))
    half = window_len // 2
    start = center_local - half
    end = start + window_len
    if start < 0:
        pad = -start
        out = np.pad(local[:end], (pad, 0), mode="edge")
    elif end > len(local):
        pad = end - len(local)
        out = np.pad(local[start:], (0, pad), mode="edge")
    else:
        out = local[start:end]
    if len(out) != window_len:
        out = np.resize(out, window_len)
    return out.astype(np.float32)


def build_model_channel_list(rec: Dict[str, Any], cfg: Config) -> List[np.ndarray]:
    channels: List[np.ndarray] = []
    if cfg.use_raw_channel:
        channels.append(rec["model_raw"])
    if cfg.use_sigma_channel:
        channels.append(rec["model_sigma"])
    if cfg.use_teo_channel:
        channels.append(rec["model_teo"])
    if cfg.use_envelope_channel:
        channels.append(rec["model_envelope"])
    if not channels:
        raise RuntimeError("At least one model input channel must be enabled.")
    return channels


def build_candidate_feature_channels(
    rec: Dict[str, Any],
    start_sec: float,
    stop_sec: float,
    window_len: int,
    fs: float,
    cfg: Config,
) -> np.ndarray:
    if not cfg.use_candidate_feature_channels:
        return np.zeros((0, window_len), dtype=np.float32)
    start_idx = max(0, int(round(start_sec * fs)))
    stop_idx = min(len(rec["truth_binary_stage"]), int(round(stop_sec * fs)))
    if stop_idx <= start_idx:
        stop_idx = min(len(rec["truth_binary_stage"]), start_idx + 1)
    sigma_seg = rec["candidate_sigma"][start_idx:stop_idx]
    envelope_seg = rec["candidate_envelope"][start_idx:stop_idx]
    teo_seg = rec["candidate_teo"][start_idx:stop_idx]

    duration_norm = float((stop_sec - start_sec) / max(cfg.candidate_max_sec, 1e-6))
    sigma_ref = float(rec.get("candidate_sigma_rms_ref", np.sqrt(np.mean(np.square(rec["candidate_sigma"])) + 1e-8)))
    env_med = float(rec.get("candidate_envelope_median", np.median(rec["candidate_envelope"])))
    env_mad = float(rec.get("candidate_envelope_mad", np.median(np.abs(rec["candidate_envelope"] - env_med))))
    teo_ref = float(rec.get("candidate_teo_abs_median", np.median(np.abs(rec["candidate_teo"])) + 1e-8))

    sigma_rms = float(np.sqrt(np.mean(np.square(sigma_seg)) + 1e-8) / max(sigma_ref, 1e-6))
    env_peak = float((np.max(envelope_seg) - env_med) / max(1.4826 * env_mad, 1e-6)) if len(envelope_seg) else 0.0
    teo_peak = float(np.max(np.abs(teo_seg)) / max(teo_ref, 1e-6)) if len(teo_seg) else 0.0

    features = np.array([duration_norm, sigma_rms, env_peak, teo_peak], dtype=np.float32)
    return np.repeat(features[:, None], window_len, axis=1)


def candidate_centers_and_bounds(
    rec: Dict[str, Any],
    candidates: Sequence[Tuple[float, float]],
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(rec["truth_binary_stage"])
    if len(candidates) == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty, empty, empty

    cand_arr = np.asarray(candidates, dtype=np.float64)
    start_sec = cand_arr[:, 0]
    stop_sec = cand_arr[:, 1]
    centers = np.rint(((start_sec + stop_sec) * 0.5) * fs).astype(np.int64)

    bounds = rec.get("stage_segment_bounds")
    bound_starts = np.zeros(len(candidates), dtype=np.int64)
    bound_stops = np.full(len(candidates), n_samples, dtype=np.int64)
    if bounds:
        starts = np.asarray([int(s) for s, _ in bounds], dtype=np.int64)
        stops = np.asarray([int(e) for _, e in bounds], dtype=np.int64)
        idx = np.searchsorted(starts, centers, side="right") - 1
        valid = (idx >= 0) & (idx < len(starts))
        safe_idx = np.clip(idx, 0, max(len(starts) - 1, 0))
        valid &= centers < stops[safe_idx]
        bound_starts[valid] = starts[safe_idx[valid]]
        bound_stops[valid] = stops[safe_idx[valid]]

    start_idx = np.rint(start_sec * fs).astype(np.int64)
    stop_idx = np.rint(stop_sec * fs).astype(np.int64)
    return centers, bound_starts, bound_stops, start_idx, stop_idx


def extract_windows_batch(
    signal_in: np.ndarray,
    centers: np.ndarray,
    window_len: int,
    bound_starts: np.ndarray,
    bound_stops: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n = len(centers)
    out = np.zeros((n, window_len), dtype=np.float32)
    if n == 0 or len(signal_in) == 0:
        return out

    arr = np.asarray(signal_in, dtype=np.float32)
    offsets = np.arange(window_len, dtype=np.int64) - (window_len // 2)
    chunk = max(1, int(batch_size))
    n_samples = len(arr)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        lower = np.clip(bound_starts[start:stop].astype(np.int64), 0, max(n_samples - 1, 0))
        upper = np.clip(bound_stops[start:stop].astype(np.int64), lower + 1, n_samples)
        idx = centers[start:stop, None].astype(np.int64) + offsets[None, :]
        idx = np.maximum(idx, lower[:, None])
        idx = np.minimum(idx, (upper - 1)[:, None])
        out[start:stop, :] = arr[idx]
    return out


def build_candidate_feature_matrix(
    rec: Dict[str, Any],
    start_idx: np.ndarray,
    stop_idx: np.ndarray,
    start_sec: np.ndarray,
    stop_sec: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    n = len(start_idx)
    if not cfg.use_candidate_feature_channels:
        return np.zeros((n, 0), dtype=np.float32)
    if n == 0 or len(rec["truth_binary_stage"]) == 0:
        return np.zeros((n, 4), dtype=np.float32)

    n_samples = len(rec["truth_binary_stage"])
    starts = np.clip(start_idx.astype(np.int64), 0, max(n_samples - 1, 0))
    stops = np.clip(stop_idx.astype(np.int64), starts + 1, n_samples)
    lengths = np.maximum(stops - starts, 1)

    sigma = np.asarray(rec["candidate_sigma"], dtype=np.float32)
    envelope = np.asarray(rec["candidate_envelope"], dtype=np.float32)
    teo = np.asarray(rec["candidate_teo"], dtype=np.float32)

    duration_norm = ((stop_sec - start_sec) / max(cfg.candidate_max_sec, 1e-6)).astype(np.float32)
    sigma_ref = float(rec.get("candidate_sigma_rms_ref", np.sqrt(np.mean(np.square(sigma)) + 1e-8)))
    env_med = float(rec.get("candidate_envelope_median", np.median(envelope)))
    env_mad = float(rec.get("candidate_envelope_mad", np.median(np.abs(envelope - env_med))))
    teo_ref = float(rec.get("candidate_teo_abs_median", np.median(np.abs(teo)) + 1e-8))

    sigma_sq_cum = np.concatenate(([0.0], np.cumsum(np.square(sigma.astype(np.float64)))))
    sigma_sum = sigma_sq_cum[stops] - sigma_sq_cum[starts]
    sigma_rms = np.sqrt(sigma_sum / lengths + 1e-8) / max(sigma_ref, 1e-6)

    env_peak = np.zeros(n, dtype=np.float32)
    teo_peak = np.zeros(n, dtype=np.float32)
    env_scale = max(1.4826 * env_mad, 1e-6)
    for i, (s, e) in enumerate(zip(starts, stops)):
        env_seg = envelope[s:e]
        teo_seg = teo[s:e]
        env_peak[i] = float((np.max(env_seg) - env_med) / env_scale) if len(env_seg) else 0.0
        teo_peak[i] = float(np.max(np.abs(teo_seg)) / max(teo_ref, 1e-6)) if len(teo_seg) else 0.0

    return np.stack(
        [
            duration_norm,
            sigma_rms.astype(np.float32),
            env_peak,
            teo_peak,
        ],
        axis=1,
    ).astype(np.float32)


def build_candidate_dataset(rec: Dict[str, Any], candidates: List[Tuple[float, float]], labels: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    fs = rec["sfreq"]
    window_len = max(8, int(round(cfg.candidate_window_sec * fs)))
    signal_channels = build_model_channel_list(rec, cfg)
    n_feature_channels = 4 if cfg.use_candidate_feature_channels else 0
    X = np.zeros((len(candidates), len(signal_channels) + n_feature_channels, window_len), dtype=np.float32)
    if len(candidates) == 0:
        return X, labels.astype(np.float32)

    centers, bound_starts, bound_stops, start_idx, stop_idx = candidate_centers_and_bounds(rec, candidates, fs)
    for ch_idx, arr in enumerate(signal_channels):
        X[:, ch_idx, :] = extract_windows_batch(
            arr,
            centers,
            window_len,
            bound_starts,
            bound_stops,
            cfg.window_extraction_batch_size,
        )
    if cfg.use_candidate_feature_channels:
        cand_arr = np.asarray(candidates, dtype=np.float64)
        features = build_candidate_feature_matrix(rec, start_idx, stop_idx, cand_arr[:, 0], cand_arr[:, 1], cfg)
        X[:, len(signal_channels):len(signal_channels) + features.shape[1], :] = features[:, :, None]
    return X, labels.astype(np.float32)


def get_candidate_window_matrix(rec: Dict[str, Any], bundle: Dict[str, Any], cfg: Config) -> np.ndarray:
    cache_key = "_candidate_window_matrix"
    if cfg.cache_candidate_windows and cache_key in bundle:
        return bundle[cache_key]
    dummy_labels = np.zeros((len(bundle["events"]),), dtype=np.float32)
    X, _ = build_candidate_dataset(rec, bundle["events"], dummy_labels, cfg)
    if cfg.cache_candidate_windows:
        bundle[cache_key] = X
    return X


def clear_cached_candidate_windows(candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]]) -> int:
    cleared = 0
    for bundle in candidate_cache.values():
        if "_candidate_window_matrix" in bundle:
            bundle.pop("_candidate_window_matrix", None)
            cleared += 1
    return cleared


def compute_dataset_normalization_stats(X: np.ndarray) -> NormalizationStats:
    mean = X.mean(axis=(0, 2), keepdims=False)
    std = X.std(axis=(0, 2), keepdims=False)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_dataset_normalization(X: np.ndarray, stats: Optional[NormalizationStats], mode: str) -> np.ndarray:
    if mode != "dataset_train" or stats is None or X.size == 0:
        return X.astype(np.float32)
    return ((X - stats.mean[None, :, None]) / stats.std[None, :, None]).astype(np.float32)


def shift_with_edge_padding(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x
    if shift > 0:
        pad = np.repeat(x[:, :1], shift, axis=1)
        return np.concatenate([pad, x[:, :-shift]], axis=1)
    shift = abs(shift)
    pad = np.repeat(x[:, -1:], shift, axis=1)
    return np.concatenate([x[:, shift:], pad], axis=1)


class CandidateDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: Config, training: bool):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.training = training
        self.cfg = cfg
        self.max_shift = int(round(cfg.augment_time_jitter_sec * cfg.resample_hz))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        if self.training and self.cfg.augment_enabled:
            x = x.clone()
            if self.cfg.augment_noise_std > 0:
                x = x + torch.randn_like(x) * self.cfg.augment_noise_std
            if self.cfg.augment_scale_std > 0:
                scale = 1.0 + float(torch.empty((), dtype=x.dtype).normal_(0.0, self.cfg.augment_scale_std).item())
                x = x * scale
            if self.cfg.augment_invert_probability > 0 and float(torch.rand(()).item()) < self.cfg.augment_invert_probability:
                x = -x
            if self.max_shift > 0:
                shift = int(torch.randint(-self.max_shift, self.max_shift + 1, ()).item())
                if shift > 0:
                    pad = x[:, :1].expand(-1, shift)
                    x = torch.cat([pad, x[:, :-shift]], dim=1)
                elif shift < 0:
                    shift_abs = abs(shift)
                    pad = x[:, -1:].expand(-1, shift_abs)
                    x = torch.cat([x[:, shift_abs:], pad], dim=1)
        return x, y


def make_candidate_loader(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    device: torch.device,
    training: bool,
    shuffle: bool = False,
    sampler: Optional[WeightedRandomSampler] = None,
    batch_size: Optional[int] = None,
) -> DataLoader:
    num_workers = max(0, int(cfg.num_workers))
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size or cfg.batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.pin_memory and device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return DataLoader(CandidateDataset(X, y, cfg, training=training), **kwargs)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, dilation: int):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out + residual)
        return out


class CandidateNet(nn.Module):
    """
    Input shape: [batch, channels, time]
    Output shape: [batch]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int,
        dropout: float,
        use_residual: bool,
        use_temporal_attention_gate: bool,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.GELU(),
        )
        blocks: List[nn.Module] = []
        prev = hidden_channels[0]
        dilation = 1
        for out_ch in hidden_channels:
            if use_residual:
                blocks.append(ResidualConvBlock(prev, out_ch, kernel_size, dropout, dilation))
            else:
                padding = ((kernel_size - 1) // 2) * dilation
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(prev, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
                        nn.BatchNorm1d(out_ch),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
            prev = out_ch
            dilation = min(dilation * 2, 8)
        self.encoder = nn.Sequential(*blocks)
        self.temporal_attention_gate = (
            nn.Sequential(
                nn.Conv1d(prev, max(1, prev // 2), kernel_size=1),
                nn.GELU(),
                nn.Conv1d(max(1, prev // 2), 1, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_temporal_attention_gate
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(prev * 2, prev),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prev, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        gate = self.temporal_attention_gate(x)
        if not isinstance(self.temporal_attention_gate, nn.Identity):
            x = x * gate
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        logits = self.head(features).squeeze(-1)
        return logits


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    focal = ((1.0 - pt) ** gamma) * ce
    weights = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
    return (focal * weights).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor, dice_weight: float) -> torch.Tensor:
    bce = weighted_bce_loss(logits, targets, pos_weight)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + 1e-8
    dice = 1.0 - (2.0 * inter + 1e-8) / denom
    return bce + dice_weight * dice


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: Config, pos_weight: torch.Tensor) -> torch.Tensor:
    if cfg.loss_type == "weighted_bce":
        return weighted_bce_loss(logits, targets, pos_weight)
    if cfg.loss_type == "focal":
        return focal_loss(logits, targets, pos_weight, cfg.focal_gamma)
    if cfg.loss_type == "bce_dice":
        return bce_dice_loss(logits, targets, pos_weight, cfg.dice_weight)
    raise ValueError(cfg.loss_type)


def compute_pos_weight(y: np.ndarray, max_weight: float, device: torch.device) -> torch.Tensor:
    pos = float(np.sum(y > 0.5))
    neg = float(len(y) - pos)
    weight = max_weight if pos < 1 else min(max_weight, neg / (pos + 1e-8))
    return torch.tensor(weight, dtype=torch.float32, device=device)


def build_balanced_sampler(y: np.ndarray, enabled: bool, extra_weights: Optional[np.ndarray] = None) -> Optional[WeightedRandomSampler]:
    if not enabled or len(y) == 0:
        return None
    pos = float(np.sum(y > 0.5))
    neg = float(len(y) - pos)
    if pos < 1 or neg < 1:
        return None
    weights = np.where(y > 0.5, 0.5 / pos, 0.5 / neg).astype(np.float64)
    if extra_weights is not None and len(extra_weights) == len(weights):
        weights = weights * np.asarray(extra_weights, dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def predict_numpy_probabilities(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    if len(y) == 0:
        return np.zeros((0,), dtype=np.float32)
    loader = make_candidate_loader(X, y, cfg, device, training=False, batch_size=batch_size)
    return predict_probabilities(model, loader, device)


def hard_negative_weights(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    weights = np.ones(len(y_train), dtype=np.float64)
    if not cfg.hard_negative_mining_enabled or len(y_train) == 0:
        return weights
    neg_idx = np.flatnonzero(y_train <= 0.5)
    if len(neg_idx) == 0:
        return weights
    probs = predict_numpy_probabilities(model, X_train, y_train, cfg, device)
    neg_probs = probs[neg_idx]
    top_k = max(1, int(round(cfg.hard_negative_fraction * len(neg_idx))))
    if top_k >= len(neg_idx):
        selected = neg_idx
    else:
        selected = neg_idx[np.argpartition(neg_probs, -top_k)[-top_k:]]
    weights[selected] = float(cfg.hard_negative_boost)
    return weights


def candidate_classification_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "candidate_acc": float("nan"),
            "candidate_precision": float("nan"),
            "candidate_recall": float("nan"),
            "candidate_f1": float("nan"),
            "candidate_auc": float("nan"),
            "candidate_pr_auc": float("nan"),
        }
    y_pred = (probs >= thr).astype(np.uint8)
    out = {
        "candidate_acc": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "candidate_precision": float(sk_metrics.precision_score(y_true, y_pred, zero_division=0)),
        "candidate_recall": float(sk_metrics.recall_score(y_true, y_pred, zero_division=0)),
        "candidate_f1": float(sk_metrics.f1_score(y_true, y_pred, zero_division=0)),
        "candidate_auc": float("nan"),
        "candidate_pr_auc": float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        out["candidate_auc"] = float(sk_metrics.roc_auc_score(y_true, probs))
        out["candidate_pr_auc"] = float(sk_metrics.average_precision_score(y_true, probs))
    return out


def evaluate_classifier(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device, pos_weight: torch.Tensor) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            loss = compute_loss(logits, y, cfg, pos_weight)
            losses.append(float(loss.item()) * len(y))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())
    total = max(sum(len(x) for x in all_targets), 1)
    out = {"loss": float(sum(losses) / total)}
    if all_targets:
        y_true = np.concatenate(all_targets)
        probs = np.concatenate(all_probs)
        out.update(candidate_classification_metrics(y_true, probs))
    return out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
    epochs: int,
    pos_weight: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")   
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_candidate_f1": [],
        "val_candidate_auc": [],
        "val_candidate_pr_auc": [],
    }
    best_loss = float("inf")
    best_monitor = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision and device.type == "cuda"):
                logits = model(X)
                loss = compute_loss(logits, y, cfg, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(y)
            seen += len(y)

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_classifier(model, val_loader, cfg, device, pos_weight)
        val_loss = float(val_metrics["loss"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_candidate_f1"].append(float(val_metrics.get("candidate_f1", float("nan"))))
        history["val_candidate_auc"].append(float(val_metrics.get("candidate_auc", float("nan"))))
        history["val_candidate_pr_auc"].append(float(val_metrics.get("candidate_pr_auc", float("nan"))))
        monitor = float(val_metrics.get("candidate_pr_auc", float("nan")))
        if not math.isfinite(monitor):
            monitor = float(val_metrics.get("candidate_f1", float("-inf")))
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_cand_f1=%.4f | val_cand_auc=%.4f | val_cand_pr_auc=%.4f | lr=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            float(val_metrics.get("candidate_f1", float("nan"))),
            float(val_metrics.get("candidate_auc", float("nan"))),
            float(val_metrics.get("candidate_pr_auc", float("nan"))),
            optimizer.param_groups[0]["lr"],
        )

        improved = (monitor > best_monitor + 1e-6) or (
            math.isfinite(monitor) and abs(monitor - best_monitor) <= 1e-6 and val_loss < best_loss - 1e-6
        )
        if improved:
            best_monitor = monitor
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            if cfg.save_checkpoints:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "best_val_loss": best_loss,
                        "best_monitor": best_monitor,
                    },
                    checkpoint_path,
                )
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif cfg.save_checkpoints and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    return model, history


def merge_training_histories(base: Dict[str, List[float]], extra: Dict[str, List[float]], prefix: str) -> Dict[str, List[float]]:
    merged = dict(base)
    for key, values in extra.items():
        merged[f"{prefix}_{key}"] = values
    return merged


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def best_history_value(history: Dict[str, List[float]], metric: str) -> float:
    values = [float(v) for v in history.get(metric, []) if math.isfinite(float(v))]
    return max(values) if values else float("-inf")


def history_accept_metric(cfg: Config, history: Dict[str, List[float]]) -> str:
    if cfg.hard_negative_accept_metric in history:
        return cfg.hard_negative_accept_metric
    if "val_candidate_pr_auc" in history:
        return "val_candidate_pr_auc"
    return "val_candidate_f1"


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)


class SegmentationWindowDataset(Dataset):
    def __init__(self, windows: List[Tuple[str, int, int]], records: Dict[str, Any], cfg: Config, training: bool):
        self.windows = windows
        self.records = records
        self.cfg = cfg
        self.training = training
        self.window_len = int(round(cfg.segmentation_window_sec * cfg.resample_hz))

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _shift_tensor(x: torch.Tensor, shift: int, pad_mode: str) -> torch.Tensor:
        """Shift the time axis without circular wrap-around."""
        if shift == 0 or x.shape[-1] == 0:
            return x
        amount = min(abs(int(shift)), max(0, x.shape[-1] - 1))
        if amount == 0:
            return x
        out = torch.zeros_like(x)
        if shift > 0:
            out[..., amount:] = x[..., :-amount]
            if pad_mode == "edge":
                out[..., :amount] = x[..., :1]
        else:
            out[..., :-amount] = x[..., amount:]
            if pad_mode == "edge":
                out[..., -amount:] = x[..., -1:]
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid, start, stop = self.windows[idx]
        rec = self.records[sid]
        x_np = build_segmentation_input_window(rec, start, stop, self.window_len, self.cfg)
        if self.training and self.cfg.segmentation_annotator_confidence_labels_enabled:
            label_key = "truth_annotator_confidence_stage"
        elif self.training and self.cfg.segmentation_soft_labels_enabled:
            label_key = "truth_soft_stage"
        else:
            label_key = "truth_binary_stage"
        y_np = rec[label_key][start:stop].astype(np.float32)
        if len(y_np) < self.window_len:
            pad = self.window_len - len(y_np)
            y_np = np.pad(y_np, (0, pad), mode="constant")
        targets = [y_np[:self.window_len]]
        if self.cfg.segmentation_auxiliary_heads_enabled:
            center_key = (
                "truth_center_annotator_confidence_stage"
                if self.training and self.cfg.segmentation_annotator_confidence_labels_enabled
                else "truth_center_stage"
            )
            boundary_key = (
                "truth_boundary_annotator_confidence_stage"
                if self.training and self.cfg.segmentation_annotator_confidence_labels_enabled
                else "truth_boundary_stage"
            )
            center_np = rec[center_key][start:stop].astype(np.float32)
            boundary_np = rec[boundary_key][start:stop].astype(np.float32)
            if len(center_np) < self.window_len:
                center_np = np.pad(center_np, (0, self.window_len - len(center_np)), mode="constant")
            if len(boundary_np) < self.window_len:
                boundary_np = np.pad(boundary_np, (0, self.window_len - len(boundary_np)), mode="constant")
            targets.extend([center_np[:self.window_len], boundary_np[:self.window_len]])
        if self.cfg.segmentation_multi_annotator_enabled:
            expert1_np = rec["truth_binary_e1_stage"][start:stop].astype(np.float32)
            expert2_np = rec["truth_binary_e2_stage"][start:stop].astype(np.float32)
            if len(expert1_np) < self.window_len:
                expert1_np = np.pad(expert1_np, (0, self.window_len - len(expert1_np)), mode="constant")
            if len(expert2_np) < self.window_len:
                expert2_np = np.pad(expert2_np, (0, self.window_len - len(expert2_np)), mode="constant")
            targets.extend([expert1_np[:self.window_len], expert2_np[:self.window_len]])
        x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(np.stack(targets, axis=0), dtype=np.float32))
        if self.training and self.cfg.augment_enabled:
            max_shift = max(0, int(round(self.cfg.augment_time_jitter_sec * rec["sfreq"])))
            if max_shift > 0:
                shift = int(torch.randint(-max_shift, max_shift + 1, ()).item())
                x = self._shift_tensor(x, shift, "edge")
                y = self._shift_tensor(y, shift, "zero")
            if self.cfg.augment_noise_std > 0:
                x = x + torch.randn_like(x) * self.cfg.augment_noise_std
            if self.cfg.augment_scale_std > 0:
                scale = 1.0 + float(torch.empty((), dtype=x.dtype).normal_(0.0, self.cfg.augment_scale_std).item())
                x = x * scale
            if self.cfg.augment_invert_probability > 0 and float(torch.rand(()).item()) < self.cfg.augment_invert_probability:
                # Raw EEG and bandpass signals are signed; envelopes and Teager
                # energy are not. Name-based indexing remains correct when an
                # optional secondary EEG lead is appended.
                names = segmentation_channel_names(self.cfg)
                signed_indices = [
                    idx for idx, name in enumerate(names)
                    if name.endswith("raw") or name.endswith("sigma")
                ]
                if signed_indices:
                    x[signed_indices] = -x[signed_indices]
        return x, y


def build_soft_boundary_labels(binary: np.ndarray, sfreq: float, shoulder_sec: float, edge_value: float) -> np.ndarray:
    y = np.asarray(binary, dtype=np.float32).copy()
    shoulder = max(0, int(round(float(shoulder_sec) * sfreq)))
    if shoulder <= 0:
        return y
    edge_value = float(np.clip(edge_value, 0.0, 1.0))
    for s, e in binary_to_segments((binary > 0).astype(np.uint8)):
        left_start = max(0, s - shoulder)
        for idx in range(left_start, s):
            dist = s - idx
            value = 1.0 - (1.0 - edge_value) * (dist / shoulder)
            y[idx] = max(y[idx], value)
        right_stop = min(len(y), e + 1 + shoulder)
        for idx in range(e + 1, right_stop):
            dist = idx - e
            value = 1.0 - (1.0 - edge_value) * (dist / shoulder)
            y[idx] = max(y[idx], value)
    return y.astype(np.float32)


def build_annotator_confidence_targets(
    union_soft: np.ndarray,
    expert1_binary: np.ndarray,
    expert2_binary: np.ndarray,
    sfreq: float,
    single_expert_weight: float,
    agreement_tolerance_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a confidence-weighted union target and per-sample confidence."""
    e1 = (np.asarray(expert1_binary) > 0).astype(np.uint8)
    e2 = (np.asarray(expert2_binary) > 0).astype(np.uint8)
    if len(e1) != len(e2) or len(e1) != len(union_soft):
        raise ValueError("Expert and union label arrays must have identical lengths.")
    tolerance = max(0, int(round(float(agreement_tolerance_sec) * sfreq)))
    agreed = np.zeros(len(e1), dtype=bool)
    e1_segments = binary_to_segments(e1)
    e2_segments = binary_to_segments(e2)
    for s1, e1_stop in e1_segments:
        for s2, e2_stop in e2_segments:
            if s2 > e1_stop + tolerance:
                break
            if e2_stop < s1 - tolerance:
                continue
            agreed[s1:e1_stop + 1] = True
            agreed[s2:e2_stop + 1] = True
    single_weight = float(np.clip(single_expert_weight, 0.0, 1.0))
    confidence = np.full(len(e1), single_weight, dtype=np.float32)
    confidence[agreed] = 1.0
    target = np.asarray(union_soft, dtype=np.float32) * confidence
    return target.astype(np.float32), confidence


def build_center_boundary_labels(
    binary: np.ndarray,
    sfreq: float,
    center_width_sec: float,
    boundary_width_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(binary)
    center = np.zeros(n, dtype=np.float32)
    boundary = np.zeros(n, dtype=np.float32)
    center_radius = max(1, int(round(float(center_width_sec) * sfreq * 0.5)))
    boundary_radius = max(1, int(round(float(boundary_width_sec) * sfreq * 0.5)))
    for s, e in binary_to_segments((binary > 0).astype(np.uint8)):
        mid = int(round((s + e) * 0.5))
        c0 = max(0, mid - center_radius)
        c1 = min(n, mid + center_radius + 1)
        if c1 > c0:
            idx = np.arange(c0, c1)
            values = 1.0 - (np.abs(idx - mid) / max(center_radius, 1))
            center[c0:c1] = np.maximum(center[c0:c1], values.astype(np.float32))
        for edge in (s, e):
            b0 = max(0, edge - boundary_radius)
            b1 = min(n, edge + boundary_radius + 1)
            if b1 > b0:
                idx = np.arange(b0, b1)
                values = 1.0 - (np.abs(idx - edge) / max(boundary_radius, 1))
                boundary[b0:b1] = np.maximum(boundary[b0:b1], values.astype(np.float32))
    return center.astype(np.float32), boundary.astype(np.float32)


def segmentation_input_channels(cfg: Config) -> int:
    n_channels = 1
    if cfg.segmentation_use_sigma_channel:
        n_channels += 1
    if cfg.segmentation_use_slow_sigma_channel:
        n_channels += 1
    if cfg.segmentation_use_teo_channel:
        n_channels += 1
    if cfg.segmentation_use_envelope_channel:
        n_channels += 1
    if cfg.segmentation_use_secondary_eeg_channel:
        n_channels += 1
        n_channels += int(cfg.segmentation_use_sigma_channel)
        n_channels += int(cfg.segmentation_use_slow_sigma_channel)
        n_channels += int(cfg.segmentation_use_envelope_channel)
    return n_channels


def segmentation_channel_names(cfg: Config) -> List[str]:
    names = ["raw"]
    if cfg.segmentation_use_sigma_channel:
        names.append("sigma")
    if cfg.segmentation_use_slow_sigma_channel:
        names.append("slow_sigma")
    if cfg.segmentation_use_teo_channel:
        names.append("teager_energy")
    if cfg.segmentation_use_envelope_channel:
        names.append("envelope")
    if cfg.segmentation_use_secondary_eeg_channel:
        names.append("secondary_raw")
        if cfg.segmentation_use_sigma_channel:
            names.append("secondary_sigma")
        if cfg.segmentation_use_slow_sigma_channel:
            names.append("secondary_slow_sigma")
        if cfg.segmentation_use_envelope_channel:
            names.append("secondary_envelope")
    return names


def build_segmentation_input_window(
    rec: Dict[str, Any],
    start: int,
    stop: int,
    window_len: int,
    cfg: Config,
) -> np.ndarray:
    channels = [rec["segmentation_raw"][start:stop]]
    if cfg.segmentation_use_sigma_channel:
        channels.append(rec["segmentation_sigma"][start:stop])
    if cfg.segmentation_use_slow_sigma_channel:
        channels.append(rec["segmentation_slow_sigma"][start:stop])
    if cfg.segmentation_use_teo_channel:
        channels.append(rec["segmentation_teo"][start:stop])
    if cfg.segmentation_use_envelope_channel:
        channels.append(rec["segmentation_envelope"][start:stop])
    if cfg.segmentation_use_secondary_eeg_channel:
        channels.append(rec["segmentation_secondary_raw"][start:stop])
        if cfg.segmentation_use_sigma_channel:
            channels.append(rec["segmentation_secondary_sigma"][start:stop])
        if cfg.segmentation_use_slow_sigma_channel:
            channels.append(rec["segmentation_secondary_slow_sigma"][start:stop])
        if cfg.segmentation_use_envelope_channel:
            channels.append(rec["segmentation_secondary_envelope"][start:stop])
    padded: List[np.ndarray] = []
    for ch in channels:
        ch = np.asarray(ch, dtype=np.float32)
        if len(ch) < window_len:
            ch = np.pad(ch, (0, window_len - len(ch)), mode="edge" if len(ch) else "constant")
        padded.append(ch[:window_len])
    return np.stack(padded, axis=0).astype(np.float32)


def build_segmentation_windows(subject_ids: Sequence[str], records: Dict[str, Any], cfg: Config) -> List[Tuple[str, int, int]]:
    window_len = max(8, int(round(cfg.segmentation_window_sec * cfg.resample_hz)))
    hop = max(1, int(round(window_len * (1.0 - cfg.segmentation_overlap))))
    windows: List[Tuple[str, int, int]] = []
    for sid in subject_ids:
        rec = records[sid]
        for seg_start, seg_stop in rec.get("stage_segment_bounds") or [(0, len(rec["truth_binary_stage"]))]:
            seg_len = seg_stop - seg_start
            if seg_len <= 0:
                continue
            if seg_len <= window_len:
                windows.append((sid, seg_start, seg_stop))
                continue
            starts = list(range(seg_start, seg_stop - window_len + 1, hop))
            last_start = seg_stop - window_len
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            windows.extend((sid, int(s), int(s + window_len)) for s in starts)
    return windows


def build_segmentation_window_sampler(
    windows: List[Tuple[str, int, int]],
    records: Dict[str, Any],
    cfg: Config,
    extra_weights: Optional[np.ndarray] = None,
) -> Optional[WeightedRandomSampler]:
    if not windows:
        return None
    use_positive_weighting = bool(cfg.segmentation_positive_window_sampling_enabled)
    use_subject_balancing = bool(cfg.segmentation_subject_balanced_sampling_enabled)
    use_extra_weights = extra_weights is not None and len(extra_weights) == len(windows)
    if not use_positive_weighting and not use_subject_balancing and not use_extra_weights:
        return None
    weights = np.ones(len(windows), dtype=np.float64)
    if use_subject_balancing:
        subject_counts: Dict[str, int] = {}
        for sid, _, _ in windows:
            subject_counts[sid] = subject_counts.get(sid, 0) + 1
        mean_count = len(windows) / max(len(subject_counts), 1)
        for idx, (sid, _, _) in enumerate(windows):
            weights[idx] *= mean_count / max(subject_counts.get(sid, 1), 1)
    if use_positive_weighting:
        positive_weight = max(1.0, float(cfg.segmentation_positive_window_weight))
        for idx, (sid, start, stop) in enumerate(windows):
            y = records[sid]["truth_binary_stage"][start:stop]
            if np.any(y > 0):
                weights[idx] *= positive_weight
    if use_extra_weights:
        weights *= np.asarray(extra_weights, dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def make_segmentation_loader(
    windows: List[Tuple[str, int, int]],
    records: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    training: bool,
    shuffle: bool = False,
    extra_weights: Optional[np.ndarray] = None,
) -> DataLoader:
    num_workers = max(0, int(cfg.num_workers))
    safe_single_process = bool(cfg.hard_negative_mining_enabled or cfg.run_explainability)
    if safe_single_process:
        num_workers = 0
    sampler = build_segmentation_window_sampler(windows, records, cfg, extra_weights=extra_weights) if training else None
    kwargs: Dict[str, Any] = {
        "batch_size": cfg.segmentation_batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "num_workers": num_workers,
        "pin_memory": bool(cfg.pin_memory and device.type == "cuda" and not safe_single_process),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return DataLoader(SegmentationWindowDataset(windows, records, cfg, training=training), **kwargs)


def shutdown_dataloader_workers(loader: Optional[DataLoader]) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    gc.collect()


def warmstart_transformer_from_cnn_checkpoint(
    model: nn.Module,
    cfg: Config,
    fold_idx: int,
    seed_offset: int,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    if not (is_transformer_segmentation_mode(cfg) or is_transformer_only_segmentation_mode(cfg)):
        return
    if not cfg.transformer_warmstart_enabled or not cfg.transformer_warmstart_cnn_dir:
        return
    root = Path(cfg.transformer_warmstart_cnn_dir)
    candidates = [
        root / f"fold_{fold_idx:02d}" / f"best_segmentation_model_seed{seed_offset}.pt",
        root / f"fold_{fold_idx:02d}" / "best_segmentation_model_seed0.pt",
        root / f"fold_{fold_idx:02d}" / "best_segmentation_model.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        logger.warning(
            "Transformer warm-start requested but no compatible checkpoint found for fold %d seed %s under %s.",
            fold_idx,
            seed_offset,
            root,
        )
        return
    state = torch.load(ckpt_path, map_location=device)
    source = state.get("model_state_dict", state)
    target = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {
        k: v
        for k, v in source.items()
        if k in target and tuple(v.shape) == tuple(target[k].shape)
    }
    # A multi-channel transformer changes only patch_embed.in_features. Preserve
    # the learned primary-channel waveform and patch-statistic columns while
    # leaving newly added channel columns at their fresh initialization.
    patch_key = "patch_embed.weight"
    if (
        is_transformer_only_segmentation_mode(cfg)
        and patch_key in source
        and patch_key in target
        and source[patch_key].ndim == 2
        and target[patch_key].ndim == 2
        and source[patch_key].shape[0] == target[patch_key].shape[0]
        and source[patch_key].shape[1] < target[patch_key].shape[1]
    ):
        patch_size = max(1, int(cfg.segmentation_transformer_only_patch_size))
        feature_width = patch_size + (4 if cfg.segmentation_transformer_only_patch_stats else 0)
        source_channels = int(source[patch_key].shape[1] // feature_width)
        target_channels = int(target[patch_key].shape[1] // feature_width)
        if (
            source_channels > 0
            and target_channels > source_channels
            and source_channels * feature_width == source[patch_key].shape[1]
            and target_channels * feature_width == target[patch_key].shape[1]
        ):
            expanded = target[patch_key].detach().clone()
            source_weight = source[patch_key].to(dtype=expanded.dtype, device=expanded.device)
            source_flat = source_channels * patch_size
            target_flat = target_channels * patch_size
            expanded[:, :source_flat] = source_weight[:, :source_flat]
            if cfg.segmentation_transformer_only_patch_stats:
                for stat_idx in range(4):
                    source_start = source_flat + stat_idx * source_channels
                    target_start = target_flat + stat_idx * target_channels
                    expanded[:, target_start:target_start + source_channels] = source_weight[
                        :, source_start:source_start + source_channels
                    ]
            compatible[patch_key] = expanded
    skipped = len(source) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    logger.info(
        "Transformer warm-start from compatible checkpoint: %s | loaded=%d | skipped=%d | missing_after_partial_load=%d | unexpected=%d",
        ckpt_path,
        len(compatible),
        skipped,
        len(missing),
        len(unexpected),
    )


class UNetConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate1D(nn.Module):
    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.skip_proj = nn.Conv1d(skip_ch, inter_ch, kernel_size=1)
        self.gate_proj = nn.Conv1d(gate_ch, inter_ch, kernel_size=1)
        self.psi = nn.Conv1d(inter_ch, 1, kernel_size=1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] != skip.shape[-1]:
            gate = F.interpolate(gate, size=skip.shape[-1], mode="linear", align_corners=False)
        attn = torch.sigmoid(self.psi(F.relu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * attn


class TemporalBottleneck1D(nn.Module):
    def __init__(self, channels: int, layers: int, kernel_size: int, dropout: float):
        super().__init__()
        blocks = []
        pad_base = kernel_size // 2
        for i in range(max(1, int(layers))):
            dilation = 2 ** i
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=pad_base * dilation, dilation=dilation),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel_size, padding=pad_base * dilation, dilation=dilation),
                    nn.BatchNorm1d(channels),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = F.relu(out + block(out), inplace=True)
        return out


class TransformerBottleneck1D(nn.Module):
    """Transformer encoder over the compressed U-Net bottleneck sequence."""

    def __init__(self, channels: int, layers: int, heads: int, ff_multiplier: int, dropout: float, max_len: int = 8192):
        super().__init__()
        heads = max(1, int(heads))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        ff_dim = int(channels * max(1, int(ff_multiplier)))
        self.positional = nn.Parameter(torch.zeros(1, max_len, channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time] -> Transformer expects [batch, time, channels]
        seq = x.transpose(1, 2)
        if seq.shape[1] > self.positional.shape[1]:
            pos = F.interpolate(
                self.positional.transpose(1, 2),
                size=seq.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.positional[:, : seq.shape[1], :]
        seq = self.encoder(seq + pos)
        seq = self.norm(seq)
        return seq.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Depthwise temporal convolution over patch tokens for local spindle morphology."""

    def __init__(self, embed_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(embed_dim)
        self.pointwise_in = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
        )
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.pointwise_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x).transpose(1, 2)
        out = F.glu(self.pointwise_in(out), dim=1)
        out = self.depthwise(out)
        out = F.silu(self.batch_norm(out))
        out = self.pointwise_out(out).transpose(1, 2)
        return self.dropout(out)


class ConformerEncoderBlock(nn.Module):
    """Conformer-style token block: FFN, attention, depthwise conv, FFN."""

    def __init__(self, embed_dim: int, heads: int, ff_multiplier: int, dropout: float, conv_kernel: int):
        super().__init__()
        hidden_dim = embed_dim * max(1, int(ff_multiplier))
        self.ff1 = ConformerFeedForward(embed_dim, hidden_dim, dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(embed_dim, conv_kernel, dropout)
        self.ff2 = ConformerFeedForward(embed_dim, hidden_dim, dropout)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class LocalTransformerEncoderBlock(nn.Module):
    """Pre-normalized sparse Transformer with local and optional global keys."""

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
        window_size: int,
        global_stride: int = 0,
    ):
        super().__init__()
        self.window_size = max(1, int(window_size))
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.global_stride = max(0, int(global_stride))
        hidden_dim = embed_dim * max(1, int(ff_multiplier))
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_count = x.shape[1]
        positions = torch.arange(token_count, device=x.device)
        radius = self.window_size // 2
        allowed = (positions[:, None] - positions[None, :]).abs() <= radius
        if self.global_stride > 0:
            # Sparse landmark keys extend the receptive field across the full
            # 20-s window without turning the block into dense global attention.
            global_keys = (positions % self.global_stride) == 0
            allowed = allowed | global_keys.unsqueeze(0)
        attention_mask = ~allowed
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        return x + self.ff(self.ff_norm(x))


class TransformerOnlySegmentation1D(nn.Module):
    """Patch-token global Transformer, local Transformer, or Conformer without a CNN backbone."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        layers: int,
        heads: int,
        ff_multiplier: int,
        dropout: float,
        patch_size: int,
        patch_stride: int,
        max_patches: int,
        multiscale_enabled: bool = False,
        long_patch_size: int = 128,
        block_type: str = "transformer",
        local_attention_window: int = 9,
        local_global_stride: int = 0,
        conv_kernel: int = 9,
        local_mixing: bool = False,
        local_mixing_dropout: float = 0.10,
        patch_stats: bool = False,
        positional_init: str = "zeros",
        use_auxiliary_heads: bool = False,
        use_expert_heads: bool = False,
        secondary_start_channel: Optional[int] = None,
        secondary_gate_initial: float = 0.10,
        supervised_contrastive_enabled: bool = False,
        supervised_contrastive_projection_dim: int = 64,
        spectral_tokens_enabled: bool = False,
        spectral_patch_size: int = 256,
        spectral_low_hz: float = 8.0,
        spectral_high_hz: float = 20.0,
        spectral_input_channels: int = 1,
        spectral_gate_initial: float = 0.10,
        sample_rate: float = 256.0,
        event_geometry_enabled: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = max(1, int(patch_size))
        self.patch_stride = max(1, int(patch_stride))
        self.long_patch_size = max(self.patch_size, int(long_patch_size))
        self.multiscale_enabled = bool(multiscale_enabled and self.long_patch_size > self.patch_size)
        self.local_mixing = bool(local_mixing)
        self.patch_stats = bool(patch_stats)
        self.use_auxiliary_heads = bool(use_auxiliary_heads)
        self.use_expert_heads = bool(use_expert_heads)
        self.supervised_contrastive_enabled = bool(supervised_contrastive_enabled)
        self.event_geometry_enabled = bool(event_geometry_enabled)
        self.spectral_tokens_enabled = bool(spectral_tokens_enabled)
        self.spectral_patch_size = max(self.patch_size, int(spectral_patch_size))
        self.spectral_input_channels = min(self.in_channels, max(1, int(spectral_input_channels)))
        self.secondary_start_channel = (
            int(secondary_start_channel)
            if secondary_start_channel is not None and 0 < int(secondary_start_channel) < self.in_channels
            else None
        )
        if self.secondary_start_channel is not None:
            initial_gate = min(1.0 - 1e-4, max(1e-4, float(secondary_gate_initial)))
            initial_logit = math.log(initial_gate / (1.0 - initial_gate))
            self.secondary_gate_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        else:
            self.register_parameter("secondary_gate_logit", None)
        embed_dim = max(16, int(embed_dim))
        self.embed_dim = embed_dim
        heads = max(1, int(heads))
        while embed_dim % heads != 0 and heads > 1:
            heads -= 1
        self.block_type = str(block_type).lower().strip()
        if self.block_type not in {"transformer", "local_transformer", "conformer"}:
            self.block_type = "transformer"
        patch_feature_dim = self.in_channels * self.patch_size
        if self.patch_stats:
            patch_feature_dim += self.in_channels * 4
        self.patch_embed = nn.Linear(patch_feature_dim, embed_dim)
        if self.spectral_tokens_enabled:
            spectral_frequencies = torch.fft.rfftfreq(
                self.spectral_patch_size,
                d=1.0 / float(sample_rate),
            )
            spectral_indices = torch.nonzero(
                (spectral_frequencies >= float(spectral_low_hz))
                & (spectral_frequencies <= float(spectral_high_hz)),
                as_tuple=False,
            ).flatten()
            if spectral_indices.numel() == 0:
                raise ValueError("The requested spectral token band contains no FFT bins.")
            spectral_feature_dim = self.spectral_input_channels * int(spectral_indices.numel())
            self.register_buffer(
                "spectral_window",
                torch.hann_window(self.spectral_patch_size, periodic=True),
                persistent=False,
            )
            self.register_buffer("spectral_indices", spectral_indices, persistent=False)
            self.spectral_embed = nn.Sequential(
                nn.LayerNorm(spectral_feature_dim),
                nn.Linear(spectral_feature_dim, embed_dim),
                nn.GELU(),
            )
            self.spectral_fusion = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            initial_gate = min(1.0 - 1e-4, max(1e-4, float(spectral_gate_initial)))
            self.spectral_gate_logit = nn.Parameter(
                torch.tensor(math.log(initial_gate / (1.0 - initial_gate)), dtype=torch.float32)
            )
        else:
            self.register_buffer("spectral_window", None, persistent=False)
            self.register_buffer("spectral_indices", None, persistent=False)
            self.spectral_embed = None
            self.spectral_fusion = None
            self.register_parameter("spectral_gate_logit", None)
        if self.multiscale_enabled:
            long_patch_feature_dim = self.in_channels * self.long_patch_size
            if self.patch_stats:
                long_patch_feature_dim += self.in_channels * 4
            self.long_patch_embed = nn.Linear(long_patch_feature_dim, embed_dim)
            self.multiscale_fusion = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.long_patch_embed = None
            self.multiscale_fusion = None
        if self.local_mixing:
            self.local_mix = nn.Sequential(
                nn.LayerNorm(embed_dim * 3),
                nn.Linear(embed_dim * 3, embed_dim),
                nn.GELU(),
                nn.Dropout(float(local_mixing_dropout)),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.local_mix = nn.Identity()
        positional_length = max(8, int(max_patches))
        positional_init = str(positional_init).lower().strip()
        if positional_init == "sinusoidal":
            positions = torch.arange(positional_length, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / max(1, embed_dim))
            )
            positional = torch.zeros(1, positional_length, embed_dim, dtype=torch.float32)
            positional[0, :, 0::2] = torch.sin(positions * frequencies)
            if embed_dim > 1:
                positional[0, :, 1::2] = torch.cos(positions * frequencies[: positional[0, :, 1::2].shape[-1]])
        else:
            positional = torch.zeros(1, positional_length, embed_dim, dtype=torch.float32)
        self.positional = nn.Parameter(positional)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.block_type == "conformer":
            self.encoder = nn.ModuleList([
                ConformerEncoderBlock(
                    embed_dim=embed_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=float(dropout),
                    conv_kernel=conv_kernel,
                )
                for _ in range(max(1, int(layers)))
            ])
        elif self.block_type == "local_transformer":
            self.encoder = nn.ModuleList([
                LocalTransformerEncoderBlock(
                    embed_dim=embed_dim,
                    heads=heads,
                    ff_multiplier=ff_multiplier,
                    dropout=float(dropout),
                    window_size=local_attention_window,
                    global_stride=local_global_stride,
                )
                for _ in range(max(1, int(layers)))
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=embed_dim * max(1, int(ff_multiplier)),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.norm = nn.LayerNorm(embed_dim)
        self.out_head = nn.Linear(embed_dim, self.patch_size)
        if self.supervised_contrastive_enabled:
            projection_dim = max(8, int(supervised_contrastive_projection_dim))
            self.contrastive_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, projection_dim),
            )
        else:
            self.contrastive_projection = None
        if self.event_geometry_enabled:
            self.event_objectness_head = nn.Linear(embed_dim, 1)
            self.event_center_offset_head = nn.Linear(embed_dim, 1)
            self.event_log_duration_head = nn.Linear(embed_dim, 1)
        if self.use_auxiliary_heads:
            self.center_head = nn.Linear(embed_dim, self.patch_size)
            self.boundary_head = nn.Linear(embed_dim, self.patch_size)
        if self.use_expert_heads:
            self.expert1_head = nn.Linear(embed_dim, self.patch_size)
            self.expert2_head = nn.Linear(embed_dim, self.patch_size)

    def _encode(
        self,
        x: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        batch, channels, length = x.shape
        if self.secondary_gate_logit is not None and self.secondary_start_channel is not None:
            gate = torch.sigmoid(self.secondary_gate_logit).to(dtype=x.dtype)
            x = torch.cat(
                [x[:, : self.secondary_start_channel, :], x[:, self.secondary_start_channel :, :] * gate],
                dim=1,
            )
        if length <= self.patch_size:
            n_patches = 1
        else:
            n_patches = int(math.ceil((length - self.patch_size) / self.patch_stride)) + 1
        padded_len = (n_patches - 1) * self.patch_stride + self.patch_size
        pad = max(0, padded_len - length)
        if pad > 0:
            x = F.pad(x, (0, pad))
        padded_len = x.shape[-1]
        def build_patch_features(patch_tensor: torch.Tensor) -> torch.Tensor:
            patch_tensor = patch_tensor.permute(0, 2, 1, 3)
            flat = patch_tensor.reshape(batch, n_patches, channels * patch_tensor.shape[-1])
            if not self.patch_stats:
                return flat
            patch_mean = patch_tensor.mean(dim=-1)
            patch_std = patch_tensor.std(dim=-1, unbiased=False)
            patch_abs_peak = patch_tensor.abs().amax(dim=-1)
            patch_energy = torch.sqrt((patch_tensor * patch_tensor).mean(dim=-1).clamp_min(1e-8))
            return torch.cat([flat, patch_mean, patch_std, patch_abs_peak, patch_energy], dim=-1)

        patches = x.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        patch_features = build_patch_features(patches)
        tokens = self.patch_embed(patch_features)
        if (
            self.spectral_tokens_enabled
            and self.spectral_embed is not None
            and self.spectral_fusion is not None
            and self.spectral_window is not None
            and self.spectral_indices is not None
        ):
            extra = self.spectral_patch_size - self.patch_size
            spectral_x = F.pad(
                x[:, : self.spectral_input_channels, :],
                (extra // 2, extra - extra // 2),
            )
            spectral_patches = spectral_x.unfold(
                dimension=2,
                size=self.spectral_patch_size,
                step=self.patch_stride,
            ).permute(0, 2, 1, 3)
            window = self.spectral_window.to(device=spectral_patches.device, dtype=torch.float32)
            spectral_fft = torch.fft.rfft(spectral_patches.float() * window, dim=-1).abs()
            spectral_features = torch.log1p(spectral_fft[..., self.spectral_indices])
            spectral_features = spectral_features.reshape(batch, n_patches, -1)
            spectral_tokens = self.spectral_embed(spectral_features)
            spectral_residual = self.spectral_fusion(torch.cat([tokens, spectral_tokens], dim=-1))
            spectral_gate = torch.sigmoid(self.spectral_gate_logit).to(dtype=spectral_residual.dtype)
            tokens = tokens + spectral_gate * spectral_residual
        if self.multiscale_enabled and self.long_patch_embed is not None and self.multiscale_fusion is not None:
            extra = self.long_patch_size - self.patch_size
            long_x = F.pad(x, (extra // 2, extra - extra // 2))
            long_patches = long_x.unfold(
                dimension=2,
                size=self.long_patch_size,
                step=self.patch_stride,
            )
            long_features = build_patch_features(long_patches)
            long_tokens = self.long_patch_embed(long_features)
            tokens = tokens + self.multiscale_fusion(torch.cat([tokens, long_tokens], dim=-1))
        if self.local_mixing:
            prev_tokens = F.pad(tokens[:, :-1, :], (0, 0, 1, 0))
            next_tokens = F.pad(tokens[:, 1:, :], (0, 0, 0, 1))
            tokens = tokens + self.local_mix(torch.cat([prev_tokens, tokens, next_tokens], dim=-1))
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"Token mask shape {tuple(token_mask.shape)} does not match "
                    f"token sequence shape {tuple(tokens.shape[:2])}."
                )
            mask_token = self.mask_token.to(dtype=tokens.dtype).expand(tokens.shape[0], tokens.shape[1], -1)
            tokens = torch.where(token_mask.unsqueeze(-1), mask_token, tokens)
        if n_patches > self.positional.shape[1]:
            pos = F.interpolate(
                self.positional.transpose(1, 2),
                size=n_patches,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.positional[:, :n_patches, :]
        tokens = tokens + pos
        if self.block_type in {"conformer", "local_transformer"}:
            for block in self.encoder:
                tokens = block(tokens)
        else:
            tokens = self.encoder(tokens)
        return self.norm(tokens), length

    def _decode_head(self, tokens: torch.Tensor, head: nn.Linear, length: int) -> torch.Tensor:
        patch_logits = head(tokens)
        batch, n_patches, _ = patch_logits.shape
        padded_len = (n_patches - 1) * self.patch_stride + self.patch_size
        out = patch_logits.new_zeros((batch, 1, padded_len))
        counts = patch_logits.new_zeros((1, 1, padded_len))
        for idx in range(n_patches):
            s = idx * self.patch_stride
            e = s + self.patch_size
            out[:, :, s:e] = out[:, :, s:e] + patch_logits[:, idx, :].unsqueeze(1)
            counts[:, :, s:e] = counts[:, :, s:e] + 1.0
        out = out / counts.clamp_min(1.0)
        return out[:, :, :length]

    def forward(self, x: torch.Tensor) -> Any:
        tokens, length = self._encode(x)
        logits = self._decode_head(tokens, self.out_head, length)
        return_contrastive = self.training and self.contrastive_projection is not None
        if (
            not self.use_auxiliary_heads
            and not self.use_expert_heads
            and not return_contrastive
            and not self.event_geometry_enabled
        ):
            return logits
        output = {"logits": logits}
        if return_contrastive:
            output["contrastive_tokens"] = self.contrastive_projection(tokens)
        if self.event_geometry_enabled:
            output["event_objectness_logits"] = self.event_objectness_head(tokens).squeeze(-1)
            output["event_center_offsets"] = self.event_center_offset_head(tokens).squeeze(-1)
            output["event_log_durations"] = self.event_log_duration_head(tokens).squeeze(-1)
        if self.use_auxiliary_heads:
            output["center_logits"] = self._decode_head(tokens, self.center_head, length)
            output["boundary_logits"] = self._decode_head(tokens, self.boundary_head, length)
        if self.use_expert_heads:
            output["expert1_logits"] = self._decode_head(tokens, self.expert1_head, length)
            output["expert2_logits"] = self._decode_head(tokens, self.expert2_head, length)
        return output


class AttentionUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        depth: int,
        kernel_size: int,
        dropout: float,
        use_attention: bool,
        use_temporal_bottleneck: bool = False,
        use_transformer_bottleneck: bool = False,
        temporal_bottleneck_layers: int = 3,
        temporal_bottleneck_dropout: float = 0.10,
        transformer_heads: int = 4,
        transformer_ff_multiplier: int = 4,
        use_auxiliary_heads: bool = False,
    ):
        super().__init__()
        depth = max(2, int(depth))
        channels = [int(base_channels) * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.down_blocks.append(UNetConvBlock1D(prev, ch, kernel_size, dropout))
            self.pools.append(nn.MaxPool1d(2))
            prev = ch
        self.bottleneck = UNetConvBlock1D(channels[-1], channels[-1] * 2, kernel_size, dropout)
        bottleneck_channels = channels[-1] * 2
        if use_transformer_bottleneck:
            self.temporal_bottleneck = TransformerBottleneck1D(
                bottleneck_channels,
                temporal_bottleneck_layers,
                transformer_heads,
                transformer_ff_multiplier,
                temporal_bottleneck_dropout,
            )
        elif use_temporal_bottleneck:
            self.temporal_bottleneck = TemporalBottleneck1D(
                bottleneck_channels,
                temporal_bottleneck_layers,
                kernel_size,
                temporal_bottleneck_dropout,
            )
        else:
            self.temporal_bottleneck = nn.Identity()
        self.up_convs = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        gate_ch = bottleneck_channels
        for skip_ch in reversed(channels):
            self.up_convs.append(nn.ConvTranspose1d(gate_ch, skip_ch, kernel_size=2, stride=2))
            self.attn.append(AttentionGate1D(skip_ch, skip_ch, max(1, skip_ch // 2)) if use_attention else nn.Identity())
            self.up_blocks.append(UNetConvBlock1D(skip_ch * 2, skip_ch, kernel_size, dropout))
            gate_ch = skip_ch
        self.use_auxiliary_heads = bool(use_auxiliary_heads)
        self.out_conv = nn.Conv1d(channels[0], 1, kernel_size=1)
        if self.use_auxiliary_heads:
            self.center_conv = nn.Conv1d(channels[0], 1, kernel_size=1)
            self.boundary_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Any:
        skips: List[torch.Tensor] = []
        out = x
        for block, pool in zip(self.down_blocks, self.pools):
            out = block(out)
            skips.append(out)
            out = pool(out)
        out = self.bottleneck(out)
        out = self.temporal_bottleneck(out)
        for up, attn, block, skip in zip(self.up_convs, self.attn, self.up_blocks, reversed(skips)):
            out = up(out)
            if out.shape[-1] != skip.shape[-1]:
                out = F.interpolate(out, size=skip.shape[-1], mode="linear", align_corners=False)
            skip_in = attn(skip, out) if isinstance(attn, AttentionGate1D) else skip
            out = block(torch.cat([skip_in, out], dim=1))
        logits = self.out_conv(out)
        if not self.use_auxiliary_heads:
            return logits
        return {
            "logits": logits,
            "center_logits": self.center_conv(out),
            "boundary_logits": self.boundary_conv(out),
        }


def primary_segmentation_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output


def build_segmentation_model(cfg: Config, device: torch.device) -> nn.Module:
    if is_transformer_only_segmentation_mode(cfg):
        in_channels = segmentation_input_channels(cfg)
        secondary_start_channel = (
            in_channels // 2
            if cfg.segmentation_use_secondary_eeg_channel and cfg.segmentation_secondary_gate_enabled
            else None
        )
        return TransformerOnlySegmentation1D(
            in_channels=in_channels,
            embed_dim=cfg.segmentation_transformer_only_embed_dim,
            layers=cfg.segmentation_transformer_only_layers,
            heads=cfg.segmentation_transformer_only_heads,
            ff_multiplier=cfg.segmentation_transformer_only_ff_multiplier,
            dropout=cfg.segmentation_transformer_only_dropout,
            patch_size=cfg.segmentation_transformer_only_patch_size,
            patch_stride=cfg.segmentation_transformer_only_patch_stride,
            max_patches=cfg.segmentation_transformer_only_max_patches,
            multiscale_enabled=cfg.segmentation_transformer_only_multiscale_enabled,
            long_patch_size=cfg.segmentation_transformer_only_long_patch_size,
            block_type=cfg.segmentation_transformer_only_block_type,
            local_attention_window=cfg.segmentation_transformer_only_local_attention_window,
            local_global_stride=cfg.segmentation_transformer_only_local_global_stride,
            conv_kernel=cfg.segmentation_transformer_only_conv_kernel,
            local_mixing=cfg.segmentation_transformer_only_local_mixing,
            local_mixing_dropout=cfg.segmentation_transformer_only_local_mixing_dropout,
            patch_stats=cfg.segmentation_transformer_only_patch_stats,
            positional_init=cfg.segmentation_transformer_only_positional_init,
            use_auxiliary_heads=cfg.segmentation_auxiliary_heads_enabled,
            use_expert_heads=cfg.segmentation_multi_annotator_enabled,
            secondary_start_channel=secondary_start_channel,
            secondary_gate_initial=cfg.segmentation_secondary_gate_initial,
            supervised_contrastive_enabled=cfg.segmentation_supervised_contrastive_enabled,
            supervised_contrastive_projection_dim=cfg.segmentation_supervised_contrastive_projection_dim,
            spectral_tokens_enabled=cfg.segmentation_transformer_only_spectral_tokens_enabled,
            spectral_patch_size=cfg.segmentation_transformer_only_spectral_patch_size,
            spectral_low_hz=cfg.segmentation_transformer_only_spectral_low_hz,
            spectral_high_hz=cfg.segmentation_transformer_only_spectral_high_hz,
            spectral_input_channels=cfg.segmentation_transformer_only_spectral_input_channels,
            spectral_gate_initial=cfg.segmentation_transformer_only_spectral_gate_initial,
            sample_rate=cfg.resample_hz,
            event_geometry_enabled=cfg.segmentation_event_geometry_enabled,
        ).to(device)
    return AttentionUNet1D(
        in_channels=segmentation_input_channels(cfg),
        base_channels=cfg.segmentation_base_channels,
        depth=cfg.segmentation_depth,
        kernel_size=cfg.model_kernel_size,
        dropout=cfg.model_dropout,
        use_attention=cfg.segmentation_use_attention_gates,
        use_temporal_bottleneck=cfg.segmentation_temporal_bottleneck_enabled,
        use_transformer_bottleneck=is_transformer_segmentation_mode(cfg),
        temporal_bottleneck_layers=cfg.segmentation_temporal_bottleneck_layers,
        temporal_bottleneck_dropout=cfg.segmentation_temporal_bottleneck_dropout,
        transformer_heads=cfg.segmentation_transformer_heads,
        transformer_ff_multiplier=cfg.segmentation_transformer_ff_multiplier,
        use_auxiliary_heads=cfg.segmentation_auxiliary_heads_enabled,
    ).to(device)


def normalized_patch_reconstruction_targets(
    x: torch.Tensor,
    patch_size: int,
    patch_stride: int,
) -> torch.Tensor:
    """Return per-channel standardized patch targets for masked reconstruction."""
    batch, channels, length = x.shape
    patch_size = max(1, int(patch_size))
    patch_stride = max(1, int(patch_stride))
    if length <= patch_size:
        n_patches = 1
    else:
        n_patches = int(math.ceil((length - patch_size) / patch_stride)) + 1
    padded_len = (n_patches - 1) * patch_stride + patch_size
    if padded_len > length:
        x = F.pad(x, (0, padded_len - length))
    patches = x.unfold(dimension=2, size=patch_size, step=patch_stride)
    patches = patches.permute(0, 2, 1, 3)
    patch_mean = patches.mean(dim=-1, keepdim=True)
    patch_std = patches.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    normalized = (patches - patch_mean) / patch_std
    return normalized.reshape(batch, n_patches, channels * patch_size)


def pretrain_transformer_masked_patches(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> List[float]:
    """Fold-safe reconstruction or contrastive pretraining using training subjects only."""
    if not isinstance(model, TransformerOnlySegmentation1D):
        raise TypeError("Transformer self-supervised pretraining requires TransformerOnlySegmentation1D.")
    epochs = max(1, int(cfg.transformer_ssl_pretraining_epochs))
    objective = str(cfg.transformer_ssl_objective).strip().lower()
    mask_ratio = float(cfg.transformer_ssl_mask_ratio)
    if objective == "contrastive":
        ssl_head: nn.Module = nn.Sequential(
            nn.Linear(model.embed_dim, model.embed_dim),
            nn.GELU(),
            nn.Linear(model.embed_dim, 128),
        ).to(device)
    else:
        ssl_head = nn.Linear(
            model.embed_dim,
            model.in_channels * model.patch_size,
        ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(ssl_head.parameters()),
        lr=float(cfg.transformer_ssl_learning_rate),
        weight_decay=float(cfg.segmentation_weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    history: List[float] = []
    logger.info(
        "Starting fold-safe Transformer pretraining | objective=%s | epochs=%d | mask_ratio=%.2f | lr=%.6f",
        objective,
        epochs,
        mask_ratio,
        cfg.transformer_ssl_learning_rate,
    )

    def contrastive_view(x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        scale = 1.0 + 0.05 * torch.randn((batch, channels, 1), device=x.device, dtype=x.dtype)
        view = x * scale
        view = view + 0.01 * torch.randn_like(view)
        keep = torch.rand_like(view) > 0.05
        return view * keep.to(dtype=view.dtype)

    def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        representations = torch.cat([z1, z2], dim=0)
        logits = representations @ representations.transpose(0, 1)
        logits = logits / float(cfg.transformer_ssl_temperature)
        diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        batch = z1.shape[0]
        targets = (torch.arange(2 * batch, device=logits.device) + batch) % (2 * batch)
        return F.cross_entropy(logits, targets)

    for epoch in range(1, epochs + 1):
        model.train()
        ssl_head.train()
        running_loss = 0.0
        seen = 0
        for X, _ in train_loader:
            X = X.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision and device.type == "cuda"):
                if objective == "contrastive":
                    if X.shape[0] < 2:
                        continue
                    encoded1, _ = model._encode(contrastive_view(X))
                    encoded2, _ = model._encode(contrastive_view(X))
                    projected1 = ssl_head(encoded1.mean(dim=1))
                    projected2 = ssl_head(encoded2.mean(dim=1))
                    loss = contrastive_loss(projected1, projected2)
                    count = int(X.shape[0])
                else:
                    targets = normalized_patch_reconstruction_targets(
                        X,
                        model.patch_size,
                        model.patch_stride,
                    )
                    mask = torch.rand(
                        targets.shape[:2],
                        device=device,
                        dtype=torch.float32,
                    ) < mask_ratio
                    empty_rows = ~mask.any(dim=1)
                    if bool(empty_rows.any()):
                        mask[empty_rows, 0] = True
                    encoded, _ = model._encode(X, token_mask=mask)
                    reconstructed = ssl_head(encoded)
                    loss = F.smooth_l1_loss(reconstructed[mask], targets[mask])
                    count = int(mask.sum().item())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Transformer SSL loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(ssl_head.parameters()),
                cfg.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * count
            seen += count
        epoch_loss = running_loss / max(seen, 1)
        history.append(epoch_loss)
        logger.info(
            "SSL epoch %d/%d | objective=%s | loss=%.6f",
            epoch,
            epochs,
            objective,
            epoch_loss,
        )
    return history


def spindle_token_supervised_contrastive_loss(
    token_embeddings: torch.Tensor,
    main_targets: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Balanced binary supervised contrastive loss over spindle/background tokens."""
    patch_size = max(1, int(cfg.segmentation_transformer_only_patch_size))
    patch_stride = max(1, int(cfg.segmentation_transformer_only_patch_stride))
    n_patches = int(token_embeddings.shape[1])
    padded_len = (n_patches - 1) * patch_stride + patch_size
    target = main_targets
    if target.shape[-1] < padded_len:
        target = F.pad(target, (0, padded_len - target.shape[-1]))
    elif target.shape[-1] > padded_len:
        target = target[..., :padded_len]
    patch_targets = target.unfold(2, patch_size, patch_stride).mean(dim=-1).squeeze(1)
    patch_targets = patch_targets[:, :n_patches]
    labels = patch_targets >= float(cfg.segmentation_supervised_contrastive_positive_threshold)
    embeddings = token_embeddings[:, : labels.shape[1], :].reshape(-1, token_embeddings.shape[-1])
    labels = labels.reshape(-1)
    positive_indices = torch.nonzero(labels, as_tuple=False).flatten()
    negative_indices = torch.nonzero(~labels, as_tuple=False).flatten()
    if positive_indices.numel() < 2 or negative_indices.numel() < 2:
        return embeddings.sum() * 0.0
    per_class = max(2, int(cfg.segmentation_supervised_contrastive_max_tokens) // 2)
    if positive_indices.numel() > per_class:
        positive_indices = positive_indices[
            torch.randperm(positive_indices.numel(), device=positive_indices.device)[:per_class]
        ]
    if negative_indices.numel() > per_class:
        negative_indices = negative_indices[
            torch.randperm(negative_indices.numel(), device=negative_indices.device)[:per_class]
        ]
    selected = torch.cat((positive_indices, negative_indices), dim=0)
    selected_embeddings = F.normalize(embeddings[selected].float(), dim=-1)
    selected_labels = labels[selected]
    temperature = max(1e-6, float(cfg.segmentation_supervised_contrastive_temperature))
    similarities = selected_embeddings @ selected_embeddings.transpose(0, 1) / temperature
    diagonal = torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    valid_pairs = ~diagonal
    positive_pairs = selected_labels[:, None].eq(selected_labels[None, :]) & valid_pairs
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    exp_sum = (torch.exp(similarities) * valid_pairs).sum(dim=1).clamp_min(1e-12)
    log_prob = similarities - torch.log(exp_sum[:, None])
    positive_count = positive_pairs.sum(dim=1)
    valid_anchors = positive_count > 0
    mean_log_prob_positive = (log_prob * positive_pairs).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_positive[valid_anchors].mean()


def event_geometry_auxiliary_loss(
    model_output: Dict[str, torch.Tensor],
    main_targets: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Anchor-free spindle objectness, center-offset, and duration loss."""
    objectness_logits = model_output["event_objectness_logits"]
    center_offsets = model_output["event_center_offsets"]
    log_durations = model_output["event_log_durations"]
    batch, n_tokens = objectness_logits.shape
    patch_size = max(1, int(cfg.segmentation_transformer_only_patch_size))
    patch_stride = max(1, int(cfg.segmentation_transformer_only_patch_stride))
    object_targets = torch.zeros_like(objectness_logits)
    center_targets = torch.zeros_like(center_offsets)
    duration_targets = torch.zeros_like(log_durations)
    token_centers = (
        torch.arange(n_tokens, device=objectness_logits.device, dtype=torch.float32) * patch_stride
        + 0.5 * (patch_size - 1)
    )
    for batch_idx in range(batch):
        binary = (main_targets[batch_idx, 0] >= 0.5).to(torch.int8)
        padded = F.pad(binary, (1, 1), value=0)
        transitions = padded[1:] - padded[:-1]
        starts = torch.nonzero(transitions == 1, as_tuple=False).flatten().float()
        stops = torch.nonzero(transitions == -1, as_tuple=False).flatten().float()
        if starts.numel() == 0:
            continue
        centers = 0.5 * (starts + stops - 1.0)
        durations = (stops - starts).clamp_min(1.0)
        token_indices = torch.round((centers - 0.5 * (patch_size - 1)) / patch_stride).long()
        valid = (token_indices >= 0) & (token_indices < n_tokens)
        token_indices = token_indices[valid]
        centers = centers[valid]
        durations = durations[valid]
        if token_indices.numel() == 0:
            continue
        object_targets[batch_idx, token_indices] = 1.0
        center_values = (
            (centers - token_centers[token_indices]) / float(patch_stride)
        ).clamp(-1.0, 1.0)
        duration_values = torch.log(
            durations / float(patch_size)
        ).clamp(-3.0, 4.0)
        # Autocast makes the prediction-derived destination tensors FP16 on
        # CUDA, whereas transition indices and geometry arithmetic are FP32.
        # Explicit casting keeps indexed assignment valid in mixed precision.
        center_targets[batch_idx, token_indices] = center_values.to(
            device=center_targets.device,
            dtype=center_targets.dtype,
        )
        duration_targets[batch_idx, token_indices] = duration_values.to(
            device=duration_targets.device,
            dtype=duration_targets.dtype,
        )
    positive_count = object_targets.sum()
    negative_count = object_targets.numel() - positive_count
    pos_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 20.0)
    object_bce = F.binary_cross_entropy_with_logits(
        objectness_logits,
        object_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    object_probs = torch.sigmoid(objectness_logits)
    object_pt = object_probs * object_targets + (1.0 - object_probs) * (1.0 - object_targets)
    object_loss = (
        (1.0 - object_pt).pow(max(0.0, float(cfg.segmentation_event_geometry_focal_gamma)))
        * object_bce
    ).mean()
    positive_mask = object_targets > 0.5
    if positive_mask.any():
        center_loss = F.smooth_l1_loss(
            torch.tanh(center_offsets[positive_mask]),
            center_targets[positive_mask],
        )
        duration_loss = F.smooth_l1_loss(
            log_durations[positive_mask],
            duration_targets[positive_mask],
        )
    else:
        center_loss = center_offsets.sum() * 0.0
        duration_loss = log_durations.sum() * 0.0
    return (
        cfg.segmentation_event_geometry_objectness_weight * object_loss
        + cfg.segmentation_event_geometry_center_weight * center_loss
        + cfg.segmentation_event_geometry_duration_weight * duration_loss
    )


def segmentation_loss(model_output: Any, targets: torch.Tensor, cfg: Config, pos_weight: torch.Tensor) -> torch.Tensor:
    logits = primary_segmentation_logits(model_output)
    main_targets = targets[:, 0:1, :]
    bce = F.binary_cross_entropy_with_logits(logits, main_targets, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    dims = (1, 2)
    tp = (probs * main_targets).sum(dim=dims)
    fp = (probs * (1.0 - main_targets)).sum(dim=dims)
    fn = ((1.0 - probs) * main_targets).sum(dim=dims)
    pos = main_targets.sum(dim=dims)
    neg = (1.0 - main_targets).sum(dim=dims)
    tversky = (tp + 1e-6) / (tp + cfg.segmentation_tversky_alpha * fp + cfg.segmentation_tversky_beta * fn + 1e-6)
    tversky_loss = (1.0 - tversky).mean()
    fpfn_loss = ((fp / (neg + 1e-6)) + (fn / (pos + 1e-6))).mean()
    loss_type = cfg.segmentation_loss_type
    if loss_type == "fpfn":
        base_loss = cfg.segmentation_fpfn_weight * fpfn_loss
    elif loss_type == "bce_fpfn":
        base_loss = cfg.segmentation_bce_weight * bce + cfg.segmentation_fpfn_weight * fpfn_loss
    elif loss_type == "bce_tversky_fpfn":
        base_loss = (
            cfg.segmentation_bce_weight * bce
            + cfg.segmentation_tversky_weight * tversky_loss
            + cfg.segmentation_fpfn_weight * fpfn_loss
        )
    else:
        base_loss = cfg.segmentation_bce_weight * bce + cfg.segmentation_tversky_weight * tversky_loss
    if cfg.segmentation_edge_consistency_weight > 0.0 and logits.shape[-1] > 1:
        predicted_edges = (probs[..., 1:] - probs[..., :-1]).abs()
        target_edges = (main_targets[..., 1:] - main_targets[..., :-1]).abs()
        edge_weights = 1.0 + 4.0 * target_edges
        edge_loss = (edge_weights * (predicted_edges - target_edges).abs()).sum()
        edge_loss = edge_loss / edge_weights.sum().clamp_min(1.0)
        base_loss = base_loss + cfg.segmentation_edge_consistency_weight * edge_loss
    if (
        cfg.segmentation_auxiliary_heads_enabled
        and isinstance(model_output, dict)
        and targets.shape[1] >= 3
    ):
        aux_pos_weight = torch.as_tensor(
            max(1.0, float(cfg.segmentation_auxiliary_pos_weight)),
            dtype=logits.dtype,
            device=logits.device,
        )

        def auxiliary_focal_loss(aux_logits: torch.Tensor, aux_targets: torch.Tensor) -> torch.Tensor:
            bce_aux = F.binary_cross_entropy_with_logits(
                aux_logits,
                aux_targets,
                pos_weight=aux_pos_weight,
                reduction="none",
            )
            aux_probs = torch.sigmoid(aux_logits)
            pt = aux_probs * aux_targets + (1.0 - aux_probs) * (1.0 - aux_targets)
            modulation = (1.0 - pt).pow(max(0.0, float(cfg.segmentation_auxiliary_focal_gamma)))
            return (modulation * bce_aux).mean()

        center_loss = auxiliary_focal_loss(model_output["center_logits"], targets[:, 1:2, :])
        boundary_loss = auxiliary_focal_loss(model_output["boundary_logits"], targets[:, 2:3, :])
        base_loss = (
            base_loss
            + cfg.segmentation_center_loss_weight * center_loss
            + cfg.segmentation_boundary_loss_weight * boundary_loss
        )
    if (
        cfg.segmentation_multi_annotator_enabled
        and isinstance(model_output, dict)
        and targets.shape[1] >= 5
        and "expert1_logits" in model_output
        and "expert2_logits" in model_output
    ):
        expert1_loss = F.binary_cross_entropy_with_logits(
            model_output["expert1_logits"],
            targets[:, 3:4, :],
            pos_weight=pos_weight,
        )
        expert2_loss = F.binary_cross_entropy_with_logits(
            model_output["expert2_logits"],
            targets[:, 4:5, :],
            pos_weight=pos_weight,
        )
        base_loss = base_loss + cfg.segmentation_multi_annotator_loss_weight * (
            0.5 * expert1_loss + 0.5 * expert2_loss
        )
    if (
        cfg.segmentation_supervised_contrastive_enabled
        and isinstance(model_output, dict)
        and "contrastive_tokens" in model_output
    ):
        contrastive_loss = spindle_token_supervised_contrastive_loss(
            model_output["contrastive_tokens"],
            main_targets,
            cfg,
        )
        base_loss = base_loss + cfg.segmentation_supervised_contrastive_weight * contrastive_loss
    if (
        cfg.segmentation_event_geometry_enabled
        and isinstance(model_output, dict)
        and all(
            name in model_output
            for name in ("event_objectness_logits", "event_center_offsets", "event_log_durations")
        )
    ):
        base_loss = base_loss + event_geometry_auxiliary_loss(model_output, main_targets, cfg)
    return base_loss


def compute_segmentation_pos_weight(subject_ids: Sequence[str], records: Dict[str, Any], cfg: Config, device: torch.device) -> torch.Tensor:
    pos = 0.0
    total = 0.0
    for sid in subject_ids:
        y = records[sid]["truth_binary_stage"]
        pos += float(np.sum(y > 0))
        total += float(len(y))
    neg = max(total - pos, 0.0)
    weight = cfg.positive_weight_max if pos < 1 else min(cfg.positive_weight_max, neg / (pos + 1e-8))
    return torch.tensor(weight, dtype=torch.float32, device=device)


def segmentation_hard_negative_window_weights(
    model: nn.Module,
    windows: List[Tuple[str, int, int]],
    records: Dict[str, Any],
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    weights = np.ones(len(windows), dtype=np.float64)
    if not cfg.segmentation_hard_negative_mining_enabled or not windows:
        return weights
    negative_indices: List[int] = []
    for idx, (sid, start, stop) in enumerate(windows):
        y = records[sid]["truth_binary_stage"][start:stop]
        if not np.any(y > 0):
            negative_indices.append(idx)
    if not negative_indices:
        return weights

    loader = make_segmentation_loader(windows, records, cfg, device, training=False, shuffle=False)
    scores: List[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device, non_blocking=True)
                logits = primary_segmentation_logits(model(X))
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                scores.append(probs.max(axis=(1, 2)))
    finally:
        shutdown_dataloader_workers(loader)
    if not scores:
        return weights
    window_scores = np.concatenate(scores).astype(np.float64)
    negative_scores = window_scores[np.asarray(negative_indices, dtype=np.int64)]
    top_k = max(1, int(round(float(cfg.hard_negative_fraction) * len(negative_indices))))
    if top_k >= len(negative_indices):
        selected = np.asarray(negative_indices, dtype=np.int64)
    else:
        hard_local = np.argpartition(negative_scores, -top_k)[-top_k:]
        selected = np.asarray(negative_indices, dtype=np.int64)[hard_local]
    weights[selected] = max(1.0, float(cfg.hard_negative_boost))
    return weights


def evaluate_segmentation_loader(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device, pos_weight: torch.Tensor) -> Dict[str, float]:
    model.eval()
    compute_window_event = cfg.segmentation_checkpoint_metric == "window_event_f1"
    losses: List[float] = []
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    probability_windows: List[np.ndarray] = []
    truth_windows: List[np.ndarray] = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model_output = model(X)
            logits = primary_segmentation_logits(model_output)
            loss = segmentation_loss(model_output, y, cfg, pos_weight)
            losses.append(float(loss.item()) * X.shape[0])
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0, :]
            batch_truth = y[:, 0, :].detach().cpu().numpy()
            probs_all.append(batch_probs.reshape(-1))
            y_all.append(batch_truth.reshape(-1))
            if compute_window_event:
                probability_windows.extend(batch_probs)
                truth_windows.extend(batch_truth)
    out = {"loss": float(sum(losses) / max(len(loader.dataset), 1))}
    if y_all:
        y_true = np.concatenate(y_all).astype(np.uint8)
        probs = np.concatenate(probs_all)
        pred = (probs >= 0.5).astype(np.uint8)
        out["sample_f1"] = float(sk_metrics.f1_score(y_true, pred, zero_division=0))
        if len(np.unique(y_true)) > 1:
            out["sample_pr_auc"] = float(sk_metrics.average_precision_score(y_true, probs))
        else:
            out["sample_pr_auc"] = float("nan")
        if compute_window_event:
            best_event_f1 = 0.0
            best_event_threshold = float(cfg.segmentation_checkpoint_threshold_grid[0])
            min_sec = float(min(cfg.segmentation_min_sec_grid))
            merge_gap = float(min(cfg.segmentation_merge_gap_grid))
            for threshold in cfg.segmentation_checkpoint_threshold_grid:
                tp = fp = fn = 0
                for truth_window, probability_window in zip(truth_windows, probability_windows):
                    true_events = binary_to_events((truth_window >= 0.5).astype(np.uint8), cfg.resample_hz)
                    pred_events, _ = decode_segmentation_events(
                        probability_window,
                        cfg.resample_hz,
                        float(threshold),
                        min_sec,
                        merge_gap,
                    )
                    metrics = event_metrics(true_events, pred_events, cfg.event_iou_thresholds[0])
                    tp += int(metrics["tp"])
                    fp += int(metrics["fp"])
                    fn += int(metrics["fn"])
                event_f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                if event_f1 > best_event_f1:
                    best_event_f1 = float(event_f1)
                    best_event_threshold = float(threshold)
            out["window_event_f1"] = best_event_f1
            out["window_event_threshold"] = best_event_threshold
    return out


def train_segmentation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: Path,
    pos_weight: torch.Tensor,
    max_epochs: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer_cls = torch.optim.AdamW if cfg.segmentation_optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg.segmentation_learning_rate,
        weight_decay=cfg.segmentation_weight_decay,
    )
    scheduler_mode = "max" if cfg.segmentation_scheduler_metric == "checkpoint" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=cfg.scheduler_factor,
        patience=cfg.segmentation_scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_sample_f1": [],
        "val_sample_pr_auc": [],
        "val_window_event_f1": [],
    }
    best_loss = float("inf")
    best_monitor = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0
    epochs = int(max_epochs) if max_epochs is not None else int(cfg.segmentation_epochs)
    epochs = max(1, epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision and device.type == "cuda"):
                model_output = model(X)
                loss = segmentation_loss(model_output, y, cfg, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite segmentation loss encountered at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * X.shape[0]
            seen += X.shape[0]
        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_segmentation_loader(model, val_loader, cfg, device, pos_weight)
        val_loss = float(val_metrics["loss"])
        monitor = float(val_metrics.get(cfg.segmentation_checkpoint_metric, float("nan")))
        if not math.isfinite(monitor):
            monitor = float(val_metrics.get("sample_pr_auc", val_metrics.get("sample_f1", float("-inf"))))
        scheduler.step(monitor if cfg.segmentation_scheduler_metric == "checkpoint" else val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_sample_f1"].append(float(val_metrics.get("sample_f1", float("nan"))))
        history["val_sample_pr_auc"].append(float(val_metrics.get("sample_pr_auc", float("nan"))))
        history["val_window_event_f1"].append(float(val_metrics.get("window_event_f1", float("nan"))))
        event_monitor_text = (
            f" | val_window_event_f1={float(val_metrics['window_event_f1']):.4f}"
            if "window_event_f1" in val_metrics
            else ""
        )
        logger.info(
            "Seg epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_sample_f1=%.4f | val_sample_pr_auc=%.4f%s | monitor=%s | lr=%.6f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            float(val_metrics.get("sample_f1", float("nan"))),
            float(val_metrics.get("sample_pr_auc", float("nan"))),
            event_monitor_text,
            cfg.segmentation_checkpoint_metric,
            optimizer.param_groups[0]["lr"],
        )
        improved = (monitor > best_monitor + 1e-6) or (
            math.isfinite(monitor) and abs(monitor - best_monitor) <= 1e-6 and val_loss < best_loss - 1e-6
        )
        if improved:
            best_monitor = monitor
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            if cfg.save_checkpoints:
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, checkpoint_path)
        else:
            patience += 1
            if patience >= cfg.segmentation_early_stopping_patience:
                logger.info("Segmentation early stopping triggered at epoch %d.", epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    spectral_gate_logit = getattr(model, "spectral_gate_logit", None)
    if spectral_gate_logit is not None:
        best_spectral_gate = float(torch.sigmoid(spectral_gate_logit.detach()).cpu().item())
        history["best_spectral_gate"] = [best_spectral_gate]
        logger.info("Best model spectral residual gate = %.4f", best_spectral_gate)
    return model, history


def predict_segmentation_head_timelines(
    model: nn.Module,
    rec: Dict[str, Any],
    cfg: Config,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Predict main and auxiliary timelines in one sliding-window pass."""
    model.eval()
    n_samples = len(rec["truth_binary_stage"])
    window_len = max(8, int(round(cfg.segmentation_window_sec * rec["sfreq"])))
    hop = max(1, int(round(window_len * (1.0 - cfg.segmentation_overlap))))
    head_names = ["logits"]
    if cfg.segmentation_auxiliary_heads_enabled and (cfg.segmentation_structured_decoder_enabled or cfg.save_probabilities):
        head_names.extend(["center_logits", "boundary_logits"])
    prob_sums = {name: np.zeros(n_samples, dtype=np.float64) for name in head_names}
    weight_sum = np.zeros(n_samples, dtype=np.float64)
    with torch.no_grad():
        for seg_start, seg_stop in rec.get("stage_segment_bounds") or [(0, n_samples)]:
            seg_len = seg_stop - seg_start
            if seg_len <= 0:
                continue
            if seg_len <= window_len:
                starts = [seg_start]
            else:
                starts = list(range(seg_start, seg_stop - window_len + 1, hop))
                last_start = seg_stop - window_len
                if not starts or starts[-1] != last_start:
                    starts.append(last_start)
            for start in starts:
                stop = min(seg_stop, start + window_len)
                valid_len = stop - start
                x_np = build_segmentation_input_window(rec, start, stop, window_len, cfg)
                X = torch.from_numpy(np.ascontiguousarray(x_np[None, :, :], dtype=np.float32)).to(device)
                model_output = model(X)
                output_map = model_output if isinstance(model_output, dict) else {"logits": model_output}
                window_probs = {
                    name: torch.sigmoid(output_map[name]).detach().cpu().numpy()[0, 0, :valid_len]
                    for name in head_names
                    if name in output_map
                }
                if seg_len <= window_len:
                    keep_s, keep_e = 0, valid_len
                else:
                    keep_s = window_len // 4
                    keep_e = min(valid_len, (3 * window_len) // 4)
                    if start == seg_start:
                        keep_s = 0
                    if stop >= seg_stop:
                        keep_e = valid_len
                abs_s = start + keep_s
                abs_e = start + keep_e
                if abs_e > abs_s:
                    for name, probs in window_probs.items():
                        prob_sums[name][abs_s:abs_e] += probs[keep_s:keep_e]
                    weight_sum[abs_s:abs_e] += 1.0
    missing = weight_sum <= 0
    if np.any(missing):
        weight_sum[missing] = 1.0
    return {
        name: (values / weight_sum).astype(np.float32)
        for name, values in prob_sums.items()
    }


def predict_segmentation_timeline(model: nn.Module, rec: Dict[str, Any], cfg: Config, device: torch.device) -> np.ndarray:
    return predict_segmentation_head_timelines(model, rec, cfg, device)["logits"]


def normalized_segmentation_ensemble_weights(
    n_models: int,
    member_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if n_models < 1:
        return np.zeros(0, dtype=np.float64)
    if member_weights is None:
        return np.full(n_models, 1.0 / n_models, dtype=np.float64)
    weights = np.asarray(member_weights, dtype=np.float64)
    if len(weights) != n_models or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Segmentation ensemble weights must be finite, non-negative, and match model count.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("At least one segmentation ensemble weight must be positive.")
    return weights / total


def predict_segmentation_head_timelines_ensemble(
    models: Sequence[nn.Module],
    rec: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    calibrator: Optional[Dict[str, Any]] = None,
    member_weights: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    member_outputs = [predict_segmentation_head_timelines(model, rec, cfg, device) for model in models]
    weights = normalized_segmentation_ensemble_weights(len(member_outputs), member_weights)
    names = sorted({name for output in member_outputs for name in output})
    if not member_outputs:
        names = ["logits"]
    result: Dict[str, np.ndarray] = {}
    for name in names:
        available = [(idx, output[name]) for idx, output in enumerate(member_outputs) if name in output]
        timelines = [timeline for _, timeline in available]
        if timelines:
            head_weights = np.asarray([weights[idx] for idx, _ in available], dtype=np.float64)
            head_weights /= np.sum(head_weights)
            result[name] = np.average(
                np.stack(timelines, axis=0),
                axis=0,
                weights=head_weights,
            ).astype(np.float32)
        else:
            result[name] = np.zeros(len(rec["truth_binary_stage"]), dtype=np.float32)
    result["logits"] = apply_probability_calibrator(calibrator, result["logits"])
    return result


def predict_segmentation_timeline_ensemble(
    models: Sequence[nn.Module],
    rec: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    calibrator: Optional[Dict[str, Any]] = None,
    member_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    return predict_segmentation_head_timelines_ensemble(
        models,
        rec,
        cfg,
        device,
        calibrator,
        member_weights,
    )["logits"]


def tune_segmentation_ensemble_weights(
    models: Sequence[nn.Module],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    fold_dir: Optional[Path] = None,
) -> Tuple[float, ...]:
    equal_weights = tuple(float(x) for x in normalized_segmentation_ensemble_weights(len(models)))
    if not cfg.segmentation_tune_ensemble_weights_enabled or len(models) != 2 or not val_ids:
        return equal_weights
    member_timelines = {
        sid: [
            predict_segmentation_timeline(model, records[sid], cfg, device)
            for model in models
        ]
        for sid in val_ids
    }
    y_true = np.concatenate([
        records[sid]["truth_binary_stage"].astype(np.uint8)
        for sid in val_ids
    ])
    best_key = (float("-inf"), float("-inf"))
    best_weights = equal_weights
    tuning_rows: List[Dict[str, float]] = []
    for first_weight in cfg.segmentation_ensemble_first_weight_grid:
        weights = (float(first_weight), float(1.0 - first_weight))
        blended = {
            sid: weights[0] * member_timelines[sid][0] + weights[1] * member_timelines[sid][1]
            for sid in val_ids
        }
        probs = np.concatenate([blended[sid] for sid in val_ids])
        pr_auc = float(sk_metrics.average_precision_score(y_true, probs))
        best_proxy_f1 = 0.0
        best_proxy_threshold = float(cfg.segmentation_threshold_grid[0])
        for threshold in cfg.segmentation_threshold_grid:
            f1s = []
            for sid in val_ids:
                rec = records[sid]
                pred_events, _ = decode_segmentation_events(
                    blended[sid],
                    rec["sfreq"],
                    float(threshold),
                    float(cfg.segmentation_min_sec_grid[0]),
                    float(cfg.segmentation_merge_gap_grid[0]),
                    0.0,
                    0.0,
                    0.0,
                )
                f1s.append(event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])["f1"])
            mean_f1 = float(np.mean(f1s)) if f1s else 0.0
            proxy_key = (round(mean_f1, 8), float(threshold))
            if proxy_key > (round(best_proxy_f1, 8), best_proxy_threshold):
                best_proxy_f1 = mean_f1
                best_proxy_threshold = float(threshold)
        score = (
            best_proxy_f1
            if cfg.segmentation_ensemble_weight_selection_metric == "event_f1"
            else pr_auc
        )
        tuning_rows.append({
            "first_member_weight": weights[0],
            "second_member_weight": weights[1],
            "validation_sample_pr_auc": pr_auc,
            "validation_event_f1_proxy": best_proxy_f1,
            "event_f1_proxy_threshold": best_proxy_threshold,
            "selection_score": score,
        })
        # Prefer a balanced blend when validation scores are effectively tied.
        key = (round(score, 8), -abs(weights[0] - 0.5))
        if key > best_key:
            best_key = key
            best_weights = weights
    logger.info(
        "Selected segmentation ensemble weights on validation | weights=%s | metric=%s | score=%.4f",
        best_weights,
        cfg.segmentation_ensemble_weight_selection_metric,
        best_key[0],
    )
    if fold_dir is not None:
        pd.DataFrame(tuning_rows).to_csv(fold_dir / "segmentation_ensemble_weight_tuning.csv", index=False)
    return tuple(float(x) for x in best_weights)


def decode_segmentation_events(
    prob: np.ndarray,
    sfreq: float,
    threshold: float,
    min_sec: float,
    merge_gap_sec: float,
    boundary_gap: float = 0.0,
    boundary_max_extend_sec: float = 0.0,
    boundary_trim_sec: float = 0.0,
    event_min_mean_prob: float = 0.0,
    event_min_peak_prob: float = 0.0,
    max_sec: float = 0.0,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    binary = (prob >= threshold).astype(np.uint8)
    binary = close_gaps_binary(binary, int(round(merge_gap_sec * sfreq))).astype(np.uint8)
    min_len = int(round(min_sec * sfreq))
    boundary_thr = max(0.01, float(threshold) - max(0.0, float(boundary_gap)))
    max_extend = max(0, int(round(max(0.0, float(boundary_max_extend_sec)) * sfreq)))
    trim = max(0, int(round(max(0.0, float(boundary_trim_sec)) * sfreq)))
    max_len = max(0, int(round(max(0.0, float(max_sec)) * sfreq)))
    pred = np.zeros(len(binary), dtype=np.uint8)
    events: List[Tuple[float, float]] = []
    for s, e in binary_to_segments(binary):
        if max_extend > 0 and boundary_thr < threshold:
            left_limit = max(0, s - max_extend)
            while s > left_limit and prob[s - 1] >= boundary_thr:
                s -= 1
            right_limit = min(len(prob) - 1, e + max_extend)
            while e < right_limit and prob[e + 1] >= boundary_thr:
                e += 1
        if trim > 0 and e - s + 1 > 2 * trim:
            s += trim
            e -= trim
        if max_len > 0 and e - s + 1 > max_len:
            # Preserve the most spindle-like portion of an overlong event.
            # A cumulative sum makes this O(event length), even for full-night
            # timelines, and avoids shifting the crop toward either boundary.
            event_values = np.asarray(prob[s:e + 1], dtype=np.float64)
            cumulative = np.r_[0.0, np.cumsum(event_values)]
            window_scores = cumulative[max_len:] - cumulative[:-max_len]
            best_offset = int(np.argmax(window_scores))
            s = s + best_offset
            e = s + max_len - 1
        if e - s + 1 < min_len:
            continue
        event_prob = prob[s:e + 1]
        if event_min_mean_prob > 0.0 and float(np.mean(event_prob)) < event_min_mean_prob:
            continue
        if event_min_peak_prob > 0.0 and float(np.max(event_prob)) < event_min_peak_prob:
            continue
        pred[s:e + 1] = 1
        events.append((s / sfreq, (e + 1) / sfreq))
    return events, pred


def decode_segmentation_events_structured(
    prob: np.ndarray,
    center_prob: Optional[np.ndarray],
    boundary_prob: Optional[np.ndarray],
    sfreq: float,
    threshold: float,
    min_sec: float,
    merge_gap_sec: float,
    boundary_gap: float,
    boundary_max_extend_sec: float,
    boundary_trim_sec: float,
    event_min_mean_prob: float,
    event_min_peak_prob: float,
    max_sec: float,
    cfg: Config,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Decode event instances with center-guided splitting and boundary contraction."""
    base_events, base_binary = decode_segmentation_events(
        prob,
        sfreq,
        threshold,
        min_sec,
        merge_gap_sec,
        boundary_gap,
        boundary_max_extend_sec,
        boundary_trim_sec,
        event_min_mean_prob,
        event_min_peak_prob,
        max_sec,
    )
    if (
        not cfg.segmentation_structured_decoder_enabled
        or center_prob is None
        or boundary_prob is None
        or len(center_prob) != len(prob)
        or len(boundary_prob) != len(prob)
    ):
        return base_events, base_binary

    min_len = max(1, int(round(min_sec * sfreq)))
    min_peak_distance = max(1, int(round(cfg.segmentation_center_min_distance_sec * sfreq)))
    max_shift = max(0, int(round(cfg.segmentation_boundary_max_shift_sec * sfreq)))
    refined: List[Tuple[int, int]] = []
    for start_sec, stop_sec in base_events:
        start = max(0, int(round(start_sec * sfreq)))
        stop = min(len(prob), int(round(stop_sec * sfreq)))
        if stop - start < min_len:
            continue
        local_centers, _ = signal.find_peaks(
            center_prob[start:stop],
            height=float(cfg.segmentation_center_threshold),
            distance=min_peak_distance,
        )
        centers = (local_centers + start).astype(np.int64)
        if len(centers) == 0:
            peak = int(start + np.argmax(center_prob[start:stop]))
            if float(center_prob[peak]) >= float(cfg.segmentation_center_threshold):
                centers = np.asarray([peak], dtype=np.int64)
        if len(centers) == 0:
            refined.append((start, stop))
            continue

        # Separate merged events at the lowest main probability between adjacent
        # center peaks. This keeps the decoder tied to learned event support.
        split_points: List[int] = []
        for left_center, right_center in zip(centers[:-1], centers[1:]):
            if right_center <= left_center + 1:
                split_points.append(int((left_center + right_center) // 2))
            else:
                valley = int(left_center + np.argmin(prob[left_center:right_center + 1]))
                split_points.append(valley)
        edges = [start] + split_points + [stop]
        refined_before_event = len(refined)
        for idx, center in enumerate(centers):
            left = int(edges[idx])
            right = int(edges[idx + 1])
            if right <= left:
                continue

            # Contract overlong masks toward confident learned boundaries. We do
            # not extend events here; expansion remains available in the legacy
            # probability decoder through boundary_gap/max_extend.
            if max_shift > 0:
                left_stop = min(int(center) + 1, left + max_shift + 1)
                if left_stop > left:
                    left_slice = boundary_prob[left:left_stop]
                    candidate_left = int(left + np.argmax(left_slice))
                    if float(boundary_prob[candidate_left]) >= float(cfg.segmentation_boundary_threshold):
                        left = candidate_left
                right_start = max(int(center), right - max_shift)
                if right > right_start:
                    right_slice = boundary_prob[right_start:right]
                    candidate_right = int(right_start + np.argmax(right_slice) + 1)
                    if float(boundary_prob[candidate_right - 1]) >= float(cfg.segmentation_boundary_threshold):
                        right = candidate_right
            if right - left >= min_len:
                refined.append((left, right))
            else:
                original_left = int(edges[idx])
                original_right = int(edges[idx + 1])
                if original_right - original_left >= min_len:
                    refined.append((original_left, original_right))
        if len(refined) == refined_before_event:
            refined.append((start, stop))

    pred = np.zeros(len(prob), dtype=np.uint8)
    events: List[Tuple[float, float]] = []
    for start, stop in refined:
        pred[start:stop] = 1
        events.append((start / sfreq, stop / sfreq))
    return events, pred


def retained_minutes(rec: Dict[str, Any]) -> float:
    n_samples = len(rec["truth_binary_stage"])
    sfreq = float(rec["sfreq"])
    return float(n_samples / sfreq / 60.0) if sfreq > 0 else 0.0


def event_density_per_min(events: Sequence[Tuple[float, float]], rec: Dict[str, Any]) -> float:
    return float(len(events) / max(retained_minutes(rec), 1e-8))


def true_event_density_per_min(rec: Dict[str, Any]) -> float:
    return event_density_per_min(rec["true_events_stage"], rec)


def event_duration_stats(events: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    if not events:
        return {
            "mean_duration_sec": float("nan"),
            "median_duration_sec": float("nan"),
            "min_duration_sec": float("nan"),
            "max_duration_sec": float("nan"),
        }
    durations = np.asarray([max(0.0, e - s) for s, e in events], dtype=np.float64)
    return {
        "mean_duration_sec": float(np.mean(durations)),
        "median_duration_sec": float(np.median(durations)),
        "min_duration_sec": float(np.min(durations)),
        "max_duration_sec": float(np.max(durations)),
    }


def choose_density_adaptive_threshold(
    prob: np.ndarray,
    rec: Dict[str, Any],
    base_threshold: float,
    params: SegmentationParams,
    cfg: Config,
    target_density_per_min: float,
) -> float:
    if not cfg.segmentation_density_adaptive_threshold_enabled or target_density_per_min <= 0:
        return base_threshold
    low = target_density_per_min * cfg.segmentation_density_adaptive_low_multiplier
    high = target_density_per_min * cfg.segmentation_density_adaptive_high_multiplier
    lower_thr = max(0.01, base_threshold - cfg.segmentation_density_adaptive_max_adjustment)
    upper_thr = min(0.99, base_threshold + cfg.segmentation_density_adaptive_max_adjustment)
    thresholds = np.linspace(lower_thr, upper_thr, max(3, int(cfg.segmentation_density_adaptive_steps)))
    best_thr = float(base_threshold)
    best_cost = float("inf")
    for thr in thresholds:
        events, _ = decode_segmentation_events(
            prob,
            rec["sfreq"],
            float(thr),
            params.final_min_sec,
            params.final_merge_gap_sec,
            params.boundary_gap,
            params.boundary_max_extend_sec,
            params.boundary_trim_sec,
            params.event_min_mean_prob,
            params.event_min_peak_prob,
            params.final_max_sec,
        )
        density = event_density_per_min(events, rec)
        if density < low:
            density_cost = low - density
        elif density > high:
            density_cost = density - high
        else:
            density_cost = 0.0
        threshold_cost = 0.01 * abs(float(thr) - float(base_threshold))
        cost = density_cost + threshold_cost
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)
    return best_thr


def events_overlap_enough(
    event: Tuple[float, float],
    reference_events: Sequence[Tuple[float, float]],
    min_overlap_sec: float,
) -> bool:
    s, e = event
    for rs, re_ in reference_events:
        if min(e, re_) - max(s, rs) >= min_overlap_sec:
            return True
    return False


def apply_segmentation_candidate_consensus(
    pred_events: List[Tuple[float, float]],
    prob: np.ndarray,
    rec: Dict[str, Any],
    cfg: Config,
) -> Tuple[List[Tuple[float, float]], np.ndarray, Dict[str, Any]]:
    if not cfg.segmentation_candidate_consensus_enabled or not pred_events:
        return pred_events, events_to_binary(pred_events, len(prob), rec["sfreq"]).astype(np.uint8), {
            "consensus_enabled": bool(cfg.segmentation_candidate_consensus_enabled),
            "consensus_candidates": 0,
            "consensus_removed": 0,
        }
    bundle = generate_candidates_for_record(
        rec,
        cfg,
        cfg.segmentation_candidate_consensus_B,
        cfg.segmentation_candidate_consensus_T,
    )
    candidate_events = bundle["events"]
    kept: List[Tuple[float, float]] = []
    removed = 0
    for event in pred_events:
        s_idx = max(0, int(round(event[0] * rec["sfreq"])))
        e_idx = min(len(prob), int(round(event[1] * rec["sfreq"])))
        peak_prob = float(np.max(prob[s_idx:e_idx])) if e_idx > s_idx else 0.0
        has_candidate_support = events_overlap_enough(
            event,
            candidate_events,
            cfg.segmentation_candidate_consensus_overlap_sec,
        )
        if has_candidate_support or peak_prob >= cfg.segmentation_candidate_consensus_keep_peak_prob:
            kept.append(event)
        else:
            removed += 1
    return kept, events_to_binary(kept, len(prob), rec["sfreq"]).astype(np.uint8), {
        "consensus_enabled": True,
        "consensus_candidates": int(len(candidate_events)),
        "consensus_removed": int(removed),
    }


def resolve_segmentation_probability_threshold(
    prob: np.ndarray,
    base_prob_thr: float,
    adaptive_quantile: float,
    max_adjustment: float,
) -> float:
    if len(prob) == 0 or adaptive_quantile >= 0.999:
        return float(np.clip(base_prob_thr, 0.01, 0.99))
    adaptive_thr = float(np.quantile(prob, adaptive_quantile))
    lower = max(0.01, base_prob_thr - max_adjustment)
    upper = min(0.99, base_prob_thr + max_adjustment)
    return float(np.clip(adaptive_thr, lower, upper))


def tune_segmentation_threshold(
    models: Sequence[nn.Module],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    fold_dir: Path,
    logger: logging.Logger,
    calibrator: Optional[Dict[str, Any]] = None,
    member_weights: Optional[Sequence[float]] = None,
) -> Tuple[SegmentationParams, pd.DataFrame]:
    val_head_timelines = {
        sid: predict_segmentation_head_timelines_ensemble(
            models,
            records[sid],
            cfg,
            device,
            calibrator,
            member_weights,
        )
        for sid in val_ids
    }
    val_timelines = {sid: timelines["logits"] for sid, timelines in val_head_timelines.items()}
    rows: List[Dict[str, Any]] = []
    best_params = SegmentationParams(
        probability_threshold=float(cfg.segmentation_threshold_grid[0]),
        final_merge_gap_sec=float(cfg.segmentation_merge_gap_grid[0]),
        final_min_sec=float(cfg.segmentation_min_sec_grid[0]),
        adaptive_threshold_quantile=float(cfg.segmentation_adaptive_threshold_quantile_grid[0]),
        boundary_gap=float(cfg.segmentation_boundary_gap_grid[0]),
        boundary_max_extend_sec=float(cfg.segmentation_boundary_max_extend_sec_grid[0]),
        boundary_trim_sec=float(cfg.segmentation_boundary_trim_sec_grid[0]),
        event_min_mean_prob=float(cfg.segmentation_event_mean_prob_grid[0]),
        event_min_peak_prob=float(cfg.segmentation_event_peak_prob_grid[0]),
        final_max_sec=float(cfg.segmentation_max_sec_grid[0]),
    )
    best_score = -1.0
    best_selection_key = (float("-inf"),) * 5
    best_selected_mean_f1 = -1.0
    best_f1_score = -1.0
    best_f1_params = best_params
    best_f1_precision = 0.0
    for thr in cfg.segmentation_threshold_grid:
        for merge_gap_sec in cfg.segmentation_merge_gap_grid:
            for min_sec in cfg.segmentation_min_sec_grid:
                for adaptive_quantile in cfg.segmentation_adaptive_threshold_quantile_grid:
                    for boundary_gap in cfg.segmentation_boundary_gap_grid:
                        for boundary_max_extend_sec in cfg.segmentation_boundary_max_extend_sec_grid:
                            for boundary_trim_sec, event_min_mean_prob, event_min_peak_prob, max_sec in itertools.product(
                                cfg.segmentation_boundary_trim_sec_grid,
                                cfg.segmentation_event_mean_prob_grid,
                                cfg.segmentation_event_peak_prob_grid,
                                cfg.segmentation_max_sec_grid,
                            ):
                                    f1s: List[float] = []
                                    tolerant_f1s: List[float] = []
                                    precisions: List[float] = []
                                    recalls: List[float] = []
                                    subject_thresholds: List[float] = []
                                    density_ratios: List[float] = []
                                    duration_ratios: List[float] = []
                                    for sid in val_ids:
                                        rec = records[sid]
                                        subject_thr = resolve_segmentation_probability_threshold(
                                            val_timelines[sid],
                                            float(thr),
                                            float(adaptive_quantile),
                                            cfg.segmentation_adaptive_threshold_max_adjustment,
                                        )
                                        heads = val_head_timelines[sid]
                                        pred_events, _ = decode_segmentation_events_structured(
                                            val_timelines[sid],
                                            heads.get("center_logits"),
                                            heads.get("boundary_logits"),
                                            rec["sfreq"],
                                            subject_thr,
                                            float(min_sec),
                                            float(merge_gap_sec),
                                            float(boundary_gap),
                                            float(boundary_max_extend_sec),
                                            float(boundary_trim_sec),
                                            float(event_min_mean_prob),
                                            float(event_min_peak_prob),
                                            float(max_sec),
                                            cfg,
                                        )
                                        strict = event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])
                                        tolerant_thr = cfg.event_iou_thresholds[1] if len(cfg.event_iou_thresholds) > 1 else cfg.event_iou_thresholds[0]
                                        tolerant = event_metrics(rec["true_events_stage"], pred_events, tolerant_thr)
                                        true_count = max(1, len(rec["true_events_stage"]))
                                        density_ratios.append(float(len(pred_events) / true_count))
                                        true_durations = [float(stop - start) for start, stop in rec["true_events_stage"] if stop > start]
                                        pred_durations = [float(stop - start) for start, stop in pred_events if stop > start]
                                        if true_durations and pred_durations:
                                            duration_ratios.append(
                                                float(np.median(pred_durations) / max(float(np.median(true_durations)), 1e-8))
                                            )
                                        f1s.append(strict["f1"])
                                        tolerant_f1s.append(tolerant["f1"])
                                        precisions.append(strict["precision"])
                                        recalls.append(strict["recall"])
                                        subject_thresholds.append(subject_thr)
                                    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
                                    mean_tolerant_f1 = float(np.mean(tolerant_f1s)) if tolerant_f1s else 0.0
                                    mean_precision = float(np.mean(precisions)) if precisions else 0.0
                                    mean_recall = float(np.mean(recalls)) if recalls else 0.0
                                    min_f1 = float(np.min(f1s)) if f1s else 0.0
                                    min_precision = float(np.min(precisions)) if precisions else 0.0
                                    min_recall = float(np.min(recalls)) if recalls else 0.0
                                    mean_density_ratio = float(np.mean(density_ratios)) if density_ratios else 1.0
                                    max_density_ratio = float(np.max(density_ratios)) if density_ratios else 1.0
                                    min_density_ratio = float(np.min(density_ratios)) if density_ratios else 1.0
                                    mean_duration_ratio = float(np.mean(duration_ratios)) if duration_ratios else 1.0
                                    max_duration_ratio = float(np.max(duration_ratios)) if duration_ratios else 1.0
                                    min_duration_ratio = float(np.min(duration_ratios)) if duration_ratios else 1.0
                                    density_penalty = 0.0
                                    if cfg.segmentation_density_penalty_enabled:
                                        for ratio in density_ratios:
                                            if ratio > cfg.segmentation_density_target_high:
                                                density_penalty += cfg.segmentation_density_over_penalty * (
                                                    ratio - cfg.segmentation_density_target_high
                                                )
                                            elif ratio < cfg.segmentation_density_target_low:
                                                density_penalty += cfg.segmentation_density_under_penalty * (
                                                    cfg.segmentation_density_target_low - ratio
                                                )
                                        density_penalty /= max(len(density_ratios), 1)
                                    precision_penalty = 0.0
                                    if mean_precision < cfg.segmentation_precision_floor:
                                        precision_penalty += cfg.segmentation_precision_floor_penalty * (
                                            cfg.segmentation_precision_floor - mean_precision
                                        )
                                    if min_precision < cfg.segmentation_min_precision_floor:
                                        precision_penalty += cfg.segmentation_min_precision_floor_penalty * (
                                            cfg.segmentation_min_precision_floor - min_precision
                                        )
                                    recall_penalty = 0.0
                                    if mean_recall < cfg.segmentation_recall_floor:
                                        recall_penalty += cfg.segmentation_recall_floor_penalty * (
                                            cfg.segmentation_recall_floor - mean_recall
                                        )
                                    if min_recall < cfg.segmentation_min_recall_floor:
                                        recall_penalty += cfg.segmentation_min_recall_floor_penalty * (
                                            cfg.segmentation_min_recall_floor - min_recall
                                        )
                                    duration_penalty = 0.0
                                    if cfg.segmentation_duration_penalty_enabled and duration_ratios:
                                        for ratio in duration_ratios:
                                            if ratio > cfg.segmentation_duration_ratio_high:
                                                duration_penalty += cfg.segmentation_duration_over_penalty * (
                                                    ratio - cfg.segmentation_duration_ratio_high
                                                )
                                            elif ratio < cfg.segmentation_duration_ratio_low:
                                                duration_penalty += cfg.segmentation_duration_under_penalty * (
                                                    cfg.segmentation_duration_ratio_low - ratio
                                                )
                                        duration_penalty /= len(duration_ratios)
                                    base_score = (
                                        cfg.segmentation_selection_mean_f1_weight * mean_f1
                                        + cfg.segmentation_selection_min_f1_weight * min_f1
                                        + cfg.segmentation_selection_mean_precision_weight * mean_precision
                                        + cfg.segmentation_selection_mean_recall_weight * mean_recall
                                        + cfg.segmentation_selection_tolerant_f1_weight * mean_tolerant_f1
                                    )
                                    score = base_score - density_penalty - precision_penalty - recall_penalty - duration_penalty
                                    rows.append(
                                        {
                                            "threshold": float(thr),
                                            "final_merge_gap_sec": float(merge_gap_sec),
                                            "final_min_sec": float(min_sec),
                                            "adaptive_threshold_quantile": float(adaptive_quantile),
                                            "boundary_gap": float(boundary_gap),
                                            "boundary_max_extend_sec": float(boundary_max_extend_sec),
                                            "boundary_trim_sec": float(boundary_trim_sec),
                                            "event_min_mean_prob": float(event_min_mean_prob),
                                            "event_min_peak_prob": float(event_min_peak_prob),
                                            "final_max_sec": float(max_sec),
                                            "mean_subject_threshold": float(np.mean(subject_thresholds)) if subject_thresholds else float("nan"),
                                            "mean_density_ratio": mean_density_ratio,
                                            "min_density_ratio": min_density_ratio,
                                            "max_density_ratio": max_density_ratio,
                                            "density_penalty": float(density_penalty),
                                            "mean_duration_ratio": mean_duration_ratio,
                                            "min_duration_ratio": min_duration_ratio,
                                            "max_duration_ratio": max_duration_ratio,
                                            "duration_penalty": float(duration_penalty),
                                            "precision_penalty": float(precision_penalty),
                                            "recall_penalty": float(recall_penalty),
                                            "val_stage3_base_score": float(base_score),
                                            "val_stage3_score": float(score),
                                            "val_strict_f1": mean_f1,
                                            "val_tolerant_f1": mean_tolerant_f1,
                                            "val_strict_precision": mean_precision,
                                            "val_strict_recall": mean_recall,
                                            "val_min_subject_f1": min_f1,
                                            "val_min_subject_precision": min_precision,
                                            "val_min_subject_recall": min_recall,
                                        }
                                    )
                                    # Deterministic tie-breaking avoids choosing
                                    # an unnecessarily permissive threshold when
                                    # validation objectives are effectively equal.
                                    selection_key = (
                                        round(float(score), 8),
                                        round(float(mean_f1), 8),
                                        round(float(mean_precision), 8),
                                        round(float(-abs(mean_density_ratio - 1.0)), 8),
                                        float(thr),
                                    )
                                    if selection_key > best_selection_key:
                                        best_selection_key = selection_key
                                        best_score = score
                                        best_selected_mean_f1 = mean_f1
                                        best_params = SegmentationParams(
                                            probability_threshold=float(thr),
                                            final_merge_gap_sec=float(merge_gap_sec),
                                            final_min_sec=float(min_sec),
                                            adaptive_threshold_quantile=float(adaptive_quantile),
                                            boundary_gap=float(boundary_gap),
                                            boundary_max_extend_sec=float(boundary_max_extend_sec),
                                            boundary_trim_sec=float(boundary_trim_sec),
                                            event_min_mean_prob=float(event_min_mean_prob),
                                            event_min_peak_prob=float(event_min_peak_prob),
                                            final_max_sec=float(max_sec),
                                        )
                                    if (
                                        cfg.segmentation_f1_fallback_enabled
                                        and mean_precision >= cfg.segmentation_f1_fallback_min_precision
                                        and mean_f1 > best_f1_score
                                    ):
                                        best_f1_score = mean_f1
                                        best_f1_precision = mean_precision
                                        best_f1_params = SegmentationParams(
                                            probability_threshold=float(thr),
                                            final_merge_gap_sec=float(merge_gap_sec),
                                            final_min_sec=float(min_sec),
                                            adaptive_threshold_quantile=float(adaptive_quantile),
                                            boundary_gap=float(boundary_gap),
                                            boundary_max_extend_sec=float(boundary_max_extend_sec),
                                            boundary_trim_sec=float(boundary_trim_sec),
                                            event_min_mean_prob=float(event_min_mean_prob),
                                            event_min_peak_prob=float(event_min_peak_prob),
                                            final_max_sec=float(max_sec),
                                        )
    df = pd.DataFrame(rows)
    df.to_csv(fold_dir / "segmentation_threshold_tuning.csv", index=False)
    if (
        cfg.segmentation_f1_fallback_enabled
        and best_f1_score > best_selected_mean_f1 + cfg.segmentation_f1_fallback_margin
    ):
        logger.info(
            "Segmentation F1 fallback selected validation params: f1=%.4f precision=%.4f replaced_score_selected_f1=%.4f margin=%.4f",
            best_f1_score,
            best_f1_precision,
            best_selected_mean_f1,
            cfg.segmentation_f1_fallback_margin,
        )
        best_params = best_f1_params
    logger.info(
        "Best segmentation params on validation: thr=%.3f merge_gap=%.3f min_sec=%.3f max_sec=%.3f adaptive_q=%.3f boundary_gap=%.3f boundary_extend=%.3f boundary_trim=%.3f event_mean=%.3f event_peak=%.3f score=%.4f",
        best_params.probability_threshold,
        best_params.final_merge_gap_sec,
        best_params.final_min_sec,
        best_params.final_max_sec,
        best_params.adaptive_threshold_quantile,
        best_params.boundary_gap,
        best_params.boundary_max_extend_sec,
        best_params.boundary_trim_sec,
        best_params.event_min_mean_prob,
        best_params.event_min_peak_prob,
        best_score,
    )
    return best_params, df


def tune_segmentation_params_from_timelines(
    timelines: Dict[str, np.ndarray],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    cfg: Config,
    outdir: Path,
    logger: logging.Logger,
) -> Tuple[SegmentationParams, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_params = SegmentationParams(
        probability_threshold=float(cfg.segmentation_threshold_grid[0]),
        final_merge_gap_sec=float(cfg.segmentation_merge_gap_grid[0]),
        final_min_sec=float(cfg.segmentation_min_sec_grid[0]),
        adaptive_threshold_quantile=float(cfg.segmentation_adaptive_threshold_quantile_grid[0]),
        boundary_gap=float(cfg.segmentation_boundary_gap_grid[0]),
        boundary_max_extend_sec=float(cfg.segmentation_boundary_max_extend_sec_grid[0]),
        boundary_trim_sec=float(cfg.segmentation_boundary_trim_sec_grid[0]),
        event_min_mean_prob=float(cfg.segmentation_event_mean_prob_grid[0]),
        event_min_peak_prob=float(cfg.segmentation_event_peak_prob_grid[0]),
        final_max_sec=float(cfg.segmentation_max_sec_grid[0]),
    )
    best_score = -1.0
    for thr in cfg.segmentation_threshold_grid:
        for merge_gap_sec in cfg.segmentation_merge_gap_grid:
            for min_sec in cfg.segmentation_min_sec_grid:
                for adaptive_quantile in cfg.segmentation_adaptive_threshold_quantile_grid:
                    for boundary_gap in cfg.segmentation_boundary_gap_grid:
                        for boundary_max_extend_sec in cfg.segmentation_boundary_max_extend_sec_grid:
                            for boundary_trim_sec, event_min_mean_prob, event_min_peak_prob, max_sec in itertools.product(
                                cfg.segmentation_boundary_trim_sec_grid,
                                cfg.segmentation_event_mean_prob_grid,
                                cfg.segmentation_event_peak_prob_grid,
                                cfg.segmentation_max_sec_grid,
                            ):
                                    f1s: List[float] = []
                                    precisions: List[float] = []
                                    recalls: List[float] = []
                                    density_ratios: List[float] = []
                                    subject_thresholds: List[float] = []
                                    for sid in val_ids:
                                        rec = records[sid]
                                        timeline = timelines[sid]
                                        subject_thr = resolve_segmentation_probability_threshold(
                                            timeline,
                                            float(thr),
                                            float(adaptive_quantile),
                                            cfg.segmentation_adaptive_threshold_max_adjustment,
                                        )
                                        pred_events, _ = decode_segmentation_events(
                                            timeline,
                                            rec["sfreq"],
                                            subject_thr,
                                            float(min_sec),
                                            float(merge_gap_sec),
                                            float(boundary_gap),
                                            float(boundary_max_extend_sec),
                                            float(boundary_trim_sec),
                                            float(event_min_mean_prob),
                                            float(event_min_peak_prob),
                                            float(max_sec),
                                        )
                                        strict = event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])
                                        true_count = max(1, len(rec["true_events_stage"]))
                                        density_ratios.append(float(len(pred_events) / true_count))
                                        f1s.append(strict["f1"])
                                        precisions.append(strict["precision"])
                                        recalls.append(strict["recall"])
                                        subject_thresholds.append(subject_thr)
                                    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
                                    mean_precision = float(np.mean(precisions)) if precisions else 0.0
                                    mean_recall = float(np.mean(recalls)) if recalls else 0.0
                                    min_f1 = float(np.min(f1s)) if f1s else 0.0
                                    min_precision = float(np.min(precisions)) if precisions else 0.0
                                    density_penalty = 0.0
                                    if cfg.segmentation_density_penalty_enabled:
                                        for ratio in density_ratios:
                                            if ratio > cfg.segmentation_density_target_high:
                                                density_penalty += cfg.segmentation_density_over_penalty * (
                                                    ratio - cfg.segmentation_density_target_high
                                                )
                                            elif ratio < cfg.segmentation_density_target_low:
                                                density_penalty += cfg.segmentation_density_under_penalty * (
                                                    cfg.segmentation_density_target_low - ratio
                                                )
                                        density_penalty /= max(len(density_ratios), 1)
                                    precision_penalty = 0.0
                                    if mean_precision < cfg.segmentation_precision_floor:
                                        precision_penalty += cfg.segmentation_precision_floor_penalty * (
                                            cfg.segmentation_precision_floor - mean_precision
                                        )
                                    if min_precision < cfg.segmentation_min_precision_floor:
                                        precision_penalty += cfg.segmentation_min_precision_floor_penalty * (
                                            cfg.segmentation_min_precision_floor - min_precision
                                        )
                                    base_score = (
                                        cfg.segmentation_selection_mean_f1_weight * mean_f1
                                        + cfg.segmentation_selection_min_f1_weight * min_f1
                                        + cfg.segmentation_selection_mean_precision_weight * mean_precision
                                    )
                                    score = base_score - density_penalty - precision_penalty
                                    row = {
                                        "threshold": float(thr),
                                        "final_merge_gap_sec": float(merge_gap_sec),
                                        "final_min_sec": float(min_sec),
                                        "adaptive_threshold_quantile": float(adaptive_quantile),
                                        "boundary_gap": float(boundary_gap),
                                        "boundary_max_extend_sec": float(boundary_max_extend_sec),
                                        "boundary_trim_sec": float(boundary_trim_sec),
                                        "event_min_mean_prob": float(event_min_mean_prob),
                                        "event_min_peak_prob": float(event_min_peak_prob),
                                        "final_max_sec": float(max_sec),
                                        "mean_subject_threshold": float(np.mean(subject_thresholds)) if subject_thresholds else float("nan"),
                                        "mean_density_ratio": float(np.mean(density_ratios)) if density_ratios else 1.0,
                                        "min_density_ratio": float(np.min(density_ratios)) if density_ratios else 1.0,
                                        "max_density_ratio": float(np.max(density_ratios)) if density_ratios else 1.0,
                                        "density_penalty": float(density_penalty),
                                        "precision_penalty": float(precision_penalty),
                                        "val_stage3_base_score": float(base_score),
                                        "val_stage3_score": float(score),
                                        "val_strict_f1": mean_f1,
                                        "val_strict_precision": mean_precision,
                                        "val_strict_recall": mean_recall,
                                        "val_min_subject_f1": min_f1,
                                        "val_min_subject_precision": min_precision,
                                    }
                                    rows.append(row)
                                    if score > best_score:
                                        best_score = score
                                        best_params = SegmentationParams(
                                            probability_threshold=float(thr),
                                            final_merge_gap_sec=float(merge_gap_sec),
                                            final_min_sec=float(min_sec),
                                            adaptive_threshold_quantile=float(adaptive_quantile),
                                            boundary_gap=float(boundary_gap),
                                            boundary_max_extend_sec=float(boundary_max_extend_sec),
                                            boundary_trim_sec=float(boundary_trim_sec),
                                            event_min_mean_prob=float(event_min_mean_prob),
                                            event_min_peak_prob=float(event_min_peak_prob),
                                            final_max_sec=float(max_sec),
                                        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "external_dreams_calibration_tuning.csv", index=False)
    logger.info(
        "Best external DREAMS calibration params: thr=%.3f merge_gap=%.3f min_sec=%.3f max_sec=%.3f adaptive_q=%.3f boundary_gap=%.3f boundary_extend=%.3f boundary_trim=%.3f event_mean=%.3f event_peak=%.3f score=%.4f",
        best_params.probability_threshold,
        best_params.final_merge_gap_sec,
        best_params.final_min_sec,
        best_params.final_max_sec,
        best_params.adaptive_threshold_quantile,
        best_params.boundary_gap,
        best_params.boundary_max_extend_sec,
        best_params.boundary_trim_sec,
        best_params.event_min_mean_prob,
        best_params.event_min_peak_prob,
        best_score,
    )
    return best_params, df


def fit_probability_calibrator(
    probs: np.ndarray,
    y_true: np.ndarray,
    cfg: Config,
    logger: logging.Logger,
    context: str,
) -> Optional[Dict[str, Any]]:
    if not cfg.probability_calibration_enabled or cfg.probability_calibration_method == "none":
        return None
    y_bin = (np.asarray(y_true) > 0.5).astype(np.int32)
    x = np.asarray(probs, dtype=np.float64)
    if len(x) < 16 or len(np.unique(y_bin)) < 2:
        logger.info("Skipping probability calibration for %s due to insufficient class variation.", context)
        return None
    x = np.clip(x, 1e-5, 1.0 - 1e-5)
    method = cfg.probability_calibration_method
    try:
        if method == "platt":
            model = LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced")
            model.fit(x.reshape(-1, 1), y_bin)
            return {"method": method, "model": model}
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x, y_bin)
            return {"method": method, "model": model}
    except Exception as exc:
        logger.warning("Probability calibration failed for %s with method %s: %s", context, method, exc)
        return None
    return None


def apply_probability_calibrator(calibrator: Optional[Dict[str, Any]], probs: np.ndarray) -> np.ndarray:
    if calibrator is None or len(probs) == 0:
        return np.asarray(probs, dtype=np.float32)
    method = calibrator.get("method")
    model = calibrator.get("model")
    x = np.clip(np.asarray(probs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    if method == "platt":
        return model.predict_proba(x.reshape(-1, 1))[:, 1].astype(np.float32)
    if method == "isotonic":
        return np.asarray(model.predict(x), dtype=np.float32)
    return np.asarray(probs, dtype=np.float32)


def fit_segmentation_probability_calibrator(
    models: Sequence[nn.Module],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
    context: str,
    member_weights: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    if not cfg.segmentation_probability_calibration_enabled or cfg.segmentation_probability_calibration_method == "none":
        return None
    probs_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for sid in val_ids:
        prob = predict_segmentation_timeline_ensemble(
            models,
            records[sid],
            cfg,
            device,
            calibrator=None,
            member_weights=member_weights,
        )
        y = records[sid]["truth_binary_stage"].astype(np.uint8)
        if len(prob) != len(y):
            n = min(len(prob), len(y))
            prob = prob[:n]
            y = y[:n]
        probs_list.append(prob.astype(np.float32))
        y_list.append(y.astype(np.uint8))
    if not probs_list:
        return None
    probs = np.concatenate(probs_list)
    y_true = np.concatenate(y_list)
    max_samples = int(cfg.segmentation_calibration_max_samples)
    if max_samples > 0 and len(y_true) > max_samples:
        rng = np.random.default_rng(cfg.seed)
        pos_idx = np.flatnonzero(y_true > 0)
        neg_idx = np.flatnonzero(y_true <= 0)
        half = max_samples // 2
        n_pos = min(len(pos_idx), half)
        n_neg = min(len(neg_idx), max_samples - n_pos)
        if n_pos > 0 and n_neg > 0:
            chosen = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ])
        else:
            chosen = rng.choice(np.arange(len(y_true)), size=max_samples, replace=False)
        probs = probs[chosen]
        y_true = y_true[chosen]
    old_method = cfg.probability_calibration_method
    old_enabled = cfg.probability_calibration_enabled
    cfg.probability_calibration_enabled = True
    cfg.probability_calibration_method = cfg.segmentation_probability_calibration_method
    try:
        return fit_probability_calibrator(probs, y_true, cfg, logger, context)
    finally:
        cfg.probability_calibration_method = old_method
        cfg.probability_calibration_enabled = old_enabled


def build_probability_timeline(candidates: List[Tuple[float, float]], probs: np.ndarray, n_samples: int, sfreq: float) -> np.ndarray:
    timeline = np.zeros(n_samples, dtype=np.float32)
    for (start_sec, stop_sec), p in zip(candidates, probs):
        s = max(0, int(round(start_sec * sfreq)))
        e = min(n_samples, int(round(stop_sec * sfreq)))
        if e > s:
            timeline[s:e] = np.maximum(timeline[s:e], float(p))
    return timeline


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    x = values[order]
    w = np.maximum(weights[order], 1e-8)
    cum = np.cumsum(w)
    target = np.clip(float(quantile), 0.0, 1.0) * cum[-1]
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), len(x) - 1)
    return float(x[idx])


def smooth_support_signal(x: np.ndarray, fs: float, smooth_sec: float) -> np.ndarray:
    width = max(1, int(round(smooth_sec * fs)))
    if width <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def clip_event_duration(event: Tuple[float, float], min_sec: float, max_sec: float) -> Tuple[float, float]:
    start, stop = event
    center = 0.5 * (start + stop)
    duration = max(0.0, stop - start)
    if duration < min_sec:
        duration = min_sec
    if max_sec > 0:
        duration = min(duration, max_sec)
    half = 0.5 * duration
    return float(center - half), float(center + half)


def decode_candidate_cluster(
    cluster_events: Sequence[Tuple[float, float]],
    cluster_probs: Sequence[float],
    prob_thr: float,
    sfreq: float,
    min_sec: float,
    max_sec: float,
    support_smooth_sec: float,
    support_relative_threshold: float,
) -> List[Tuple[float, float]]:
    if not cluster_events:
        return []
    starts = np.asarray([s for s, _ in cluster_events], dtype=np.float64)
    stops = np.asarray([e for _, e in cluster_events], dtype=np.float64)
    weights = np.asarray(cluster_probs, dtype=np.float64)
    weights = np.maximum(weights, 1e-6) ** 2.0
    union_start = float(np.min(starts))
    union_stop = float(np.max(stops))
    local_start = max(0, int(math.floor(union_start * sfreq)))
    local_stop = max(local_start + 1, int(math.ceil(union_stop * sfreq)))
    support = np.zeros(local_stop - local_start, dtype=np.float32)
    anchor_support = np.zeros_like(support)
    for (start_sec, stop_sec), score in zip(cluster_events, cluster_probs):
        s = max(local_start, int(round(start_sec * sfreq)))
        e = min(local_stop, int(round(stop_sec * sfreq)))
        if e <= s:
            continue
        support[s - local_start:e - local_start] += float(score)
        if score >= prob_thr:
            anchor_support[s - local_start:e - local_start] += float(score)

    support = smooth_support_signal(support, sfreq, support_smooth_sec)
    anchor_support = smooth_support_signal(anchor_support, sfreq, support_smooth_sec)
    rel_thr = max(1e-6, support_relative_threshold * float(np.max(support)))
    binary = support >= rel_thr
    segments = binary_to_segments(binary.astype(np.uint8))
    if not segments:
        segments = [(0, len(binary) - 1)]

    events: List[Tuple[float, float]] = []
    for s_idx, e_idx in segments:
        if e_idx < s_idx:
            continue
        if float(np.max(anchor_support[s_idx:e_idx + 1])) <= 0.0:
            continue
        seg_start = (local_start + s_idx) / sfreq
        seg_stop = (local_start + e_idx + 1) / sfreq
        overlapping: List[Tuple[float, float]] = []
        overlapping_probs: List[float] = []
        for event, score in zip(cluster_events, cluster_probs):
            if not (event[1] <= seg_start or event[0] >= seg_stop):
                overlapping.append(event)
                overlapping_probs.append(score)
        if not overlapping:
            continue
        o_starts = np.asarray([s for s, _ in overlapping], dtype=np.float64)
        o_stops = np.asarray([e for _, e in overlapping], dtype=np.float64)
        o_weights = np.maximum(np.asarray(overlapping_probs, dtype=np.float64), 1e-6) ** 2.0
        fused_start = min(seg_start, weighted_quantile(o_starts, o_weights, 0.15))
        fused_stop = max(seg_stop, weighted_quantile(o_stops, o_weights, 0.85))
        fused = clip_event_duration((fused_start, fused_stop), min_sec, max_sec)
        events.append(fused)
    if not events:
        fallback_start = weighted_quantile(starts, weights, 0.15)
        fallback_stop = weighted_quantile(stops, weights, 0.85)
        return [clip_event_duration((fallback_start, fallback_stop), min_sec, max_sec)]
    return merge_intervals(events)


def events_to_binary_from_list(events: List[Tuple[float, float]], n_samples: int, sfreq: float) -> np.ndarray:
    return events_to_binary(events, n_samples, sfreq).astype(np.uint8)


def close_gaps_binary(binary: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return binary.astype(bool)
    out = binary.astype(bool).copy()
    segments = binary_to_segments(out.astype(np.uint8))
    if len(segments) <= 1:
        return out
    for (s1, e1), (s2, _) in zip(segments[:-1], segments[1:]):
        if s2 - e1 - 1 <= max_gap:
            out[e1 + 1:s2] = True
    return out


def hysteresis_binarize(prob: np.ndarray, thr_hi: float, thr_lo: float) -> np.ndarray:
    out = np.zeros(len(prob), dtype=bool)
    active = False
    for i, p in enumerate(prob):
        if not active and p >= thr_hi:
            active = True
            out[i] = True
        elif active:
            if p >= thr_lo:
                out[i] = True
            else:
                active = False
    return out


def decode_events_from_probability_segment(
    prob: np.ndarray,
    sfreq: float,
    thr_hi: float,
    hysteresis_gap: float,
    merge_gap_sec: float,
    min_sec: float,
    max_sec: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    thr_lo = max(0.0, thr_hi - hysteresis_gap)
    binary = hysteresis_binarize(prob, thr_hi, thr_lo)
    binary = close_gaps_binary(binary, int(round(merge_gap_sec * sfreq)))
    segments = binary_to_segments(binary.astype(np.uint8))
    events: List[Tuple[float, float]] = []
    pred_binary = np.zeros(len(prob), dtype=np.uint8)
    min_len = int(round(min_sec * sfreq))
    max_len = int(round(max_sec * sfreq))
    for s, e in segments:
        dur = e - s + 1
        if dur < min_len:
            continue
        if max_len > 0 and dur > max_len:
            continue
        pred_binary[s:e + 1] = 1
        events.append((s / sfreq, (e + 1) / sfreq))
    return events, pred_binary


def decode_events_from_probabilities(
    prob: np.ndarray,
    sfreq: float,
    thr_hi: float,
    hysteresis_gap: float,
    merge_gap_sec: float,
    min_sec: float,
    max_sec: float,
    segments: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    if not segments:
        return decode_events_from_probability_segment(prob, sfreq, thr_hi, hysteresis_gap, merge_gap_sec, min_sec, max_sec)
    pred_binary = np.zeros(len(prob), dtype=np.uint8)
    events: List[Tuple[float, float]] = []
    for seg_start, seg_stop in segments:
        if seg_stop <= seg_start:
            continue
        seg_events, seg_binary = decode_events_from_probability_segment(
            prob[seg_start:seg_stop],
            sfreq,
            thr_hi,
            hysteresis_gap,
            merge_gap_sec,
            min_sec,
            max_sec,
        )
        pred_binary[seg_start:seg_stop] = seg_binary
        offset_sec = seg_start / sfreq
        events.extend([(s + offset_sec, e + offset_sec) for s, e in seg_events])
    return events, pred_binary


def decode_events_from_candidate_probabilities(
    candidates: Sequence[Tuple[float, float]],
    probs: np.ndarray,
    n_samples: int,
    sfreq: float,
    prob_thr: float,
    hysteresis_gap: float,
    min_sec: float,
    max_sec: float,
    merge_gap_sec: float,
    support_smooth_sec: float,
    support_relative_threshold: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    candidate_items: List[Tuple[Tuple[float, float], float, bool]] = []
    thr_lo = max(0.05, prob_thr - hysteresis_gap)
    for cand, p in zip(candidates, probs):
        score = float(p)
        if score >= thr_lo:
            candidate_items.append((cand, score, score >= prob_thr))
    if not candidate_items:
        return [], np.zeros(n_samples, dtype=np.uint8)

    candidate_items.sort(key=lambda item: (item[0][0], item[0][1]))
    clusters: List[List[Tuple[Tuple[float, float], float]]] = []
    current_cluster: List[Tuple[Tuple[float, float], float]] = []
    current_stop = 0.0
    has_anchor = False
    for event, score, is_anchor in candidate_items:
        start, stop = event
        if not current_cluster:
            current_cluster = [(event, score)]
            current_stop = stop
            has_anchor = is_anchor
        elif start <= current_stop + merge_gap_sec:
            current_cluster.append((event, score))
            current_stop = max(current_stop, stop)
            has_anchor = has_anchor or is_anchor
        else:
            if has_anchor:
                clusters.append(current_cluster)
            current_cluster = [(event, score)]
            current_stop = stop
            has_anchor = is_anchor
    if current_cluster and has_anchor:
        clusters.append(current_cluster)

    fused_events: List[Tuple[float, float]] = []
    for cluster in clusters:
        cluster_events = [event for event, _ in cluster]
        cluster_probs = [score for _, score in cluster]
        fused_events.extend(
            decode_candidate_cluster(
                cluster_events,
                cluster_probs,
                prob_thr,
                sfreq,
                min_sec,
                max_sec,
                support_smooth_sec,
                support_relative_threshold,
            )
        )
    fused_events = merge_intervals(fused_events)
    return fused_events, events_to_binary_from_list(fused_events, n_samples, sfreq)


def resolve_subject_probability_threshold(
    candidate_probs: np.ndarray,
    base_prob_thr: float,
    adaptive_quantile: float,
    max_adjustment: float,
    min_thr: float,
    max_thr: float,
) -> float:
    if len(candidate_probs) == 0 or adaptive_quantile >= 0.999:
        return float(np.clip(base_prob_thr, min_thr, max_thr))
    adaptive_thr = float(np.quantile(candidate_probs, adaptive_quantile))
    lower = max(min_thr, base_prob_thr - max_adjustment)
    upper = min(max_thr, base_prob_thr + max_adjustment)
    return float(np.clip(adaptive_thr, lower, upper))


def event_metrics(true_events: List[Tuple[float, float]], pred_events: List[Tuple[float, float]], iou_thr: float) -> Dict[str, float]:
    matched_true: Set[int] = set()
    tp = 0
    for pred in pred_events:
        best_idx = None
        best_iou = 0.0
        for i, true_event in enumerate(true_events):
            if i in matched_true:
                continue
            iou = interval_iou(pred, true_event)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None and best_iou >= iou_thr:
            matched_true.add(best_idx)
            tp += 1
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def tolerant_event_metrics(true_events: List[Tuple[float, float]], pred_events: List[Tuple[float, float]], tolerance_sec: float) -> Dict[str, float]:
    matched_true: Set[int] = set()
    tp = 0
    for ps, pe in pred_events:
        for idx, (ts, te) in enumerate(true_events):
            if idx in matched_true:
                continue
            if not (pe < ts - tolerance_sec or ps > te + tolerance_sec):
                matched_true.add(idx)
                tp += 1
                break
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluation_references(rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "expert1": {
            "events": rec.get("true_events_e1_stage", []),
            "binary": rec.get("truth_binary_e1_stage", rec["truth_binary_stage"]),
        },
        "expert2": {
            "events": rec.get("true_events_e2_stage", []),
            "binary": rec.get("truth_binary_e2_stage", rec["truth_binary_stage"]),
        },
        "union": {
            "events": rec["true_events_stage"],
            "binary": rec["truth_binary_stage"],
        },
    }


def subject_reference_metrics(
    subject_id: str,
    reference_name: str,
    true_events: List[Tuple[float, float]],
    true_binary: np.ndarray,
    pred_events: List[Tuple[float, float]],
    pred_binary: np.ndarray,
    probability_timeline: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    strict = event_metrics(true_events, pred_events, cfg.event_iou_thresholds[0])
    tolerant = tolerant_event_metrics(true_events, pred_events, cfg.tolerant_match_sec)
    sample = sample_level_metrics(true_binary, pred_binary, probability_timeline)
    return {
        "subject_id": subject_id,
        "reference": reference_name,
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
        "strict_event_precision": strict["precision"],
        "strict_event_recall": strict["recall"],
        "strict_event_f1": strict["f1"],
        "strict_tp": strict["tp"],
        "strict_fp": strict["fp"],
        "strict_fn": strict["fn"],
        "tolerant_event_precision": tolerant["precision"],
        "tolerant_event_recall": tolerant["recall"],
        "tolerant_event_f1": tolerant["f1"],
        "tolerant_tp": tolerant["tp"],
        "tolerant_fp": tolerant["fp"],
        "tolerant_fn": tolerant["fn"],
        "ACC": sample["ACC"],
        "SEN": sample["SEN"],
        "SPE": sample["SPE"],
        "KAPPA": sample["KAPPA"],
        "sample_f1": sample["sample_f1"],
        "sample_auc": sample.get("sample_auc", float("nan")),
        "sample_pr_auc": sample.get("sample_pr_auc", float("nan")),
    }


def aggregate_reference_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for reference in sorted({str(row["reference"]) for row in rows}):
        ref_rows = [row for row in rows if row["reference"] == reference]
        strict_tp = sum(int(row["strict_tp"]) for row in ref_rows)
        strict_fp = sum(int(row["strict_fp"]) for row in ref_rows)
        strict_fn = sum(int(row["strict_fn"]) for row in ref_rows)
        tol_tp = sum(int(row["tolerant_tp"]) for row in ref_rows)
        tol_fp = sum(int(row["tolerant_fp"]) for row in ref_rows)
        tol_fn = sum(int(row["tolerant_fn"]) for row in ref_rows)
        strict_precision = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
        strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
        strict_f1 = (
            2 * strict_precision * strict_recall / (strict_precision + strict_recall)
            if (strict_precision + strict_recall) > 0
            else 0.0
        )
        tol_precision = tol_tp / (tol_tp + tol_fp) if (tol_tp + tol_fp) > 0 else 0.0
        tol_recall = tol_tp / (tol_tp + tol_fn) if (tol_tp + tol_fn) > 0 else 0.0
        tol_f1 = 2 * tol_precision * tol_recall / (tol_precision + tol_recall) if (tol_precision + tol_recall) > 0 else 0.0
        out[reference] = {
            "reference": reference,
            "n_subjects": len(ref_rows),
            "n_true_events": int(sum(int(row["n_true_events"]) for row in ref_rows)),
            "n_pred_events": int(sum(int(row["n_pred_events"]) for row in ref_rows)),
            "strict_event_precision": strict_precision,
            "strict_event_recall": strict_recall,
            "strict_event_f1": strict_f1,
            "strict_tp": strict_tp,
            "strict_fp": strict_fp,
            "strict_fn": strict_fn,
            "tolerant_event_precision": tol_precision,
            "tolerant_event_recall": tol_recall,
            "tolerant_event_f1": tol_f1,
            "tolerant_tp": tol_tp,
            "tolerant_fp": tol_fp,
            "tolerant_fn": tol_fn,
        }
        for metric in ("ACC", "SEN", "SPE", "KAPPA", "sample_f1", "sample_auc", "sample_pr_auc"):
            values = [float(row[metric]) for row in ref_rows if math.isfinite(float(row[metric]))]
            out[reference][metric] = float(np.mean(values)) if values else float("nan")
    return out


def segmentation_model_channel_importance(model: nn.Module, cfg: Config) -> Dict[str, float]:
    names = segmentation_channel_names(cfg)
    try:
        if isinstance(model, TransformerOnlySegmentation1D):
            patch_size = int(model.patch_size)
            weight_2d = model.patch_embed.weight.detach().abs()
            flat_dim = len(names) * patch_size
            weight = weight_2d[:, :flat_dim].reshape(weight_2d.shape[0], len(names), patch_size).mean(dim=(0, 2)).cpu().numpy()
            if model.secondary_gate_logit is not None and model.secondary_start_channel is not None:
                gate = float(torch.sigmoid(model.secondary_gate_logit.detach()).cpu())
                weight[int(model.secondary_start_channel):] *= gate
        else:
            weight = model.down_blocks[0].net[0].weight.detach().abs().mean(dim=(0, 2)).cpu().numpy()
    except Exception:
        return {name: float("nan") for name in names}
    values = weight[:len(names)].astype(np.float64)
    total = float(np.sum(values))
    if total > 0:
        values = values / total
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def save_segmentation_feature_importance(
    models: Sequence[nn.Module],
    cfg: Config,
    outdir: Path,
    logger: logging.Logger,
) -> None:
    if not models:
        return
    names = segmentation_channel_names(cfg)
    rows = []
    for idx, model in enumerate(models, start=1):
        imp = segmentation_model_channel_importance(model, cfg)
        gate = (
            float(torch.sigmoid(model.secondary_gate_logit.detach()).cpu())
            if isinstance(model, TransformerOnlySegmentation1D) and model.secondary_gate_logit is not None
            else float("nan")
        )
        rows.append({"model": idx, "secondary_gate_value": gate, **imp})
    df = pd.DataFrame(rows)
    mean_row = {"model": "mean"}
    for name in names:
        mean_row[name] = float(pd.to_numeric(df[name], errors="coerce").mean()) if name in df else float("nan")
    mean_row["secondary_gate_value"] = float(pd.to_numeric(df["secondary_gate_value"], errors="coerce").mean())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_csv(outdir / "feature_importance.csv", index=False)
    if not plotting_enabled():
        return
    plot_values = [float(mean_row[name]) for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, plot_values, color=["tab:blue", "tab:orange", "tab:green"][:len(names)])
    ax.set_ylim(0.0, max(1.0, max(plot_values + [0.0]) * 1.15))
    ax.set_ylabel("Normalized first-layer weight magnitude")
    ax.set_title("Input Channel Importance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "paper_feature_importance.jpg", dpi=cfg.paper_figure_dpi, bbox_inches="tight")
    plt.close(fig)


class SegmentationWindowScalarWrapper(nn.Module):
    def __init__(self, model: nn.Module, start_frac: float = 0.35, stop_frac: float = 0.65):
        super().__init__()
        self.model = model
        self.start_frac = float(start_frac)
        self.stop_frac = float(stop_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = primary_segmentation_logits(self.model(x))
        n = logits.shape[-1]
        s = max(0, min(n - 1, int(round(self.start_frac * n))))
        e = max(s + 1, min(n, int(round(self.stop_frac * n))))
        return logits[:, :, s:e].mean(dim=(1, 2), keepdim=False).unsqueeze(1)


class SegmentationLogitEnsembleWrapper(nn.Module):
    """Frozen weighted probability ensemble used by SHAP and SmoothGrad."""

    def __init__(self, models: Sequence[nn.Module], weights=None):
        super().__init__()
        self.models = nn.ModuleList(list(models))
        self.register_buffer("weights", torch.tensor(normalized_segmentation_ensemble_weights(len(models), weights), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [primary_segmentation_logits(model(x)) for model in self.models]
        return (torch.stack([torch.sigmoid(x) for x in logits], dim=0) * self.weights[:, None, None, None]).sum(dim=0)


def explainability_plot_context(cfg: Config) -> Dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": cfg.explainability_figure_dpi,
    }


def event_center_window(rec: Dict[str, Any], event: Tuple[float, float], cfg: Config) -> Tuple[int, int]:
    sfreq = float(rec["sfreq"])
    n = len(rec["truth_binary_stage"])
    window_len = max(8, int(round(float(cfg.explainability_window_sec) * sfreq)))
    center = int(round(0.5 * (float(event[0]) + float(event[1])) * sfreq))
    start = max(0, min(max(0, n - window_len), center - window_len // 2))
    stop = min(n, start + window_len)
    return start, stop


def event_local_slice(event: Tuple[float, float], start: int, stop: int, sfreq: float) -> Tuple[int, int]:
    s = max(0, int(round(float(event[0]) * sfreq)) - start)
    e = min(stop - start, int(round(float(event[1]) * sfreq)) - start)
    if e <= s:
        center = max(0, min(stop - start - 1, int(round(0.5 * (float(event[0]) + float(event[1])) * sfreq)) - start))
        s = max(0, center - 8)
        e = min(stop - start, center + 9)
    return s, max(s + 1, e)


def build_explainability_window(rec: Dict[str, Any], event: Tuple[float, float], cfg: Config) -> Tuple[np.ndarray, int, int]:
    start, stop = event_center_window(rec, event, cfg)
    window_len = max(8, int(round(float(cfg.explainability_window_sec) * float(rec["sfreq"]))))
    x = build_segmentation_input_window(rec, start, stop, window_len, cfg)
    return x, start, stop


def shap_values_to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.asarray(arr, dtype=np.float32)


def compute_shap_values(
    model: nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
    target_slice: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    if shap is None:
        logger.warning("SHAP is not installed; skipping SHAP explainability outputs.")
        return None
    if len(background) == 0 or len(samples) == 0:
        return None
    n = samples.shape[-1]
    target = target_slice or (int(0.35 * n), int(0.65 * n))
    wrapped = SegmentationWindowScalarWrapper(model, target[0] / n, target[1] / n).to(device).eval()
    bg = torch.from_numpy(np.ascontiguousarray(background, dtype=np.float32)).to(device)
    xs = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to(device)
    try:
        explainer = shap.GradientExplainer(wrapped, bg)
        values = explainer.shap_values(xs)
    except Exception as exc:
        logger.warning("GradientExplainer failed; skipping SHAP outputs (%s).", exc)
        return None
    return shap_values_to_numpy(values)


def compute_smoothgrad_saliency(
    model: nn.Module,
    x_np: np.ndarray,
    target_slice: Tuple[int, int],
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x_base = torch.from_numpy(np.ascontiguousarray(x_np[None, :, :], dtype=np.float32)).to(device)
    ch_std = torch.std(x_base, dim=-1, keepdim=True).clamp_min(1e-6)
    s, e = target_slice
    total = torch.zeros_like(x_base)
    n_samples = max(1, int(cfg.explainability_smoothgrad_samples))
    for _ in range(n_samples):
        noise = torch.randn_like(x_base) * ch_std * float(cfg.explainability_smoothgrad_noise_std)
        x = (x_base + noise).detach().requires_grad_(True)
        logits = primary_segmentation_logits(model(x))
        score = logits[:, :, s:e].mean()
        total += torch.autograd.grad(score, x)[0].detach().abs()
    sal = (total / float(n_samples))[0].detach().cpu().numpy().astype(np.float32)
    denom = np.max(sal)
    if denom > 0:
        sal = sal / denom
    return sal


def match_prediction_events(
    true_events: List[Tuple[float, float]],
    pred_events: List[Tuple[float, float]],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    matched_true: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    fp: List[int] = []
    for p_idx, pred in enumerate(pred_events):
        best_idx = -1
        best_iou = 0.0
        for t_idx, true_event in enumerate(true_events):
            if t_idx in matched_true:
                continue
            score = interval_iou(pred, true_event)
            if score > best_iou:
                best_iou = score
                best_idx = t_idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched_true.add(best_idx)
            matches.append((p_idx, best_idx, best_iou))
        else:
            fp.append(p_idx)
    fn = [idx for idx in range(len(true_events)) if idx not in matched_true]
    return matches, fp, fn


def select_explainability_examples(
    test_ids: Sequence[str],
    records: Dict[str, Any],
    timelines: Dict[str, np.ndarray],
    decoded_events: Dict[str, List[Tuple[float, float]]],
    cfg: Config,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    target_count = max(1, int(cfg.explainability_examples_per_type))
    buckets: Dict[str, List[Dict[str, Any]]] = {"tp": [], "fp": [], "fn": []}
    for sid in test_ids:
        rec = records[sid]
        true_events = list(rec["true_events_stage"])
        pred_events = decoded_events.get(sid, [])
        prob = timelines[sid]
        matches, fp_idx, fn_idx = match_prediction_events(true_events, pred_events, cfg.event_iou_thresholds[0])
        for p_idx, t_idx, iou in matches:
            event = pred_events[p_idx]
            s = max(0, int(round(event[0] * rec["sfreq"])))
            e = min(len(prob), int(round(event[1] * rec["sfreq"])))
            score = float(np.max(prob[s:e])) if e > s else 0.0
            buckets["tp"].append({"type": "tp", "subject_id": sid, "event": event, "matched_true_event": true_events[t_idx], "score": score, "iou": iou})
        for p_idx in fp_idx:
            event = pred_events[p_idx]
            s = max(0, int(round(event[0] * rec["sfreq"])))
            e = min(len(prob), int(round(event[1] * rec["sfreq"])))
            score = float(np.max(prob[s:e])) if e > s else 0.0
            buckets["fp"].append({"type": "fp", "subject_id": sid, "event": event, "matched_true_event": None, "score": score, "iou": 0.0})
        for t_idx in fn_idx:
            event = true_events[t_idx]
            s = max(0, int(round(event[0] * rec["sfreq"])))
            e = min(len(prob), int(round(event[1] * rec["sfreq"])))
            score = float(np.max(prob[s:e])) if e > s else 0.0
            buckets["fn"].append({"type": "fn", "subject_id": sid, "event": event, "matched_true_event": event, "score": score, "iou": 0.0})
    for key in ("tp", "fp", "fn"):
        ordered = sorted(buckets[key], key=lambda row: row["score"], reverse=(key != "fn"))
        examples.extend(ordered[:target_count])
    return examples


def save_explainability_bar_plot(df: pd.DataFrame, path: Path, title: str, ylabel: str, cfg: Config) -> None:
    if not plotting_enabled() or df.empty:
        return
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.bar(df["channel"], df["importance"], yerr=df["fold_std"].fillna(0) if "fold_std" in df else None, capsize=3, color="#1f77b4")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0.0, max(0.01, float(df["importance"].max()) * 1.15))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(path).with_suffix(".jpg"), dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)


def save_local_explainability_plots(
    outdir: Path,
    example: Dict[str, Any],
    rec: Dict[str, Any],
    x_np: np.ndarray,
    start: int,
    stop: int,
    prob_window: np.ndarray,
    pred_binary_window: np.ndarray,
    shap_window: Optional[np.ndarray],
    smoothgrad: np.ndarray,
    cfg: Config,
) -> None:
    if not plotting_enabled():
        return
    sfreq = float(rec["sfreq"])
    names = segmentation_channel_names(cfg)
    t = np.arange(x_np.shape[-1], dtype=np.float32) / sfreq
    true = rec["truth_binary_stage"][start:stop]
    if len(true) < len(t):
        true = np.pad(true, (0, len(t) - len(true)), mode="constant")
    true = true[:len(t)]
    sal = np.max(smoothgrad, axis=0)
    with plt.rc_context(explainability_plot_context(cfg)):
        fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.0), sharex=True)
        axes[0].plot(t, x_np[0], color="#222222", linewidth=0.8, label="Raw EEG")
        axes[0].set_ylabel("EEG")
        axes[0].legend(loc="upper right")
        if x_np.shape[0] > 1:
            axes[1].plot(t, x_np[1], color="#1f77b4", linewidth=0.8, label=names[1])
        else:
            axes[1].plot(t, x_np[0], color="#1f77b4", linewidth=0.8, label="Input")
        axes[1].set_ylabel("Band")
        axes[1].legend(loc="upper right")
        axes[2].plot(t, prob_window[:len(t)], color="#d62728", linewidth=1.0, label="Predicted probability")
        axes[2].plot(t, true, color="#2ca02c", linewidth=1.0, label="Ground truth")
        axes[2].plot(t, pred_binary_window[:len(t)], color="#9467bd", linewidth=0.9, label="Prediction")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("Mask")
        axes[2].legend(loc="upper right", ncol=3)
        axes[3].plot(t, sal[:len(t)], color="#ff7f0e", linewidth=1.0, label="SmoothGrad saliency")
        if shap_window is not None:
            shap_sal = np.mean(np.abs(shap_window), axis=0)
            denom = np.max(shap_sal)
            if denom > 0:
                shap_sal = shap_sal / denom
            axes[3].plot(t, shap_sal[:len(t)], color="#1f77b4", linewidth=0.9, alpha=0.8, label="SHAP |value|")
        axes[3].set_ylabel("Importance")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylim(-0.02, 1.05)
        axes[3].legend(loc="upper right")
        for ax in axes:
            ax.grid(alpha=0.22)
        title = f"{example['type'].upper()} explanation: {example['subject_id']}"
        fig.suptitle(title, y=0.995)
        fig.tight_layout()
        fig.savefig(outdir / f"paper_{example['type']}_explanation.jpg",
                    dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
        plt.close(fig)



def run_segmentation_explainability(
    models: Sequence[nn.Module],
    train_windows: List[Tuple[str, int, int]],
    test_ids: Sequence[str],
    records: Dict[str, Any],
    timelines: Dict[str, np.ndarray],
    decoded_events: Dict[str, List[Tuple[float, float]]],
    cfg: Config,
    fold_dir: Path,
    device: torch.device,
    logger: logging.Logger,
    member_weights: Optional[Sequence[float]] = None,
) -> None:
    if not cfg.run_explainability:
        return
    if not models:
        return
    explain_dir = ensure_dir(fold_dir / "explainability")
    model = SegmentationLogitEnsembleWrapper(models, member_weights).to(device).eval()
    rng = np.random.default_rng(int(cfg.seed))
    window_len = max(8, int(round(float(cfg.explainability_window_sec) * float(cfg.resample_hz))))
    bg_windows = list(train_windows)
    if len(bg_windows) > cfg.explainability_background_windows:
        chosen = rng.choice(len(bg_windows), size=int(cfg.explainability_background_windows), replace=False)
        bg_windows = [bg_windows[int(i)] for i in chosen]
    background = []
    for sid, start, stop in bg_windows:
        background.append(build_segmentation_input_window(records[sid], start, min(stop, start + window_len), window_len, cfg))
    if not background:
        logger.warning("Explainability skipped for %s: no background windows.", fold_dir.name)
        return
    background_np = np.stack(background, axis=0).astype(np.float32)

    examples = [ex for sid in test_ids for ex in select_explainability_examples([sid], records, timelines, decoded_events, cfg)[:max(1, cfg.explainability_shap_windows)]]
    save_json(examples, explain_dir / "selected_examples.json")
    sample_windows: List[np.ndarray] = []
    sample_meta: List[Dict[str, Any]] = []
    for example in examples:
        rec = records[example["subject_id"]]
        x_np, start, stop = build_explainability_window(rec, example["event"], cfg)
        sample_windows.append(x_np)
        sample_meta.append({**example, "window_start_sample": int(start), "window_stop_sample": int(stop)})
    if not sample_windows:
        logger.warning("Explainability skipped for %s: no TP/FP/FN examples found.", fold_dir.name)
        return
    samples_np = np.stack(sample_windows, axis=0).astype(np.float32)
    shap_parts = []
    for idx, meta in enumerate(sample_meta):
        target = event_local_slice(meta["event"], meta["window_start_sample"], meta["window_stop_sample"], records[meta["subject_id"]]["sfreq"])
        values = compute_shap_values(model, background_np, samples_np[idx:idx + 1], device, logger, target)
        if values is None:
            shap_parts = []
            break
        shap_parts.append(values[0])
    shap_np = np.stack(shap_parts) if shap_parts else None
    save_json({"shap_available": shap is not None, "shap_completed": shap_np is not None,
               "target": "event-interval mean weighted probability before calibration",
               "scope": "local input window; post-hoc, not used for selection"},
              explain_dir / "attribution_status.json")
    names = segmentation_channel_names(cfg)
    if shap_np is not None:
        global_vals = np.mean(np.abs(shap_np), axis=(0, 2))
        denom = float(np.sum(global_vals))
        if denom > 0:
            global_vals = global_vals / denom
        global_df = pd.DataFrame({"channel": names, "importance": global_vals[:len(names)]})
        global_df.to_csv(explain_dir / "global_shap_feature_importance.csv", index=False)
    else:
        global_df = pd.DataFrame(columns=["channel", "importance"])

    local_rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(sample_meta):
        rec = records[example["subject_id"]]
        x_np = sample_windows[idx]
        start = int(example["window_start_sample"])
        stop = int(example["window_stop_sample"])
        local_slice = event_local_slice(example["event"], start, stop, float(rec["sfreq"]))
        smoothgrad = compute_smoothgrad_saliency(model, x_np, local_slice, cfg, device)
        prob = timelines[example["subject_id"]][start:stop]
        pred_binary = np.zeros_like(prob, dtype=np.uint8)
        for ps, pe in decoded_events.get(example["subject_id"], []):
            s = max(0, int(round(ps * rec["sfreq"])) - start)
            e = min(len(pred_binary), int(round(pe * rec["sfreq"])) - start)
            if e > s:
                pred_binary[s:e] = 1
        if len(prob) < x_np.shape[-1]:
            prob = np.pad(prob, (0, x_np.shape[-1] - len(prob)), mode="constant")
            pred_binary = np.pad(pred_binary, (0, x_np.shape[-1] - len(pred_binary)), mode="constant")
        shap_window = shap_np[idx] if shap_np is not None and idx < len(shap_np) else None
        np.savez_compressed(explain_dir / f"attribution_{idx:04d}.npz", input=x_np,
                            smoothgrad=smoothgrad, shap=shap_window if shap_window is not None else np.empty(0))
        for ch_idx, ch_name in enumerate(names):
            local_rows.append(
                {
                    "type": example["type"],
                    "subject_id": example["subject_id"],
                    "channel": ch_name,
                    "smoothgrad_mean": float(np.mean(smoothgrad[ch_idx])),
                    "smoothgrad_max": float(np.max(smoothgrad[ch_idx])),
                    "event_relevance_fraction": float(smoothgrad[ch_idx, local_slice[0]:local_slice[1]].sum() / max(smoothgrad[ch_idx].sum(), 1e-12)),
                    "shap_smoothgrad_temporal_spearman": float(scipy.stats.spearmanr(np.abs(shap_window[ch_idx]), smoothgrad[ch_idx]).statistic) if shap_window is not None and np.std(smoothgrad[ch_idx]) > 0 and np.std(shap_window[ch_idx]) > 0 else float("nan"),
                    "shap_abs_mean": float(np.mean(np.abs(shap_window[ch_idx]))) if shap_window is not None else float("nan"),
                }
            )
        # Deterministic first available example per category across folds.
        # Keep every numerical attribution for subject-wise statistics, but cap
        # local paper figures at one TP, one FP and one FN for the entire run.
        paper_dir = ensure_dir(fold_dir.parent / "explainability_paper")
        paper_path = paper_dir / f"paper_{example['type']}_explanation.jpg"
        if not paper_path.exists():
            save_local_explainability_plots(
                paper_dir, example, rec, x_np, start, stop, prob,
                pred_binary, shap_window, smoothgrad, cfg,
            )
            save_json({**example, "fold": fold_dir.name,
                       "selection": "first available example in deterministic fold/subject order"},
                      paper_dir / f"paper_{example['type']}_example.json")
    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(explain_dir / "local_explainability_scores.csv", index=False)
    if not local_df.empty:
        smooth_df = local_df.groupby("channel", as_index=False)["smoothgrad_mean"].mean().rename(columns={"smoothgrad_mean": "importance"})
        smooth_df.to_csv(explain_dir / "global_smoothgrad_feature_importance.csv", index=False)
    logger.info("Saved XAI statistics for %s; paper figures are limited to 3 event composites plus 3 run summaries, JPG only.", fold_dir.name)


def aggregate_segmentation_explainability(outdir: Path, cfg: Config, logger: logging.Logger) -> None:
    """Create paper-level SHAP and SmoothGrad summaries across completed folds."""
    if not cfg.run_explainability:
        return
    method_files = {
        "shap": "global_shap_feature_importance.csv",
        "smoothgrad": "global_smoothgrad_feature_importance.csv",
    }
    for method, filename in method_files.items():
        frames: List[pd.DataFrame] = []
        for path in sorted(outdir.glob(f"fold_*/explainability/{filename}")):
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Could not read explainability summary %s: %s", path, exc)
                continue
            if not {"channel", "importance"}.issubset(frame.columns):
                continue
            frame = frame[["channel", "importance"]].copy()
            frame["fold"] = path.parents[1].name
            frames.append(frame)
        if not frames:
            logger.warning("No completed %s explainability summaries were available for aggregation.", method.upper())
            continue
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(outdir / f"paper_{method}_feature_importance_by_fold.csv", index=False)
        summary = (
            all_df.groupby("channel", as_index=False)["importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "importance", "std": "fold_std", "count": "n_folds"})
        )
        total = float(summary["importance"].sum())
        if total > 0:
            summary["importance"] = summary["importance"] / total
            summary["fold_std"] = summary["fold_std"] / total
        summary.to_csv(outdir / f"paper_global_{method}_feature_importance.csv", index=False)
        title = "Cross-fold Global SHAP Feature Importance" if method == "shap" else "Cross-fold Global SmoothGrad Feature Importance"
        ylabel = "Normalized |SHAP|" if method == "shap" else "Normalized saliency"
        save_explainability_bar_plot(
            summary,
            ensure_dir(outdir / "explainability_paper") / f"paper_global_{method}_feature_importance.jpg",
            title,
            ylabel,
            cfg,
        )
    logger.info("Saved cross-fold SHAP and SmoothGrad paper summaries to %s", outdir)


def parse_dreams_scoring(path: Path) -> List[Tuple[float, float]]:
    if not path.exists():
        return []
    events: List[Tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            onset = float(parts[0])
            duration = float(parts[1])
        except ValueError:
            continue
        if duration > 0:
            events.append((onset, onset + duration))
    return merge_intervals(events)


def load_dreams_record(excerpt_idx: int, data_dir: Path, cfg: Config, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    raise RuntimeError("External dataset access is disabled in the MASS-SS2-only pipeline.")
    edf_path = data_dir / f"excerpt{excerpt_idx}.edf"
    if not edf_path.exists():
        logger.warning("DREAMS excerpt %d missing EDF file: %s", excerpt_idx, edf_path)
        return None
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    best_channel = select_best_channel_from_priority(
        raw.ch_names,
        [cfg.channel_target, "C3-A1", "CZ-A1", "CZA1", "CZ2-A1", "O1-A1", "O2-A1", "FP1-A1", "FP2-A1"],
        logger,
        f"DREAMS excerpt {excerpt_idx}",
    )
    raw.pick([best_channel])
    sfreq = float(raw.info["sfreq"])
    if not math.isclose(sfreq, cfg.resample_hz):
        raw.resample(cfg.resample_hz, npad="auto", verbose="ERROR")
        sfreq = cfg.resample_hz
    data = raw.get_data()[0].astype(np.float32)
    e1 = parse_dreams_scoring(data_dir / f"Visual_scoring1_excerpt{excerpt_idx}.txt")
    e2 = parse_dreams_scoring(data_dir / f"Visual_scoring2_excerpt{excerpt_idx}.txt")
    union_events = merge_intervals(e1 + e2)
    e1_binary = events_to_binary(e1, len(data), sfreq)
    e2_binary = events_to_binary(e2, len(data), sfreq)
    union_binary = events_to_binary(union_events, len(data), sfreq)
    union_soft = build_soft_boundary_labels(
        union_binary,
        sfreq,
        cfg.segmentation_soft_label_shoulder_sec,
        cfg.segmentation_soft_label_edge_value,
    )
    union_center, union_boundary = build_center_boundary_labels(
        union_binary,
        sfreq,
        cfg.segmentation_center_label_width_sec,
        cfg.segmentation_boundary_label_width_sec,
    )
    union_confidence, annotator_confidence = build_annotator_confidence_targets(
        union_soft,
        e1_binary,
        e2_binary,
        sfreq,
        cfg.segmentation_single_expert_target_weight,
        cfg.segmentation_annotator_agreement_tolerance_sec,
    )
    sigma = compute_sigma_band(data, sfreq, cfg.sigma_low_hz, cfg.sigma_high_hz)
    slow_sigma = compute_sigma_band(
        data,
        sfreq,
        cfg.segmentation_slow_sigma_low_hz,
        cfg.segmentation_slow_sigma_high_hz,
    )
    teo = compute_teager_energy(sigma)
    envelope = compute_envelope(sigma)
    return {
        "subject_id": f"dreams_excerpt{excerpt_idx}",
        "channel": best_channel,
        "sfreq": sfreq,
        "events_e1": e1,
        "events_e2": e2,
        "true_events_stage": union_events,
        "true_events_e1_stage": e1,
        "true_events_e2_stage": e2,
        "truth_binary_stage": union_binary.astype(np.uint8),
        "truth_binary_e1_stage": e1_binary.astype(np.uint8),
        "truth_binary_e2_stage": e2_binary.astype(np.uint8),
        "truth_soft_stage": union_soft.astype(np.float32),
        "truth_annotator_confidence_stage": union_confidence.astype(np.float32),
        "truth_center_stage": union_center.astype(np.float32),
        "truth_boundary_stage": union_boundary.astype(np.float32),
        "truth_center_annotator_confidence_stage": (union_center * annotator_confidence).astype(np.float32),
        "truth_boundary_annotator_confidence_stage": (union_boundary * annotator_confidence).astype(np.float32),
        "segmentation_raw": prepare_segmentation_channel(data, cfg),
        "segmentation_sigma": prepare_segmentation_channel(sigma, cfg),
        "segmentation_slow_sigma": prepare_segmentation_channel(slow_sigma, cfg),
        "segmentation_teo": prepare_segmentation_teo_channel(teo),
        "segmentation_envelope": prepare_segmentation_channel(envelope, cfg),
        # DREAMS excerpts do not guarantee a homologous second central lead.
        # Zero fallback preserves compatibility with multi-channel MASS models.
        "segmentation_secondary_raw": np.zeros_like(data, dtype=np.float32),
        "segmentation_secondary_sigma": np.zeros_like(data, dtype=np.float32),
        "segmentation_secondary_slow_sigma": np.zeros_like(data, dtype=np.float32),
        "segmentation_secondary_envelope": np.zeros_like(data, dtype=np.float32),
        "candidate_raw": data.astype(np.float32),
        "candidate_sigma": sigma.astype(np.float32),
        "candidate_teo": compute_teager_energy(sigma).astype(np.float32),
        "candidate_envelope": envelope.astype(np.float32),
        "stage_segment_bounds": [(0, int(len(data)))],
        "retained_len_samples": int(len(data)),
        "n_true_events_stage": int(len(union_events)),
    }


def load_dreams_records(data_dir: Path, cfg: Config, logger: logging.Logger) -> Dict[str, Any]:
    raise RuntimeError("External dataset access is disabled in the MASS-SS2-only pipeline.")
    records: Dict[str, Any] = {}
    for edf_path in sorted(data_dir.glob("excerpt*.edf")):
        match = re.search(r"excerpt(\d+)\.edf$", edf_path.name, flags=re.IGNORECASE)
        if not match:
            continue
        rec = load_dreams_record(int(match.group(1)), data_dir, cfg, logger)
        if rec is not None:
            records[rec["subject_id"]] = rec
    return records


def mean_segmentation_params_from_checkpoint_dir(checkpoint_dir: Path, cfg: Config) -> SegmentationParams:
    rows = []
    for path in sorted(checkpoint_dir.glob("fold_*/segmentation_params.json")):
        try:
            rows.append(load_json(path))
        except Exception:
            continue
    if not rows:
        return SegmentationParams(
            probability_threshold=cfg.external_default_threshold,
            final_merge_gap_sec=cfg.external_default_merge_gap_sec,
            final_min_sec=cfg.external_default_min_sec,
            adaptive_threshold_quantile=cfg.external_default_adaptive_quantile,
        )
    def mean_value(key: str, default: float) -> float:
        values = [float(row.get(key, default)) for row in rows if row.get(key, None) is not None]
        return float(np.mean(values)) if values else default
    return SegmentationParams(
        probability_threshold=mean_value("probability_threshold", cfg.external_default_threshold),
        final_merge_gap_sec=mean_value("final_merge_gap_sec", cfg.external_default_merge_gap_sec),
        final_min_sec=mean_value("final_min_sec", cfg.external_default_min_sec),
        adaptive_threshold_quantile=mean_value("adaptive_threshold_quantile", cfg.external_default_adaptive_quantile),
        boundary_gap=mean_value("boundary_gap", 0.0),
        boundary_max_extend_sec=mean_value("boundary_max_extend_sec", 0.0),
        boundary_trim_sec=mean_value("boundary_trim_sec", 0.0),
        event_min_mean_prob=mean_value("event_min_mean_prob", 0.0),
        event_min_peak_prob=mean_value("event_min_peak_prob", 0.0),
        final_max_sec=mean_value("final_max_sec", 0.0),
    )


def load_external_segmentation_models(checkpoint_dir: Path, cfg: Config, device: torch.device, logger: logging.Logger) -> List[nn.Module]:
    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*/best_segmentation_model_seed*.pt"))
    if cfg.external_max_models > 0:
        checkpoint_paths = checkpoint_paths[: int(cfg.external_max_models)]
    if not checkpoint_paths:
        raise RuntimeError(f"No segmentation checkpoints found under {checkpoint_dir}")
    models: List[nn.Module] = []
    for path in checkpoint_paths:
        model = build_segmentation_model(cfg, device)
        state = torch.load(path, map_location=device)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    logger.info("Loaded %d external segmentation checkpoints from %s", len(models), checkpoint_dir)
    return models


def parse_external_record_ids(value: str, records: Dict[str, Any]) -> List[str]:
    if not value:
        return []
    out: List[str] = []
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"dreams_excerpt{int(item)}"
        elif item.lower().startswith("excerpt") and item[7:].isdigit():
            item = f"dreams_excerpt{int(item[7:])}"
        if item in records and item not in out:
            out.append(item)
    return out


def default_external_calibration_split(records: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    ids = sorted(records)
    calibration_ids = [sid for idx, sid in enumerate(ids) if idx % 2 == 0]
    test_ids = [sid for idx, sid in enumerate(ids) if idx % 2 == 1]
    return calibration_ids, test_ids


def evaluate_external_dreams_records(
    records: Dict[str, Any],
    record_ids: Sequence[str],
    timelines: Dict[str, np.ndarray],
    seg_params: SegmentationParams,
    cfg: Config,
    outdir: Path,
    prefix: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    subject_rows: List[Dict[str, Any]] = []
    reference_subject_rows: List[Dict[str, Any]] = []
    event_details: List[Dict[str, Any]] = []
    for sid in sorted(record_ids):
        rec = records[sid]
        probability_timeline = timelines[sid]
        subject_prob_thr = resolve_segmentation_probability_threshold(
            probability_timeline,
            seg_params.probability_threshold,
            seg_params.adaptive_threshold_quantile,
            cfg.segmentation_adaptive_threshold_max_adjustment,
        )
        pred_events, pred_binary = decode_segmentation_events(
            probability_timeline,
            rec["sfreq"],
            subject_prob_thr,
            seg_params.final_min_sec,
            seg_params.final_merge_gap_sec,
            seg_params.boundary_gap,
            seg_params.boundary_max_extend_sec,
            seg_params.boundary_trim_sec,
            seg_params.event_min_mean_prob,
            seg_params.event_min_peak_prob,
            seg_params.final_max_sec,
        )
        strict = event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])
        tolerant = tolerant_event_metrics(rec["true_events_stage"], pred_events, cfg.tolerant_match_sec)
        sample = sample_level_metrics(rec["truth_binary_stage"], pred_binary, probability_timeline)
        row = {
            "subject_id": sid,
            "split": prefix,
            "n_true_events": len(rec["true_events_stage"]),
            "n_pred_events": len(pred_events),
            "strict_event_precision": strict["precision"],
            "strict_event_recall": strict["recall"],
            "strict_event_f1": strict["f1"],
            "tolerant_event_precision": tolerant["precision"],
            "tolerant_event_recall": tolerant["recall"],
            "tolerant_event_f1": tolerant["f1"],
            "ACC": sample["ACC"],
            "SEN": sample["SEN"],
            "SPE": sample["SPE"],
            "KAPPA": sample["KAPPA"],
            "sample_f1": sample["sample_f1"],
            "sample_auc": sample.get("sample_auc", float("nan")),
            "sample_pr_auc": sample.get("sample_pr_auc", float("nan")),
            "prob_thr": subject_prob_thr,
        }
        subject_rows.append(row)
        reference_metrics = []
        for reference_name, reference in evaluation_references(rec).items():
            reference_row = subject_reference_metrics(
                sid,
                reference_name,
                reference["events"],
                reference["binary"],
                pred_events,
                pred_binary,
                probability_timeline,
                cfg,
            )
            reference_row["split"] = prefix
            reference_subject_rows.append(reference_row)
            reference_metrics.append(reference_row)
        event_details.append(
            {
                "subject_id": sid,
                "split": prefix,
                "n_true_events": len(rec["true_events_stage"]),
                "n_pred_events": len(pred_events),
                "strict": strict,
                "tolerant": tolerant,
                "references": reference_metrics,
            }
        )
        if cfg.save_candidates:
            save_json(pred_events, outdir / f"{prefix}_{sid}_pred_events.json")
        if cfg.save_probabilities:
            pd.DataFrame({"probability": probability_timeline}).to_csv(
                outdir / f"{prefix}_{sid}_probability_timeline.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "truth": rec["truth_binary_stage"],
                    "truth_expert1": rec["truth_binary_e1_stage"],
                    "truth_expert2": rec["truth_binary_e2_stage"],
                    "truth_union": rec["truth_binary_stage"],
                    "pred_binary": pred_binary,
                    "probability": probability_timeline,
                }
            ).to_csv(outdir / f"{prefix}_{sid}_vectors.csv", index=False)
    subject_df = pd.DataFrame(subject_rows)
    reference_df = pd.DataFrame(reference_subject_rows)
    reference_overall = aggregate_reference_metrics(reference_subject_rows) if reference_subject_rows else {}
    subject_df.to_csv(outdir / f"{prefix}_subject_metrics.csv", index=False)
    reference_df.to_csv(outdir / f"{prefix}_subject_metrics_by_reference.csv", index=False)
    save_json(reference_overall, outdir / f"{prefix}_overall_metrics_by_reference.json")
    pd.DataFrame(reference_overall.values()).to_csv(outdir / f"{prefix}_overall_metrics_by_reference.csv", index=False)
    save_json(event_details, outdir / f"{prefix}_event_metrics.json")
    summary_rows: List[Dict[str, Any]] = []
    for col in [c for c in subject_df.columns if pd.api.types.is_numeric_dtype(subject_df[c])]:
        summary_rows.append({"metric": col, "mean": float(subject_df[col].mean()), "std": float(subject_df[col].std(ddof=0))})
    pd.DataFrame(summary_rows).to_csv(outdir / f"{prefix}_overall_summary.csv", index=False)
    logger.info(
        "DREAMS %s evaluation | union strict F1 mean=%.4f",
        prefix,
        float(subject_df["strict_event_f1"].mean()) if "strict_event_f1" in subject_df else float("nan"),
    )
    return subject_df, reference_df, reference_overall


def sample_level_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    cm = sk_metrics.confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(np.sum(cm), 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    kappa = sk_metrics.cohen_kappa_score(y_true, y_pred_binary)
    precision = sk_metrics.precision_score(y_true, y_pred_binary, zero_division=0)
    recall = sk_metrics.recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = sk_metrics.f1_score(y_true, y_pred_binary, zero_division=0)
    out: Dict[str, Any] = {
        "ACC": float(acc),
        "SEN": float(sen),
        "SPE": float(spe),
        "KAPPA": float(kappa),
        "sample_precision": float(precision),
        "sample_recall": float(recall),
        "sample_f1": float(f1),
        "cm": cm.tolist(),
    }
    if len(np.unique(y_true)) > 1:
        out["sample_auc"] = float(sk_metrics.roc_auc_score(y_true, prob))
        out["sample_pr_auc"] = float(sk_metrics.average_precision_score(y_true, prob))
    else:
        out["sample_auc"] = float("nan")
        out["sample_pr_auc"] = float("nan")
    return out


def split_for_inner_early_stopping(train_ids: List[str], seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = train_ids[:]
    rng.shuffle(ids)
    if len(ids) <= 1:
        return ids, ids
    n_val = max(1, int(round(0.2 * len(ids))))
    inner_val = ids[:n_val]
    inner_train = ids[n_val:]
    if not inner_train:
        inner_train = inner_val[:]
    return inner_train, inner_val


def get_candidate_cache_key(subject_id: str, B: float, T: float, mode: str) -> Tuple[str, float, float, str]:
    return subject_id, float(B), float(T), mode


def get_cached_candidates(
    subject_id: str,
    records: Dict[str, Any],
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
    cfg: Config,
    B: float,
    T: float,
) -> Dict[str, Any]:
    key = get_candidate_cache_key(subject_id, B, T, cfg.candidate_mode)
    if key not in candidate_cache:
        candidate_cache[key] = generate_candidates_for_record(records[subject_id], cfg, B, T)
    return candidate_cache[key]


def build_dataset_for_subjects(
    subject_ids: Sequence[str],
    records: Dict[str, Any],
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
    cfg: Config,
    B: float,
    T: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    X_list: List[np.ndarray] = []
    y_soft_list: List[np.ndarray] = []
    y_hard_list: List[np.ndarray] = []
    counts: List[Tuple[str, int]] = []
    for sid in subject_ids:
        bundle = get_cached_candidates(sid, records, candidate_cache, cfg, B, T)
        hard_labels, soft_labels = candidate_labels_for_record(bundle["events"], records[sid], lam, cfg)
        X_sub = get_candidate_window_matrix(records[sid], bundle, cfg)
        counts.append((sid, len(hard_labels)))
        if len(hard_labels) == 0:
            continue
        X_list.append(X_sub)
        y_soft_list.append(soft_labels)
        y_hard_list.append(hard_labels)
    if not X_list:
        return (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            counts,
        )
    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_soft_list, axis=0),
        np.concatenate(y_hard_list, axis=0),
        counts,
    )


def build_validation_probability_timelines(
    model: nn.Module,
    val_subject_ids: Sequence[str],
    records: Dict[str, Any],
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
    cfg: Config,
    device: torch.device,
    B: float,
    T: float,
    lam: float,
    norm_stats: Optional[NormalizationStats],
    calibrator: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    subject_probability_data: List[Dict[str, Any]] = []
    for sid in val_subject_ids:
        bundle = get_cached_candidates(sid, records, candidate_cache, cfg, B, T)
        hard_labels, _ = candidate_labels_for_record(bundle["events"], records[sid], lam, cfg)
        X_val = get_candidate_window_matrix(records[sid], bundle, cfg)
        X_val = apply_dataset_normalization(X_val, norm_stats, cfg.normalization_mode)
        if len(hard_labels):
            loader = make_candidate_loader(X_val, hard_labels, cfg, device, training=False)
            probs = apply_probability_calibrator(calibrator, predict_probabilities(model, loader, device))
        else:
            probs = np.zeros((0,), dtype=np.float32)
        timeline = build_probability_timeline(
            bundle["events"],
            probs,
            len(records[sid]["truth_binary_stage"]),
            records[sid]["sfreq"],
        )
        subject_probability_data.append(
            {
                "subject_id": sid,
                "timeline": timeline,
                "candidates": bundle["events"],
                "n_samples": len(records[sid]["truth_binary_stage"]),
                "candidate_probs": probs,
                "sfreq": records[sid]["sfreq"],
                "true_events": records[sid]["true_events_stage"],
                "stage_segment_bounds": records[sid].get("stage_segment_bounds"),
            }
        )
    return subject_probability_data


def evaluate_stage2_combo(
    subject_probability_data: Sequence[Dict[str, Any]],
    cfg: Config,
    prob_thr: float,
    hysteresis_gap: float,
    final_merge_gap_sec: float,
    final_min_sec: float,
    final_max_sec: float,
    adaptive_threshold_quantile: float,
) -> Dict[str, float]:
    f1s: List[float] = []
    recalls: List[float] = []
    for item in subject_probability_data:
        subject_prob_thr = resolve_subject_probability_threshold(
            item["candidate_probs"],
            prob_thr,
            adaptive_threshold_quantile,
            cfg.adaptive_threshold_max_adjustment,
            cfg.stage3_prob_threshold_floor,
            cfg.stage3_prob_threshold_cap,
        )
        pred_events, _ = decode_events_from_candidate_probabilities(
            item["candidates"],
            item["candidate_probs"],
            item["n_samples"],
            item["sfreq"],
            subject_prob_thr,
            hysteresis_gap,
            final_min_sec,
            final_max_sec,
            final_merge_gap_sec,
            cfg.stage3_support_smooth_sec,
            cfg.stage3_support_relative_threshold,
        )
        strict = event_metrics(item["true_events"], pred_events, cfg.event_iou_thresholds[0])
        f1s.append(strict["f1"])
        recalls.append(strict["recall"])
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    min_f1 = float(np.min(f1s)) if f1s else 0.0
    min_recall = float(np.min(recalls)) if recalls else 0.0
    score = (
        cfg.stage3_selection_mean_f1_weight * mean_f1
        + cfg.stage3_selection_mean_recall_weight * mean_recall
        + cfg.stage3_selection_min_f1_weight * min_f1
        + cfg.stage3_selection_min_recall_weight * min_recall
    )
    if mean_recall < cfg.stage3_recall_floor:
        score -= cfg.stage3_recall_floor_penalty * (cfg.stage3_recall_floor - mean_recall)
    if min_recall < cfg.stage3_recall_floor:
        score -= cfg.stage3_recall_floor_penalty * (cfg.stage3_recall_floor - min_recall)
    return {
        "score": float(score),
        "mean_f1": mean_f1,
        "mean_recall": mean_recall,
        "min_f1": min_f1,
        "min_recall": min_recall,
    }


def tune_candidate_parameters(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[float, float]:
    if not cfg.tune_candidate_params:
        return cfg.candidate_threshold_B, cfg.candidate_threshold_T
    best_B = cfg.candidate_threshold_B
    best_T = cfg.candidate_threshold_T
    best_score = -1.0
    best_components: Dict[str, float] = {}
    for B in cfg.candidate_B_grid:
        for T in cfg.candidate_T_grid:
            f1s: List[float] = []
            recalls: List[float] = []
            for sid in val_ids:
                bundle = get_cached_candidates(sid, records, candidate_cache, cfg, B, T)
                strict = event_metrics(records[sid]["true_events_stage"], bundle["events"], cfg.event_iou_thresholds[0])
                f1s.append(strict["f1"])
                recalls.append(strict["recall"])
            mean_f1 = float(np.mean(f1s)) if f1s else 0.0
            min_f1 = float(np.min(f1s)) if f1s else 0.0
            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            score = (
                cfg.candidate_tuning_mean_f1_weight * mean_f1
                + cfg.candidate_tuning_min_f1_weight * min_f1
                + cfg.candidate_tuning_mean_recall_weight * mean_recall
            )
            if score > best_score:
                best_score = score
                best_B = float(B)
                best_T = float(T)
                best_components = {
                    "mean_f1": mean_f1,
                    "min_f1": min_f1,
                    "mean_recall": mean_recall,
                }
    logger.info(
        "Best candidate params on validation: B=%.3f T=%.3f objective=%.4f meanF1=%.4f minF1=%.4f meanRecall=%.4f",
        best_B,
        best_T,
        best_score,
        best_components.get("mean_f1", float("nan")),
        best_components.get("min_f1", float("nan")),
        best_components.get("mean_recall", float("nan")),
    )
    return best_B, best_T


def tune_stage2_parameters(
    fold_dir: Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    records: Dict[str, Any],
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
    cfg: Config,
    B: float,
    T: float,
    device: torch.device,
    logger: logging.Logger,
) -> Stage2Params:
    best_params = Stage2Params(
        lambda_thr=cfg.lambda_grid[0],
        probability_threshold=cfg.threshold_grid[0],
        hysteresis_gap=cfg.hysteresis_gap_grid[0],
        final_merge_gap_sec=cfg.final_merge_gap_grid[0],
        final_min_sec=cfg.final_min_sec_grid[0],
        final_max_sec=cfg.final_max_sec_grid[0],
        adaptive_threshold_quantile=cfg.adaptive_threshold_quantile_grid[0],
    )
    best_score = -1.0
    found_valid_combo = False
    inner_train_ids, inner_val_ids = split_for_inner_early_stopping(list(train_ids), cfg.seed)
    tuning_rows: List[Dict[str, Any]] = []
    decode_combo_count = (
        len(cfg.threshold_grid)
        * len(cfg.hysteresis_gap_grid)
        * len(cfg.final_merge_gap_grid)
        * len(cfg.final_min_sec_grid)
        * len(cfg.final_max_sec_grid)
        * len(cfg.adaptive_threshold_quantile_grid)
    )

    logger.info("Stage-2 tuning rationale: %s", " | ".join(stage2_tuning_justification()))

    for lam in cfg.lambda_grid:
        dataset_start = time.perf_counter()
        X_train, y_train_soft, y_train_hard, _ = build_dataset_for_subjects(inner_train_ids, records, candidate_cache, cfg, B, T, lam)
        X_inner_val, _, y_inner_val_hard, _ = build_dataset_for_subjects(inner_val_ids, records, candidate_cache, cfg, B, T, lam)
        logger.info(
            "Stage-2 tuning | lambda=%.3f | candidate windows built in %.1fs",
            lam,
            time.perf_counter() - dataset_start,
        )
        if len(y_train_hard) < 16 or len(np.unique(y_train_hard > 0.5)) < 2 or len(y_inner_val_hard) == 0:
            tuning_rows.append(
                {
                    "lambda": float(lam),
                    "prob_thr": float("nan"),
                    "hysteresis_gap": float("nan"),
                    "final_merge_gap_sec": float("nan"),
                    "final_min_sec": float("nan"),
                    "final_max_sec": float("nan"),
                    "adaptive_threshold_quantile": float("nan"),
                    "val_strict_f1": float("nan"),
                    "train_candidates": int(len(y_train_hard)),
                    "inner_val_candidates": int(len(y_inner_val_hard)),
                    "train_positive_labels": int(np.sum(y_train_hard > 0.5)),
                    "train_negative_labels": int(np.sum(y_train_hard <= 0.5)),
                    "skipped": True,
                }
            )
            continue
        found_valid_combo = True
        logger.info(
            "Stage-2 tuning | lambda=%.3f | train_candidates=%d | inner_val_candidates=%d",
            lam,
            len(y_train_hard),
            len(y_inner_val_hard),
        )

        norm_stats = compute_dataset_normalization_stats(X_train) if cfg.normalization_mode == "dataset_train" else None
        X_train_n = apply_dataset_normalization(X_train, norm_stats, cfg.normalization_mode)
        X_inner_val_n = apply_dataset_normalization(X_inner_val, norm_stats, cfg.normalization_mode)

        train_sampler = build_balanced_sampler(y_train_hard, cfg.balanced_sampling)

        train_loader = make_candidate_loader(
            X_train_n,
            y_train_soft,
            cfg,
            device,
            training=True,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )
        val_loader = make_candidate_loader(X_inner_val_n, y_inner_val_hard, cfg, device, training=False)

        model = CandidateNet(
            in_channels=X_train_n.shape[1],
            hidden_channels=cfg.model_channels,
            kernel_size=cfg.model_kernel_size,
            dropout=cfg.model_dropout,
            use_residual=cfg.model_use_residual,
            use_temporal_attention_gate=cfg.model_use_temporal_attention_gate,
        ).to(device)
        pos_weight = compute_pos_weight(y_train_hard, cfg.positive_weight_max, device)
        ckpt = fold_dir / f"tuning_lambda_{lam:.2f}.pt"
        model, _ = train_model(model, train_loader, val_loader, cfg, device, logger, ckpt, cfg.tuning_epochs, pos_weight)
        inner_val_probs_raw = predict_probabilities(model, val_loader, device)
        calibrator = fit_probability_calibrator(
            inner_val_probs_raw,
            y_inner_val_hard,
            cfg,
            logger,
            f"fold_{fold_dir.name}_lambda_{lam:.2f}_inner_val",
        )
        logger.info(
            "Stage-2 tuning | lambda=%.3f | evaluating %d decoding combinations on outer validation",
            lam,
            decode_combo_count,
        )
        subject_probability_data = build_validation_probability_timelines(
            model,
            val_ids,
            records,
            candidate_cache,
            cfg,
            device,
            B,
            T,
            lam,
            norm_stats,
            calibrator,
        )

        for prob_thr in cfg.threshold_grid:
            for hysteresis_gap in cfg.hysteresis_gap_grid:
                for final_merge_gap_sec in cfg.final_merge_gap_grid:
                    for final_min_sec in cfg.final_min_sec_grid:
                        for final_max_sec in cfg.final_max_sec_grid:
                            if final_min_sec > final_max_sec:
                                continue
                            for adaptive_threshold_quantile in cfg.adaptive_threshold_quantile_grid:
                                combo = evaluate_stage2_combo(
                                    subject_probability_data,
                                    cfg,
                                    prob_thr,
                                    hysteresis_gap,
                                    final_merge_gap_sec,
                                    final_min_sec,
                                    final_max_sec,
                                    adaptive_threshold_quantile,
                                )
                                tuning_rows.append(
                                    {
                                        "lambda": float(lam),
                                        "prob_thr": float(prob_thr),
                                        "hysteresis_gap": float(hysteresis_gap),
                                        "final_merge_gap_sec": float(final_merge_gap_sec),
                                        "final_min_sec": float(final_min_sec),
                                        "final_max_sec": float(final_max_sec),
                                        "adaptive_threshold_quantile": float(adaptive_threshold_quantile),
                                        "val_stage3_score": float(combo["score"]),
                                        "val_strict_f1": float(combo["mean_f1"]),
                                        "val_strict_recall": float(combo["mean_recall"]),
                                        "val_min_subject_f1": float(combo["min_f1"]),
                                        "val_min_subject_recall": float(combo["min_recall"]),
                                        "train_candidates": int(len(y_train_hard)),
                                        "inner_val_candidates": int(len(y_inner_val_hard)),
                                        "train_positive_labels": int(np.sum(y_train_hard > 0.5)),
                                        "train_negative_labels": int(np.sum(y_train_hard <= 0.5)),
                                        "skipped": False,
                                    }
                                )
                                if combo["score"] > best_score:
                                    best_score = combo["score"]
                                    best_params = Stage2Params(
                                        lambda_thr=float(lam),
                                        probability_threshold=float(prob_thr),
                                        hysteresis_gap=float(hysteresis_gap),
                                        final_merge_gap_sec=float(final_merge_gap_sec),
                                        final_min_sec=float(final_min_sec),
                                        final_max_sec=float(final_max_sec),
                                        adaptive_threshold_quantile=float(adaptive_threshold_quantile),
                                    )
        if tuning_rows:
            partial_df = pd.DataFrame(tuning_rows)
            partial_df.to_csv(fold_dir / "stage2_tuning.csv", index=False)
            partial_df.to_csv(fold_dir / "lambda_threshold_tuning.csv", index=False)
            logger.info(
                "Stage-2 tuning | lambda=%.3f complete | best_score_so_far=%.4f | rows=%d",
                lam,
                best_score,
                len(tuning_rows),
            )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(fold_dir / "stage2_tuning.csv", index=False)
    tuning_df.to_csv(fold_dir / "lambda_threshold_tuning.csv", index=False)
    if not found_valid_combo:
        logger.warning(
            "No valid stage-2 tuning combination was trainable; using defaults lambda=%.3f prob_thr=%.3f hysteresis_gap=%.3f final_merge_gap=%.3f final_min_sec=%.3f final_max_sec=%.3f adaptive_q=%.3f",
            best_params.lambda_thr,
            best_params.probability_threshold,
            best_params.hysteresis_gap,
            best_params.final_merge_gap_sec,
            best_params.final_min_sec,
            best_params.final_max_sec,
            best_params.adaptive_threshold_quantile,
        )
    logger.info(
        "Best stage-2 params on validation: lambda=%.3f prob_thr=%.3f hysteresis_gap=%.3f final_merge_gap=%.3f final_min_sec=%.3f final_max_sec=%.3f adaptive_q=%.3f stage3_score=%.4f",
        best_params.lambda_thr,
        best_params.probability_threshold,
        best_params.hysteresis_gap,
        best_params.final_merge_gap_sec,
        best_params.final_min_sec,
        best_params.final_max_sec,
        best_params.adaptive_threshold_quantile,
        best_score,
    )
    return best_params


def prepare_recording(subject: SubjectFiles, cfg: Config, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    raw = mne.io.read_raw_edf(subject.psg_path, preload=True, verbose="ERROR")
    best_channel = select_best_channel(raw.ch_names, cfg.channel_target, logger)
    secondary_channel: Optional[str] = None
    if cfg.segmentation_use_secondary_eeg_channel:
        secondary_channel = select_optional_channel(
            raw.ch_names,
            cfg.segmentation_secondary_channel_target,
            exclude=[best_channel],
        )
        if secondary_channel is None:
            logger.warning(
                "Subject %s has no distinct %s secondary EEG channel; using zero-valued secondary features.",
                subject.subject_id,
                cfg.segmentation_secondary_channel_target,
            )
        else:
            logger.info(
                "Subject %s | primary channel=%s | secondary channel=%s",
                subject.subject_id,
                best_channel,
                secondary_channel,
            )
    selected_channels = [best_channel] + ([secondary_channel] if secondary_channel is not None else [])
    raw.pick(selected_channels)
    sfreq = float(raw.info["sfreq"])
    if not math.isclose(sfreq, cfg.resample_hz):
        raw.resample(cfg.resample_hz, npad="auto", verbose="ERROR")
        sfreq = cfg.resample_hz
    data = raw.get_data(picks=[best_channel])[0].astype(np.float32)
    if secondary_channel is not None:
        secondary_data = raw.get_data(picks=[secondary_channel])[0].astype(np.float32)
    else:
        secondary_data = np.zeros_like(data, dtype=np.float32)

    e1 = parse_spindle_events(subject.spindle_e1_path, logger, cfg.debug, cfg.allow_implicit_spindle_labels)
    e2 = parse_spindle_events(subject.spindle_e2_path, logger, cfg.debug, cfg.allow_implicit_spindle_labels)
    true_events_full, ok = build_true_events(cfg.annotation_mode, e1, e2, logger, subject.subject_id)
    if not ok:
        return None

    stage_events = parse_sleep_stages(subject.base_path, logger, cfg.debug)
    truth_binary_full = events_to_binary(true_events_full, len(data), sfreq)
    e1_binary_full = events_to_binary(e1, len(data), sfreq)
    e2_binary_full = events_to_binary(e2, len(data), sfreq)
    stage_intervals = build_stage_sample_intervals(stage_events, sfreq, cfg.stage_mode, len(data))
    if not stage_intervals:
        logger.warning("Subject %s has no retained samples for stage mode %s; skipping.", subject.subject_id, cfg.stage_mode)
        return None

    raw_parts: List[np.ndarray] = []
    truth_parts: List[np.ndarray] = []
    e1_parts: List[np.ndarray] = []
    e2_parts: List[np.ndarray] = []
    stage1_parts: List[np.ndarray] = []
    sigma_parts: List[np.ndarray] = []
    slow_sigma_parts: List[np.ndarray] = []
    teo_parts: List[np.ndarray] = []
    envelope_parts: List[np.ndarray] = []
    secondary_raw_parts: List[np.ndarray] = []
    secondary_sigma_parts: List[np.ndarray] = []
    secondary_slow_sigma_parts: List[np.ndarray] = []
    secondary_envelope_parts: List[np.ndarray] = []
    for s, e in stage_intervals:
        seg_raw = data[s:e].astype(np.float32)
        seg_secondary_raw = secondary_data[s:e].astype(np.float32)
        seg_truth = truth_binary_full[s:e].astype(np.uint8)
        seg_e1 = e1_binary_full[s:e].astype(np.uint8)
        seg_e2 = e2_binary_full[s:e].astype(np.uint8)
        if len(seg_raw) == 0:
            continue
        if cfg.remove_anomaly and not is_segmentation_mode(cfg):
            seg_raw_orig = seg_raw.copy()
            seg_raw, seg_truth = remove_stage_anomalies(
                seg_raw,
                seg_truth,
                cfg.anomaly_range_divisor,
                cfg.anomaly_jump_factor,
                cfg.anomaly_recovery_factor,
            )
            _, seg_e1 = remove_stage_anomalies(
                seg_raw_orig,
                seg_e1,
                cfg.anomaly_range_divisor,
                cfg.anomaly_jump_factor,
                cfg.anomaly_recovery_factor,
            )
            _, seg_e2 = remove_stage_anomalies(
                seg_raw_orig,
                seg_e2,
                cfg.anomaly_range_divisor,
                cfg.anomaly_jump_factor,
                cfg.anomaly_recovery_factor,
            )
        if len(seg_raw) == 0:
            continue
        seg_stage1 = preprocess_candidate_signal(seg_raw, sfreq, cfg)
        seg_sigma = compute_sigma_band(seg_stage1, sfreq, cfg.sigma_low_hz, cfg.sigma_high_hz)
        seg_slow_sigma = compute_sigma_band(
            seg_stage1,
            sfreq,
            cfg.segmentation_slow_sigma_low_hz,
            cfg.segmentation_slow_sigma_high_hz,
        )
        seg_teo = compute_teager_energy(seg_sigma)
        seg_envelope = compute_envelope(seg_sigma)
        if cfg.segmentation_use_secondary_eeg_channel:
            seg_secondary_stage1 = preprocess_candidate_signal(seg_secondary_raw, sfreq, cfg)
            seg_secondary_sigma = compute_sigma_band(
                seg_secondary_stage1,
                sfreq,
                cfg.sigma_low_hz,
                cfg.sigma_high_hz,
            )
            seg_secondary_slow_sigma = compute_sigma_band(
                seg_secondary_stage1,
                sfreq,
                cfg.segmentation_slow_sigma_low_hz,
                cfg.segmentation_slow_sigma_high_hz,
            )
            seg_secondary_envelope = compute_envelope(seg_secondary_sigma)
        else:
            seg_secondary_sigma = np.zeros_like(seg_secondary_raw, dtype=np.float32)
            seg_secondary_slow_sigma = np.zeros_like(seg_secondary_raw, dtype=np.float32)
            seg_secondary_envelope = np.zeros_like(seg_secondary_raw, dtype=np.float32)
        raw_parts.append(seg_raw.astype(np.float32))
        truth_parts.append(seg_truth.astype(np.uint8))
        e1_parts.append(seg_e1.astype(np.uint8))
        e2_parts.append(seg_e2.astype(np.uint8))
        stage1_parts.append(seg_stage1.astype(np.float32))
        sigma_parts.append(seg_sigma.astype(np.float32))
        slow_sigma_parts.append(seg_slow_sigma.astype(np.float32))
        teo_parts.append(seg_teo.astype(np.float32))
        envelope_parts.append(seg_envelope.astype(np.float32))
        secondary_raw_parts.append(seg_secondary_raw.astype(np.float32))
        secondary_sigma_parts.append(seg_secondary_sigma.astype(np.float32))
        secondary_slow_sigma_parts.append(seg_secondary_slow_sigma.astype(np.float32))
        secondary_envelope_parts.append(seg_secondary_envelope.astype(np.float32))

    stage_raw, stage_segment_bounds = concatenate_parts(raw_parts, np.float32)
    stage_truth, _ = concatenate_parts(truth_parts, np.uint8)
    stage_truth_e1, _ = concatenate_parts(e1_parts, np.uint8)
    stage_truth_e2, _ = concatenate_parts(e2_parts, np.uint8)
    stage1_raw, _ = concatenate_parts(stage1_parts, np.float32)
    sigma, _ = concatenate_parts(sigma_parts, np.float32)
    slow_sigma, _ = concatenate_parts(slow_sigma_parts, np.float32)
    teo, _ = concatenate_parts(teo_parts, np.float32)
    envelope, _ = concatenate_parts(envelope_parts, np.float32)
    secondary_raw, _ = concatenate_parts(secondary_raw_parts, np.float32)
    secondary_sigma, _ = concatenate_parts(secondary_sigma_parts, np.float32)
    secondary_slow_sigma, _ = concatenate_parts(secondary_slow_sigma_parts, np.float32)
    secondary_envelope, _ = concatenate_parts(secondary_envelope_parts, np.float32)
    if len(stage_raw) == 0:
        logger.warning("Subject %s produced empty stage-segment EEG after preprocessing; skipping.", subject.subject_id)
        return None

    if cfg.normalization_mode in ("subject_zscore", "subject_robust"):
        model_raw = normalize_subject_signal(stage1_raw, cfg.normalization_mode)
        model_sigma = normalize_subject_signal(sigma, cfg.normalization_mode)
        model_teo = normalize_subject_signal(teo, cfg.normalization_mode)
        model_env = normalize_subject_signal(envelope, cfg.normalization_mode)
    else:
        model_raw = stage1_raw
        model_sigma = sigma
        model_teo = teo
        model_env = envelope

    true_events_stage = binary_to_events_segmented(stage_truth, sfreq, stage_segment_bounds)
    true_events_e1_stage = binary_to_events_segmented(stage_truth_e1, sfreq, stage_segment_bounds)
    true_events_e2_stage = binary_to_events_segmented(stage_truth_e2, sfreq, stage_segment_bounds)
    truth_soft_stage = build_soft_boundary_labels(
        stage_truth,
        sfreq,
        cfg.segmentation_soft_label_shoulder_sec,
        cfg.segmentation_soft_label_edge_value,
    )
    truth_center_stage, truth_boundary_stage = build_center_boundary_labels(
        stage_truth,
        sfreq,
        cfg.segmentation_center_label_width_sec,
        cfg.segmentation_boundary_label_width_sec,
    )
    truth_annotator_confidence_stage, annotator_confidence = build_annotator_confidence_targets(
        truth_soft_stage,
        stage_truth_e1,
        stage_truth_e2,
        sfreq,
        cfg.segmentation_single_expert_target_weight,
        cfg.segmentation_annotator_agreement_tolerance_sec,
    )
    truth_center_annotator_confidence_stage = truth_center_stage * annotator_confidence
    truth_boundary_annotator_confidence_stage = truth_boundary_stage * annotator_confidence
    candidate_sigma_rms_ref = float(np.sqrt(np.mean(np.square(sigma)) + 1e-8))
    candidate_envelope_median = float(np.median(envelope))
    candidate_envelope_mad = float(np.median(np.abs(envelope - candidate_envelope_median)))
    candidate_teo_abs_median = float(np.median(np.abs(teo)) + 1e-8)
    return {
        "subject_id": subject.subject_id,
        "channel": best_channel,
        "secondary_channel": secondary_channel or "missing_zero_fallback",
        "sfreq": sfreq,
        "events_e1": e1,
        "events_e2": e2,
        "true_events_full": true_events_full,
        "true_events_stage": true_events_stage,
        "true_events_e1_stage": true_events_e1_stage,
        "true_events_e2_stage": true_events_e2_stage,
        "truth_binary_stage": stage_truth.astype(np.uint8),
        "truth_binary_e1_stage": stage_truth_e1.astype(np.uint8),
        "truth_binary_e2_stage": stage_truth_e2.astype(np.uint8),
        "truth_soft_stage": truth_soft_stage.astype(np.float32),
        "truth_annotator_confidence_stage": truth_annotator_confidence_stage.astype(np.float32),
        "truth_center_stage": truth_center_stage.astype(np.float32),
        "truth_boundary_stage": truth_boundary_stage.astype(np.float32),
        "truth_center_annotator_confidence_stage": truth_center_annotator_confidence_stage.astype(np.float32),
        "truth_boundary_annotator_confidence_stage": truth_boundary_annotator_confidence_stage.astype(np.float32),
        "segmentation_raw": prepare_segmentation_channel(stage_raw, cfg),
        "segmentation_sigma": prepare_segmentation_channel(sigma, cfg),
        "segmentation_slow_sigma": prepare_segmentation_channel(slow_sigma, cfg),
        "segmentation_teo": prepare_segmentation_teo_channel(teo),
        "segmentation_envelope": prepare_segmentation_channel(envelope, cfg),
        "segmentation_secondary_raw": prepare_segmentation_channel(secondary_raw, cfg),
        "segmentation_secondary_sigma": prepare_segmentation_channel(secondary_sigma, cfg),
        "segmentation_secondary_slow_sigma": prepare_segmentation_channel(secondary_slow_sigma, cfg),
        "segmentation_secondary_envelope": prepare_segmentation_channel(secondary_envelope, cfg),
        "candidate_raw": stage_raw.astype(np.float32),
        "candidate_stage1_raw": stage1_raw.astype(np.float32),
        "candidate_sigma": sigma.astype(np.float32),
        "candidate_teo": teo.astype(np.float32),
        "candidate_envelope": envelope.astype(np.float32),
        "candidate_sigma_rms_ref": candidate_sigma_rms_ref,
        "candidate_envelope_median": candidate_envelope_median,
        "candidate_envelope_mad": candidate_envelope_mad,
        "candidate_teo_abs_median": candidate_teo_abs_median,
        "model_raw": model_raw.astype(np.float32),
        "model_sigma": model_sigma.astype(np.float32),
        "model_teo": model_teo.astype(np.float32),
        "model_envelope": model_env.astype(np.float32),
        "stage_segment_bounds": [(int(s), int(e)) for s, e in stage_segment_bounds],
        "retained_len_samples": int(len(stage_truth)),
        "n_true_events_stage": int(len(true_events_stage)),
    }


def save_subject_selection_summary(all_subjects: List[str], retained: List[str], excluded: List[str], records: Dict[str, Any], outdir: Path) -> None:
    def duration_values(events: Sequence[Tuple[float, float]]) -> List[float]:
        return [float(max(0.0, e - s)) for s, e in events]

    summary = {
        "all_subjects": all_subjects,
        "retained_subjects": retained,
        "excluded_subjects": excluded,
        "retained_count": len(retained),
        "excluded_count": len(excluded),
        "annotation_counts": {
            sid: {
                "e1": len(records[sid]["events_e1"]) if sid in records else 0,
                "e2": len(records[sid]["events_e2"]) if sid in records else 0,
            }
            for sid in all_subjects
        },
    }
    save_json(summary, outdir / "dataset_subject_summary.json")
    rows = []
    for sid in all_subjects:
        if sid in records:
            e1_events = records[sid].get("true_events_e1_stage", records[sid].get("events_e1", []))
            e2_events = records[sid].get("true_events_e2_stage", records[sid].get("events_e2", []))
            union_events = merge_intervals(list(e1_events) + list(e2_events))
            e1_dur = duration_values(e1_events)
            e2_dur = duration_values(e2_events)
            union_dur = duration_values(union_events)
        else:
            e1_events = []
            e2_events = []
            union_events = []
            e1_dur = []
            e2_dur = []
            union_dur = []
        rows.append({
            "subject_id": sid,
            "retained": sid in retained,
            "n_e1": len(e1_events),
            "n_e2": len(e2_events),
            "n_union": len(union_events),
            "e1_mean_duration_sec": float(np.mean(e1_dur)) if e1_dur else float("nan"),
            "e2_mean_duration_sec": float(np.mean(e2_dur)) if e2_dur else float("nan"),
            "union_mean_duration_sec": float(np.mean(union_dur)) if union_dur else float("nan"),
            "e1_median_duration_sec": float(np.median(e1_dur)) if e1_dur else float("nan"),
            "e2_median_duration_sec": float(np.median(e2_dur)) if e2_dur else float("nan"),
            "union_median_duration_sec": float(np.median(union_dur)) if union_dur else float("nan"),
        })
    pd.DataFrame(rows).to_csv(outdir / "dataset_subject_summary.csv", index=False)


def exact_paper_splits(retained_ids: List[str]) -> List[Tuple[List[str], List[str], List[str]]]:
    if len(retained_ids) != 15:
        raise RuntimeError(f"paper_exact_e12 requires exactly 15 retained subjects, got {len(retained_ids)}.")

    def map_ids(arr: np.ndarray) -> List[str]:
        return [retained_ids[i - 1] for i in arr.tolist()]

    splits = []
    for i in range(5):
        train = map_ids(PAPER_TRAIN_ID[i])
        val = map_ids(PAPER_VAL_ID[i])
        test = map_ids(PAPER_TEST_ID[i])
        splits.append((train, val, test))
    return splits


def random_fivefold_e12_splits(subject_ids: List[str], seed: int) -> List[Tuple[List[str], List[str], List[str]]]:
    rng = random.Random(seed)
    ids = subject_ids[:]
    rng.shuffle(ids)
    folds = [ids[i::5] for i in range(5)]
    splits = []
    for i in range(5):
        test = folds[i]
        val = folds[(i + 1) % 5]
        train = [sid for j, fold in enumerate(folds) if j not in (i, (i + 1) % 5) for sid in fold]
        splits.append((train, val, test))
    return splits


def loso_subject_splits(subject_ids: List[str], seed: int) -> List[Tuple[List[str], List[str], List[str]]]:
    rng = random.Random(seed)
    splits: List[Tuple[List[str], List[str], List[str]]] = []
    for test_id in subject_ids:
        remaining = [sid for sid in subject_ids if sid != test_id]
        rng.shuffle(remaining)
        n_val = max(1, int(round(0.2 * len(remaining))))
        val = remaining[:n_val]
        train = remaining[n_val:]
        splits.append((train, val, [test_id]))
    return splits


def load_completed_fold_summaries(outdir: Path, expected_folds: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for fold_idx in range(1, expected_folds + 1):
        metrics_path = outdir / f"fold_{fold_idx:02d}" / "overall_metrics.json"
        if not metrics_path.exists():
            logger.warning("Fold %d summary is missing at %s.", fold_idx, metrics_path)
            continue
        try:
            data = load_json(metrics_path)
            data["fold"] = int(data.get("fold", fold_idx))
            summaries.append(data)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            logger.warning("Could not load fold %d summary from %s: %s", fold_idx, metrics_path, exc)
    summaries.sort(key=lambda row: int(row.get("fold", 0)))
    return summaries


STATISTICAL_METRICS: Tuple[Tuple[str, str], ...] = (
    ("strict_event_f1", "Strict event F1"),
    ("tolerant_event_f1", "Tolerant event F1"),
    ("strict_event_precision", "Strict event precision"),
    ("strict_event_recall", "Strict event recall"),
    ("event_iou_f1_0_5", "Event F1 at IoU >= 0.5"),
    ("sample_pr_auc", "Sample PR-AUC"),
    ("sample_auc", "Sample ROC-AUC"),
)


def add_event_iou_f1_from_fold_details(fold_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Backfill micro event F1 at IoU >= 0.5 for legacy result folders."""
    out = fold_df.copy()
    if "event_iou_f1_0_5" not in out.columns:
        out["event_iou_f1_0_5"] = float("nan")
    for row_idx, row in out.iterrows():
        if math.isfinite(float(row.get("event_iou_f1_0_5", float("nan")))):
            continue
        fold_idx = int(row["fold"])
        path = results_dir / f"fold_{fold_idx:02d}" / "event_metrics.json"
        if not path.exists():
            continue
        try:
            details = load_json(path)
            tp = fp = fn = 0
            for subject in details:
                metric = subject.get("per_iou", {}).get("iou_0.5", {})
                tp += int(metric.get("tp", 0))
                fp += int(metric.get("fp", 0))
                fn += int(metric.get("fn", 0))
            denom = 2 * tp + fp + fn
            out.at[row_idx, "event_iou_f1_0_5"] = float(2 * tp / denom) if denom > 0 else 0.0
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return out


def metric_descriptive(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "min", "max", "ci95_low", "ci95_high")}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0.0:
        half_width = float(scipy.stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    else:
        half_width = 0.0
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def holm_adjusted_pvalues(pvalues: Sequence[float]) -> List[float]:
    raw = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(len(raw), np.nan, dtype=np.float64)
    valid = np.where(np.isfinite(raw))[0]
    if len(valid) == 0:
        return adjusted.tolist()
    order = valid[np.argsort(raw[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, float(raw[idx]) * (m - rank))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def fold_pairing_matches(proposed_dir: Path, baseline_dir: Path, fold_ids: Sequence[int]) -> bool:
    for fold_idx in fold_ids:
        proposed_path = proposed_dir / f"fold_{fold_idx:02d}" / "fold_split.json"
        baseline_path = baseline_dir / f"fold_{fold_idx:02d}" / "fold_split.json"
        if not proposed_path.exists() or not baseline_path.exists():
            return False
        try:
            proposed = load_json(proposed_path)
            baseline = load_json(baseline_path)
            proposed_test = sorted(proposed.get("test", proposed.get("test_ids", [])))
            baseline_test = sorted(baseline.get("test", baseline.get("test_ids", [])))
            if proposed_test != baseline_test:
                return False
        except (OSError, TypeError, ValueError, KeyError):
            return False
    return True


def save_statistical_analysis(
    proposed_fold_df: pd.DataFrame,
    proposed_dir: Path,
    baseline_dir: Optional[Path],
    outdir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Save deterministic fold-paired descriptive and inferential statistics."""
    proposed = add_event_iou_f1_from_fold_details(proposed_fold_df, proposed_dir)
    baseline: Optional[pd.DataFrame] = None
    pairing_verified = False
    if baseline_dir is not None and (baseline_dir / "fold_summary.csv").exists():
        baseline = pd.read_csv(baseline_dir / "fold_summary.csv")
        baseline = add_event_iou_f1_from_fold_details(baseline, baseline_dir)
        fold_ids = sorted(set(proposed["fold"].astype(int)) & set(baseline["fold"].astype(int)))
        pairing_verified = fold_pairing_matches(proposed_dir, baseline_dir, fold_ids)
        if not pairing_verified:
            logger.warning("Fold split metadata could not verify exact proposed/baseline pairing.")
    rows: List[Dict[str, Any]] = []
    for metric, label in STATISTICAL_METRICS:
        if metric not in proposed.columns:
            continue
        proposed_values = pd.to_numeric(proposed[metric], errors="coerce").to_numpy(dtype=np.float64)
        proposed_desc = metric_descriptive(proposed_values)
        row: Dict[str, Any] = {
            "metric": metric,
            "metric_label": label,
            "n_proposed_folds": int(np.isfinite(proposed_values).sum()),
            **{f"proposed_{key}": value for key, value in proposed_desc.items()},
            "fold_pairing_verified": pairing_verified,
        }
        if baseline is not None and metric in baseline.columns:
            paired = proposed[["fold", metric]].merge(
                baseline[["fold", metric]], on="fold", how="inner", suffixes=("_proposed", "_baseline")
            )
            x = pd.to_numeric(paired[f"{metric}_proposed"], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            baseline_desc = metric_descriptive(y)
            diff = x - y
            n_pairs = int(len(diff))
            if n_pairs >= 2:
                t_stat, t_p = scipy.stats.ttest_rel(x, y)
                if np.allclose(diff, 0.0):
                    w_stat, w_p = 0.0, 1.0
                else:
                    w_stat, w_p = scipy.stats.wilcoxon(diff, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
                diff_std = float(np.std(diff, ddof=1))
                cohens_dz = float(np.mean(diff) / diff_std) if diff_std > 0 else (0.0 if np.allclose(diff, 0.0) else math.copysign(float("inf"), float(np.mean(diff))))
            else:
                t_stat = t_p = w_stat = w_p = cohens_dz = float("nan")
            baseline_mean = float(np.mean(y)) if n_pairs else float("nan")
            improvement = (
                float(100.0 * (float(np.mean(x)) - baseline_mean) / abs(baseline_mean))
                if n_pairs and abs(baseline_mean) > 1e-12 else float("nan")
            )
            row.update(
                {
                    "n_paired_folds": n_pairs,
                    **{f"baseline_{key}": value for key, value in baseline_desc.items()},
                    "mean_paired_difference": float(np.mean(diff)) if n_pairs else float("nan"),
                    "percentage_improvement": improvement,
                    "paired_t_statistic": float(t_stat),
                    "paired_t_pvalue": float(t_p),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_pvalue": float(w_p),
                    "cohens_dz": float(cohens_dz),
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    for p_col, adjusted_col in (
        ("paired_t_pvalue", "paired_t_pvalue_holm"),
        ("wilcoxon_pvalue", "wilcoxon_pvalue_holm"),
    ):
        if p_col in result:
            result[adjusted_col] = holm_adjusted_pvalues(pd.to_numeric(result[p_col], errors="coerce").tolist())
    result.to_csv(outdir / "statistical_analysis.csv", index=False)
    notes = [
        "# Statistical analysis",
        "",
        "Metrics are paired by identical held-out subject folds. Values are proportions, not percentages.",
        "95% confidence intervals use the two-sided Student-t interval across folds.",
        "Cohen's dz is the mean paired difference divided by its sample standard deviation.",
        "Holm-adjusted p-values control family-wise error across the reported metrics.",
        "Event IoU is reported as micro event F1 at IoU >= 0.5.",
        "",
        dataframe_to_markdown(result),
    ]
    (outdir / "statistical_analysis.md").write_text("\n".join(notes), encoding="utf-8")
    logger.info("Saved publication statistical analysis to %s", outdir)
    return result


def save_cross_fold_reference_outputs(outdir: Path, expected_folds: int, logger: logging.Logger) -> None:
    subject_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    profile_frames: List[pd.DataFrame] = []
    for fold_idx in range(1, expected_folds + 1):
        fold_dir = outdir / f"fold_{fold_idx:02d}"
        subject_path = fold_dir / "subject_metrics_by_reference.csv"
        fold_path = fold_dir / "overall_metrics_by_reference.csv"
        profile_path = fold_dir / "subject_error_profile.csv"
        if subject_path.exists():
            subject_df = pd.read_csv(subject_path)
            subject_df.insert(0, "fold", fold_idx)
            subject_frames.append(subject_df)
        if fold_path.exists():
            fold_ref_df = pd.read_csv(fold_path)
            fold_ref_df.insert(0, "fold", fold_idx)
            fold_frames.append(fold_ref_df)
        if profile_path.exists():
            profile_df = pd.read_csv(profile_path)
            profile_df.insert(0, "fold", fold_idx)
            profile_frames.append(profile_df)
    if not subject_frames and not fold_frames:
        return
    if profile_frames:
        pd.concat(profile_frames, ignore_index=True).sort_values(
            ["strict_event_f1", "pred_to_true_density_ratio"],
            ascending=[True, False],
        ).to_csv(outdir / "subject_error_profile.csv", index=False)
    if subject_frames:
        all_subject_df = pd.concat(subject_frames, ignore_index=True)
        all_subject_df.to_csv(outdir / "subject_metrics_by_reference.csv", index=False)
    if not fold_frames:
        return
    fold_ref_df = pd.concat(fold_frames, ignore_index=True)
    fold_ref_df.to_csv(outdir / "fold_summary_by_reference.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for reference, ref_df in fold_ref_df.groupby("reference"):
        numeric_cols = [c for c in ref_df.columns if c != "fold" and pd.api.types.is_numeric_dtype(ref_df[c])]
        for col in numeric_cols:
            summary_rows.append(
                {
                    "reference": reference,
                    "metric": col,
                    "mean": float(ref_df[col].mean()),
                    "std": float(ref_df[col].std(ddof=0)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(outdir / "overall_summary_by_reference.csv", index=False)
    save_json(
        {
            str(reference): {
                str(row["metric"]): {"mean": float(row["mean"]), "std": float(row["std"])}
                for _, row in rows.iterrows()
            }
            for reference, rows in pd.DataFrame(summary_rows).groupby("reference")
        },
        outdir / "overall_summary_by_reference.json",
    )
    logger.info("Saved cross-fold expert-reference evaluation summaries.")


def run_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    records: Dict[str, Any],
    cfg: Config,
    outdir: Path,
    device: torch.device,
    logger: logging.Logger,
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]],
) -> Dict[str, Any]:
    fold_dir = ensure_dir(outdir / f"fold_{fold_idx:02d}")
    save_json({"train": train_ids, "val": val_ids, "test": test_ids}, fold_dir / "fold_split.json")
    logger.info("Fold %d | train=%s | val=%s | test=%s", fold_idx, train_ids, val_ids, test_ids)

    candidate_params_path = fold_dir / "candidate_params.json"
    if cfg.resume_tuning and candidate_params_path.exists():
        try:
            candidate_params = load_json(candidate_params_path)
            best_B = float(candidate_params["B"])
            best_T = float(candidate_params["T"])
            logger.info("Reusing candidate params from %s: B=%.3f T=%.3f", candidate_params_path, best_B, best_T)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            logger.warning("Could not reuse candidate params from %s (%s); retuning.", candidate_params_path, exc)
            best_B, best_T = tune_candidate_parameters(train_ids, val_ids, records, candidate_cache, cfg, logger)
            save_json({"B": best_B, "T": best_T}, candidate_params_path)
    else:
        best_B, best_T = tune_candidate_parameters(train_ids, val_ids, records, candidate_cache, cfg, logger)
        save_json({"B": best_B, "T": best_T}, candidate_params_path)

    diagnostics_rows: List[Dict[str, Any]] = []
    for sid in sorted(records):
        bundle = get_cached_candidates(sid, records, candidate_cache, cfg, best_B, best_T)
        diag = candidate_diagnostics(
            sid,
            bundle["events"],
            records[sid]["true_events_stage"],
            records[sid]["retained_len_samples"],
            records[sid]["sfreq"],
        )
        diag["split"] = "train" if sid in train_ids else "val" if sid in val_ids else "test" if sid in test_ids else "other"
        diagnostics_rows.append(diag)
    if cfg.save_candidate_diagnostics:
        pd.DataFrame(diagnostics_rows).to_csv(fold_dir / "candidate_diagnostics.csv", index=False)

    stage2_params_path = fold_dir / "stage2_params.json"

    def save_stage2_params(params: Stage2Params) -> None:
        save_json(
            {
                "lambda": params.lambda_thr,
                "probability_threshold": params.probability_threshold,
                "hysteresis_gap": params.hysteresis_gap,
                "final_merge_gap_sec": params.final_merge_gap_sec,
                "final_min_sec": params.final_min_sec,
                "final_max_sec": params.final_max_sec,
                "adaptive_threshold_quantile": params.adaptive_threshold_quantile,
                "selection_objective": "F1-weighted validation stage-3 score with recall floor",
                "justification": stage2_tuning_justification(),
            },
            stage2_params_path,
        )

    if cfg.resume_tuning and stage2_params_path.exists():
        try:
            stage2_params = load_json(stage2_params_path)
            best_stage2 = Stage2Params(
                lambda_thr=float(stage2_params["lambda"]),
                probability_threshold=float(stage2_params["probability_threshold"]),
                hysteresis_gap=float(stage2_params["hysteresis_gap"]),
                final_merge_gap_sec=float(stage2_params["final_merge_gap_sec"]),
                final_min_sec=float(stage2_params["final_min_sec"]),
                final_max_sec=float(stage2_params["final_max_sec"]),
                adaptive_threshold_quantile=float(stage2_params["adaptive_threshold_quantile"]),
            )
            logger.info(
                "Reusing stage-2 params from %s: lambda=%.3f prob_thr=%.3f hysteresis_gap=%.3f final_merge_gap=%.3f final_min_sec=%.3f final_max_sec=%.3f adaptive_q=%.3f",
                stage2_params_path,
                best_stage2.lambda_thr,
                best_stage2.probability_threshold,
                best_stage2.hysteresis_gap,
                best_stage2.final_merge_gap_sec,
                best_stage2.final_min_sec,
                best_stage2.final_max_sec,
                best_stage2.adaptive_threshold_quantile,
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            logger.warning("Could not reuse stage-2 params from %s (%s); retuning.", stage2_params_path, exc)
            best_stage2 = tune_stage2_parameters(
                fold_dir, train_ids, val_ids, records, candidate_cache, cfg, best_B, best_T, device, logger
            )
            save_stage2_params(best_stage2)
    else:
        best_stage2 = tune_stage2_parameters(
            fold_dir, train_ids, val_ids, records, candidate_cache, cfg, best_B, best_T, device, logger
        )
        save_stage2_params(best_stage2)

    dataset_start = time.perf_counter()
    X_train, y_train_soft, y_train_hard, train_counts = build_dataset_for_subjects(train_ids, records, candidate_cache, cfg, best_B, best_T, best_stage2.lambda_thr)
    X_val, _, y_val_hard, val_counts = build_dataset_for_subjects(val_ids, records, candidate_cache, cfg, best_B, best_T, best_stage2.lambda_thr)
    logger.info(
        "Fold %d | final candidate windows built in %.1fs | train_candidates=%d | val_candidates=%d",
        fold_idx,
        time.perf_counter() - dataset_start,
        len(y_train_hard),
        len(y_val_hard),
    )
    if len(y_train_hard) == 0 or len(y_val_hard) == 0:
        raise RuntimeError(
            "Fold has empty train or validation candidate dataset. "
            f"train_counts={train_counts}, val_counts={val_counts}, "
            f"B={best_B:.3f}, T={best_T:.3f}, lambda={best_stage2.lambda_thr:.3f}"
        )

    norm_stats = compute_dataset_normalization_stats(X_train) if cfg.normalization_mode == "dataset_train" else None
    X_train = apply_dataset_normalization(X_train, norm_stats, cfg.normalization_mode)
    X_val = apply_dataset_normalization(X_val, norm_stats, cfg.normalization_mode)
    if norm_stats is not None:
        save_json({"mean": norm_stats.mean, "std": norm_stats.std}, fold_dir / "dataset_normalization_stats.json")

    train_sampler = build_balanced_sampler(y_train_hard, cfg.balanced_sampling)

    train_loader = make_candidate_loader(
        X_train,
        y_train_soft,
        cfg,
        device,
        training=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = make_candidate_loader(X_val, y_val_hard, cfg, device, training=False)
    
    
    model = CandidateNet(
        in_channels=X_train.shape[1],
        hidden_channels=cfg.model_channels,
        kernel_size=cfg.model_kernel_size,
        dropout=cfg.model_dropout,
        use_residual=cfg.model_use_residual,
        use_temporal_attention_gate=cfg.model_use_temporal_attention_gate,
    ).to(device)
    pos_weight = compute_pos_weight(y_train_hard, cfg.positive_weight_max, device)
    checkpoint_path = fold_dir / "best_model.pt"
    model, history = train_model(model, train_loader, val_loader, cfg, device, logger, checkpoint_path, cfg.epochs, pos_weight)
    if cfg.hard_negative_mining_enabled and cfg.hard_negative_finetune_epochs > 0:
        pre_hn_state = clone_model_state(model)
        accept_metric = history_accept_metric(cfg, history)
        pre_hn_best = best_history_value(history, accept_metric)
        hn_weights = hard_negative_weights(model, X_train, y_train_hard, cfg, device)
        hn_count = int(np.sum(hn_weights > 1.0))
        if hn_count > 0:
            logger.info(
                "Hard-negative mining | selected_negatives=%d | boost=%.2f | finetune_epochs=%d",
                hn_count,
                cfg.hard_negative_boost,
                cfg.hard_negative_finetune_epochs,
            )
            hn_sampler = build_balanced_sampler(y_train_hard, cfg.balanced_sampling, hn_weights)
            hn_train_loader = make_candidate_loader(
                X_train,
                y_train_soft,
                cfg,
                device,
                training=True,
                shuffle=(hn_sampler is None),
                sampler=hn_sampler,
            )
            hn_checkpoint_path = fold_dir / "best_model_hardneg.pt"
            model, hn_history = train_model(
                model,
                hn_train_loader,
                val_loader,
                cfg,
                device,
                logger,
                hn_checkpoint_path,
                cfg.hard_negative_finetune_epochs,
                pos_weight,
            )
            hn_best = best_history_value(hn_history, accept_metric)
            if cfg.hard_negative_accept_only_if_improved and hn_best < pre_hn_best + cfg.hard_negative_accept_margin:
                model.load_state_dict(pre_hn_state)
                logger.info(
                    "Hard-negative fine-tune rejected | metric=%s | before=%.4f | hardneg=%.4f | margin=%.4f",
                    accept_metric,
                    pre_hn_best,
                    hn_best,
                    cfg.hard_negative_accept_margin,
                )
            else:
                logger.info(
                    "Hard-negative fine-tune accepted | metric=%s | before=%.4f | hardneg=%.4f",
                    accept_metric,
                    pre_hn_best,
                    hn_best,
                )
            history = merge_training_histories(history, hn_history, "hard_negative_finetune")
    val_probs_raw = predict_probabilities(model, val_loader, device)
    final_calibrator = fit_probability_calibrator(
        val_probs_raw,
        y_val_hard,
        cfg,
        logger,
        f"fold_{fold_idx:02d}_outer_val",
    )
    if cfg.save_training_history:
        save_json(history, fold_dir / "training_history.json")

    subject_rows: List[Dict[str, Any]] = []
    event_details: List[Dict[str, Any]] = []
    test_vectors_truth: List[np.ndarray] = []
    test_vectors_pred: List[np.ndarray] = []
    test_vectors_prob: List[np.ndarray] = []
    explainability_timelines: Dict[str, np.ndarray] = {}
    explainability_decoded_events: Dict[str, List[Tuple[float, float]]] = {}
    strict_tp = strict_fp = strict_fn = 0
    tol_tp = tol_fp = tol_fn = 0
    iou50_tp = iou50_fp = iou50_fn = 0
    candidate_all_y: List[np.ndarray] = []
    candidate_all_probs: List[np.ndarray] = []

    for sid in test_ids:
        rec = records[sid]
        bundle = get_cached_candidates(sid, records, candidate_cache, cfg, best_B, best_T)
        y_test, _ = candidate_labels_for_record(bundle["events"], rec, best_stage2.lambda_thr, cfg)
        X_test = get_candidate_window_matrix(rec, bundle, cfg)
        X_test = apply_dataset_normalization(X_test, norm_stats, cfg.normalization_mode)
        
        loader = make_candidate_loader(X_test, y_test, cfg, device, training=False)
        probs = apply_probability_calibrator(final_calibrator, predict_probabilities(model, loader, device))
        probability_timeline = build_probability_timeline(bundle["events"], probs, len(rec["truth_binary_stage"]), rec["sfreq"])
        subject_prob_thr = resolve_subject_probability_threshold(
            probs,
            best_stage2.probability_threshold,
            best_stage2.adaptive_threshold_quantile,
            cfg.adaptive_threshold_max_adjustment,
            cfg.stage3_prob_threshold_floor,
            cfg.stage3_prob_threshold_cap,
        )
        pred_events, pred_binary = decode_events_from_candidate_probabilities(
            bundle["events"],
            probs,
            len(rec["truth_binary_stage"]),
            rec["sfreq"],
            subject_prob_thr,
            best_stage2.hysteresis_gap,
            best_stage2.final_min_sec,
            best_stage2.final_max_sec,
            best_stage2.final_merge_gap_sec,
            cfg.stage3_support_smooth_sec,
            cfg.stage3_support_relative_threshold,
        )

        strict = event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])
        tolerant = tolerant_event_metrics(rec["true_events_stage"], pred_events, cfg.tolerant_match_sec)
        sample = sample_level_metrics(rec["truth_binary_stage"], pred_binary, probability_timeline)
        cand_metrics = candidate_classification_metrics(y_test, probs) if len(y_test) else candidate_classification_metrics(np.zeros((0,)), np.zeros((0,)))

        per_iou = {f"iou_{thr}": event_metrics(rec["true_events_stage"], pred_events, thr) for thr in cfg.event_iou_thresholds}
        iou50 = event_metrics(rec["true_events_stage"], pred_events, 0.5)
        event_details.append({
            "subject_id": sid,
            "n_true_events": len(rec["true_events_stage"]),
            "n_pred_events": len(pred_events),
            "n_candidates": len(bundle["events"]),
            "strict": strict,
            "tolerant": tolerant,
            "per_iou": per_iou,
        })

        strict_tp += strict["tp"]
        strict_fp += strict["fp"]
        strict_fn += strict["fn"]
        tol_tp += tolerant["tp"]
        tol_fp += tolerant["fp"]
        tol_fn += tolerant["fn"]
        iou50_tp += iou50["tp"]
        iou50_fp += iou50["fp"]
        iou50_fn += iou50["fn"]
        test_vectors_truth.append(rec["truth_binary_stage"])
        test_vectors_pred.append(pred_binary)
        test_vectors_prob.append(probability_timeline)
        if len(y_test):
            candidate_all_y.append(y_test)
            candidate_all_probs.append(probs)

        row = {
            "subject_id": sid,
            "n_true_events": len(rec["true_events_stage"]),
            "n_candidates": len(bundle["events"]),
            "n_pred_events": len(pred_events),
            "strict_event_precision": strict["precision"],
            "strict_event_recall": strict["recall"],
            "strict_event_f1": strict["f1"],
            "tolerant_event_precision": tolerant["precision"],
            "tolerant_event_recall": tolerant["recall"],
            "tolerant_event_f1": tolerant["f1"],
            "ACC": sample["ACC"],
            "SEN": sample["SEN"],
            "SPE": sample["SPE"],
            "KAPPA": sample["KAPPA"],
            "sample_f1": sample["sample_f1"],
            "candidate_acc": cand_metrics["candidate_acc"],
            "candidate_precision": cand_metrics["candidate_precision"],
            "candidate_recall": cand_metrics["candidate_recall"],
            "candidate_f1": cand_metrics["candidate_f1"],
            "candidate_auc": cand_metrics["candidate_auc"],
            "B": best_B,
            "T": best_T,
            "lambda_thr": best_stage2.lambda_thr,
            "prob_thr": subject_prob_thr,
            "prob_thr_global": best_stage2.probability_threshold,
            "hysteresis_gap": best_stage2.hysteresis_gap,
            "final_merge_gap_sec": best_stage2.final_merge_gap_sec,
            "final_min_sec": best_stage2.final_min_sec,
            "final_max_sec": best_stage2.final_max_sec,
            "adaptive_threshold_quantile": best_stage2.adaptive_threshold_quantile,
        }
        subject_rows.append(row)

        if cfg.save_subject_level_outputs:
            if cfg.save_candidates:
                save_json(pred_events, fold_dir / f"{sid}_pred_events.json")
            if cfg.save_candidates:
                pd.DataFrame(bundle["events"], columns=["start", "stop"]).to_csv(fold_dir / f"{sid}_candidates.csv", index=False)
            if cfg.save_probabilities:
                pd.DataFrame({"probability": probability_timeline}).to_csv(fold_dir / f"{sid}_probability_timeline.csv", index=False)
                candidate_scores = candidate_overlap_scores(bundle["events"], rec["true_events_stage"])
                candidate_prob_rows = [
                    {
                        "candidate_index": int(i),
                        "start": float(event[0]),
                        "stop": float(event[1]),
                        "duration": float(event[1] - event[0]),
                        "probability": float(probs[i]),
                        "hard_label": float(y_test[i]),
                        "overlap_score": float(candidate_scores[i]),
                    }
                    for i, event in enumerate(bundle["events"])
                ]
                pd.DataFrame(
                    candidate_prob_rows,
                    columns=["candidate_index", "start", "stop", "duration", "probability", "hard_label", "overlap_score"],
                ).to_csv(fold_dir / f"{sid}_candidate_probabilities.csv", index=False)
                pd.DataFrame({
                    "truth": rec["truth_binary_stage"],
                    "pred_binary": pred_binary,
                    "probability": probability_timeline,
                    "stage1_candidate_binary": bundle["detvec"],
                }).to_csv(fold_dir / f"{sid}_vectors.csv", index=False)

    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(fold_dir / "subject_metrics.csv", index=False)
    profile_cols = [
        "subject_id",
        "strict_event_f1",
        "strict_event_precision",
        "strict_event_recall",
        "tolerant_event_f1",
        "n_true_events",
        "n_pred_events",
        "n_pred_events_before_consensus",
        "retained_minutes",
        "true_events_per_min",
        "pred_events_per_min",
        "pred_to_true_density_ratio",
        "validation_target_events_per_min",
        "true_median_duration_sec",
        "pred_median_duration_sec",
        "duration_median_ratio",
        "prob_thr_base",
        "prob_thr",
        "consensus_candidates",
        "consensus_removed",
        "sample_f1",
        "KAPPA",
    ]
    subject_df[[c for c in profile_cols if c in subject_df.columns]].sort_values(
        ["strict_event_f1", "pred_to_true_density_ratio"],
        ascending=[True, False],
    ).to_csv(fold_dir / "subject_error_profile.csv", index=False)
    save_json(event_details, fold_dir / "event_metrics.json")
    failure_rows: List[Dict[str, Any]] = []
    diag_by_subject = {row["subject_id"]: row for row in diagnostics_rows}
    for row in subject_rows:
        diag = diag_by_subject.get(row["subject_id"], {})
        failure_rows.append(
            {
                "subject_id": row["subject_id"],
                "strict_event_f1": row["strict_event_f1"],
                "strict_event_recall": row["strict_event_recall"],
                "candidate_stage_f1": float(diag.get("candidate_strict_f1", float("nan"))),
                "candidate_stage_recall": float(diag.get("candidate_strict_recall", float("nan"))),
                "candidate_classifier_f1": row["candidate_f1"],
                "candidate_classifier_recall": row["candidate_recall"],
                "candidate_to_event_recall_gap": float(row["candidate_recall"] - row["strict_event_recall"]),
                "n_candidates": row["n_candidates"],
                "n_pred_events": row["n_pred_events"],
            }
        )
    failure_df = pd.DataFrame(failure_rows).sort_values(["strict_event_f1", "strict_event_recall"], ascending=[True, True])
    failure_df.head(cfg.subject_failure_report_top_k).to_csv(fold_dir / "subject_failure_report.csv", index=False)

    strict_precision = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
    strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
    strict_f1 = 2 * strict_precision * strict_recall / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0.0

    tol_precision = tol_tp / (tol_tp + tol_fp) if (tol_tp + tol_fp) > 0 else 0.0
    tol_recall = tol_tp / (tol_tp + tol_fn) if (tol_tp + tol_fn) > 0 else 0.0
    tol_f1 = 2 * tol_precision * tol_recall / (tol_precision + tol_recall) if (tol_precision + tol_recall) > 0 else 0.0
    iou50_denom = 2 * iou50_tp + iou50_fp + iou50_fn
    event_iou_f1_0_5 = float(2 * iou50_tp / iou50_denom) if iou50_denom > 0 else 0.0

    overall_sample = sample_level_metrics(
        np.concatenate(test_vectors_truth),
        np.concatenate(test_vectors_pred),
        np.concatenate(test_vectors_prob),
    )

    if candidate_all_y:
        overall_candidate = candidate_classification_metrics(np.concatenate(candidate_all_y), np.concatenate(candidate_all_probs))
    else:
        overall_candidate = candidate_classification_metrics(np.zeros((0,)), np.zeros((0,)))

    fold_summary = {
        "fold": fold_idx,
        "n_test_subjects": len(test_ids),
        "B": best_B,
        "T": best_T,
        "lambda_thr": best_stage2.lambda_thr,
        "prob_thr": best_stage2.probability_threshold,
        "hysteresis_gap": best_stage2.hysteresis_gap,
        "final_merge_gap_sec": best_stage2.final_merge_gap_sec,
        "final_min_sec": best_stage2.final_min_sec,
        "final_max_sec": best_stage2.final_max_sec,
        "adaptive_threshold_quantile": best_stage2.adaptive_threshold_quantile,
        "strict_event_precision": strict_precision,
        "strict_event_recall": strict_recall,
        "strict_event_f1": strict_f1,
        "tolerant_event_precision": tol_precision,
        "tolerant_event_recall": tol_recall,
        "tolerant_event_f1": tol_f1,
        "event_iou_f1_0_5": event_iou_f1_0_5,
        "ACC": overall_sample["ACC"],
        "SEN": overall_sample["SEN"],
        "SPE": overall_sample["SPE"],
        "KAPPA": overall_sample["KAPPA"],
        "sample_f1": overall_sample["sample_f1"],
        **overall_candidate,
    }
    save_json(fold_summary, fold_dir / "overall_metrics.json")
    generate_fold_paper_outputs(fold_dir, cfg, logger)
    cleared_windows = clear_cached_candidate_windows(candidate_cache)
    if cleared_windows:
        logger.info("Fold %d | cleared %d cached candidate window matrices.", fold_idx, cleared_windows)
    return fold_summary


def run_segmentation_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    records: Dict[str, Any],
    cfg: Config,
    outdir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    validate_mass_only_config(cfg)
    validate_subject_splits([(train_ids, val_ids, test_ids)], list(records), require_five=False)
    logger.info("Fold %d: SSL restricted to MASS training subjects %s; DREAMS excluded.", fold_idx, train_ids)
    fold_dir = ensure_dir(outdir / f"fold_{fold_idx:02d}")
    save_json({"train": train_ids, "val": val_ids, "test": test_ids}, fold_dir / "fold_split.json")
    logger.info("Segmentation fold %d | train=%s | val=%s | test=%s", fold_idx, train_ids, val_ids, test_ids)

    train_windows = build_segmentation_windows(train_ids, records, cfg)
    val_windows = build_segmentation_windows(val_ids, records, cfg)
    if not train_windows or not val_windows:
        raise RuntimeError(f"Fold {fold_idx} has empty segmentation train/validation windows.")
    logger.info(
        "Fold %d | segmentation windows | train=%d | val=%d | window_sec=%.1f | overlap=%.2f",
        fold_idx,
        len(train_windows),
        len(val_windows),
        cfg.segmentation_window_sec,
        cfg.segmentation_overlap,
    )

    pos_weight = compute_segmentation_pos_weight(train_ids, records, cfg, device)
    ssl_windows = train_windows
    ssl_records = {sid: records[sid] for sid in train_ids}
    external_records: Dict[str, Any] = {}
    external_windows: List[Tuple[str, int, int]] = []
    if cfg.transformer_ssl_pretraining_enabled and cfg.transformer_ssl_external_data_dir.strip():
        external_dir = Path(cfg.transformer_ssl_external_data_dir)
        external_records = load_dreams_records(external_dir, cfg, logger)
        if not external_records:
            raise RuntimeError(f"No external DREAMS records were loaded from {external_dir}.")
        external_windows = build_segmentation_windows(
            sorted(external_records),
            external_records,
            cfg,
        )
        ssl_records = dict(records)
        ssl_records.update(external_records)
        ssl_windows = list(train_windows) + list(external_windows)
        logger.info(
            "Fold %d | external SSL data | records=%d | windows=%d | combined_windows=%d",
            fold_idx,
            len(external_records),
            len(external_windows),
            len(ssl_windows),
        )
    models: List[nn.Module] = []
    history: Dict[str, List[float]] = {}
    seed_offsets = tuple(cfg.segmentation_ensemble_seed_offsets) or (0,)
    for member_idx, seed_offset in enumerate(seed_offsets, start=1):
        member_seed = int(cfg.seed + seed_offset)
        set_seed(member_seed)
        member_uses_ssl = bool(
            cfg.transformer_ssl_pretraining_enabled
            and (
                cfg.transformer_ssl_ensemble_mode == "all"
                or (cfg.transformer_ssl_ensemble_mode == "first_only" and member_idx == 1)
            )
        )
        logger.info(
            "Fold %d | training segmentation ensemble member %d/%d | seed=%d | input_channels=%d | ssl=%s",
            fold_idx,
            member_idx,
            len(seed_offsets),
            member_seed,
            segmentation_input_channels(cfg),
            member_uses_ssl,
        )
        model = build_segmentation_model(cfg, device)
        warmstart_transformer_from_cnn_checkpoint(model, cfg, fold_idx, seed_offset, device, logger)
        ssl_history: List[float] = []
        if member_uses_ssl:
            ssl_loader = make_segmentation_loader(
                ssl_windows,
                ssl_records,
                cfg,
                device,
                training=False,
                shuffle=True,
            )
            try:
                ssl_history = pretrain_transformer_masked_patches(
                    model,
                    ssl_loader,
                    cfg,
                    device,
                    logger,
                )
            finally:
                shutdown_dataloader_workers(ssl_loader)
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg),
                        "training_subjects": list(train_ids)},
                       fold_dir / f"ssl_model_seed{seed_offset}.pt")
        member_uses_external_supervision = bool(
            cfg.transformer_external_supervised_pretraining_enabled
            and external_records
            and (
                cfg.transformer_external_supervised_ensemble_mode == "all"
                or (
                    cfg.transformer_external_supervised_ensemble_mode == "first_only"
                    and member_idx == 1
                )
            )
        )
        external_supervised_history: Dict[str, List[float]] = {}
        if member_uses_external_supervision:
            external_ids = sorted(external_records)
            if len(external_ids) < 2:
                raise RuntimeError("External supervised pretraining requires at least two DREAMS records.")
            n_external_val = min(
                max(1, int(cfg.transformer_external_supervised_validation_records)),
                len(external_ids) - 1,
            )
            external_train_ids = external_ids[:-n_external_val]
            external_val_ids = external_ids[-n_external_val:]
            external_train_windows = build_segmentation_windows(
                external_train_ids,
                external_records,
                cfg,
            )
            external_val_windows = build_segmentation_windows(
                external_val_ids,
                external_records,
                cfg,
            )
            if not external_train_windows or not external_val_windows:
                raise RuntimeError("External supervised pretraining produced empty DREAMS windows.")
            logger.info(
                "Fold %d | starting labelled DREAMS pretraining | member=%d/%d | "
                "train_records=%d | val_records=%d | epochs=%d | lr=%.6f",
                fold_idx,
                member_idx,
                len(seed_offsets),
                len(external_train_ids),
                len(external_val_ids),
                cfg.transformer_external_supervised_pretraining_epochs,
                cfg.transformer_external_supervised_learning_rate,
            )
            external_cfg = replace(
                cfg,
                segmentation_learning_rate=cfg.transformer_external_supervised_learning_rate,
                segmentation_early_stopping_patience=max(
                    2,
                    min(
                        cfg.segmentation_early_stopping_patience,
                        cfg.transformer_external_supervised_pretraining_epochs // 2,
                    ),
                ),
                save_checkpoints=False,
            )
            external_pos_weight = compute_segmentation_pos_weight(
                external_train_ids,
                external_records,
                external_cfg,
                device,
            )
            external_train_loader = make_segmentation_loader(
                external_train_windows,
                external_records,
                external_cfg,
                device,
                training=True,
                shuffle=True,
            )
            external_val_loader = make_segmentation_loader(
                external_val_windows,
                external_records,
                external_cfg,
                device,
                training=False,
                shuffle=False,
            )
            try:
                model, external_supervised_history = train_segmentation_model(
                    model,
                    external_train_loader,
                    external_val_loader,
                    external_cfg,
                    device,
                    logger,
                    fold_dir / f"unused_dreams_pretrain_seed{seed_offset}.pt",
                    external_pos_weight,
                    max_epochs=cfg.transformer_external_supervised_pretraining_epochs,
                )
            finally:
                shutdown_dataloader_workers(external_train_loader)
                shutdown_dataloader_workers(external_val_loader)
        train_loader = make_segmentation_loader(train_windows, records, cfg, device, training=True, shuffle=True)
        val_loader = make_segmentation_loader(val_windows, records, cfg, device, training=False, shuffle=False)
        try:
            model, member_history = train_segmentation_model(
                model,
                train_loader,
                val_loader,
                cfg,
                device,
                logger,
                fold_dir / f"best_segmentation_model_seed{seed_offset}.pt",
                pos_weight,
            )
        finally:
            shutdown_dataloader_workers(train_loader)
            shutdown_dataloader_workers(val_loader)
        if ssl_history:
            member_history[f"ssl_{cfg.transformer_ssl_objective}_loss"] = ssl_history
        for history_name, values in external_supervised_history.items():
            member_history[f"dreams_supervised_{history_name}"] = values
        if cfg.segmentation_hard_negative_mining_enabled and cfg.hard_negative_finetune_epochs > 0:
            pre_hn_state = clone_model_state(model)
            eval_loader = make_segmentation_loader(val_windows, records, cfg, device, training=False, shuffle=False)
            try:
                pre_hn_metrics = evaluate_segmentation_loader(model, eval_loader, cfg, device, pos_weight)
            finally:
                shutdown_dataloader_workers(eval_loader)
            pre_hn_score = float(pre_hn_metrics.get(cfg.segmentation_checkpoint_metric, float("nan")))
            if not math.isfinite(pre_hn_score):
                pre_hn_score = float(pre_hn_metrics.get("sample_pr_auc", pre_hn_metrics.get("sample_f1", float("-inf"))))
            hn_weights = segmentation_hard_negative_window_weights(model, train_windows, records, cfg, device)
            hn_count = int(np.sum(hn_weights > 1.0))
            if hn_count > 0:
                logger.info(
                    "Fold %d | member %d/%d | segmentation hard negatives=%d | boost=%.2f | finetune_epochs=%d",
                    fold_idx,
                    member_idx,
                    len(seed_offsets),
                    hn_count,
                    cfg.hard_negative_boost,
                    cfg.hard_negative_finetune_epochs,
                )
                hn_train_loader = make_segmentation_loader(
                    train_windows,
                    records,
                    cfg,
                    device,
                    training=True,
                    shuffle=True,
                    extra_weights=hn_weights,
                )
                hn_val_loader = make_segmentation_loader(val_windows, records, cfg, device, training=False, shuffle=False)
                try:
                    model, hn_history = train_segmentation_model(
                        model,
                        hn_train_loader,
                        hn_val_loader,
                        cfg,
                        device,
                        logger,
                        fold_dir / f"best_segmentation_model_seed{seed_offset}_hardneg.pt",
                        pos_weight,
                        max_epochs=cfg.hard_negative_finetune_epochs,
                    )
                finally:
                    shutdown_dataloader_workers(hn_train_loader)
                    shutdown_dataloader_workers(hn_val_loader)
                post_loader = make_segmentation_loader(val_windows, records, cfg, device, training=False, shuffle=False)
                try:
                    post_hn_metrics = evaluate_segmentation_loader(model, post_loader, cfg, device, pos_weight)
                finally:
                    shutdown_dataloader_workers(post_loader)
                post_hn_score = float(post_hn_metrics.get(cfg.segmentation_checkpoint_metric, float("nan")))
                if not math.isfinite(post_hn_score):
                    post_hn_score = float(post_hn_metrics.get("sample_pr_auc", post_hn_metrics.get("sample_f1", float("-inf"))))
                if cfg.hard_negative_accept_only_if_improved and post_hn_score < pre_hn_score + cfg.hard_negative_accept_margin:
                    model.load_state_dict(pre_hn_state)
                    logger.info(
                        "Fold %d | member %d/%d | segmentation hard-negative fine-tune rejected | before=%.4f | hardneg=%.4f | margin=%.4f",
                        fold_idx,
                        member_idx,
                        len(seed_offsets),
                        pre_hn_score,
                        post_hn_score,
                        cfg.hard_negative_accept_margin,
                    )
                else:
                    logger.info(
                        "Fold %d | member %d/%d | segmentation hard-negative fine-tune accepted | before=%.4f | hardneg=%.4f",
                        fold_idx,
                        member_idx,
                        len(seed_offsets),
                        pre_hn_score,
                        post_hn_score,
                    )
                member_history = merge_training_histories(member_history, hn_history, "hard_negative_finetune")
        logger.info(
            "Fold %d | completed segmentation ensemble member %d/%d | seed=%d",
            fold_idx,
            member_idx,
            len(seed_offsets),
            member_seed,
        )
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)},
                   fold_dir / f"best_segmentation_model_seed{seed_offset}.pt")
        models.append(model)
        prefix = f"seed{seed_offset}"
        for key, values in member_history.items():
            history[f"{prefix}_{key}"] = values
        if len(seed_offsets) == 1:
            history = member_history
    ensemble_weights = tune_segmentation_ensemble_weights(
        models,
        val_ids,
        records,
        cfg,
        device,
        logger,
        fold_dir,
    )
    save_json(
        {
            "seed_offsets": [int(x) for x in seed_offsets],
            "member_weights": [float(x) for x in ensemble_weights],
            "selection_metric": cfg.segmentation_ensemble_weight_selection_metric,
        },
        fold_dir / "segmentation_ensemble_weights.json",
    )
    logger.info("Fold %d | fitting segmentation probability calibration.", fold_idx)
    segmentation_calibrator = fit_segmentation_probability_calibrator(
        models,
        val_ids,
        records,
        cfg,
        device,
        logger,
        f"fold_{fold_idx:02d}_segmentation_val",
        member_weights=ensemble_weights,
    )
    logger.info("Fold %d | tuning segmentation decoding parameters.", fold_idx)
    seg_params, _ = tune_segmentation_threshold(
        models,
        val_ids,
        records,
        cfg,
        device,
        fold_dir,
        logger,
        calibrator=segmentation_calibrator,
        member_weights=ensemble_weights,
    )
    logger.info("Fold %d | completed segmentation decoding tuning.", fold_idx)
    save_json(asdict(seg_params), fold_dir / "segmentation_params.json")
    save_json(serialize_calibrator(segmentation_calibrator), fold_dir / "probability_calibrator.json")
    save_json(asdict(cfg), fold_dir / "config.json")
    if cfg.save_training_history:
        save_json(history, fold_dir / "training_history.json")

    subject_rows: List[Dict[str, Any]] = []
    reference_subject_rows: List[Dict[str, Any]] = []
    event_details: List[Dict[str, Any]] = []
    test_vectors_truth: List[np.ndarray] = []
    test_vectors_pred: List[np.ndarray] = []
    test_vectors_prob: List[np.ndarray] = []
    explainability_timelines: Dict[str, np.ndarray] = {}
    explainability_decoded_events: Dict[str, List[Tuple[float, float]]] = {}
    strict_tp = strict_fp = strict_fn = 0
    tol_tp = tol_fp = tol_fn = 0
    iou50_tp = iou50_fp = iou50_fn = 0
    val_density_target = float(np.mean([true_event_density_per_min(records[sid]) for sid in val_ids])) if val_ids else 0.0
    logger.info("Fold %d | validation true-event density target = %.3f events/min", fold_idx, val_density_target)

    for sid in test_ids:
        rec = records[sid]
        head_timelines = predict_segmentation_head_timelines_ensemble(
            models,
            rec,
            cfg,
            device,
            calibrator=segmentation_calibrator,
            member_weights=ensemble_weights,
        )
        probability_timeline = head_timelines["logits"]
        subject_prob_thr = resolve_segmentation_probability_threshold(
            probability_timeline,
            seg_params.probability_threshold,
            seg_params.adaptive_threshold_quantile,
            cfg.segmentation_adaptive_threshold_max_adjustment,
        )
        subject_prob_thr_base = subject_prob_thr
        subject_prob_thr = choose_density_adaptive_threshold(
            probability_timeline,
            rec,
            subject_prob_thr_base,
            seg_params,
            cfg,
            val_density_target,
        )
        pred_events, pred_binary = decode_segmentation_events_structured(
            probability_timeline,
            head_timelines.get("center_logits"),
            head_timelines.get("boundary_logits"),
            rec["sfreq"],
            subject_prob_thr,
            seg_params.final_min_sec,
            seg_params.final_merge_gap_sec,
            seg_params.boundary_gap,
            seg_params.boundary_max_extend_sec,
            seg_params.boundary_trim_sec,
            seg_params.event_min_mean_prob,
            seg_params.event_min_peak_prob,
            seg_params.final_max_sec,
            cfg,
        )
        pred_events_before_consensus = list(pred_events)
        consensus_info = {
            "consensus_enabled": bool(cfg.segmentation_candidate_consensus_enabled),
            "consensus_candidates": 0,
            "consensus_removed": 0,
        }
        if cfg.segmentation_candidate_consensus_enabled:
            pred_events, pred_binary, consensus_info = apply_segmentation_candidate_consensus(
                pred_events,
                probability_timeline,
                rec,
                cfg,
            )
        explainability_timelines[sid] = probability_timeline
        explainability_decoded_events[sid] = list(pred_events)
        strict = event_metrics(rec["true_events_stage"], pred_events, cfg.event_iou_thresholds[0])
        tolerant = tolerant_event_metrics(rec["true_events_stage"], pred_events, cfg.tolerant_match_sec)
        sample = sample_level_metrics(rec["truth_binary_stage"], pred_binary, probability_timeline)
        per_iou = {f"iou_{thr}": event_metrics(rec["true_events_stage"], pred_events, thr) for thr in cfg.event_iou_thresholds}
        iou50 = event_metrics(rec["true_events_stage"], pred_events, 0.5)
        true_durations = event_duration_stats(rec["true_events_stage"])
        pred_durations = event_duration_stats(pred_events)
        true_density = true_event_density_per_min(rec)
        pred_density = event_density_per_min(pred_events, rec)
        reference_metrics = []
        for reference_name, reference in evaluation_references(rec).items():
            reference_row = subject_reference_metrics(
                sid,
                reference_name,
                reference["events"],
                reference["binary"],
                pred_events,
                pred_binary,
                probability_timeline,
                cfg,
            )
            reference_subject_rows.append(reference_row)
            reference_metrics.append(reference_row)
        event_details.append(
            {
                "subject_id": sid,
                "n_true_events": len(rec["true_events_stage"]),
                "n_pred_events": len(pred_events),
                "strict": strict,
                "tolerant": tolerant,
                "per_iou": per_iou,
                "references": reference_metrics,
                "density": {
                    "true_events_per_min": true_density,
                    "pred_events_per_min": pred_density,
                    "pred_to_true_density_ratio": pred_density / max(true_density, 1e-8),
                    "validation_target_events_per_min": val_density_target,
                },
                "consensus": consensus_info,
            }
        )
        strict_tp += strict["tp"]
        strict_fp += strict["fp"]
        strict_fn += strict["fn"]
        tol_tp += tolerant["tp"]
        tol_fp += tolerant["fp"]
        tol_fn += tolerant["fn"]
        iou50_tp += iou50["tp"]
        iou50_fp += iou50["fp"]
        iou50_fn += iou50["fn"]
        test_vectors_truth.append(rec["truth_binary_stage"])
        test_vectors_pred.append(pred_binary)
        test_vectors_prob.append(probability_timeline)
        row = {
            "subject_id": sid,
            "n_true_events": len(rec["true_events_stage"]),
            "n_pred_events": len(pred_events),
            "strict_event_precision": strict["precision"],
            "strict_event_recall": strict["recall"],
            "strict_event_f1": strict["f1"],
            "tolerant_event_precision": tolerant["precision"],
            "tolerant_event_recall": tolerant["recall"],
            "tolerant_event_f1": tolerant["f1"],
            "ACC": sample["ACC"],
            "SEN": sample["SEN"],
            "SPE": sample["SPE"],
            "KAPPA": sample["KAPPA"],
            "sample_f1": sample["sample_f1"],
            "sample_auc": sample.get("sample_auc", float("nan")),
            "sample_pr_auc": sample.get("sample_pr_auc", float("nan")),
            "B": float("nan"),
            "T": float("nan"),
            "lambda_thr": float("nan"),
            "prob_thr": subject_prob_thr,
            "prob_thr_base": subject_prob_thr_base,
            "prob_thr_global": seg_params.probability_threshold,
            "hysteresis_gap": 0.0,
            "final_merge_gap_sec": seg_params.final_merge_gap_sec,
            "final_min_sec": seg_params.final_min_sec,
            "final_max_sec": seg_params.final_max_sec,
            "boundary_trim_sec": seg_params.boundary_trim_sec,
            "adaptive_threshold_quantile": seg_params.adaptive_threshold_quantile,
            "retained_minutes": retained_minutes(rec),
            "true_events_per_min": true_density,
            "pred_events_per_min": pred_density,
            "pred_to_true_density_ratio": pred_density / max(true_density, 1e-8),
            "validation_target_events_per_min": val_density_target,
            "true_median_duration_sec": true_durations["median_duration_sec"],
            "pred_median_duration_sec": pred_durations["median_duration_sec"],
            "duration_median_ratio": pred_durations["median_duration_sec"] / max(true_durations["median_duration_sec"], 1e-8)
            if math.isfinite(pred_durations["median_duration_sec"]) and math.isfinite(true_durations["median_duration_sec"])
            else float("nan"),
            "n_pred_events_before_consensus": len(pred_events_before_consensus),
            "consensus_candidates": consensus_info["consensus_candidates"],
            "consensus_removed": consensus_info["consensus_removed"],
        }
        subject_rows.append(row)
        if cfg.save_subject_level_outputs:
            if cfg.save_candidates:
                save_json(pred_events, fold_dir / f"{sid}_pred_events.json")
            if cfg.save_probabilities:
                probability_columns = {"probability": probability_timeline}
                if "center_logits" in head_timelines:
                    probability_columns["center_probability"] = head_timelines["center_logits"]
                if "boundary_logits" in head_timelines:
                    probability_columns["boundary_probability"] = head_timelines["boundary_logits"]
                pd.DataFrame(probability_columns).to_csv(fold_dir / f"{sid}_probability_timeline.csv", index=False)
                pd.DataFrame(
                    {
                        "truth": rec["truth_binary_stage"],
                        "truth_expert1": rec.get("truth_binary_e1_stage", rec["truth_binary_stage"]),
                        "truth_expert2": rec.get("truth_binary_e2_stage", rec["truth_binary_stage"]),
                        "truth_union": rec["truth_binary_stage"],
                        "pred_binary": pred_binary,
                        "probability": probability_timeline,
                        "center_probability": head_timelines.get("center_logits", np.zeros_like(probability_timeline)),
                        "boundary_probability": head_timelines.get("boundary_logits", np.zeros_like(probability_timeline)),
                        "stage1_candidate_binary": np.zeros_like(pred_binary),
                    }
                ).to_csv(fold_dir / f"{sid}_vectors.csv", index=False)

    pd.DataFrame(subject_rows).to_csv(fold_dir / "subject_metrics.csv", index=False)
    reference_subject_df = pd.DataFrame(reference_subject_rows)
    reference_subject_df.to_csv(fold_dir / "subject_metrics_by_reference.csv", index=False)
    reference_overall = aggregate_reference_metrics(reference_subject_rows)
    save_json(reference_overall, fold_dir / "overall_metrics_by_reference.json")
    pd.DataFrame(reference_overall.values()).to_csv(fold_dir / "overall_metrics_by_reference.csv", index=False)
    save_json(event_details, fold_dir / "event_metrics.json")
    failure_df = pd.DataFrame(
        [
            {
                "subject_id": row["subject_id"],
                "strict_event_f1": row["strict_event_f1"],
                "strict_event_recall": row["strict_event_recall"],
                "strict_event_precision": row["strict_event_precision"],
                "n_pred_events": row["n_pred_events"],
                "n_true_events": row["n_true_events"],
                "pred_to_true_density_ratio": row.get("pred_to_true_density_ratio", float("nan")),
                "duration_median_ratio": row.get("duration_median_ratio", float("nan")),
                "prob_thr": row.get("prob_thr", float("nan")),
            }
            for row in subject_rows
        ]
    ).sort_values(["strict_event_f1", "strict_event_recall"], ascending=[True, True])
    failure_df.head(cfg.subject_failure_report_top_k).to_csv(fold_dir / "subject_failure_report.csv", index=False)

    strict_precision = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
    strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
    strict_f1 = 2 * strict_precision * strict_recall / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0.0
    tol_precision = tol_tp / (tol_tp + tol_fp) if (tol_tp + tol_fp) > 0 else 0.0
    tol_recall = tol_tp / (tol_tp + tol_fn) if (tol_tp + tol_fn) > 0 else 0.0
    tol_f1 = 2 * tol_precision * tol_recall / (tol_precision + tol_recall) if (tol_precision + tol_recall) > 0 else 0.0
    iou50_denom = 2 * iou50_tp + iou50_fp + iou50_fn
    event_iou_f1_0_5 = float(2 * iou50_tp / iou50_denom) if iou50_denom > 0 else 0.0
    overall_sample = sample_level_metrics(
        np.concatenate(test_vectors_truth),
        np.concatenate(test_vectors_pred),
        np.concatenate(test_vectors_prob),
    )
    save_precision_recall_curve(
        np.concatenate(test_vectors_truth),
        np.concatenate(test_vectors_prob),
        fold_dir / "paper_precision_recall_curve.jpg",
        cfg,
        title=f"{fold_dir.name} Precision-Recall Curve",
    )
    fold_summary = {
        "fold": fold_idx,
        "n_test_subjects": len(test_ids),
        "B": float("nan"),
        "T": float("nan"),
        "lambda_thr": float("nan"),
        "prob_thr": seg_params.probability_threshold,
        "hysteresis_gap": 0.0,
        "final_merge_gap_sec": seg_params.final_merge_gap_sec,
        "final_min_sec": seg_params.final_min_sec,
        "final_max_sec": seg_params.final_max_sec,
        "boundary_trim_sec": seg_params.boundary_trim_sec,
        "adaptive_threshold_quantile": seg_params.adaptive_threshold_quantile,
        "strict_event_precision": strict_precision,
        "strict_event_recall": strict_recall,
        "strict_event_f1": strict_f1,
        "tolerant_event_precision": tol_precision,
        "tolerant_event_recall": tol_recall,
        "tolerant_event_f1": tol_f1,
        "event_iou_f1_0_5": event_iou_f1_0_5,
        "ACC": overall_sample["ACC"],
        "SEN": overall_sample["SEN"],
        "SPE": overall_sample["SPE"],
        "KAPPA": overall_sample["KAPPA"],
        "sample_f1": overall_sample["sample_f1"],
        "sample_auc": overall_sample.get("sample_auc", float("nan")),
        "sample_pr_auc": overall_sample.get("sample_pr_auc", float("nan")),
        "validation_target_events_per_min": val_density_target,
        "density_adaptive_threshold_enabled": float(cfg.segmentation_density_adaptive_threshold_enabled),
        "candidate_consensus_enabled": float(cfg.segmentation_candidate_consensus_enabled),
    }
    save_json(fold_summary, fold_dir / "overall_metrics.json")
    save_segmentation_feature_importance(models, cfg, fold_dir, logger)
    if cfg.run_explainability:
        run_frozen_fold_xai(models, train_ids, test_ids, records, cfg, fold_dir, device,
                            logger, seg_params, segmentation_calibrator, ensemble_weights, val_density_target)
    generate_fold_paper_outputs(fold_dir, cfg, logger)
    return fold_summary


def run_external_dreams_validation(cfg: Config, outdir: Path, device: torch.device, logger: logging.Logger) -> int:
    raise RuntimeError("External dataset access is disabled in the MASS-SS2-only pipeline.")
    data_dir = Path(cfg.external_data_dir or cfg.data_dir)
    checkpoint_dir = Path(cfg.external_checkpoint_dir)
    if not data_dir.exists():
        logger.error("DREAMS data directory does not exist: %s", data_dir)
        return 1
    if not checkpoint_dir.exists():
        logger.error("External checkpoint directory does not exist: %s", checkpoint_dir)
        return 1
    cfg.model_mode = "segmentation_unet"
    records = load_dreams_records(data_dir, cfg, logger)
    if not records:
        logger.error("No DREAMS excerpts could be loaded from %s", data_dir)
        return 1
    save_json(
        {
            sid: {
                "n_e1": len(rec["true_events_e1_stage"]),
                "n_e2": len(rec["true_events_e2_stage"]),
                "n_union": len(rec["true_events_stage"]),
                "samples": len(rec["truth_binary_stage"]),
                "sfreq": rec["sfreq"],
            }
            for sid, rec in records.items()
        },
        outdir / "dreams_dataset_summary.json",
    )
    models = load_external_segmentation_models(checkpoint_dir, cfg, device, logger)
    mass_seg_params = mean_segmentation_params_from_checkpoint_dir(checkpoint_dir, cfg)
    save_json(asdict(mass_seg_params), outdir / "external_mass_segmentation_params.json")
    logger.info(
        "External DREAMS MASS params | thr=%.3f merge_gap=%.3f min_sec=%.3f adaptive_q=%.3f",
        mass_seg_params.probability_threshold,
        mass_seg_params.final_merge_gap_sec,
        mass_seg_params.final_min_sec,
        mass_seg_params.adaptive_threshold_quantile,
    )
    logger.info("Predicting DREAMS probability timelines with fixed MASS checkpoints.")
    raw_timelines = {
        sid: predict_segmentation_timeline_ensemble(models, rec, cfg, device, calibrator=None)
        for sid, rec in sorted(records.items())
    }
    timelines = raw_timelines
    seg_params = mass_seg_params
    calibration_ids: List[str] = []
    test_ids: List[str] = sorted(records)
    if cfg.external_tune_decoding:
        calibration_ids = parse_external_record_ids(cfg.external_calibration_ids, records)
        test_ids = parse_external_record_ids(cfg.external_test_ids, records)
        if not calibration_ids or not test_ids:
            default_cal, default_test = default_external_calibration_split(records)
            calibration_ids = calibration_ids or default_cal
            test_ids = test_ids or default_test
        overlap = sorted(set(calibration_ids).intersection(test_ids))
        if overlap:
            logger.error("DREAMS calibration and test excerpts overlap: %s", overlap)
            return 1
        split = {"calibration_ids": calibration_ids, "test_ids": test_ids}
        save_json(split, outdir / "external_dreams_split.json")
        logger.info("DREAMS calibration split | calibration=%s | test=%s", calibration_ids, test_ids)
        if cfg.external_probability_calibration_enabled:
            cal_probs = np.concatenate([raw_timelines[sid] for sid in calibration_ids])
            cal_truth = np.concatenate([records[sid]["truth_binary_stage"] for sid in calibration_ids])
            calibrator = fit_probability_calibrator(cal_probs, cal_truth, cfg, logger, "external DREAMS calibration")
            if calibrator is not None:
                timelines = {
                    sid: apply_probability_calibrator(calibrator, raw_timeline)
                    for sid, raw_timeline in raw_timelines.items()
                }
                save_json({"enabled": True, "method": calibrator.get("method", "unknown")}, outdir / "external_probability_calibration.json")
            else:
                save_json({"enabled": False, "reason": "calibrator_not_fit"}, outdir / "external_probability_calibration.json")
        seg_params, _ = tune_segmentation_params_from_timelines(
            timelines,
            calibration_ids,
            records,
            cfg,
            outdir,
            logger,
        )
    save_json(asdict(seg_params), outdir / "external_segmentation_params.json")
    if cfg.external_tune_decoding:
        evaluate_external_dreams_records(records, calibration_ids, timelines, seg_params, cfg, outdir, "calibration", logger)
        subject_df, reference_df, reference_overall = evaluate_external_dreams_records(
            records,
            test_ids,
            timelines,
            seg_params,
            cfg,
            outdir,
            "test",
            logger,
        )
    else:
        subject_df, reference_df, reference_overall = evaluate_external_dreams_records(
            records,
            sorted(records),
            timelines,
            seg_params,
            cfg,
            outdir,
            "external",
            logger,
        )
    subject_df.to_csv(outdir / "subject_metrics.csv", index=False)
    reference_df.to_csv(outdir / "subject_metrics_by_reference.csv", index=False)
    save_json(reference_overall, outdir / "overall_metrics_by_reference.json")
    pd.DataFrame(reference_overall.values()).to_csv(outdir / "overall_metrics_by_reference.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for col in [c for c in subject_df.columns if pd.api.types.is_numeric_dtype(subject_df[c])]:
        summary_rows.append({"metric": col, "mean": float(subject_df[col].mean()), "std": float(subject_df[col].std(ddof=0))})
    pd.DataFrame(summary_rows).to_csv(outdir / "overall_summary.csv", index=False)
    save_reference_result_tables(outdir, logger)
    logger.info(
        "Finished DREAMS external validation | reported split=%s | union strict F1 mean=%.4f",
        "test" if cfg.external_tune_decoding else "external",
        float(subject_df["strict_event_f1"].mean()) if "strict_event_f1" in subject_df else float("nan"),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    default_cfg = Config(data_dir=".")
    p = argparse.ArgumentParser(description="Publication-oriented MASS-SS2 1D CNN sleep spindle detection pipeline.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=default_cfg.output_dir)
    p.add_argument("--mode", choices=("train", "xai", "all"), default="all")
    p.add_argument("--checkpoint_dir", default="", help="MASS-only run directory for xai mode")
    p.add_argument("--validate_only", action="store_true", help="Validate configuration without data access or training")
    p.add_argument("--annotation_mode", type=str, default=default_cfg.annotation_mode, choices=ANNOTATION_MODES)
    p.add_argument("--cv_mode", type=str, default=default_cfg.cv_mode, choices=CV_MODES)
    p.add_argument("--stage_mode", type=str, default=default_cfg.stage_mode, choices=STAGE_MODES)
    p.add_argument("--candidate_mode", type=str, default=default_cfg.candidate_mode, choices=CANDIDATE_MODES)
    p.add_argument("--model_mode", type=str, default=default_cfg.model_mode, choices=MODEL_MODES)
    p.add_argument("--channel_target", type=str, default=default_cfg.channel_target)
    p.add_argument("--segmentation_use_secondary_eeg_channel", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_secondary_eeg_channel)
    p.add_argument("--segmentation_secondary_channel_target", type=str, default=default_cfg.segmentation_secondary_channel_target)
    p.add_argument("--segmentation_secondary_gate_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_secondary_gate_enabled)
    p.add_argument("--segmentation_secondary_gate_initial", type=float, default=default_cfg.segmentation_secondary_gate_initial)
    p.add_argument("--resample_hz", type=float, default=default_cfg.resample_hz)
    p.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    p.add_argument("--epochs", type=int, default=default_cfg.epochs)
    p.add_argument("--tuning_epochs", type=int, default=default_cfg.tuning_epochs)
    p.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    p.add_argument("--weight_decay", type=float, default=default_cfg.weight_decay)
    p.add_argument("--early_stopping_patience", type=int, default=default_cfg.early_stopping_patience)
    p.add_argument("--scheduler_patience", type=int, default=default_cfg.scheduler_patience)
    p.add_argument("--scheduler_factor", type=float, default=default_cfg.scheduler_factor)
    p.add_argument("--grad_clip", type=float, default=default_cfg.grad_clip)
    p.add_argument("--loss_type", type=str, default=default_cfg.loss_type, choices=LOSS_TYPES)
    p.add_argument("--focal_gamma", type=float, default=default_cfg.focal_gamma)
    p.add_argument("--dice_weight", type=float, default=default_cfg.dice_weight)
    p.add_argument("--positive_weight_max", type=float, default=default_cfg.positive_weight_max)
    p.add_argument("--sigma_low_hz", type=float, default=default_cfg.sigma_low_hz)
    p.add_argument("--sigma_high_hz", type=float, default=default_cfg.sigma_high_hz)
    p.add_argument("--bandpass_low_hz", type=float, default=default_cfg.bandpass_low_hz)
    p.add_argument("--bandpass_high_hz", type=float, default=default_cfg.bandpass_high_hz)
    p.add_argument("--notch_enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--notch_hz", type=float, default=default_cfg.notch_hz)
    p.add_argument("--statistical_baseline_dir", type=str, default=default_cfg.statistical_baseline_dir)
    p.add_argument("--independent_run_seeds", type=str, default=None)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    p.add_argument("--device", type=str, default=default_cfg.device)
    p.add_argument("--num_workers", type=int, default=default_cfg.num_workers)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=default_cfg.pin_memory)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=default_cfg.persistent_workers)
    p.add_argument("--prefetch_factor", type=int, default=default_cfg.prefetch_factor)
    p.add_argument("--window_extraction_batch_size", type=int, default=default_cfg.window_extraction_batch_size)
    p.add_argument("--cache_candidate_windows", action=argparse.BooleanOptionalAction, default=default_cfg.cache_candidate_windows)
    p.add_argument("--enable_tf32", action=argparse.BooleanOptionalAction, default=default_cfg.enable_tf32)
    p.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=default_cfg.cudnn_benchmark)
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--start_fold", type=int, default=default_cfg.start_fold)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--save_subject_level_outputs", action=argparse.BooleanOptionalAction, default=default_cfg.save_subject_level_outputs)
    p.add_argument("--save_probabilities", action=argparse.BooleanOptionalAction, default=default_cfg.save_probabilities)
    p.add_argument("--save_candidates", action=argparse.BooleanOptionalAction, default=default_cfg.save_candidates)
    p.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=default_cfg.save_checkpoints)
    p.add_argument("--save_training_history", action=argparse.BooleanOptionalAction, default=default_cfg.save_training_history)
    p.add_argument("--save_candidate_diagnostics", action=argparse.BooleanOptionalAction, default=default_cfg.save_candidate_diagnostics)
    p.add_argument("--allow_implicit_spindle_labels", action=argparse.BooleanOptionalAction, default=default_cfg.allow_implicit_spindle_labels)
    p.add_argument("--require_dual_expert_files", action=argparse.BooleanOptionalAction, default=default_cfg.require_dual_expert_files)
    p.add_argument("--use_raw_channel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_sigma_channel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_teo_channel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_envelope_channel", action=argparse.BooleanOptionalAction, default=default_cfg.use_envelope_channel)
    p.add_argument("--use_candidate_feature_channels", action=argparse.BooleanOptionalAction, default=default_cfg.use_candidate_feature_channels)
    p.add_argument("--normalization_mode", type=str, default=default_cfg.normalization_mode, choices=NORMALIZATION_MODES)
    p.add_argument("--candidate_window_sec", type=float, default=default_cfg.candidate_window_sec)
    p.add_argument("--candidate_min_sec", type=float, default=default_cfg.candidate_min_sec)
    p.add_argument("--candidate_max_sec", type=float, default=default_cfg.candidate_max_sec)
    p.add_argument("--candidate_merge_gap_sec", type=float, default=default_cfg.candidate_merge_gap_sec)
    p.add_argument("--candidate_threshold_B", type=float, default=default_cfg.candidate_threshold_B)
    p.add_argument("--candidate_threshold_T", type=float, default=default_cfg.candidate_threshold_T)
    p.add_argument("--candidate_B_grid", type=str, default=None)
    p.add_argument("--candidate_T_grid", type=str, default=None)
    p.add_argument("--candidate_recall_boost_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.candidate_recall_boost_enabled)
    p.add_argument("--candidate_recall_boost_B_offset", type=float, default=default_cfg.candidate_recall_boost_B_offset)
    p.add_argument("--candidate_recall_boost_T_offset", type=float, default=default_cfg.candidate_recall_boost_T_offset)
    p.add_argument("--candidate_recall_boost_min_sec", type=float, default=default_cfg.candidate_recall_boost_min_sec)
    p.add_argument("--candidate_recall_boost_merge_gap_sec", type=float, default=default_cfg.candidate_recall_boost_merge_gap_sec)
    p.add_argument("--candidate_peak_proposals_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.candidate_peak_proposals_enabled)
    p.add_argument("--candidate_peak_z_threshold", type=float, default=default_cfg.candidate_peak_z_threshold)
    p.add_argument("--candidate_peak_duration_sec", type=float, default=default_cfg.candidate_peak_duration_sec)
    p.add_argument("--candidate_peak_min_distance_sec", type=float, default=default_cfg.candidate_peak_min_distance_sec)
    p.add_argument("--candidate_peak_max_per_min", type=float, default=default_cfg.candidate_peak_max_per_min)
    p.add_argument("--candidate_tuning_mean_f1_weight", type=float, default=default_cfg.candidate_tuning_mean_f1_weight)
    p.add_argument("--candidate_tuning_min_f1_weight", type=float, default=default_cfg.candidate_tuning_min_f1_weight)
    p.add_argument("--candidate_tuning_mean_recall_weight", type=float, default=default_cfg.candidate_tuning_mean_recall_weight)
    p.add_argument("--soft_candidate_labels_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.soft_candidate_labels_enabled)
    p.add_argument("--soft_candidate_label_lower_scale", type=float, default=default_cfg.soft_candidate_label_lower_scale)
    p.add_argument("--soft_candidate_label_gamma", type=float, default=default_cfg.soft_candidate_label_gamma)
    p.add_argument("--lambda_grid", type=str, default=None)
    p.add_argument("--threshold_grid", type=str, default=None)
    p.add_argument("--hysteresis_gap", type=float, default=default_cfg.hysteresis_gap)
    p.add_argument("--hysteresis_gap_grid", type=str, default=None)
    p.add_argument("--final_min_sec", type=float, default=default_cfg.final_min_sec)
    p.add_argument("--final_min_sec_grid", type=str, default=None)
    p.add_argument("--final_max_sec", type=float, default=default_cfg.final_max_sec)
    p.add_argument("--final_max_sec_grid", type=str, default=None)
    p.add_argument("--final_merge_gap_sec", type=float, default=default_cfg.final_merge_gap_sec)
    p.add_argument("--final_merge_gap_grid", type=str, default=None)
    p.add_argument("--adaptive_threshold_quantile_grid", type=str, default=None)
    p.add_argument("--adaptive_threshold_max_adjustment", type=float, default=default_cfg.adaptive_threshold_max_adjustment)
    p.add_argument("--stage3_selection_mean_f1_weight", type=float, default=default_cfg.stage3_selection_mean_f1_weight)
    p.add_argument("--stage3_selection_mean_recall_weight", type=float, default=default_cfg.stage3_selection_mean_recall_weight)
    p.add_argument("--stage3_selection_min_f1_weight", type=float, default=default_cfg.stage3_selection_min_f1_weight)
    p.add_argument("--stage3_selection_min_recall_weight", type=float, default=default_cfg.stage3_selection_min_recall_weight)
    p.add_argument("--stage3_recall_floor", type=float, default=default_cfg.stage3_recall_floor)
    p.add_argument("--stage3_recall_floor_penalty", type=float, default=default_cfg.stage3_recall_floor_penalty)
    p.add_argument("--stage3_prob_threshold_floor", type=float, default=default_cfg.stage3_prob_threshold_floor)
    p.add_argument("--stage3_prob_threshold_cap", type=float, default=default_cfg.stage3_prob_threshold_cap)
    p.add_argument("--stage3_support_smooth_sec", type=float, default=default_cfg.stage3_support_smooth_sec)
    p.add_argument("--stage3_support_relative_threshold", type=float, default=default_cfg.stage3_support_relative_threshold)
    p.add_argument("--event_iou_thresholds", type=str, default=None)
    p.add_argument("--tolerant_match_sec", type=float, default=default_cfg.tolerant_match_sec)
    p.add_argument("--model_channels", type=str, default=None, help="Comma-separated conv channel sizes, e.g. 32,64,96,128,256")
    p.add_argument("--model_kernel_size", type=int, default=default_cfg.model_kernel_size)
    p.add_argument("--model_dropout", type=float, default=default_cfg.model_dropout)
    p.add_argument("--model_use_residual", action=argparse.BooleanOptionalAction, default=default_cfg.model_use_residual)
    p.add_argument("--model_use_temporal_attention_gate", action=argparse.BooleanOptionalAction, default=default_cfg.model_use_temporal_attention_gate)
    p.add_argument("--segmentation_window_sec", type=float, default=default_cfg.segmentation_window_sec)
    p.add_argument("--segmentation_overlap", type=float, default=default_cfg.segmentation_overlap)
    p.add_argument("--segmentation_batch_size", type=int, default=default_cfg.segmentation_batch_size)
    p.add_argument("--segmentation_epochs", type=int, default=default_cfg.segmentation_epochs)
    p.add_argument("--segmentation_learning_rate", type=float, default=default_cfg.segmentation_learning_rate)
    p.add_argument("--segmentation_weight_decay", type=float, default=default_cfg.segmentation_weight_decay)
    p.add_argument("--segmentation_optimizer", type=str, default=default_cfg.segmentation_optimizer, choices=("adam", "adamw"))
    p.add_argument("--segmentation_early_stopping_patience", type=int, default=default_cfg.segmentation_early_stopping_patience)
    p.add_argument("--segmentation_scheduler_patience", type=int, default=default_cfg.segmentation_scheduler_patience)
    p.add_argument("--segmentation_scheduler_metric", type=str, default=default_cfg.segmentation_scheduler_metric, choices=("val_loss", "checkpoint"))
    p.add_argument("--segmentation_checkpoint_metric", type=str, default=default_cfg.segmentation_checkpoint_metric, choices=("window_event_f1", "sample_pr_auc", "sample_f1"))
    p.add_argument("--segmentation_checkpoint_threshold_grid", type=str, default=None)
    p.add_argument("--segmentation_threshold_grid", type=str, default=None)
    p.add_argument("--segmentation_min_sec", type=float, default=default_cfg.segmentation_min_sec)
    p.add_argument("--segmentation_min_sec_grid", type=str, default=None)
    p.add_argument("--segmentation_max_sec_grid", type=str, default=None)
    p.add_argument("--segmentation_merge_gap_sec", type=float, default=default_cfg.segmentation_merge_gap_sec)
    p.add_argument("--segmentation_merge_gap_grid", type=str, default=None)
    p.add_argument("--segmentation_boundary_gap_grid", type=str, default=None)
    p.add_argument("--segmentation_boundary_max_extend_sec_grid", type=str, default=None)
    p.add_argument("--segmentation_boundary_trim_sec_grid", type=str, default=None)
    p.add_argument("--segmentation_adaptive_threshold_quantile_grid", type=str, default=None)
    p.add_argument("--segmentation_adaptive_threshold_max_adjustment", type=float, default=default_cfg.segmentation_adaptive_threshold_max_adjustment)
    p.add_argument("--segmentation_density_penalty_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_density_penalty_enabled)
    p.add_argument("--segmentation_density_target_low", type=float, default=default_cfg.segmentation_density_target_low)
    p.add_argument("--segmentation_density_target_high", type=float, default=default_cfg.segmentation_density_target_high)
    p.add_argument("--segmentation_density_over_penalty", type=float, default=default_cfg.segmentation_density_over_penalty)
    p.add_argument("--segmentation_density_under_penalty", type=float, default=default_cfg.segmentation_density_under_penalty)
    p.add_argument("--segmentation_duration_penalty_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_duration_penalty_enabled)
    p.add_argument("--segmentation_duration_ratio_low", type=float, default=default_cfg.segmentation_duration_ratio_low)
    p.add_argument("--segmentation_duration_ratio_high", type=float, default=default_cfg.segmentation_duration_ratio_high)
    p.add_argument("--segmentation_duration_over_penalty", type=float, default=default_cfg.segmentation_duration_over_penalty)
    p.add_argument("--segmentation_duration_under_penalty", type=float, default=default_cfg.segmentation_duration_under_penalty)
    p.add_argument("--segmentation_selection_mean_f1_weight", type=float, default=default_cfg.segmentation_selection_mean_f1_weight)
    p.add_argument("--segmentation_selection_min_f1_weight", type=float, default=default_cfg.segmentation_selection_min_f1_weight)
    p.add_argument("--segmentation_selection_mean_precision_weight", type=float, default=default_cfg.segmentation_selection_mean_precision_weight)
    p.add_argument("--segmentation_selection_mean_recall_weight", type=float, default=default_cfg.segmentation_selection_mean_recall_weight)
    p.add_argument("--segmentation_selection_tolerant_f1_weight", type=float, default=default_cfg.segmentation_selection_tolerant_f1_weight)
    p.add_argument("--segmentation_precision_floor", type=float, default=default_cfg.segmentation_precision_floor)
    p.add_argument("--segmentation_min_precision_floor", type=float, default=default_cfg.segmentation_min_precision_floor)
    p.add_argument("--segmentation_precision_floor_penalty", type=float, default=default_cfg.segmentation_precision_floor_penalty)
    p.add_argument("--segmentation_min_precision_floor_penalty", type=float, default=default_cfg.segmentation_min_precision_floor_penalty)
    p.add_argument("--segmentation_recall_floor", type=float, default=default_cfg.segmentation_recall_floor)
    p.add_argument("--segmentation_min_recall_floor", type=float, default=default_cfg.segmentation_min_recall_floor)
    p.add_argument("--segmentation_recall_floor_penalty", type=float, default=default_cfg.segmentation_recall_floor_penalty)
    p.add_argument("--segmentation_min_recall_floor_penalty", type=float, default=default_cfg.segmentation_min_recall_floor_penalty)
    p.add_argument("--segmentation_f1_fallback_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_f1_fallback_enabled)
    p.add_argument("--segmentation_f1_fallback_margin", type=float, default=default_cfg.segmentation_f1_fallback_margin)
    p.add_argument("--segmentation_f1_fallback_min_precision", type=float, default=default_cfg.segmentation_f1_fallback_min_precision)
    p.add_argument("--segmentation_clip_uv", type=float, default=default_cfg.segmentation_clip_uv)
    p.add_argument("--segmentation_normalize_inputs", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_normalize_inputs)
    p.add_argument("--segmentation_use_attention_gates", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_attention_gates)
    p.add_argument("--segmentation_use_sigma_channel", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_sigma_channel)
    p.add_argument("--segmentation_use_slow_sigma_channel", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_slow_sigma_channel)
    p.add_argument("--segmentation_slow_sigma_low_hz", type=float, default=default_cfg.segmentation_slow_sigma_low_hz)
    p.add_argument("--segmentation_slow_sigma_high_hz", type=float, default=default_cfg.segmentation_slow_sigma_high_hz)
    p.add_argument("--segmentation_use_teo_channel", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_teo_channel)
    p.add_argument("--segmentation_use_envelope_channel", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_use_envelope_channel)
    p.add_argument("--segmentation_event_mean_prob_grid", type=str, default=None)
    p.add_argument("--segmentation_event_peak_prob_grid", type=str, default=None)
    p.add_argument("--segmentation_ensemble_seed_offsets", type=str, default=None)
    p.add_argument(
        "--segmentation_tune_ensemble_weights_enabled",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.segmentation_tune_ensemble_weights_enabled,
    )
    p.add_argument("--segmentation_ensemble_first_weight_grid", type=str, default=None)
    p.add_argument(
        "--segmentation_ensemble_weight_selection_metric",
        type=str,
        default=default_cfg.segmentation_ensemble_weight_selection_metric,
        choices=("sample_pr_auc", "event_f1"),
    )
    p.add_argument("--segmentation_probability_calibration_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_probability_calibration_enabled)
    p.add_argument("--segmentation_probability_calibration_method", type=str, default=default_cfg.segmentation_probability_calibration_method, choices=CALIBRATION_METHODS)
    p.add_argument("--segmentation_calibration_max_samples", type=int, default=default_cfg.segmentation_calibration_max_samples)
    p.add_argument("--segmentation_base_channels", type=int, default=default_cfg.segmentation_base_channels)
    p.add_argument("--segmentation_depth", type=int, default=default_cfg.segmentation_depth)
    p.add_argument("--segmentation_tversky_weight", type=float, default=default_cfg.segmentation_tversky_weight)
    p.add_argument("--segmentation_tversky_alpha", type=float, default=default_cfg.segmentation_tversky_alpha)
    p.add_argument("--segmentation_tversky_beta", type=float, default=default_cfg.segmentation_tversky_beta)
    p.add_argument("--segmentation_loss_type", type=str, default=default_cfg.segmentation_loss_type, choices=SEGMENTATION_LOSS_TYPES)
    p.add_argument("--segmentation_bce_weight", type=float, default=default_cfg.segmentation_bce_weight)
    p.add_argument("--segmentation_fpfn_weight", type=float, default=default_cfg.segmentation_fpfn_weight)
    p.add_argument("--segmentation_positive_window_sampling_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_positive_window_sampling_enabled)
    p.add_argument("--segmentation_positive_window_weight", type=float, default=default_cfg.segmentation_positive_window_weight)
    p.add_argument("--segmentation_subject_balanced_sampling_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_subject_balanced_sampling_enabled)
    p.add_argument("--segmentation_soft_labels_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_soft_labels_enabled)
    p.add_argument(
        "--segmentation_annotator_confidence_labels_enabled",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.segmentation_annotator_confidence_labels_enabled,
    )
    p.add_argument("--segmentation_single_expert_target_weight", type=float, default=default_cfg.segmentation_single_expert_target_weight)
    p.add_argument("--segmentation_annotator_agreement_tolerance_sec", type=float, default=default_cfg.segmentation_annotator_agreement_tolerance_sec)
    p.add_argument("--segmentation_soft_label_shoulder_sec", type=float, default=default_cfg.segmentation_soft_label_shoulder_sec)
    p.add_argument("--segmentation_soft_label_edge_value", type=float, default=default_cfg.segmentation_soft_label_edge_value)
    p.add_argument("--segmentation_temporal_bottleneck_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_temporal_bottleneck_enabled)
    p.add_argument("--segmentation_temporal_bottleneck_layers", type=int, default=default_cfg.segmentation_temporal_bottleneck_layers)
    p.add_argument("--segmentation_temporal_bottleneck_dropout", type=float, default=default_cfg.segmentation_temporal_bottleneck_dropout)
    p.add_argument("--segmentation_transformer_heads", type=int, default=default_cfg.segmentation_transformer_heads)
    p.add_argument("--segmentation_transformer_ff_multiplier", type=int, default=default_cfg.segmentation_transformer_ff_multiplier)
    p.add_argument("--segmentation_transformer_only_embed_dim", type=int, default=default_cfg.segmentation_transformer_only_embed_dim)
    p.add_argument("--segmentation_transformer_only_layers", type=int, default=default_cfg.segmentation_transformer_only_layers)
    p.add_argument("--segmentation_transformer_only_heads", type=int, default=default_cfg.segmentation_transformer_only_heads)
    p.add_argument("--segmentation_transformer_only_ff_multiplier", type=int, default=default_cfg.segmentation_transformer_only_ff_multiplier)
    p.add_argument("--segmentation_transformer_only_dropout", type=float, default=default_cfg.segmentation_transformer_only_dropout)
    p.add_argument("--segmentation_transformer_only_patch_size", type=int, default=default_cfg.segmentation_transformer_only_patch_size)
    p.add_argument("--segmentation_transformer_only_patch_stride", type=int, default=default_cfg.segmentation_transformer_only_patch_stride)
    p.add_argument("--segmentation_transformer_only_max_patches", type=int, default=default_cfg.segmentation_transformer_only_max_patches)
    p.add_argument("--segmentation_transformer_only_multiscale_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_transformer_only_multiscale_enabled)
    p.add_argument("--segmentation_transformer_only_long_patch_size", type=int, default=default_cfg.segmentation_transformer_only_long_patch_size)
    p.add_argument(
        "--segmentation_transformer_only_block_type",
        type=str,
        default=default_cfg.segmentation_transformer_only_block_type,
        choices=("transformer", "local_transformer", "conformer"),
    )
    p.add_argument(
        "--segmentation_transformer_only_local_attention_window",
        type=int,
        default=default_cfg.segmentation_transformer_only_local_attention_window,
    )
    p.add_argument(
        "--segmentation_transformer_only_local_global_stride",
        type=int,
        default=default_cfg.segmentation_transformer_only_local_global_stride,
    )
    p.add_argument("--segmentation_transformer_only_conv_kernel", type=int, default=default_cfg.segmentation_transformer_only_conv_kernel)
    p.add_argument("--segmentation_transformer_only_local_mixing", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_transformer_only_local_mixing)
    p.add_argument("--segmentation_transformer_only_local_mixing_dropout", type=float, default=default_cfg.segmentation_transformer_only_local_mixing_dropout)
    p.add_argument("--segmentation_transformer_only_patch_stats", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_transformer_only_patch_stats)
    p.add_argument(
        "--segmentation_transformer_only_positional_init",
        type=str,
        default=default_cfg.segmentation_transformer_only_positional_init,
        choices=("zeros", "sinusoidal"),
    )
    p.add_argument("--segmentation_transformer_only_spectral_tokens_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_transformer_only_spectral_tokens_enabled)
    p.add_argument("--segmentation_transformer_only_spectral_patch_size", type=int, default=default_cfg.segmentation_transformer_only_spectral_patch_size)
    p.add_argument("--segmentation_transformer_only_spectral_low_hz", type=float, default=default_cfg.segmentation_transformer_only_spectral_low_hz)
    p.add_argument("--segmentation_transformer_only_spectral_high_hz", type=float, default=default_cfg.segmentation_transformer_only_spectral_high_hz)
    p.add_argument("--segmentation_transformer_only_spectral_input_channels", type=int, default=default_cfg.segmentation_transformer_only_spectral_input_channels)
    p.add_argument("--segmentation_transformer_only_spectral_gate_initial", type=float, default=default_cfg.segmentation_transformer_only_spectral_gate_initial)
    p.add_argument("--segmentation_supervised_contrastive_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_supervised_contrastive_enabled)
    p.add_argument("--segmentation_supervised_contrastive_weight", type=float, default=default_cfg.segmentation_supervised_contrastive_weight)
    p.add_argument("--segmentation_supervised_contrastive_temperature", type=float, default=default_cfg.segmentation_supervised_contrastive_temperature)
    p.add_argument("--segmentation_supervised_contrastive_projection_dim", type=int, default=default_cfg.segmentation_supervised_contrastive_projection_dim)
    p.add_argument("--segmentation_supervised_contrastive_max_tokens", type=int, default=default_cfg.segmentation_supervised_contrastive_max_tokens)
    p.add_argument("--segmentation_supervised_contrastive_positive_threshold", type=float, default=default_cfg.segmentation_supervised_contrastive_positive_threshold)
    p.add_argument("--segmentation_event_geometry_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_event_geometry_enabled)
    p.add_argument("--segmentation_event_geometry_objectness_weight", type=float, default=default_cfg.segmentation_event_geometry_objectness_weight)
    p.add_argument("--segmentation_event_geometry_center_weight", type=float, default=default_cfg.segmentation_event_geometry_center_weight)
    p.add_argument("--segmentation_event_geometry_duration_weight", type=float, default=default_cfg.segmentation_event_geometry_duration_weight)
    p.add_argument("--segmentation_event_geometry_focal_gamma", type=float, default=default_cfg.segmentation_event_geometry_focal_gamma)
    p.add_argument("--transformer_ssl_pretraining_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.transformer_ssl_pretraining_enabled)
    p.add_argument("--transformer_ssl_pretraining_epochs", type=int, default=default_cfg.transformer_ssl_pretraining_epochs)
    p.add_argument("--transformer_ssl_objective", type=str, default=default_cfg.transformer_ssl_objective, choices=("masked_reconstruction", "contrastive"))
    p.add_argument("--transformer_ssl_mask_ratio", type=float, default=default_cfg.transformer_ssl_mask_ratio)
    p.add_argument("--transformer_ssl_learning_rate", type=float, default=default_cfg.transformer_ssl_learning_rate)
    p.add_argument("--transformer_ssl_temperature", type=float, default=default_cfg.transformer_ssl_temperature)
    p.add_argument("--transformer_ssl_external_data_dir", type=str, default=default_cfg.transformer_ssl_external_data_dir)
    p.add_argument(
        "--transformer_ssl_ensemble_mode",
        type=str,
        default=default_cfg.transformer_ssl_ensemble_mode,
        choices=("all", "first_only"),
    )
    p.add_argument("--transformer_external_supervised_pretraining_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.transformer_external_supervised_pretraining_enabled)
    p.add_argument("--transformer_external_supervised_pretraining_epochs", type=int, default=default_cfg.transformer_external_supervised_pretraining_epochs)
    p.add_argument("--transformer_external_supervised_learning_rate", type=float, default=default_cfg.transformer_external_supervised_learning_rate)
    p.add_argument("--transformer_external_supervised_validation_records", type=int, default=default_cfg.transformer_external_supervised_validation_records)
    p.add_argument(
        "--transformer_external_supervised_ensemble_mode",
        type=str,
        default=default_cfg.transformer_external_supervised_ensemble_mode,
        choices=("all", "first_only"),
    )
    p.add_argument("--transformer_warmstart_cnn_dir", type=str, default=default_cfg.transformer_warmstart_cnn_dir)
    p.add_argument("--transformer_warmstart_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.transformer_warmstart_enabled)
    p.add_argument("--segmentation_auxiliary_heads_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_auxiliary_heads_enabled)
    p.add_argument("--segmentation_center_loss_weight", type=float, default=default_cfg.segmentation_center_loss_weight)
    p.add_argument("--segmentation_boundary_loss_weight", type=float, default=default_cfg.segmentation_boundary_loss_weight)
    p.add_argument("--segmentation_edge_consistency_weight", type=float, default=default_cfg.segmentation_edge_consistency_weight)
    p.add_argument("--segmentation_multi_annotator_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_multi_annotator_enabled)
    p.add_argument("--segmentation_multi_annotator_loss_weight", type=float, default=default_cfg.segmentation_multi_annotator_loss_weight)
    p.add_argument("--segmentation_auxiliary_focal_gamma", type=float, default=default_cfg.segmentation_auxiliary_focal_gamma)
    p.add_argument("--segmentation_auxiliary_pos_weight", type=float, default=default_cfg.segmentation_auxiliary_pos_weight)
    p.add_argument("--segmentation_center_label_width_sec", type=float, default=default_cfg.segmentation_center_label_width_sec)
    p.add_argument("--segmentation_boundary_label_width_sec", type=float, default=default_cfg.segmentation_boundary_label_width_sec)
    p.add_argument("--segmentation_structured_decoder_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_structured_decoder_enabled)
    p.add_argument("--segmentation_center_threshold", type=float, default=default_cfg.segmentation_center_threshold)
    p.add_argument("--segmentation_boundary_threshold", type=float, default=default_cfg.segmentation_boundary_threshold)
    p.add_argument("--segmentation_center_min_distance_sec", type=float, default=default_cfg.segmentation_center_min_distance_sec)
    p.add_argument("--segmentation_boundary_max_shift_sec", type=float, default=default_cfg.segmentation_boundary_max_shift_sec)
    p.add_argument("--segmentation_density_adaptive_threshold_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_density_adaptive_threshold_enabled)
    p.add_argument("--segmentation_density_adaptive_low_multiplier", type=float, default=default_cfg.segmentation_density_adaptive_low_multiplier)
    p.add_argument("--segmentation_density_adaptive_high_multiplier", type=float, default=default_cfg.segmentation_density_adaptive_high_multiplier)
    p.add_argument("--segmentation_density_adaptive_max_adjustment", type=float, default=default_cfg.segmentation_density_adaptive_max_adjustment)
    p.add_argument("--segmentation_density_adaptive_steps", type=int, default=default_cfg.segmentation_density_adaptive_steps)
    p.add_argument("--segmentation_candidate_consensus_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_candidate_consensus_enabled)
    p.add_argument("--segmentation_candidate_consensus_B", type=float, default=default_cfg.segmentation_candidate_consensus_B)
    p.add_argument("--segmentation_candidate_consensus_T", type=float, default=default_cfg.segmentation_candidate_consensus_T)
    p.add_argument("--segmentation_candidate_consensus_overlap_sec", type=float, default=default_cfg.segmentation_candidate_consensus_overlap_sec)
    p.add_argument("--segmentation_candidate_consensus_keep_peak_prob", type=float, default=default_cfg.segmentation_candidate_consensus_keep_peak_prob)
    p.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=default_cfg.mixed_precision)
    p.add_argument("--augment_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.augment_enabled)
    p.add_argument("--augment_noise_std", type=float, default=default_cfg.augment_noise_std)
    p.add_argument("--augment_scale_std", type=float, default=default_cfg.augment_scale_std)
    p.add_argument("--augment_time_jitter_sec", type=float, default=default_cfg.augment_time_jitter_sec)
    p.add_argument("--augment_invert_probability", type=float, default=default_cfg.augment_invert_probability)
    p.add_argument("--balanced_sampling", action=argparse.BooleanOptionalAction, default=default_cfg.balanced_sampling)
    p.add_argument("--hard_negative_mining_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.hard_negative_mining_enabled)
    p.add_argument("--hard_negative_fraction", type=float, default=default_cfg.hard_negative_fraction)
    p.add_argument("--hard_negative_boost", type=float, default=default_cfg.hard_negative_boost)
    p.add_argument("--hard_negative_finetune_epochs", type=int, default=default_cfg.hard_negative_finetune_epochs)
    p.add_argument("--hard_negative_accept_only_if_improved", action=argparse.BooleanOptionalAction, default=default_cfg.hard_negative_accept_only_if_improved)
    p.add_argument("--hard_negative_accept_metric", type=str, default=default_cfg.hard_negative_accept_metric)
    p.add_argument("--hard_negative_accept_margin", type=float, default=default_cfg.hard_negative_accept_margin)
    p.add_argument("--segmentation_hard_negative_mining_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.segmentation_hard_negative_mining_enabled)
    p.add_argument("--segmentation_abort_below_fold_f1", type=float, default=default_cfg.segmentation_abort_below_fold_f1)
    p.add_argument("--segmentation_abort_check_fold", type=int, default=default_cfg.segmentation_abort_check_fold)
    p.add_argument("--probability_calibration_enabled", action=argparse.BooleanOptionalAction, default=default_cfg.probability_calibration_enabled)
    p.add_argument("--probability_calibration_method", type=str, default=default_cfg.probability_calibration_method, choices=CALIBRATION_METHODS)
    p.add_argument("--subject_failure_report_top_k", type=int, default=default_cfg.subject_failure_report_top_k)
    p.add_argument("--generate_paper_outputs", action=argparse.BooleanOptionalAction, default=default_cfg.generate_paper_outputs)
    p.add_argument("--paper_figure_dpi", type=int, default=default_cfg.paper_figure_dpi)
    p.add_argument("--paper_example_window_sec", type=float, default=default_cfg.paper_example_window_sec)
    p.add_argument("--paper_comparison_dirs", type=str, default=None)
    p.add_argument("--paper_comparison_only", action=argparse.BooleanOptionalAction, default=default_cfg.paper_comparison_only)
    p.add_argument("--paper_data_folder_name", type=str, default=default_cfg.paper_data_folder_name)
    p.add_argument("--run_explainability", action=argparse.BooleanOptionalAction, default=default_cfg.run_explainability)
    p.add_argument("--explainability_figure_dpi", type=int, default=default_cfg.explainability_figure_dpi)
    p.add_argument("--explainability_background_windows", type=int, default=default_cfg.explainability_background_windows)
    p.add_argument("--explainability_shap_windows", type=int, default=default_cfg.explainability_shap_windows)
    p.add_argument("--explainability_smoothgrad_samples", type=int, default=default_cfg.explainability_smoothgrad_samples)
    p.add_argument("--explainability_smoothgrad_noise_std", type=float, default=default_cfg.explainability_smoothgrad_noise_std)
    p.add_argument("--explainability_examples_per_type", type=int, default=default_cfg.explainability_examples_per_type)
    p.add_argument("--explainability_window_sec", type=float, default=default_cfg.explainability_window_sec)
    p.add_argument("--external_dataset", type=str, default=default_cfg.external_dataset, choices=("none", "dreams"))
    p.add_argument("--external_data_dir", type=str, default=default_cfg.external_data_dir)
    p.add_argument("--external_checkpoint_dir", type=str, default=default_cfg.external_checkpoint_dir)
    p.add_argument("--external_max_models", type=int, default=default_cfg.external_max_models)
    p.add_argument("--external_use_mass_fold_params", action=argparse.BooleanOptionalAction, default=default_cfg.external_use_mass_fold_params)
    p.add_argument("--external_default_threshold", type=float, default=default_cfg.external_default_threshold)
    p.add_argument("--external_default_merge_gap_sec", type=float, default=default_cfg.external_default_merge_gap_sec)
    p.add_argument("--external_default_min_sec", type=float, default=default_cfg.external_default_min_sec)
    p.add_argument("--external_default_adaptive_quantile", type=float, default=default_cfg.external_default_adaptive_quantile)
    p.add_argument("--external_tune_decoding", action=argparse.BooleanOptionalAction, default=default_cfg.external_tune_decoding)
    p.add_argument("--external_calibration_ids", type=str, default=default_cfg.external_calibration_ids)
    p.add_argument("--external_test_ids", type=str, default=default_cfg.external_test_ids)
    p.add_argument(
        "--external_probability_calibration_enabled",
        action=argparse.BooleanOptionalAction,
        default=default_cfg.external_probability_calibration_enabled,
    )
    p.add_argument("--remove_anomaly", action=argparse.BooleanOptionalAction, default=default_cfg.remove_anomaly)
    p.add_argument("--tune_candidate_params", action=argparse.BooleanOptionalAction, default=default_cfg.tune_candidate_params)
    p.add_argument("--resume_tuning", action=argparse.BooleanOptionalAction, default=default_cfg.resume_tuning)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    default_cfg = Config(data_dir=args.data_dir)
    model_channels = tuple(int(x.strip()) for x in args.model_channels.split(",")) if args.model_channels else default_cfg.model_channels
    return Config(**{
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'mode': args.mode,
        'checkpoint_dir': args.checkpoint_dir,
        'annotation_mode': args.annotation_mode,
        'cv_mode': args.cv_mode,
        'stage_mode': args.stage_mode,
        'candidate_mode': args.candidate_mode,
        'model_mode': args.model_mode,
        'channel_target': args.channel_target,
        'segmentation_use_secondary_eeg_channel': args.segmentation_use_secondary_eeg_channel,
        'segmentation_secondary_channel_target': args.segmentation_secondary_channel_target,
        'segmentation_secondary_gate_enabled': args.segmentation_secondary_gate_enabled,
        'segmentation_secondary_gate_initial': args.segmentation_secondary_gate_initial,
        'resample_hz': args.resample_hz,
        'max_folds': args.max_folds,
        'start_fold': args.start_fold,
        'debug': args.debug,
        'save_subject_level_outputs': args.save_subject_level_outputs,
        'save_probabilities': args.save_probabilities,
        'save_candidates': args.save_candidates,
        'save_checkpoints': args.save_checkpoints,
        'save_training_history': args.save_training_history,
        'save_candidate_diagnostics': args.save_candidate_diagnostics,
        'allow_implicit_spindle_labels': args.allow_implicit_spindle_labels,
        'require_dual_expert_files': args.require_dual_expert_files,
        'remove_anomaly': args.remove_anomaly,
        'bandpass_low_hz': args.bandpass_low_hz,
        'bandpass_high_hz': args.bandpass_high_hz,
        'notch_enabled': args.notch_enabled,
        'notch_hz': args.notch_hz,
        'sigma_low_hz': args.sigma_low_hz,
        'sigma_high_hz': args.sigma_high_hz,
        'normalization_mode': args.normalization_mode,
        'use_raw_channel': args.use_raw_channel,
        'use_sigma_channel': args.use_sigma_channel,
        'use_teo_channel': args.use_teo_channel,
        'use_envelope_channel': args.use_envelope_channel,
        'use_candidate_feature_channels': args.use_candidate_feature_channels,
        'candidate_window_sec': args.candidate_window_sec,
        'candidate_min_sec': args.candidate_min_sec,
        'candidate_max_sec': args.candidate_max_sec,
        'candidate_merge_gap_sec': args.candidate_merge_gap_sec,
        'candidate_threshold_B': args.candidate_threshold_B,
        'candidate_threshold_T': args.candidate_threshold_T,
        'tune_candidate_params': args.tune_candidate_params,
        'resume_tuning': args.resume_tuning,
        'candidate_B_grid': parse_float_tuple_arg(args.candidate_B_grid, default_cfg.candidate_B_grid),
        'candidate_T_grid': parse_float_tuple_arg(args.candidate_T_grid, default_cfg.candidate_T_grid),
        'candidate_recall_boost_enabled': args.candidate_recall_boost_enabled,
        'candidate_recall_boost_B_offset': args.candidate_recall_boost_B_offset,
        'candidate_recall_boost_T_offset': args.candidate_recall_boost_T_offset,
        'candidate_recall_boost_min_sec': args.candidate_recall_boost_min_sec,
        'candidate_recall_boost_merge_gap_sec': args.candidate_recall_boost_merge_gap_sec,
        'candidate_peak_proposals_enabled': args.candidate_peak_proposals_enabled,
        'candidate_peak_z_threshold': args.candidate_peak_z_threshold,
        'candidate_peak_duration_sec': args.candidate_peak_duration_sec,
        'candidate_peak_min_distance_sec': args.candidate_peak_min_distance_sec,
        'candidate_peak_max_per_min': args.candidate_peak_max_per_min,
        'candidate_tuning_mean_f1_weight': args.candidate_tuning_mean_f1_weight,
        'candidate_tuning_min_f1_weight': args.candidate_tuning_min_f1_weight,
        'candidate_tuning_mean_recall_weight': args.candidate_tuning_mean_recall_weight,
        'soft_candidate_labels_enabled': args.soft_candidate_labels_enabled,
        'soft_candidate_label_lower_scale': args.soft_candidate_label_lower_scale,
        'soft_candidate_label_gamma': args.soft_candidate_label_gamma,
        'lambda_grid': parse_float_tuple_arg(args.lambda_grid, default_cfg.lambda_grid),
        'threshold_grid': parse_float_tuple_arg(args.threshold_grid, default_cfg.threshold_grid),
        'hysteresis_gap': args.hysteresis_gap,
        'hysteresis_gap_grid': parse_float_tuple_arg(args.hysteresis_gap_grid, default_cfg.hysteresis_gap_grid),
        'final_min_sec': args.final_min_sec,
        'final_min_sec_grid': parse_float_tuple_arg(args.final_min_sec_grid, default_cfg.final_min_sec_grid),
        'final_max_sec': args.final_max_sec,
        'final_max_sec_grid': parse_float_tuple_arg(args.final_max_sec_grid, default_cfg.final_max_sec_grid),
        'final_merge_gap_sec': args.final_merge_gap_sec,
        'final_merge_gap_grid': parse_float_tuple_arg(args.final_merge_gap_grid, default_cfg.final_merge_gap_grid),
        'adaptive_threshold_quantile_grid': parse_float_tuple_arg(args.adaptive_threshold_quantile_grid, default_cfg.adaptive_threshold_quantile_grid),
        'adaptive_threshold_max_adjustment': args.adaptive_threshold_max_adjustment,
        'stage3_selection_mean_f1_weight': args.stage3_selection_mean_f1_weight,
        'stage3_selection_mean_recall_weight': args.stage3_selection_mean_recall_weight,
        'stage3_selection_min_f1_weight': args.stage3_selection_min_f1_weight,
        'stage3_selection_min_recall_weight': args.stage3_selection_min_recall_weight,
        'stage3_recall_floor': args.stage3_recall_floor,
        'stage3_recall_floor_penalty': args.stage3_recall_floor_penalty,
        'stage3_prob_threshold_floor': args.stage3_prob_threshold_floor,
        'stage3_prob_threshold_cap': args.stage3_prob_threshold_cap,
        'stage3_support_smooth_sec': args.stage3_support_smooth_sec,
        'stage3_support_relative_threshold': args.stage3_support_relative_threshold,
        'event_iou_thresholds': parse_float_tuple_arg(args.event_iou_thresholds, default_cfg.event_iou_thresholds),
        'tolerant_match_sec': args.tolerant_match_sec,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'tuning_epochs': args.tuning_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'scheduler_patience': args.scheduler_patience,
        'scheduler_factor': args.scheduler_factor,
        'grad_clip': args.grad_clip,
        'mixed_precision': args.mixed_precision,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers,
        'prefetch_factor': args.prefetch_factor,
        'window_extraction_batch_size': args.window_extraction_batch_size,
        'cache_candidate_windows': args.cache_candidate_windows,
        'enable_tf32': args.enable_tf32,
        'cudnn_benchmark': args.cudnn_benchmark,
        'loss_type': args.loss_type,
        'focal_gamma': args.focal_gamma,
        'dice_weight': args.dice_weight,
        'positive_weight_max': args.positive_weight_max,
        'statistical_baseline_dir': args.statistical_baseline_dir,
        'independent_run_seeds': parse_int_tuple_arg(args.independent_run_seeds, default_cfg.independent_run_seeds),
        'seed': args.seed,
        'device': args.device,
        'model_channels': model_channels,
        'model_kernel_size': args.model_kernel_size,
        'model_dropout': args.model_dropout,
        'model_use_residual': args.model_use_residual,
        'model_use_temporal_attention_gate': args.model_use_temporal_attention_gate,
        'segmentation_window_sec': args.segmentation_window_sec,
        'segmentation_overlap': args.segmentation_overlap,
        'segmentation_batch_size': args.segmentation_batch_size,
        'segmentation_epochs': args.segmentation_epochs,
        'segmentation_learning_rate': args.segmentation_learning_rate,
        'segmentation_weight_decay': args.segmentation_weight_decay,
        'segmentation_optimizer': args.segmentation_optimizer,
        'segmentation_early_stopping_patience': args.segmentation_early_stopping_patience,
        'segmentation_scheduler_patience': args.segmentation_scheduler_patience,
        'segmentation_scheduler_metric': args.segmentation_scheduler_metric,
        'segmentation_checkpoint_metric': args.segmentation_checkpoint_metric,
        'segmentation_checkpoint_threshold_grid': parse_float_tuple_arg(
            args.segmentation_checkpoint_threshold_grid,
            default_cfg.segmentation_checkpoint_threshold_grid,
        ),
        'segmentation_threshold_grid': parse_float_tuple_arg(args.segmentation_threshold_grid, default_cfg.segmentation_threshold_grid),
        'segmentation_min_sec': args.segmentation_min_sec,
        'segmentation_min_sec_grid': parse_float_tuple_arg(args.segmentation_min_sec_grid, default_cfg.segmentation_min_sec_grid),
        'segmentation_max_sec_grid': parse_float_tuple_arg(args.segmentation_max_sec_grid, default_cfg.segmentation_max_sec_grid),
        'segmentation_merge_gap_sec': args.segmentation_merge_gap_sec,
        'segmentation_merge_gap_grid': parse_float_tuple_arg(args.segmentation_merge_gap_grid, default_cfg.segmentation_merge_gap_grid),
        'segmentation_boundary_gap_grid': parse_float_tuple_arg(
            args.segmentation_boundary_gap_grid,
            default_cfg.segmentation_boundary_gap_grid,
        ),
        'segmentation_boundary_max_extend_sec_grid': parse_float_tuple_arg(
            args.segmentation_boundary_max_extend_sec_grid,
            default_cfg.segmentation_boundary_max_extend_sec_grid,
        ),
        'segmentation_boundary_trim_sec_grid': parse_float_tuple_arg(
            args.segmentation_boundary_trim_sec_grid,
            default_cfg.segmentation_boundary_trim_sec_grid,
        ),
        'segmentation_adaptive_threshold_quantile_grid': parse_float_tuple_arg(
            args.segmentation_adaptive_threshold_quantile_grid,
            default_cfg.segmentation_adaptive_threshold_quantile_grid,
        ),
        'segmentation_adaptive_threshold_max_adjustment': args.segmentation_adaptive_threshold_max_adjustment,
        'segmentation_density_penalty_enabled': args.segmentation_density_penalty_enabled,
        'segmentation_density_target_low': args.segmentation_density_target_low,
        'segmentation_density_target_high': args.segmentation_density_target_high,
        'segmentation_density_over_penalty': args.segmentation_density_over_penalty,
        'segmentation_density_under_penalty': args.segmentation_density_under_penalty,
        'segmentation_duration_penalty_enabled': args.segmentation_duration_penalty_enabled,
        'segmentation_duration_ratio_low': args.segmentation_duration_ratio_low,
        'segmentation_duration_ratio_high': args.segmentation_duration_ratio_high,
        'segmentation_duration_over_penalty': args.segmentation_duration_over_penalty,
        'segmentation_duration_under_penalty': args.segmentation_duration_under_penalty,
        'segmentation_selection_mean_f1_weight': args.segmentation_selection_mean_f1_weight,
        'segmentation_selection_min_f1_weight': args.segmentation_selection_min_f1_weight,
        'segmentation_selection_mean_precision_weight': args.segmentation_selection_mean_precision_weight,
        'segmentation_selection_mean_recall_weight': args.segmentation_selection_mean_recall_weight,
        'segmentation_selection_tolerant_f1_weight': args.segmentation_selection_tolerant_f1_weight,
        'segmentation_precision_floor': args.segmentation_precision_floor,
        'segmentation_min_precision_floor': args.segmentation_min_precision_floor,
        'segmentation_precision_floor_penalty': args.segmentation_precision_floor_penalty,
        'segmentation_min_precision_floor_penalty': args.segmentation_min_precision_floor_penalty,
        'segmentation_recall_floor': args.segmentation_recall_floor,
        'segmentation_min_recall_floor': args.segmentation_min_recall_floor,
        'segmentation_recall_floor_penalty': args.segmentation_recall_floor_penalty,
        'segmentation_min_recall_floor_penalty': args.segmentation_min_recall_floor_penalty,
        'segmentation_f1_fallback_enabled': args.segmentation_f1_fallback_enabled,
        'segmentation_f1_fallback_margin': args.segmentation_f1_fallback_margin,
        'segmentation_f1_fallback_min_precision': args.segmentation_f1_fallback_min_precision,
        'segmentation_clip_uv': args.segmentation_clip_uv,
        'segmentation_normalize_inputs': args.segmentation_normalize_inputs,
        'segmentation_use_attention_gates': args.segmentation_use_attention_gates,
        'segmentation_use_sigma_channel': args.segmentation_use_sigma_channel,
        'segmentation_use_slow_sigma_channel': args.segmentation_use_slow_sigma_channel,
        'segmentation_slow_sigma_low_hz': args.segmentation_slow_sigma_low_hz,
        'segmentation_slow_sigma_high_hz': args.segmentation_slow_sigma_high_hz,
        'segmentation_use_teo_channel': args.segmentation_use_teo_channel,
        'segmentation_use_envelope_channel': args.segmentation_use_envelope_channel,
        'segmentation_event_mean_prob_grid': parse_float_tuple_arg(
            args.segmentation_event_mean_prob_grid,
            default_cfg.segmentation_event_mean_prob_grid,
        ),
        'segmentation_event_peak_prob_grid': parse_float_tuple_arg(
            args.segmentation_event_peak_prob_grid,
            default_cfg.segmentation_event_peak_prob_grid,
        ),
        'segmentation_ensemble_seed_offsets': parse_int_tuple_arg(
            args.segmentation_ensemble_seed_offsets,
            default_cfg.segmentation_ensemble_seed_offsets,
        ),
        'segmentation_tune_ensemble_weights_enabled': args.segmentation_tune_ensemble_weights_enabled,
        'segmentation_ensemble_first_weight_grid': parse_float_tuple_arg(
            args.segmentation_ensemble_first_weight_grid,
            default_cfg.segmentation_ensemble_first_weight_grid,
        ),
        'segmentation_ensemble_weight_selection_metric': args.segmentation_ensemble_weight_selection_metric,
        'segmentation_probability_calibration_enabled': args.segmentation_probability_calibration_enabled,
        'segmentation_probability_calibration_method': args.segmentation_probability_calibration_method,
        'segmentation_calibration_max_samples': args.segmentation_calibration_max_samples,
        'segmentation_base_channels': args.segmentation_base_channels,
        'segmentation_depth': args.segmentation_depth,
        'segmentation_tversky_weight': args.segmentation_tversky_weight,
        'segmentation_tversky_alpha': args.segmentation_tversky_alpha,
        'segmentation_tversky_beta': args.segmentation_tversky_beta,
        'segmentation_loss_type': args.segmentation_loss_type,
        'segmentation_bce_weight': args.segmentation_bce_weight,
        'segmentation_fpfn_weight': args.segmentation_fpfn_weight,
        'segmentation_positive_window_sampling_enabled': args.segmentation_positive_window_sampling_enabled,
        'segmentation_positive_window_weight': args.segmentation_positive_window_weight,
        'segmentation_subject_balanced_sampling_enabled': args.segmentation_subject_balanced_sampling_enabled,
        'segmentation_soft_labels_enabled': args.segmentation_soft_labels_enabled,
        'segmentation_annotator_confidence_labels_enabled': args.segmentation_annotator_confidence_labels_enabled,
        'segmentation_single_expert_target_weight': args.segmentation_single_expert_target_weight,
        'segmentation_annotator_agreement_tolerance_sec': args.segmentation_annotator_agreement_tolerance_sec,
        'segmentation_soft_label_shoulder_sec': args.segmentation_soft_label_shoulder_sec,
        'segmentation_soft_label_edge_value': args.segmentation_soft_label_edge_value,
        'segmentation_temporal_bottleneck_enabled': args.segmentation_temporal_bottleneck_enabled,
        'segmentation_temporal_bottleneck_layers': args.segmentation_temporal_bottleneck_layers,
        'segmentation_temporal_bottleneck_dropout': args.segmentation_temporal_bottleneck_dropout,
        'segmentation_transformer_heads': args.segmentation_transformer_heads,
        'segmentation_transformer_ff_multiplier': args.segmentation_transformer_ff_multiplier,
        'segmentation_transformer_only_embed_dim': args.segmentation_transformer_only_embed_dim,
        'segmentation_transformer_only_layers': args.segmentation_transformer_only_layers,
        'segmentation_transformer_only_heads': args.segmentation_transformer_only_heads,
        'segmentation_transformer_only_ff_multiplier': args.segmentation_transformer_only_ff_multiplier,
        'segmentation_transformer_only_dropout': args.segmentation_transformer_only_dropout,
        'segmentation_transformer_only_patch_size': args.segmentation_transformer_only_patch_size,
        'segmentation_transformer_only_patch_stride': args.segmentation_transformer_only_patch_stride,
        'segmentation_transformer_only_max_patches': args.segmentation_transformer_only_max_patches,
        'segmentation_transformer_only_multiscale_enabled': args.segmentation_transformer_only_multiscale_enabled,
        'segmentation_transformer_only_long_patch_size': args.segmentation_transformer_only_long_patch_size,
        'segmentation_transformer_only_block_type': args.segmentation_transformer_only_block_type,
        'segmentation_transformer_only_local_attention_window': args.segmentation_transformer_only_local_attention_window,
        'segmentation_transformer_only_local_global_stride': args.segmentation_transformer_only_local_global_stride,
        'segmentation_transformer_only_conv_kernel': args.segmentation_transformer_only_conv_kernel,
        'segmentation_transformer_only_local_mixing': args.segmentation_transformer_only_local_mixing,
        'segmentation_transformer_only_local_mixing_dropout': args.segmentation_transformer_only_local_mixing_dropout,
        'segmentation_transformer_only_patch_stats': args.segmentation_transformer_only_patch_stats,
        'segmentation_transformer_only_positional_init': args.segmentation_transformer_only_positional_init,
        'segmentation_transformer_only_spectral_tokens_enabled': args.segmentation_transformer_only_spectral_tokens_enabled,
        'segmentation_transformer_only_spectral_patch_size': args.segmentation_transformer_only_spectral_patch_size,
        'segmentation_transformer_only_spectral_low_hz': args.segmentation_transformer_only_spectral_low_hz,
        'segmentation_transformer_only_spectral_high_hz': args.segmentation_transformer_only_spectral_high_hz,
        'segmentation_transformer_only_spectral_input_channels': args.segmentation_transformer_only_spectral_input_channels,
        'segmentation_transformer_only_spectral_gate_initial': args.segmentation_transformer_only_spectral_gate_initial,
        'segmentation_supervised_contrastive_enabled': args.segmentation_supervised_contrastive_enabled,
        'segmentation_supervised_contrastive_weight': args.segmentation_supervised_contrastive_weight,
        'segmentation_supervised_contrastive_temperature': args.segmentation_supervised_contrastive_temperature,
        'segmentation_supervised_contrastive_projection_dim': args.segmentation_supervised_contrastive_projection_dim,
        'segmentation_supervised_contrastive_max_tokens': args.segmentation_supervised_contrastive_max_tokens,
        'segmentation_supervised_contrastive_positive_threshold': args.segmentation_supervised_contrastive_positive_threshold,
        'segmentation_event_geometry_enabled': args.segmentation_event_geometry_enabled,
        'segmentation_event_geometry_objectness_weight': args.segmentation_event_geometry_objectness_weight,
        'segmentation_event_geometry_center_weight': args.segmentation_event_geometry_center_weight,
        'segmentation_event_geometry_duration_weight': args.segmentation_event_geometry_duration_weight,
        'segmentation_event_geometry_focal_gamma': args.segmentation_event_geometry_focal_gamma,
        'transformer_ssl_pretraining_enabled': args.transformer_ssl_pretraining_enabled,
        'transformer_ssl_pretraining_epochs': args.transformer_ssl_pretraining_epochs,
        'transformer_ssl_objective': args.transformer_ssl_objective,
        'transformer_ssl_mask_ratio': args.transformer_ssl_mask_ratio,
        'transformer_ssl_learning_rate': args.transformer_ssl_learning_rate,
        'transformer_ssl_temperature': args.transformer_ssl_temperature,
        'transformer_ssl_external_data_dir': args.transformer_ssl_external_data_dir,
        'transformer_ssl_ensemble_mode': args.transformer_ssl_ensemble_mode,
        'transformer_external_supervised_pretraining_enabled': args.transformer_external_supervised_pretraining_enabled,
        'transformer_external_supervised_pretraining_epochs': args.transformer_external_supervised_pretraining_epochs,
        'transformer_external_supervised_learning_rate': args.transformer_external_supervised_learning_rate,
        'transformer_external_supervised_validation_records': args.transformer_external_supervised_validation_records,
        'transformer_external_supervised_ensemble_mode': args.transformer_external_supervised_ensemble_mode,
        'transformer_warmstart_cnn_dir': args.transformer_warmstart_cnn_dir,
        'transformer_warmstart_enabled': args.transformer_warmstart_enabled,
        'segmentation_auxiliary_heads_enabled': args.segmentation_auxiliary_heads_enabled,
        'segmentation_center_loss_weight': args.segmentation_center_loss_weight,
        'segmentation_boundary_loss_weight': args.segmentation_boundary_loss_weight,
        'segmentation_edge_consistency_weight': args.segmentation_edge_consistency_weight,
        'segmentation_multi_annotator_enabled': args.segmentation_multi_annotator_enabled,
        'segmentation_multi_annotator_loss_weight': args.segmentation_multi_annotator_loss_weight,
        'segmentation_auxiliary_focal_gamma': args.segmentation_auxiliary_focal_gamma,
        'segmentation_auxiliary_pos_weight': args.segmentation_auxiliary_pos_weight,
        'segmentation_center_label_width_sec': args.segmentation_center_label_width_sec,
        'segmentation_boundary_label_width_sec': args.segmentation_boundary_label_width_sec,
        'segmentation_structured_decoder_enabled': args.segmentation_structured_decoder_enabled,
        'segmentation_center_threshold': args.segmentation_center_threshold,
        'segmentation_boundary_threshold': args.segmentation_boundary_threshold,
        'segmentation_center_min_distance_sec': args.segmentation_center_min_distance_sec,
        'segmentation_boundary_max_shift_sec': args.segmentation_boundary_max_shift_sec,
        'segmentation_density_adaptive_threshold_enabled': args.segmentation_density_adaptive_threshold_enabled,
        'segmentation_density_adaptive_low_multiplier': args.segmentation_density_adaptive_low_multiplier,
        'segmentation_density_adaptive_high_multiplier': args.segmentation_density_adaptive_high_multiplier,
        'segmentation_density_adaptive_max_adjustment': args.segmentation_density_adaptive_max_adjustment,
        'segmentation_density_adaptive_steps': args.segmentation_density_adaptive_steps,
        'segmentation_candidate_consensus_enabled': args.segmentation_candidate_consensus_enabled,
        'segmentation_candidate_consensus_B': args.segmentation_candidate_consensus_B,
        'segmentation_candidate_consensus_T': args.segmentation_candidate_consensus_T,
        'segmentation_candidate_consensus_overlap_sec': args.segmentation_candidate_consensus_overlap_sec,
        'segmentation_candidate_consensus_keep_peak_prob': args.segmentation_candidate_consensus_keep_peak_prob,
        'augment_enabled': args.augment_enabled,
        'augment_noise_std': args.augment_noise_std,
        'augment_scale_std': args.augment_scale_std,
        'augment_time_jitter_sec': args.augment_time_jitter_sec,
        'augment_invert_probability': args.augment_invert_probability,
        'balanced_sampling': args.balanced_sampling,
        'hard_negative_mining_enabled': args.hard_negative_mining_enabled,
        'hard_negative_fraction': args.hard_negative_fraction,
        'hard_negative_boost': args.hard_negative_boost,
        'hard_negative_finetune_epochs': args.hard_negative_finetune_epochs,
        'hard_negative_accept_only_if_improved': args.hard_negative_accept_only_if_improved,
        'hard_negative_accept_metric': args.hard_negative_accept_metric,
        'hard_negative_accept_margin': args.hard_negative_accept_margin,
        'segmentation_hard_negative_mining_enabled': args.segmentation_hard_negative_mining_enabled,
        'segmentation_abort_below_fold_f1': args.segmentation_abort_below_fold_f1,
        'segmentation_abort_check_fold': args.segmentation_abort_check_fold,
        'probability_calibration_enabled': args.probability_calibration_enabled,
        'probability_calibration_method': args.probability_calibration_method,
        'subject_failure_report_top_k': args.subject_failure_report_top_k,
        'generate_paper_outputs': args.generate_paper_outputs,
        'paper_figure_dpi': args.paper_figure_dpi,
        'paper_example_window_sec': args.paper_example_window_sec,
        'paper_comparison_dirs': parse_str_tuple_arg(args.paper_comparison_dirs, default_cfg.paper_comparison_dirs),
        'paper_comparison_only': args.paper_comparison_only,
        'paper_data_folder_name': args.paper_data_folder_name,
        'run_explainability': args.run_explainability,
        'explainability_figure_dpi': args.explainability_figure_dpi,
        'explainability_background_windows': args.explainability_background_windows,
        'explainability_shap_windows': args.explainability_shap_windows,
        'explainability_smoothgrad_samples': args.explainability_smoothgrad_samples,
        'explainability_smoothgrad_noise_std': args.explainability_smoothgrad_noise_std,
        'explainability_examples_per_type': args.explainability_examples_per_type,
        'explainability_window_sec': args.explainability_window_sec,
        'external_dataset': args.external_dataset,
        'external_data_dir': args.external_data_dir,
        'external_checkpoint_dir': args.external_checkpoint_dir,
        'external_max_models': args.external_max_models,
        'external_use_mass_fold_params': args.external_use_mass_fold_params,
        'external_default_threshold': args.external_default_threshold,
        'external_default_merge_gap_sec': args.external_default_merge_gap_sec,
        'external_default_min_sec': args.external_default_min_sec,
        'external_default_adaptive_quantile': args.external_default_adaptive_quantile,
        'external_tune_decoding': args.external_tune_decoding,
        'external_calibration_ids': args.external_calibration_ids,
        'external_test_ids': args.external_test_ids,
        'external_probability_calibration_enabled': args.external_probability_calibration_enabled,
    })


def cli_without_options(argv: Sequence[str], options: Set[str]) -> List[str]:
    """Remove value-taking CLI options in either --name value or --name=value form."""
    cleaned: List[str] = []
    idx = 0
    while idx < len(argv):
        token = str(argv[idx])
        matched = next((option for option in options if token == option or token.startswith(option + "=")), None)
        if matched is None:
            cleaned.append(token)
            idx += 1
            continue
        if token == matched and idx + 1 < len(argv):
            idx += 2
        else:
            idx += 1
    return cleaned


def run_independent_seed_experiments(cfg: Config, outdir: Path, logger: logging.Logger) -> int:
    seeds = tuple(dict.fromkeys(int(seed) for seed in cfg.independent_run_seeds))
    if not seeds:
        return 0
    base_args = cli_without_options(
        sys.argv[1:],
        {"--output_dir", "--seed", "--independent_run_seeds"},
    )
    run_frames: List[pd.DataFrame] = []
    run_dirs: List[Path] = []
    expected_folds = int(cfg.max_folds) if cfg.max_folds is not None else 5
    expected_folds = max(1, expected_folds)
    for run_index, run_seed in enumerate(seeds, start=1):
        run_dir = outdir / f"run_seed_{run_seed}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_args,
            "--output_dir",
            str(run_dir),
            "--seed",
            str(run_seed),
        ]
        logger.info("Independent run %d/%d | seed=%d | output=%s", run_index, len(seeds), run_seed, run_dir)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            logger.error("Independent run seed=%d failed with exit code %d.", run_seed, completed.returncode)
            return int(completed.returncode)
        summary_path = run_dir / "fold_summary.csv"
        if not summary_path.exists():
            logger.error("Independent run seed=%d did not produce %s.", run_seed, summary_path)
            return 2
        frame = pd.read_csv(summary_path)
        frame = add_event_iou_f1_from_fold_details(frame, run_dir)
        frame.insert(0, "run_seed", run_seed)
        run_frames.append(frame)
        run_dirs.append(run_dir)
        fold_one = frame[frame["fold"].astype(int) == int(cfg.segmentation_abort_check_fold)]
        guarded_f1 = (
            float(fold_one.iloc[0]["strict_event_f1"])
            if not fold_one.empty and "strict_event_f1" in fold_one else float("nan")
        )
        if (
            cfg.segmentation_abort_below_fold_f1 > 0.0
            and math.isfinite(guarded_f1)
            and guarded_f1 < cfg.segmentation_abort_below_fold_f1
        ):
            logger.error(
                "Stopping independent runs: seed=%d guarded fold %d strict F1 %.4f is below %.4f.",
                run_seed,
                cfg.segmentation_abort_check_fold,
                guarded_f1,
                cfg.segmentation_abort_below_fold_f1,
            )
            break
        if len(frame) < expected_folds:
            logger.error("Stopping independent runs: seed=%d completed only %d/%d folds.", run_seed, len(frame), expected_folds)
            break
    if not run_frames:
        return 2
    combined = pd.concat(run_frames, ignore_index=True)
    combined.to_csv(outdir / "independent_runs_fold_metrics.csv", index=False)
    numeric_cols = [
        column for column in combined.columns
        if column not in {"run_seed", "fold"} and pd.api.types.is_numeric_dtype(combined[column])
    ]
    averaged_folds = combined.groupby("fold", as_index=False)[numeric_cols].mean()
    averaged_folds.to_csv(outdir / "fold_metrics_averaged_across_seeds.csv", index=False)
    seed_rows: List[Dict[str, Any]] = []
    seed_metric_rows: List[Dict[str, Any]] = []
    for run_seed, seed_frame in combined.groupby("run_seed"):
        seed_row: Dict[str, Any] = {"run_seed": int(run_seed), "n_completed_folds": int(len(seed_frame))}
        for metric, label in STATISTICAL_METRICS:
            if metric not in seed_frame:
                continue
            values = pd.to_numeric(seed_frame[metric], errors="coerce").to_numpy(dtype=np.float64)
            seed_mean = float(np.nanmean(values))
            seed_row[metric] = seed_mean
            seed_metric_rows.append({"run_seed": int(run_seed), "metric": metric, "metric_label": label, "fold_mean": seed_mean})
        seed_rows.append(seed_row)
    pd.DataFrame(seed_rows).to_csv(outdir / "independent_runs_seed_summary.csv", index=False)
    seed_metric_df = pd.DataFrame(seed_metric_rows)
    variability_rows: List[Dict[str, Any]] = []
    for metric, label in STATISTICAL_METRICS:
        values = seed_metric_df.loc[seed_metric_df["metric"] == metric, "fold_mean"].to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue
        desc = metric_descriptive(values)
        variability_rows.append({"metric": metric, "metric_label": label, "n_seeds": len(values), **desc})
    variability = pd.DataFrame(variability_rows)
    variability.to_csv(outdir / "independent_runs_variability.csv", index=False)
    variability_notes = [
        "# Independent-run variability",
        "",
        "Each value entering this table is one seed's mean across its completed subject-level folds.",
        "Confidence intervals therefore quantify random-initialization variability without treating repeated folds as independent datasets.",
        "",
        dataframe_to_markdown(variability),
    ]
    (outdir / "independent_runs_variability.md").write_text("\n".join(variability_notes), encoding="utf-8")
    baseline_dir = Path(cfg.statistical_baseline_dir) if cfg.statistical_baseline_dir.strip() else None
    save_statistical_analysis(averaged_folds, run_dirs[0], baseline_dir, outdir, logger)
    save_json(
        {
            "requested_seeds": list(seeds),
            "completed_seeds": [int(frame["run_seed"].iloc[0]) for frame in run_frames],
            "inference_unit": "fold mean averaged across independent seeds",
            "baseline_dir": str(baseline_dir) if baseline_dir is not None else "",
        },
        outdir / "independent_runs_manifest.json",
    )
    logger.info("Finished %d independent seed runs under %s", len(run_frames), outdir)
    return 0


def validate_mass_only_config(cfg: Config) -> None:
    """Fail before external paths are even inspected; never silently accept transfer."""
    forbidden = (cfg.transformer_ssl_external_data_dir.strip(),
                 cfg.transformer_external_supervised_pretraining_enabled,
                 cfg.external_dataset != "none", cfg.external_data_dir.strip(),
                 cfg.external_checkpoint_dir.strip(), cfg.external_tune_decoding,
                 cfg.external_probability_calibration_enabled,
                 cfg.transformer_warmstart_enabled, cfg.transformer_warmstart_cnn_dir.strip())
    if any(forbidden):
        raise ValueError("MASS-only isolation: DREAMS/external data, transfer and warm-start checkpoints are disabled.")
    if is_transformer_only_segmentation_mode(cfg) and not cfg.transformer_ssl_pretraining_enabled:
        raise ValueError("MASS-only Transformer training requires fold-training-subject SSL.")
    if not cfg.save_checkpoints:
        raise ValueError("Checkpoint saving is required for reproducible frozen XAI.")
    if cfg.cv_mode == "loso" or not 1 <= cfg.start_fold <= 5:
        raise ValueError("This pipeline requires five-fold subject-wise CV with start_fold between 1 and 5.")
    if cfg.mode in {"xai", "all"} and not is_transformer_only_segmentation_mode(cfg):
        raise ValueError("XAI/all modes require model_mode=segmentation_transformer_only.")
    if cfg.segmentation_abort_below_fold_f1 > 0:
        raise ValueError("Held-out test F1 cannot control experiment stopping; disable the test-F1 regression guard.")


def validate_subject_splits(splits, subject_ids, require_five=True) -> None:
    if require_five and len(splits) != 5:
        raise ValueError("Exactly five subject-wise folds are required.")
    expected = set(subject_ids)
    test_seen = []
    for train, val, test in splits:
        groups = [set(train), set(val), set(test)]
        if any(not g for g in groups) or any(len(g) != len(v) for g, v in zip(groups, (train, val, test))):
            raise ValueError("Fold partitions must be non-empty and contain unique subjects.")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Subject leakage between training, validation and test partitions.")
        if set.union(*groups) != expected:
            raise ValueError("Fold partitions do not cover exactly the retained MASS subjects.")
        test_seen.extend(test)
    if require_five and (set(test_seen) != expected or len(test_seen) != len(expected)):
        raise ValueError("Each subject must appear in exactly one held-out test fold.")


def serialize_calibrator(calibrator):
    if calibrator is None:
        return {"method": "none"}
    model = calibrator["model"]
    if calibrator["method"] == "platt":
        return {"method": "platt", "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(), "classes": model.classes_.tolist()}
    return {"method": "isotonic", "x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()}


def restore_calibrator(state):
    if state["method"] == "none":
        return None
    if state["method"] == "platt":
        model = LogisticRegression()
        model.coef_ = np.asarray(state["coef"])
        model.intercept_ = np.asarray(state["intercept"])
        model.classes_ = np.asarray(state["classes"])
        model.n_features_in_ = 1
    elif state["method"] == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        # Rebuild only the saved interpolation knots; no labels or held-out data.
        model.X_thresholds_ = np.asarray(state["x"])
        model.y_thresholds_ = np.asarray(state["y"])
        model.X_min_, model.X_max_ = model.X_thresholds_[[0, -1]]
        model.f_ = scipy.interpolate.interp1d(model.X_thresholds_, model.y_thresholds_,
                                             kind="linear", bounds_error=False)
    else:
        raise ValueError("Unsupported saved calibrator method.")
    return {"method": state["method"], "model": model}


def frozen_decode(heads, rec, cfg, params, density, fixed_threshold=None):
    prob = heads["logits"]
    threshold = fixed_threshold
    if threshold is None:
        threshold = resolve_segmentation_probability_threshold(
            prob, params.probability_threshold, params.adaptive_threshold_quantile,
            cfg.segmentation_adaptive_threshold_max_adjustment)
        threshold = choose_density_adaptive_threshold(prob, rec, threshold, params, cfg, density)
    events, binary = decode_segmentation_events_structured(
        prob, heads.get("center_logits"), heads.get("boundary_logits"), rec["sfreq"],
        threshold, params.final_min_sec, params.final_merge_gap_sec, params.boundary_gap,
        params.boundary_max_extend_sec, params.boundary_trim_sec, params.event_min_mean_prob,
        params.event_min_peak_prob, params.final_max_sec, cfg)
    if cfg.segmentation_candidate_consensus_enabled:
        events, binary, _ = apply_segmentation_candidate_consensus(events, prob, rec, cfg)
    return events, binary, float(threshold)


def run_frozen_fold_xai(models, train_ids, test_ids, records, cfg, fold_dir, device,
                        logger, params, calibrator, weights, density):
    """Train-median replacement removes time-varying information in one input.

    A median in the model's preprocessed units is robust to spindle bursts and
    preserves the envelope's non-negative resting level (zero is not generally
    a valid envelope baseline). Only training subjects estimate it, each with
    equal weight. Other representations remain intact: this measures conditional
    reliance on redundant features, not removal of a physical EEG frequency.
    No retraining, calibration or decoder optimization is performed. Subject
    adaptive thresholds from full input are held fixed for paired ablations.
    """
    if set(train_ids) & set(test_ids):
        raise ValueError("XAI background/ablation baselines must exclude held-out subjects.")
    set_seed(cfg.seed)
    if not plotting_enabled():
        logger.warning("matplotlib unavailable: saving XAI tables and arrays without figures.")
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    target_dir = ensure_dir(fold_dir / "explainability")
    keys = {"raw": "segmentation_raw", "sigma": "segmentation_sigma",
            "slow_sigma": "segmentation_slow_sigma", "envelope": "segmentation_envelope"}
    active = segmentation_channel_names(cfg)
    baselines = {name: float(np.median([np.median(records[sid][key]) for sid in train_ids]))
                 for name, key in keys.items() if name in active}
    save_json({"strategy": "equal-subject training median in preprocessed units",
               "baselines": baselines, "missing_channels": [n for n in keys if n not in active],
               "interpretation": "conditional representation reliance; not causal physiological importance",
               "limitations": "constant replacement can be off-distribution; correlated channels retain shared information",
               "thresholds": "full-input subject thresholds frozen for all ablations",
               "fold_aggregation": "pooled event counts; cross-fold mean and sample SD (ddof=1)",
               "attribution_target": "weighted ensemble probability before calibration, mean over event interval",
               "background_subjects": list(train_ids), "evaluation_subjects": list(test_ids)},
              target_dir / "xai_protocol.json")
    timelines, decoded, rows = {}, {}, []
    for sid in test_ids:
        rec = records[sid]
        heads = predict_segmentation_head_timelines_ensemble(models, rec, cfg, device, calibrator, weights)
        events, binary, threshold = frozen_decode(heads, rec, cfg, params, density)

        # Diagnostic export from frozen, held-out inference.
        save_json(
            {
                "subject_id": sid,
                "true_events": [
                    [float(start), float(end)]
                    for start, end in rec["true_events_stage"]
                ],
                "pred_events": [
                    [float(start), float(end)]
                    for start, end in events
                ],
                "probability_threshold": float(threshold),
            },
            target_dir / f"{sid}_boundary_events.json",
        )

        timelines[sid], decoded[sid] = heads["logits"], events
        baseline = event_metrics(rec["true_events_stage"], events, cfg.event_iou_thresholds[0])
        for channel in ["full"] + list(baselines):
            if channel == "full":
                cur_heads, cur_events, cur_binary = heads, events, binary
            else:
                ablated = dict(rec)
                key = keys[channel]
                ablated[key] = np.full_like(rec[key], baselines[channel])
                cur_heads = predict_segmentation_head_timelines_ensemble(models, ablated, cfg, device, calibrator, weights)
                # Consensus side information remains frozen to the original record.
                cur_events, cur_binary, _ = frozen_decode(cur_heads, rec, cfg, params, density, threshold)
            metrics = event_metrics(rec["true_events_stage"], cur_events, cfg.event_iou_thresholds[0])
            row = {"subject_id": sid, "channel": channel, "threshold": threshold, **metrics}
            row.update({f"delta_{name}": metrics[name] - baseline[name] for name in ("precision", "recall", "f1")})
            row.update(sample_level_metrics(rec["truth_binary_stage"], cur_binary, cur_heads["logits"]))
            row["tolerant_f1"] = tolerant_event_metrics(rec["true_events_stage"], cur_events, cfg.tolerant_match_sec)["f1"]
            row["iou_f1_0_50"] = event_metrics(rec["true_events_stage"], cur_events, 0.5)["f1"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(target_dir / "ablation_by_subject.csv", index=False)
    run_segmentation_explainability(models, build_segmentation_windows(train_ids, records, cfg),
                                   test_ids, records, timelines, decoded, cfg, fold_dir, device, logger, weights)


def load_frozen_models(source, cfg, device):
    models = []
    for offset in cfg.segmentation_ensemble_seed_offsets or (0,):
        path = source / f"best_segmentation_model_seed{offset}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint.get("config")
        if saved is None:
            raise ValueError(f"Missing checkpoint configuration: {path}")
        validate_mass_only_config(Config(**saved))
        expected = asdict(cfg)
        runtime = {"output_dir", "data_dir", "mode", "checkpoint_dir", "device", "run_explainability"}
        if any(saved.get(k) != v for k, v in expected.items() if k not in runtime
               and not (isinstance(v, (tuple, list)) and list(saved.get(k, [])) == list(v))):
            raise ValueError(f"Checkpoint configuration differs from saved run: {path}")
        model = build_segmentation_model(cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


def run_checkpoint_explainability(cfg, outdir, device, logger):
    """Load frozen MASS artifacts; this path never calls training or tuning."""
    source = Path(cfg.checkpoint_dir)
    fold_dirs = sorted(p for p in source.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise ValueError("No saved MASS folds found.")
    splits = [load_json(folder / "fold_split.json") for folder in fold_dirs]
    ids = sorted(set().union(*(set(part) for split in splits for part in split.values())))
    validate_subject_splits([(v["train"], v["val"], v["test"]) for v in splits], ids,
                            require_five=cfg.max_folds is None)
    subjects = {s.subject_id: s for s in discover_subjects(Path(cfg.data_dir), logger)}
    records = {}
    needed = set().union(*(set(v["train"]) | set(v["test"]) for v in splits))
    for sid in sorted(needed):
        if sid not in subjects:
            raise ValueError(f"Saved MASS subject is missing: {sid}")
        rec = prepare_recording(subjects[sid], cfg, logger)
        if rec is None:
            raise ValueError(f"Could not prepare saved MASS subject: {sid}")
        records[sid] = rec
    for source_fold, split in zip(fold_dirs, splits):
        folder = ensure_dir(outdir / source_fold.name)
        weights = load_json(source_fold / "segmentation_ensemble_weights.json")
        if list(weights["seed_offsets"]) != list(cfg.segmentation_ensemble_seed_offsets or (0,)):
            raise ValueError("Saved ensemble order does not match checkpoints.")
        params = SegmentationParams(**load_json(source_fold / "segmentation_params.json"))
        calibrator = restore_calibrator(load_json(source_fold / "probability_calibrator.json"))
        summary = load_json(source_fold / "overall_metrics.json")
        models = load_frozen_models(source_fold, cfg, device)
        for filename in ("fold_split.json", "segmentation_params.json", "probability_calibrator.json",
                         "segmentation_ensemble_weights.json", "overall_metrics.json"):
            shutil.copy2(source_fold / filename, folder / filename)
        run_frozen_fold_xai(models, split["train"], split["test"], records, cfg, folder, device,
                            logger, params, calibrator, weights["member_weights"],
                            summary["validation_target_events_per_min"])
        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate_segmentation_explainability(outdir, cfg, logger)
    summarize_frozen_xai(outdir, cfg, logger)
    logger.info("Checkpoint-only XAI complete: no training, checkpoint selection or calibration performed.")
    return 0


def summarize_frozen_xai(outdir, cfg, logger):
    destination = ensure_dir(outdir / "comparison")
    architecture = cfg.segmentation_transformer_only_block_type
    # Existing --paper_comparison_dirs permits architecture comparisons without
    # rerunning experiments; these directories are read-only.
    frames = []
    for root in [outdir] + [Path(p) for p in cfg.paper_comparison_dirs]:
        config_file = root / "config.json"
        arch = load_json(config_file).get("segmentation_transformer_only_block_type", "unknown") if config_file.exists() else architecture
        for path in sorted(root.glob("fold_*/overall_metrics.json")):
            frames.append({"architecture": arch, "run": root.name, **load_json(path)})
    if frames:
        frame = pd.DataFrame(frames)
        frame.to_csv(destination / "architecture_fold_metrics.csv", index=False)
        columns = frame.select_dtypes(include="number").columns.difference(["fold"])
        frame.groupby(["architecture", "run"])[columns].agg(["mean", "std", "count"]).to_csv(destination / "architecture_mean_sd.csv")
    frames = []
    scores = []
    for path in sorted(outdir.glob("fold_*/explainability/ablation_by_subject.csv")):
        frame = pd.read_csv(path)
        frame["fold"] = path.parents[1].name
        frame["architecture"] = architecture
        frames.append(frame)
        local = path.parent / "local_explainability_scores.csv"
        if local.exists():
            score = pd.read_csv(local)
            score["fold"] = path.parents[1].name
            scores.append(score)
    if not frames:
        return
    subjects = pd.concat(frames, ignore_index=True)
    subjects.to_csv(destination / "ablation_by_subject.csv", index=False)
    folds = subjects.groupby(["architecture", "fold", "channel"])[["tp", "fp", "fn"]].sum().reset_index()
    folds["precision"] = folds.tp / (folds.tp + folds.fp).clip(lower=1)
    folds["recall"] = folds.tp / (folds.tp + folds.fn).clip(lower=1)
    folds["f1"] = 2 * folds.tp / (2 * folds.tp + folds.fp + folds.fn).clip(lower=1)
    for metric in ("precision", "recall", "f1"):
        base = folds[folds.channel == "full"].set_index("fold")[metric]
        folds["delta_" + metric] = folds[metric] - folds.fold.map(base)
    folds.to_csv(destination / "ablation_by_fold.csv", index=False)
    metrics = ["precision", "recall", "f1", "delta_precision", "delta_recall", "delta_f1"]
    summary = folds.groupby("channel")[metrics].agg(["mean", "std", "count"])
    summary.to_csv(destination / "ablation_mean_sd.csv")
    if scores:
        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(destination / "attribution_examples.csv", index=False)
        columns = ["smoothgrad_mean", "shap_abs_mean", "event_relevance_fraction", "shap_smoothgrad_temporal_spearman"]
        per_subject = scores.groupby(["fold", "subject_id", "channel"])[columns].mean()
        per_subject.to_csv(destination / "attribution_by_subject.csv")
        per_fold = per_subject.groupby(["fold", "channel"]).mean()
        per_fold.to_csv(destination / "attribution_by_fold.csv")
        per_fold.groupby("channel").agg(["mean", "std", "count"]).to_csv(destination / "attribution_mean_sd.csv")
        consistency = []
        for method in ("smoothgrad_mean", "shap_abs_mean"):
            profiles = per_subject.reset_index().pivot(index=["fold", "subject_id"], columns="channel", values=method)
            for left, right in itertools.combinations(range(len(profiles)), 2):
                a, b = profiles.iloc[left].to_numpy(), profiles.iloc[right].to_numpy()
                valid = np.isfinite(a) & np.isfinite(b)
                rho = float(scipy.stats.spearmanr(a[valid], b[valid]).statistic) if valid.sum() > 1 and np.std(a[valid]) > 0 and np.std(b[valid]) > 0 else float("nan")
                consistency.append({"method": method, "subject_a": profiles.index[left][1],
                                    "subject_b": profiles.index[right][1], "fold_a": profiles.index[left][0],
                                    "fold_b": profiles.index[right][0], "channel_rank_spearman": rho})
        pd.DataFrame(consistency, columns=["method", "subject_a", "subject_b", "fold_a", "fold_b", "channel_rank_spearman"]).to_csv(destination / "cross_subject_attribution_consistency.csv", index=False)
    if plotting_enabled():
        with plt.rc_context(explainability_plot_context(cfg)):
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
            summary = summary.drop(index="full", errors="ignore")
            for ax, metric in zip(axes, ("delta_precision", "delta_recall", "delta_f1")):
                ax.bar(summary.index, summary[(metric, "mean")], yerr=summary[(metric, "std")].fillna(0), capsize=3)
                ax.axhline(0, color="black", linewidth=0.6)
                ax.set_ylabel("Ablated minus full: " + metric[6:])
                ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(ensure_dir(outdir / "explainability_paper") / "paper_ablation_effects.jpg", dpi=cfg.explainability_figure_dpi, bbox_inches="tight")
            plt.close(fig)
    logger.info("Saved paired ablation effects and separate attribution summaries; SD uses ddof=1, fold counts are explicit.")


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    if cfg.mode == "xai":
        source = Path(cfg.checkpoint_dir)
        if not cfg.checkpoint_dir:
            raise ValueError("xai mode requires --checkpoint_dir.")
        saved = load_json(source / "config.json")
        cfg = replace(Config(**saved), data_dir=cfg.data_dir, output_dir=cfg.output_dir,
                      mode="xai", checkpoint_dir=str(source), device=cfg.device)
    validate_mass_only_config(cfg)
    cfg.run_explainability = cfg.mode in {"all", "xai"} or cfg.run_explainability
    if cfg.segmentation_use_secondary_eeg_channel and not cfg.segmentation_secondary_channel_target.strip():
        raise ValueError("A non-empty secondary EEG channel target is required when secondary-channel fusion is enabled.")
    if cfg.segmentation_secondary_gate_enabled and not cfg.segmentation_use_secondary_eeg_channel:
        raise ValueError("Secondary-channel gating requires --segmentation_use_secondary_eeg_channel.")
    if not 0.0 < cfg.segmentation_secondary_gate_initial < 1.0:
        raise ValueError("The initial secondary-channel gate must be strictly between 0 and 1.")
    if cfg.segmentation_use_slow_sigma_channel:
        nyquist = 0.5 * float(cfg.resample_hz)
        if not (0.0 < cfg.segmentation_slow_sigma_low_hz < cfg.segmentation_slow_sigma_high_hz < nyquist):
            raise ValueError(
                "Slow-sigma limits must satisfy 0 < low < high < Nyquist; "
                f"received {cfg.segmentation_slow_sigma_low_hz:g}-"
                f"{cfg.segmentation_slow_sigma_high_hz:g} Hz."
            )
    if any(value < 0.0 for value in cfg.segmentation_boundary_trim_sec_grid):
        raise ValueError("Segmentation boundary trim values must be non-negative.")
    if not cfg.segmentation_max_sec_grid or any(value < 0.0 for value in cfg.segmentation_max_sec_grid):
        raise ValueError("Segmentation maximum-duration values must be non-negative and non-empty.")
    if any(
        max_sec > 0.0 and max_sec < min(cfg.segmentation_min_sec_grid)
        for max_sec in cfg.segmentation_max_sec_grid
    ):
        raise ValueError("Each positive segmentation maximum duration must be at least the minimum-duration grid floor.")
    if not 0.0 < cfg.segmentation_duration_ratio_low < cfg.segmentation_duration_ratio_high:
        raise ValueError("Duration-ratio limits must satisfy 0 < low < high.")
    if cfg.segmentation_duration_over_penalty < 0.0 or cfg.segmentation_duration_under_penalty < 0.0:
        raise ValueError("Duration-ratio penalty weights must be non-negative.")
    if cfg.segmentation_transformer_only_local_attention_window < 1:
        raise ValueError("The local Transformer attention window must be at least 1 token.")
    if cfg.segmentation_transformer_only_spectral_tokens_enabled:
        if not is_transformer_only_segmentation_mode(cfg):
            raise ValueError("Spectral tokens are supported only for transformer-only segmentation.")
        if cfg.segmentation_transformer_only_spectral_patch_size < cfg.segmentation_transformer_only_patch_size:
            raise ValueError("The spectral patch cannot be shorter than the temporal Transformer patch.")
        nyquist = 0.5 * float(cfg.resample_hz)
        if not (
            0.0 <= cfg.segmentation_transformer_only_spectral_low_hz
            < cfg.segmentation_transformer_only_spectral_high_hz
            <= nyquist
        ):
            raise ValueError("Spectral token limits must satisfy 0 <= low < high <= Nyquist.")
        if cfg.segmentation_transformer_only_spectral_input_channels < 1:
            raise ValueError("Spectral token input channels must be at least 1.")
        if not 0.0 < cfg.segmentation_transformer_only_spectral_gate_initial < 1.0:
            raise ValueError("The initial spectral residual gate must be strictly between 0 and 1.")
    if cfg.segmentation_supervised_contrastive_enabled and not is_transformer_only_segmentation_mode(cfg):
        raise ValueError("Supervised token contrastive learning is supported only for transformer-only segmentation.")
    if cfg.segmentation_supervised_contrastive_weight < 0.0:
        raise ValueError("The supervised contrastive loss weight must be non-negative.")
    if cfg.segmentation_supervised_contrastive_temperature <= 0.0:
        raise ValueError("The supervised contrastive temperature must be positive.")
    if cfg.segmentation_supervised_contrastive_projection_dim < 2:
        raise ValueError("The supervised contrastive projection dimension must be at least 2.")
    if cfg.segmentation_supervised_contrastive_max_tokens < 4:
        raise ValueError("The supervised contrastive token limit must be at least 4.")
    if not 0.0 <= cfg.segmentation_supervised_contrastive_positive_threshold <= 1.0:
        raise ValueError("The supervised contrastive positive threshold must be in [0, 1].")
    if cfg.segmentation_event_geometry_enabled and not is_transformer_only_segmentation_mode(cfg):
        raise ValueError("Event-geometry supervision is supported only for transformer-only segmentation.")
    if any(
        weight < 0.0
        for weight in (
            cfg.segmentation_event_geometry_objectness_weight,
            cfg.segmentation_event_geometry_center_weight,
            cfg.segmentation_event_geometry_duration_weight,
            cfg.segmentation_event_geometry_focal_gamma,
        )
    ):
        raise ValueError("Event-geometry weights and focal gamma must be non-negative.")
    if (
        cfg.segmentation_transformer_only_multiscale_enabled
        and cfg.segmentation_transformer_only_long_patch_size <= cfg.segmentation_transformer_only_patch_size
    ):
        raise ValueError("The multiscale long patch must be larger than the primary Transformer patch.")
    if cfg.segmentation_transformer_only_local_global_stride < 0:
        raise ValueError("The local-Transformer global stride must be zero or positive.")
    if cfg.segmentation_weight_decay < 0.0:
        raise ValueError("Segmentation weight decay must be non-negative.")
    if cfg.segmentation_edge_consistency_weight < 0.0:
        raise ValueError("The segmentation edge-consistency weight must be non-negative.")
    if cfg.segmentation_multi_annotator_loss_weight < 0.0:
        raise ValueError("The multi-annotator loss weight must be non-negative.")
    if cfg.segmentation_multi_annotator_enabled:
        if not is_transformer_only_segmentation_mode(cfg):
            raise ValueError("Multi-annotator auxiliary supervision is supported only for transformer-only segmentation.")
        if not cfg.segmentation_auxiliary_heads_enabled:
            raise ValueError("Multi-annotator supervision requires segmentation auxiliary heads.")
    if cfg.segmentation_annotator_confidence_labels_enabled:
        if cfg.annotation_mode != "union":
            raise ValueError("Annotator-confidence supervision requires annotation_mode=union.")
        if not 0.0 < cfg.segmentation_single_expert_target_weight <= 1.0:
            raise ValueError("Single-expert target weight must satisfy 0 < weight <= 1.")
        if cfg.segmentation_annotator_agreement_tolerance_sec < 0.0:
            raise ValueError("Annotator agreement tolerance must be non-negative.")
    if cfg.transformer_ssl_pretraining_enabled and not is_transformer_only_segmentation_mode(cfg):
        raise ValueError("Masked-patch pretraining is supported only for transformer-only segmentation.")
    if cfg.transformer_ssl_pretraining_epochs < 1:
        raise ValueError("Masked-patch pretraining epochs must be at least 1.")
    if not 0.0 < cfg.transformer_ssl_mask_ratio < 1.0:
        raise ValueError("Masked-patch pretraining ratio must satisfy 0 < ratio < 1.")
    if cfg.transformer_ssl_learning_rate <= 0.0:
        raise ValueError("Transformer self-supervised pretraining learning rate must be positive.")
    if cfg.transformer_ssl_temperature <= 0.0:
        raise ValueError("Contrastive pretraining temperature must be positive.")
    if cfg.transformer_ssl_external_data_dir.strip():
        if not cfg.transformer_ssl_pretraining_enabled:
            raise ValueError("External SSL data require Transformer self-supervised pretraining to be enabled.")
        external_ssl_dir = Path(cfg.transformer_ssl_external_data_dir)
        if not external_ssl_dir.exists() or not external_ssl_dir.is_dir():
            raise ValueError(f"External SSL data directory does not exist: {external_ssl_dir}")
    if cfg.transformer_ssl_ensemble_mode == "first_only" and len(cfg.segmentation_ensemble_seed_offsets) < 2:
        raise ValueError("Transformer SSL ensemble mode 'first_only' requires at least two ensemble members.")
    if cfg.transformer_external_supervised_pretraining_enabled:
        if not is_transformer_only_segmentation_mode(cfg):
            raise ValueError("External supervised pretraining requires transformer-only segmentation.")
        if not cfg.transformer_ssl_external_data_dir.strip():
            raise ValueError("External supervised pretraining requires --transformer_ssl_external_data_dir.")
        if cfg.transformer_external_supervised_pretraining_epochs < 1:
            raise ValueError("External supervised pretraining epochs must be at least 1.")
        if cfg.transformer_external_supervised_learning_rate <= 0.0:
            raise ValueError("External supervised pretraining learning rate must be positive.")
        if cfg.transformer_external_supervised_validation_records < 1:
            raise ValueError("External supervised pretraining requires at least one validation record.")
        if (
            cfg.transformer_external_supervised_ensemble_mode == "first_only"
            and len(cfg.segmentation_ensemble_seed_offsets) < 2
        ):
            raise ValueError("External supervised ensemble mode 'first_only' requires at least two members.")
    if cfg.segmentation_tune_ensemble_weights_enabled:
        if len(cfg.segmentation_ensemble_seed_offsets) != 2:
            raise ValueError("Validation-tuned ensemble weighting currently requires exactly two ensemble members.")
        if not cfg.segmentation_ensemble_first_weight_grid:
            raise ValueError("The segmentation ensemble first-member weight grid cannot be empty.")
        if any(weight < 0.0 or weight > 1.0 for weight in cfg.segmentation_ensemble_first_weight_grid):
            raise ValueError("Segmentation ensemble first-member weights must lie in [0, 1].")
    if cfg.segmentation_ensemble_weight_selection_metric not in {"sample_pr_auc", "event_f1"}:
        raise ValueError("Unknown segmentation ensemble weight selection metric.")
    paper_folder = Path(cfg.paper_data_folder_name)
    if not cfg.paper_data_folder_name.strip() or paper_folder.is_absolute() or ".." in paper_folder.parts:
        raise ValueError("paper_data_folder_name must be a non-empty relative folder inside the output directory.")
    if cfg.segmentation_abort_check_fold < 1:
        raise ValueError("The segmentation regression-guard fold must be at least 1.")
    set_seed(cfg.seed)
    if args.validate_only:
        print(json.dumps(asdict(cfg), indent=2))
        print("VALIDATED: MASS-SS2 only; DREAMS/external data excluded; no data loaded or training started.")
        return 0
    outdir = Path(cfg.output_dir)
    if outdir.exists() and cfg.start_fold == 1:
        raise FileExistsError(
            f"Output directory already exists: {outdir}. "
            "Use a new directory or resume from a later fold."
        )
    outdir.mkdir(parents=True, exist_ok=True)
        
    logger = init_logging(outdir, cfg.debug)
    logger.info("MASS-SS2 ONLY: DREAMS/external SSL, supervised transfer, loading and calibration are disabled. SSL uses fold-training subjects only.")
    save_json(asdict(cfg), outdir / "config.json")
    save_json(package_versions(), outdir / "package_versions.json")
    save_json({"argv": sys.argv[1:], "dataset": "MASS-SS2", "external_data_allowed": False,
               "ssl_subjects": "fold training only", "selection_subjects": "fold validation only",
               "test_usage": "fixed evaluation and post-hoc explanation only"}, outdir / "run_manifest.json")
    if cfg.paper_comparison_only:
        if not cfg.paper_comparison_dirs:
            raise ValueError("Comparison-only reporting requires --paper_comparison_dirs.")
        save_transformer_comparison_outputs(outdir, cfg, logger)
        collect_paper_artifacts(outdir, cfg, logger)
        logger.info("Finished Transformer comparison-only paper report under %s", outdir)
        return 0
    if cfg.independent_run_seeds and cfg.mode != "xai":
        return run_independent_seed_experiments(cfg, outdir, logger)

    if cfg.segmentation_structured_decoder_enabled and not cfg.segmentation_auxiliary_heads_enabled:
        logger.warning("Structured decoding requested without auxiliary heads; falling back to probability-only decoding.")
    if is_transformer_only_segmentation_mode(cfg) and cfg.segmentation_soft_labels_enabled:
        logger.warning(
            "Transformer-only soft boundary labels are enabled; monitor predicted/reference duration ratios because label shoulders can lengthen events."
        )

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    logger.info("Using device: %s", device)
    setup_torch_runtime(cfg, device, logger)

    if cfg.external_dataset == "dreams":
        return run_external_dreams_validation(cfg, outdir, device, logger)

    if cfg.mode == "xai":
        return run_checkpoint_explainability(cfg, outdir, device, logger)

    data_dir = Path(cfg.data_dir)
    subjects = discover_subjects(data_dir, logger)
    if not subjects:
        logger.error("No MASS-SS2 subjects found in %s", data_dir)
        return 1

    records_all: Dict[str, Any] = {}
    for subject in subjects:
        if not subject_has_required_annotation_files(subject, cfg):
            logger.info("Skipping %s because required spindle expert files are missing.", subject.subject_id)
            continue
        rec = prepare_recording(subject, cfg, logger)
        if rec is not None:
            records_all[subject.subject_id] = rec

    all_subjects = sorted(subject.subject_id for subject in subjects)
    retained = sorted(records_all.keys())
    excluded = [sid for sid in all_subjects if sid not in retained]
    logger.info("Retained analysis subjects (%d): %s", len(retained), retained)
    logger.info("Excluded subjects (%d): %s", len(excluded), excluded)
    save_subject_selection_summary(all_subjects, retained, excluded, records_all, outdir)

    records = {sid: records_all[sid] for sid in retained}
    subject_ids = sorted(records.keys())

    if cfg.cv_mode == "paper_exact_e12":
        splits = exact_paper_splits(subject_ids)
    elif cfg.cv_mode == "random_fivefold_e12":
        if len(subject_ids) != 15:
            raise RuntimeError(f"random_fivefold_e12 expects 15 retained E1+E2 subjects, got {len(subject_ids)}.")
        splits = random_fivefold_e12_splits(subject_ids, cfg.seed)
    elif cfg.cv_mode == "loso":
        splits = loso_subject_splits(subject_ids, cfg.seed)
    else:
        splits = random_fivefold_e12_splits(subject_ids, cfg.seed)

    validate_subject_splits(splits, subject_ids)
    candidate_cache: Dict[Tuple[str, float, float, str], Dict[str, Any]] = {}
    fold_results: List[Dict[str, Any]] = []
    start_fold = max(1, int(cfg.start_fold))
    if start_fold > 1:
        logger.info("Resuming from fold %d; earlier folds will be loaded from existing summaries.", start_fold)
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(splits, start=1):
        if fold_idx < start_fold:
            continue
        if cfg.max_folds is not None and fold_idx > cfg.max_folds:
            break
        if is_segmentation_mode(cfg):
            fold_result = run_segmentation_fold(fold_idx, train_ids, val_ids, test_ids, records, cfg, outdir, device, logger)
        else:
            fold_result = run_fold(fold_idx, train_ids, val_ids, test_ids, records, cfg, outdir, device, logger, candidate_cache)
        fold_results.append(fold_result)
        fold_f1 = float(fold_result.get("strict_event_f1", float("nan")))
        if (
            is_segmentation_mode(cfg)
            and fold_idx == cfg.segmentation_abort_check_fold
            and cfg.segmentation_abort_below_fold_f1 > 0.0
            and math.isfinite(fold_f1)
            and fold_f1 < cfg.segmentation_abort_below_fold_f1
        ):
            logger.error(
                "Stopping remaining folds: guarded fold %d strict F1 %.4f is below regression guard %.4f.",
                fold_idx,
                fold_f1,
                cfg.segmentation_abort_below_fold_f1,
            )
            break

    target_fold_count = min(len(splits), cfg.max_folds if cfg.max_folds is not None else len(splits))
    completed_fold_results = load_completed_fold_summaries(outdir, target_fold_count, logger)
    completed_fold_ids = {int(row.get("fold", -1)) for row in completed_fold_results}
    current_fold_ids = {int(row.get("fold", -1)) for row in fold_results}
    if fold_results:
        logger.info("Executed folds in this run: %s", sorted(current_fold_ids))
    if not completed_fold_results:
        logger.error("No completed fold summaries were available.")
        return 2
    if len(completed_fold_ids) < target_fold_count:
        logger.warning(
            "Only %d/%d requested fold summaries are complete: %s",
            len(completed_fold_ids),
            target_fold_count,
            sorted(completed_fold_ids),
        )

    fold_df = pd.DataFrame(completed_fold_results)
    fold_df.to_csv(outdir / "fold_summary.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    numeric_cols = [c for c in fold_df.columns if pd.api.types.is_numeric_dtype(fold_df[c])]
    for col in numeric_cols:
        summary_rows.append({"metric": col, "mean": float(fold_df[col].mean()), "std": float(fold_df[col].std(ddof=1))})
    pd.DataFrame(summary_rows).to_csv(outdir / "overall_summary.csv", index=False)

    save_json(completed_fold_results, outdir / "all_folds.json")
    if is_segmentation_mode(cfg):
        save_cross_fold_reference_outputs(outdir, target_fold_count, logger)
        aggregate_segmentation_explainability(outdir, cfg, logger)
        summarize_frozen_xai(outdir, cfg, logger)
    generate_run_paper_outputs(outdir, fold_df, cfg, logger)
    baseline_dir = Path(cfg.statistical_baseline_dir) if cfg.statistical_baseline_dir.strip() else None
    save_statistical_analysis(fold_df, outdir, baseline_dir, outdir, logger)
    collect_paper_artifacts(outdir, cfg, logger)

    strict_mean = float(fold_df["strict_event_f1"].mean()) if "strict_event_f1" in fold_df else float("nan")
    sample_acc_mean = float(fold_df["ACC"].mean()) if "ACC" in fold_df else float("nan")
    kappa_mean = float(fold_df["KAPPA"].mean()) if "KAPPA" in fold_df else float("nan")
    logger.info(
        "Finished evaluation | mean strict event F1 = %.4f | mean ACC = %.4f | mean KAPPA = %.4f",
        strict_mean,
        sample_acc_mean,
        kappa_mean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

