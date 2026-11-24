# Second-round scientific verification report

This audit is read-only with respect to the original MASS-SS2 and DREAMS result directories. No inference, training, calibration, or decoder tuning was performed.

## 1. Checkpoint-to-metrics lineage — VERIFIED with one explicit archival limitation

- All 30 architecture/fold/ensemble-member records passed checkpoint byte-size, fold-configuration, calibrator, decoder, ensemble-weight, and MASS-metric/decoder consistency checks.
- The Global directory-name discrepancy is an archive alias: the local folder `results_Global_mass_xai_5fold` declares `results_transformer_mass_xai_5fold` in both `config.json` and `run_manifest.json`, matching every DREAMS manifest source root.
- Local SHA-256 values have now been recorded for all MASS checkpoints. The historical DREAMS manifests contain checkpoint byte sizes and strict-load/finite-state validation, but no hashes and no checkpoint copies. Exact binary identity therefore cannot be established cryptographically after the fact. This is an archival limitation, not an observed mismatch.
- The historical 78.46% Global result is not present in the retained checkpoint-linked MASS archive. The historical manuscript described a 4.59-M-parameter Global model, whereas the retained run contains 2,776,800 parameters and yields 64.56%. These are different model artifacts and must not be described as alternate renderings of one experiment.

## 2. Ablation consistency — CORRECTED

- The implemented intervention replaced one preprocessed representation by a fold-specific equal-subject median estimated from training subjects. It did not zero the channel.
- The envelope effect is the arithmetic mean of five paired fold differences: -60.88 percentage points (fold SD 16.02 pp).
- Training and test subject sets were disjoint in all five protocol files; thresholds remained frozen.

## 3. Explainability provenance — VERIFIED

- The displayed TP is subject `01-02-0002` from `fold_01` and is listed as a held-out test subject.
- Predicted interval: [17496.15625, 17497.703125]; matched reference: [17495.98046875, 17497.69140625]; IoU=0.891156.
- The saved attribution target is the event-interval mean of the weighted two-member ensemble probability before calibration. Gradient SHAP completion is recorded in the fold attribution status.
- Attribution ranking and replacement ablation remain distinct analyses and must not be described as interchangeable measures.

## 4. DREAMS boundary reconstruction — VERIFIED

- Reconstructed 900 architecture/fold/excerpt/reference/threshold metric rows and 15676 matched-event rows from preserved predictions and original annotations.
- Maximum absolute discrepancy from the frozen count/F1 tables: 1.11e-16. All rows: PASS.
- Signed onset/offset, absolute boundary, duration, and matched-IoU summaries are now available in `data/dreams_boundary_summary.csv`. These are conditional on the authoritative greedy matching rule and IoU threshold.

## 5. Standalone numerical-equivalence validation — UNRESOLVED

- No preserved report containing the claimed resampling, sliding-window, calibration, or decoder numerical-equivalence measurements was found in the supplied workspace.
- The Methods must not state that this standalone validation was completed unless that artifact is recovered. It should instead describe the frozen implementation and its audit limitations.

## 6. Statistical and editorial status — CORRECTED

- Fold- and subject-level paired tests, Cohen's $d_z$, and Holm adjustments were recomputed from the original result directories. Maximum discrepancy from the manuscript audit tables: 4.44e-16.
- Inferential claims remain exploratory.
- The five-fold interval should be omitted from main prose or explicitly called an approximate fold-level t interval based on five dependent folds.
- Replace `bimodal` with `highly variable across folds`; condense repeated numerical narration; move parameter counts and the internal/external comparison figure to supplementary material if space is limited.

## Overall decision

**CORRECTED, with one unresolved archival limitation.** The ablation conflict is resolved, boundary analysis is verified, and the Global alias is resolved. The Methods now explicitly qualify the missing standalone numerical-equivalence artifact. The package should not claim cryptographic checkpoint identity because the historical DREAMS manifests contain no hashes or checkpoint copies.

## Machine-readable evidence

- `data/checkpoint_lineage_second_round.csv`
- `audit/ablation_protocol_verification.csv`
- `audit/xai_provenance_verification.json`
- `audit/dreams_boundary_reconstruction_validation.csv`
- `audit/paired_statistics_recomputed.csv`
- `data/dreams_boundary_matches.csv`
- `data/dreams_boundary_summary.csv`
