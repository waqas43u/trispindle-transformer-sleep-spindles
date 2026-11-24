# Statistical analysis

Metrics are paired by identical held-out subject folds. Values are proportions, not percentages.
95% confidence intervals use the two-sided Student-t interval across folds.
Cohen's dz is the mean paired difference divided by its sample standard deviation.
Holm-adjusted p-values control family-wise error across the reported metrics.
Event IoU is reported as micro event F1 at IoU >= 0.5.

| metric | metric_label | n_proposed_folds | proposed_mean | proposed_std | proposed_median | proposed_min | proposed_max | proposed_ci95_low | proposed_ci95_high | fold_pairing_verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_event_f1 | Strict event F1 | 5 | 0.7792675427289599 | 0.05197050172437471 | 0.79031859718449 | 0.6946808510638298 | 0.8294558738704094 | 0.7147376417692172 | 0.8437974436887027 | False |
| tolerant_event_f1 | Tolerant event F1 | 5 | 0.8365549659255486 | 0.011092887467710928 | 0.8431070947894637 | 0.8178879310344828 | 0.8438829787234042 | 0.8227813269207662 | 0.8503286049303309 | False |
| strict_event_precision | Strict event precision | 5 | 0.7576292620053889 | 0.046417759702324354 | 0.7645747053201657 | 0.7023393385318634 | 0.8118178396687994 | 0.6999940009057393 | 0.8152645231050386 | False |
| strict_event_recall | Strict event recall | 5 | 0.8033307496638938 | 0.0657945131373205 | 0.8307478583005325 | 0.6871875822152065 | 0.847877358490566 | 0.7216360714219384 | 0.8850254279058493 | False |
| event_iou_f1_0_5 | Event F1 at IoU >= 0.5 | 5 | 0.4561577844029566 | 0.11279534722966937 | 0.4516341483493867 | 0.3034574468085106 | 0.5766198807921553 | 0.3161038625829835 | 0.5962117062229297 | False |
| sample_pr_auc | Sample PR-AUC | 5 | 0.6285741596181393 | 0.0987233337783504 | 0.6175444027211573 | 0.49336583050576527 | 0.7400168951454823 | 0.505992950282908 | 0.7511553689533707 | False |
| sample_auc | Sample ROC-AUC | 5 | 0.9208426057871708 | 0.023805325273539703 | 0.9170248170130135 | 0.8879629939096758 | 0.9470332348676628 | 0.8912843904294864 | 0.9504008211448552 | False |