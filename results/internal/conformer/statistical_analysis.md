# Statistical analysis

Metrics are paired by identical held-out subject folds. Values are proportions, not percentages.
95% confidence intervals use the two-sided Student-t interval across folds.
Cohen's dz is the mean paired difference divided by its sample standard deviation.
Holm-adjusted p-values control family-wise error across the reported metrics.
Event IoU is reported as micro event F1 at IoU >= 0.5.

| metric | metric_label | n_proposed_folds | proposed_mean | proposed_std | proposed_median | proposed_min | proposed_max | proposed_ci95_low | proposed_ci95_high | fold_pairing_verified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_event_f1 | Strict event F1 | 5 | 0.7957824216877893 | 0.02819797950513979 | 0.7857744817124558 | 0.7661268185561352 | 0.8318212205138001 | 0.7607700057141696 | 0.830794837661409 | False |
| tolerant_event_f1 | Tolerant event F1 | 5 | 0.8395184019675875 | 0.006343738782911881 | 0.8413395553115565 | 0.8300725434161355 | 0.8453824849584566 | 0.8316416099068369 | 0.8473951940283382 | False |
| strict_event_precision | Strict event precision | 5 | 0.7931152145078699 | 0.03951104285573655 | 0.8008608321377332 | 0.73598666388831 | 0.8423114593535749 | 0.7440557750624158 | 0.842174653953324 | False |
| strict_event_recall | Strict event recall | 5 | 0.8005252456682062 | 0.044600511278298 | 0.7964806668210234 | 0.7342804525124967 | 0.855935534591195 | 0.7451463965124626 | 0.8559040948239497 | False |
| event_iou_f1_0_5 | Event F1 at IoU >= 0.5 | 5 | 0.46742388850345956 | 0.10920982491685699 | 0.39608966795887407 | 0.3789843921741042 | 0.6157324764964893 | 0.3318219806540619 | 0.6030257963528572 | False |
| sample_pr_auc | Sample PR-AUC | 5 | 0.6508195198593125 | 0.08710705090983595 | 0.6003322871509339 | 0.577562203157292 | 0.7496840118141639 | 0.5426618307548667 | 0.7589772089637583 | False |
| sample_auc | Sample ROC-AUC | 5 | 0.9292389970170948 | 0.02506932182709826 | 0.9370941141403274 | 0.8923760378237505 | 0.952057155887748 | 0.898111322645003 | 0.9603666713891865 | False |