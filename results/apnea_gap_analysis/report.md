# Learning LOSO vs official x AUC gap: analysis of existing predictions

Analysis only: no model was retrained, the official split was not rerun, and no new predictions
were made on x. Script: `src/explain_gap.py`.

## Inputs
- Learning: `results/apnea_baseline_final/oof_predictions.csv`, column `hist_gb|leave-one-subject-out` (out-of-fold, leave-one-suspected-subject-out)
- Official x: `results/apnea_official_final_v2/x_predictions_hist_gb.csv`, column `prob` (hist_gb trained on 34 learning records)

## Validation
| set | rows | records | subject groups | labels | apnea min | invalid label | invalid prob | missing id | full-row dups | rows used |
|---|---|---|---|---|---|---|---|---|---|---|
| learning LOSO | 16558 | 34 | 23 | 0,1 | 6508 | 0 | 0 | 0 | 5 | 16558 |
| official x | 16292 | 33 | 23 | 0,1 | 6543 | 0 | 0 | 0 | 0 | 16292 |

The learning file has no minute column. Its rows line up one-to-one, in order, with
`data/processed/apnea_minutes.csv` (record, label and subject all match), so the full-row duplicates
are distinct minutes: early minutes of c02 and c09 with zero detected beats, whose imputed features
and predictions are therefore identical. They are kept. Subject IDs match
`splits/apnea_ecg_subject_map.csv` in both files.

## Metric definitions
- **Within-record AUC**: ROC AUC from one record's minute labels and probabilities. Defined only when
  the record has both apnea and normal minutes. Summarised as median and IQR (25th-75th percentile)
  over records, never pooled across records.
- **Pooled AUC**: one ROC AUC over all minutes of all records together. It mixes within-record
  ranking with between-record differences in probability level.
- **95% CI**: percentile bootstrap resampling whole suspected-subject groups with replacement
  (1000 draws, seed 0), as in `baseline_apnea.py`. The draws differ from that script's,
  so CI ends can differ slightly from the baseline table.

## 1. Within-record AUC
| | learning LOSO | official x |
|---|---|---|
| records | 34 | 33 |
| eligible (both classes) | 30 | 30 |
| excluded (one class only) | 4 (c01,c03,c04,c08) | 3 (x04,x06,x29) |
| median AUC | 0.853 | 0.891 |
| IQR | 0.677 - 0.904 | 0.843 - 0.938 |
| range | 0.292 - 0.994 | 0.291 - 0.980 |

## 2. Hard-subject sensitivity (pooled learning LOSO)
Excluded groups P01 (a01, a14), P09 (a09, a18),
P02 (a02; its other record x14 is in the test set and not in this file).

| variant | rows | records | subject groups | AUC [95% CI] |
|---|---|---|---|---|
| original | 16558 | 34 | 23 | 0.689 [0.549, 0.827] |
| without P01, P09, P02 | 14048 | 29 | 20 | 0.815 [0.743, 0.871] |

## 3. Within-record ranking (pooled learning LOSO)
| predictions | AUC [95% CI] |
|---|---|
| original probabilities | 0.689 [0.549, 0.827] |
| within-record ranks scaled to [0, 1] | 0.670 [0.601, 0.749] |

Replacing probabilities with within-record ranks removes every difference in probability level and
calibration between records, keeping only how each record's minutes are ordered. A change in pooled
AUC shows how much the pooled number depends on between-record differences. It does not show that
calibration caused the learning-vs-x gap: the rank transform also discards real between-record
information (a mostly-apnea night ranked against a mostly-normal night), and it changes nothing
about the x predictions.

## Caveats
- The two sets differ in more than evaluation scheme: LOSO models were trained on 33 or fewer
  records while the x model used all 34, and the records, subjects and apnea prevalence differ.
- Subject groups other than the documented copies are inferred from identical demographics
  (suspected same-subject groups, not identities).
- P01, P09 and P02 were named for exclusion before this analysis; analysis 2 is a sensitivity check,
  not a claim that these subjects should be dropped.
- Within-record AUCs rest on one record each and are unstable for records with few apnea or
  normal minutes; see `analysis1_within_record_auc_per_record.csv` for per-record counts.
- Minute-wise (leaky) predictions in the learning file were not used.
