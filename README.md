# Sleep apnea detection from single-lead ECG: leakage, duplicates and subject overlap in Apnea-ECG

Minute-by-minute sleep apnea detection from heart-rate (RR-interval) features of a single-lead
ECG, on the PhysioNet Apnea-ECG Database 1.0.0. The work asks how much published-style accuracy
depends on the evaluation scheme. We compare the same three models (logistic regression,
random forest, histogram gradient boosting) under a leaky minute-wise split,
leave-one-record-out, and leave-one-suspected-subject-out cross-validation on the 35 learning
records. We then run the official PhysioNet/CinC 2000 split (train a/b/c, test x) once, with
test results reported separately for test records whose suspected subject also appears in
training ("seen") and records whose suspected subject does not ("unseen"). Along the way, a
data-integrity audit found three pairs of records that hold the same raw ECG, including two
undocumented pairs in the test set, and the copies' apnea labels do not fully agree.

## Key findings

1. **Minute-wise cross-validation leaks.** With minutes of the same night in both training
   and test folds, hist_gb reaches AUC 0.976 [0.965, 0.985]. Holding out whole records drops this
   to 0.835 [0.769, 0.883], and holding out whole suspected-subject groups drops it to
   0.689 [0.542, 0.827]. **The minute-wise results are a leakage diagnostic only, not a
   performance claim.**
2. **Official split.** Trained on 34 learning records and tested on 33 x records, hist_gb has
   minute-level AUC 0.913 [0.884, 0.936] and logreg 0.875 [0.830, 0.917].
3. **No sign that subject overlap inflates the official benchmark.** The seen-minus-unseen AUC
   difference is -0.003 [-0.054, +0.082] for hist_gb and -0.075 [-0.141, +0.017] for logreg
   (23 seen and 10 unseen test records). The hypothesis that overlap inflates official-split
   scores is not supported. The unseen part is small, so the intervals are wide.
4. **The gap between learning-set LOSO (0.689) and official x (0.913) is not explained.**
   Median within-record AUC is closer (0.853 vs 0.891). Pooled LOSO AUC rises to 0.815 when
   three suspected-subject groups are removed. P01, P09 and P02 were chosen after inspecting
   per-subject results from an earlier pre-deduplication run
   (results/apnea_analysis/per_subject_accuracy.csv); this is a post-hoc sensitivity check, not
   a performance figure. Replacing probabilities with within-record ranks gives 0.670.
5. **Data integrity.** c05/c06 (documented by PhysioNet), x33/x34 and x22/x35 are copies of the
   same recording at a fixed time offset. The copies' apnea labels disagree on 7 of 13
   apnea-minute labels, and the PhysioNet table gives x22 and x35 different sexes. One copy of
   each pair was kept.

All 95% confidence intervals are percentile bootstraps that resample whole suspected-subject
groups, not minutes.

## Subject groups: what is known and what is inferred

Apnea-ECG has no subject IDs. `splits/apnea_ecg_subject_map.csv` assigns each record to an
`inferred_subject` group:

- **Verified copies.** c05/c06, x33/x34 and x22/x35 contain the same raw ECG (see the audit
  below). These are the only same-recording links that are verified.
- **Everything else is inferred.** Records are grouped when the PhysioNet table lists identical
  age, sex, height and weight. These are **suspected same-subject groups, not identities**.
  Identical demographics can occur by chance, and the same person can be listed with
  different values.
- Groups are merged transitively through the copies: P22 = c05, c06, c07, x33, x34 and
  P31 = c01, x17, x22, x35. The map has 29 groups over 70 records. After the exclusions,
  training uses 23 groups and the full deduplicated set uses 29.

## Results

The tables below are copied from the result CSVs. The run folders are named under each table.

### Learning-set cross-validation (`results/apnea_baseline_final/summary.csv`)

34 learning records (c05 excluded as a copy of c06), 16,558 minutes, 6,508 apnea minutes,
23 suspected-subject groups. Point estimate [95% CI].

| model | split | accuracy | sensitivity | specificity | balanced_acc | auc |
|---|---|---|---|---|---|---|
| logreg | minute_wise (leaky) | 0.782 [0.735, 0.830] | 0.730 [0.608, 0.842] | 0.816 [0.760, 0.864] | 0.773 [0.714, 0.828] | 0.852 [0.808, 0.896] |
| random_forest | minute_wise (leaky) | 0.912 [0.891, 0.930] | 0.918 [0.868, 0.950] | 0.907 [0.875, 0.934] | 0.913 [0.889, 0.931] | 0.971 [0.957, 0.981] |
| hist_gb | minute_wise (leaky) | 0.922 [0.904, 0.942] | 0.928 [0.891, 0.955] | 0.918 [0.883, 0.944] | 0.923 [0.903, 0.942] | 0.976 [0.965, 0.985] |
| logreg | leave-one-record-out | 0.723 [0.662, 0.777] | 0.654 [0.513, 0.779] | 0.768 [0.692, 0.837] | 0.711 [0.642, 0.773] | 0.771 [0.703, 0.836] |
| random_forest | leave-one-record-out | 0.752 [0.693, 0.804] | 0.621 [0.500, 0.719] | 0.837 [0.779, 0.887] | 0.729 [0.666, 0.780] | 0.834 [0.782, 0.878] |
| hist_gb | leave-one-record-out | 0.763 [0.704, 0.808] | 0.648 [0.507, 0.735] | 0.837 [0.780, 0.883] | 0.743 [0.676, 0.788] | 0.835 [0.769, 0.883] |
| logreg | leave-one-subject-out | 0.693 [0.617, 0.764] | 0.594 [0.443, 0.761] | 0.757 [0.666, 0.839] | 0.676 [0.596, 0.752] | 0.725 [0.635, 0.819] |
| random_forest | leave-one-subject-out | 0.674 [0.566, 0.769] | 0.424 [0.232, 0.617] | 0.837 [0.774, 0.894] | 0.630 [0.542, 0.721] | 0.681 [0.535, 0.831] |
| hist_gb | leave-one-subject-out | 0.679 [0.576, 0.766] | 0.439 [0.252, 0.625] | 0.835 [0.779, 0.889] | 0.637 [0.546, 0.728] | 0.689 [0.542, 0.827] |

The minute_wise rows put minutes of the same night in both training and test folds. They
measure leakage, not performance. In leave-one-record-out, c05 is excluded and c06 is held out
on its own. Leave-one-subject-out holds out each suspected-subject group in turn.

### Official split (`results/apnea_official_final_v2/summary.csv`)

Trained on 34 learning records and tested on 33 x records (x34 and x35 excluded as copies of
x33 and x22): 16,292 test minutes and 6,543 apnea minutes. "seen" means the record's suspected
group also appears in training (23 records), and "unseen" means it does not (10 records).
Night classes for x are derived from the minute labels: apnea night at 100 or more apnea
minutes, control night below 5, otherwise borderline (left out of night_auc).

| model | test_part | records | minute_auc | minute_acc | sensitivity | specificity | night_auc | night_spearman |
|---|---|---|---|---|---|---|---|---|
| logreg | all x | 33 | 0.875 [0.830, 0.917] | 0.795 [0.750, 0.839] | 0.805 [0.699, 0.886] | 0.788 [0.721, 0.848] | 1.000 [1.000, 1.000] | 0.899 [0.725, 0.949] |
| logreg | seen | 23 | 0.849 [0.800, 0.902] | 0.775 [0.726, 0.825] | 0.769 [0.646, 0.883] | 0.779 [0.710, 0.846] | 1.000 [1.000, 1.000] | 0.890 [0.630, 0.964] |
| logreg | unseen | 10 | 0.925 [0.851, 0.970] | 0.840 [0.764, 0.920] | 0.878 [0.683, 0.938] | 0.811 [0.627, 0.922] | 1.000 [1.000, 1.000] | 0.964 [0.515, 1.000] |
| logreg | seen - unseen | | -0.075 [-0.141, +0.017] | -0.064 [-0.155, +0.028] | -0.109 [-0.246, +0.103] | -0.031 [-0.162, +0.159] | +0.000 [+0.000, +0.000] | -0.074 [-0.298, +0.307] |
| hist_gb | all x | 33 | 0.913 [0.884, 0.936] | 0.840 [0.809, 0.871] | 0.815 [0.750, 0.869] | 0.857 [0.804, 0.899] | 1.000 [1.000, 1.000] | 0.932 [0.784, 0.974] |
| hist_gb | seen | 23 | 0.913 [0.882, 0.945] | 0.841 [0.806, 0.879] | 0.827 [0.734, 0.902] | 0.850 [0.791, 0.901] | 1.000 [1.000, 1.000] | 0.916 [0.682, 0.983] |
| hist_gb | unseen | 10 | 0.917 [0.835, 0.951] | 0.839 [0.774, 0.896] | 0.792 [0.662, 0.838] | 0.875 [0.750, 0.929] | 1.000 [1.000, 1.000] | 0.976 [0.641, 1.000] |
| hist_gb | seen - unseen | | -0.003 [-0.054, +0.082] | +0.002 [-0.065, +0.078] | +0.035 [-0.060, +0.175] | -0.025 [-0.109, +0.114] | +0.000 [+0.000, +0.000] | -0.060 [-0.262, +0.305] |

The night_auc of 1.000 ranks label-defined apnea nights (100 or more apnea minutes) against
control nights (fewer than 5). Borderline nights are left out, so this is an easy, extreme-group
comparison. It should not be read as clinical night-level diagnostic accuracy.

### Learning LOSO vs official x gap (`results/apnea_gap_analysis/`)

This analysis uses only existing predictions: nothing was retrained and nothing new was
predicted on x. The inputs are the hist_gb leave-one-subject-out out-of-fold predictions from
`apnea_baseline_final` and the hist_gb x predictions from `apnea_official_final_v2`. The CSVs
hold full-precision values, rounded here to 3 decimals as in `report.md`. The bootstrap here
uses 1000 draws with seed 0, which differ from the baseline script's draws. That is why the
pooled LOSO CI is [0.549, 0.827] here and [0.542, 0.827] in the baseline table.

**1. Within-record AUC** (`analysis1_within_record_auc_summary.csv`). This is a per-record ROC
AUC, summarised over records with both classes. Records are never pooled.

| set | records | eligible_records | excluded_one_class | excluded_records | median_auc | q25_auc | q75_auc | min_auc | max_auc |
|---|---|---|---|---|---|---|---|---|---|
| learning LOSO | 34 | 30 | 4 | c01,c03,c04,c08 | 0.853 | 0.677 | 0.904 | 0.292 | 0.994 |
| official x | 33 | 30 | 3 | x04,x06,x29 | 0.891 | 0.843 | 0.938 | 0.291 | 0.980 |

**2. Hard-subject sensitivity** (`analysis2_hard_subject_sensitivity.csv`). This is the pooled
learning LOSO AUC with suspected groups P01 (a01, a14), P09 (a09, a18) and P02 (a02) removed.
P01, P09 and P02 were chosen after inspecting per-subject results from an earlier
pre-deduplication run (results/apnea_analysis/per_subject_accuracy.csv); this is a post-hoc
sensitivity check, not a performance figure. Because the groups were picked for their poor
results, removing them is expected to raise the AUC. The analysis does not claim that these
subjects should be dropped.

| variant | excluded_groups | excluded_records | rows | records | subject_groups | apnea_minutes | auc | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|
| original pooled LOSO | | | 16558 | 34 | 23 | 6508 | 0.689 | 0.549 | 0.827 |
| without P01, P09, P02 | P01,P09,P02 | a01,a14,a09,a18,a02 | 14048 | 29 | 20 | 4416 | 0.815 | 0.743 | 0.871 |

**3. Within-record ranking** (`analysis3_within_record_rank_auc.csv`). Each record's
probabilities are replaced by within-record average ranks scaled to [0, 1].

| variant | rows | records | auc | ci_low | ci_high |
|---|---|---|---|---|---|
| original pooled LOSO probabilities | 16558 | 34 | 0.689 | 0.549 | 0.827 |
| within-record average ranks scaled to [0,1] | 16558 | 34 | 0.670 | 0.601 | 0.749 |

The rank transform removes between-record differences in probability level. It does not show
whether calibration caused the gap: it also discards real between-record information, and it
changes nothing about the x predictions. The two settings also differ in other ways. The LOSO
models were trained on 33 or fewer records while the x model used all 34, and the records,
subjects and apnea prevalence differ. See `results/apnea_gap_analysis/report.md`.

## Data-integrity audit

All findings come from `results/apnea_label_check.csv`, `results/apnea_diagnostics/` and the
PhysioNet table in `additional-information.txt`.

**Signal and labels.** All 70 records are sampled at 100 Hz and have one A/N label per minute,
with labels exactly 6000 samples apart and no other symbols.

**Annotation counts vs the PhysioNet table.** For 68 of the 70 records, the `.apn` file holds
exactly one labelled minute fewer than the "Length (minutes)" column of the PhysioNet table.
c07 and c08 differ by more:

| record | table length (min) | labelled minutes in .apn | whole signal minutes |
|---|---|---|---|
| c07 | 454 | 429 | 428 |
| c08 | 535 | 513 | 514 |

**Duplicate recordings.** First, an RR-series screen compared all 2,415 record pairs
(`pair_correlations.csv`). Three pairs scored above 0.9 (x33/x34 0.974, x22/x35 0.929,
c05/c06 0.922), and the next highest was a04/a12 at 0.702. These four pairs were then compared
at the raw-ECG level (`duplicates/duplicate_check.csv`, `duplicates/full_alignment.csv`,
overlay PNGs):

| pair | offset (s) | overlap (min) | identical samples | .qrs beats aligned (within 2 samples) | status |
|---|---|---|---|---|---|
| c05/c06 | 80 | 465.5 | 100% | 100% | copy, documented by PhysioNet |
| x33/x34 | 70 | 472.3 | 100% | 100% | copy, undocumented |
| x22/x35 | 40 | 481.2 | 100% | 100% | copy, undocumented |
| a04/a12 | - | - | - | - | not a copy (median windowed raw-ECG correlation 0.03) |

**Label disagreement between copies** (`duplicates/label_disagreements.csv`). Each apnea minute
in either copy was placed on the other copy's timeline. Of 13 apnea-minute labels, 6 have
matching apnea (at least 30 s) in the other copy. The other 7 have no apnea anywhere in the
corresponding minute of the other copy. None of the 7 is explained by the shifted minute grid
(no "shift_only" cases). By pair: c05/c06 has 2 consistent and 2 unmatched, x33/x34 has 4
consistent and 3 unmatched, and x22/x35 has 0 consistent and 2 unmatched. The same ECG was
therefore labelled differently in places.

**Demographic contradictions in the PhysioNet table** for the copies:

| record | age | sex | height (cm) | weight (kg) |
|---|---|---|---|---|
| c05 | 28 | F | 169 | 57 |
| c06 | 28 | F | 171 | 65 |
| x33 | 28 | F | 169 | 57 |
| x34 | 30 | F | 168 | 56 |
| x22 | 27 | F | 158 | 53 |
| x35 | 31 | M | 184 | 74 |

x22 and x35 hold identical ECG, yet the table lists them as a 27-year-old woman and a
31-year-old man. This is one reason the demographic grouping is treated as an inference only.

**Handling.** One copy of each pair is kept. c05 is dropped from training, and x34 and x35 are
dropped from testing.

## How to reproduce

Requirements: Windows PowerShell, Python 3.13, and the AWS CLI (for the download only).

```powershell
python -m pip install -r requirements.txt
# download the data (see data/README.md)
aws s3 sync --no-sign-request s3://physionet-open/apnea-ecg/1.0.0/ data/apnea_ecg/
# run the full pipeline into NEW folders (results\repro_<timestamp>, data\processed\repro_<timestamp>)
.\reproduce.ps1
```

**Warning:** `reproduce.ps1` runs new experiments. That includes a new evaluation on the x test
set, which the analysis plan allows once per fixed configuration. It never writes into the
reported results folders. The baseline step takes about 10-15 minutes.

If scikit-learn fails on Windows with "An Application Control policy has blocked this file",
reinstall it with `python -m pip install --force-reinstall --no-cache-dir scipy scikit-learn`.

## Repository layout

```
README.md, LICENSE, requirements.txt, reproduce.ps1, download_apnea.ps1
data/README.md                        how to obtain the dataset (the data itself is not included)
splits/apnea_ecg_subject_map.csv      record -> suspected same-subject group, with evidence column
src/                                  all scripts; each has a usage docstring
results/
  apnea_label_check.csv               per-record label/annotation check
  apnea_baseline_final/               learning-set CV (table above) and out-of-fold predictions
  apnea_official_final_v2/            official split (table above) and x predictions
  apnea_gap_analysis/                 LOSO vs x gap analysis (tables above, report.md)
  apnea_diagnostics/                  pair_correlations.csv: RR-series screen of all record pairs
    duplicates/                       raw-ECG copy checks, label disagreements, overlay PNGs
    x_per_record.csv                  HISTORICAL, pre-deduplication (see below)
  apnea_analysis/                     HISTORICAL, pre-deduplication (see below)
  audit_c03/                          signal-quality audit of record c03
```

**Historical files kept for transparency.** Two tracked items come from runs made before
deduplication and the subject-map fix. Their original inputs are not included in the
repository. **Neither is part of the final deduplicated evaluation, and neither should be
mistaken for it.**

- `results/apnea_analysis/` holds per-subject results, paired AUC differences and per-record
  summaries from the pre-deduplication baseline. Its `per_subject_accuracy.csv` is where the
  post-hoc hard-subject groups P01, P09 and P02 were picked.
- `results/apnea_diagnostics/x_per_record.csv` holds per-record x results from the
  pre-deduplication official-split run. It still lists x34 and x35, under old group labels.

Final pipeline scripts, in order:

| script | role |
|---|---|
| `src/verify_apnea_labels.py` | label/annotation check for all 70 records |
| `src/build_apnea_features.py` | minute-level RR features (`--exclude` drops duplicate copies) |
| `src/baseline_apnea.py` | three models under minute-wise, leave-one-record-out and leave-one-subject-out CV |
| `src/official_split_apnea.py` | official a/b/c to x split, with seen/unseen breakdown |
| `src/explain_gap.py` | LOSO vs x gap analysis on existing predictions |
| `src/diagnose_overlap.py` | RR-series duplicate screen over all record pairs; per-record x results |
| `src/verify_duplicates.py` | raw-ECG cross-correlation of candidate pairs, overlay plots |
| `src/full_alignment.py` | whole-overlap sample, beat and label comparison of the copies |
| `src/label_disagreement_report.py` | where the copies' apnea labels disagree |

Earlier and exploratory scripts are kept for transparency but are not part of the reported
results: `baseline_apnea_old.py` and `build_apnea_features_old.py` (superseded versions),
`analyze_apnea_results.py` (paired bootstrap and per-subject breakdown on the pre-deduplication
baseline), `night_level_apnea.py` (exploratory whole-night evaluation) and `audit_record.py`
(signal-quality audit of one record, used for c03). Earlier, superseded and exploratory runs
from these scripts are not included in the repository. These are the pre-deduplication
baseline, the record-normalised baseline, the pre-deduplication official-split runs and the
night-level runs.

## Limitations

- **Small sample.** There are 34 training records in 23 suspected-subject groups, and only 10
  test records come from unseen groups. Subject-level CIs are wide, and the seen/unseen
  comparison has little power to detect modest differences.
- **Subject identity is inferred.** Apart from the three verified copies, groups rest on
  matching demographics. Wrong merges or missed links would bias the leave-one-subject-out and
  seen/unseen analyses in unknown directions.
- **Label noise.** Copies of the same ECG carry different apnea labels, so the reference
  labels themselves are uncertain. Records not duplicated in the database may have the same
  issue, but it cannot be checked for them.
- **Machine-generated beats.** Features use the database's `.qrs` annotations, which are
  machine-generated and were not reviewed here.
- **Unexplained gap.** Learning-set LOSO AUC (0.689) is far below the official x AUC (0.913),
  and the gap analysis does not explain it. The 0.815 hard-subject figure is a post-hoc
  sensitivity check only: those groups were chosen after inspecting earlier per-subject results.
- **Night-level metrics.** The x night classes are approximated from minute labels, and
  night_auc compares only extreme nights.
- **One dataset.** All results come from Apnea-ECG, with no external validation set.

## Citation

If you use the dataset, cite it as described in `data/README.md`. Code: MIT License (see `LICENSE`).
