# TikTok TechJam — CIFAKE Detector

This project fine-tunes EfficientNet-B0 to distinguish real images from AI-generated images using the CIFAKE dataset. It includes training, evaluation, robustness testing, checkpointing, and single-image prediction.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The dataset and pretrained model weights are downloaded automatically on first use.

## Dataset Setup

CIFAKE continues to use the existing automatic setup. WildFake-style datasets
are downloaded manually because the image archives are too large for Git.

1. Extract each source directly under the label it belongs to. Any source name
   and any amount of nesting inside it are supported:

```text
data/raw/WildFake/
├── FAKE/
│   ├── ADM/
│   ├── DDPM/
│   └── Midjourney/
└── REAL/
    ├── cocofolder/
    ├── laion5b/
    └── Flickr/
```

2. Prepare every source with a deterministic 90/10 split:

```powershell
python main.py prepare-data
```

Nested extracted folders are preserved. A source that already has a clear
official `TRAIN`/`TEST` split keeps its membership; an unsafe partial or
ambiguous layout is left untouched with a warning. Customize the split with:

```powershell
python main.py prepare-data --train-ratio 0.9 --seed 42
```

3. Run the existing training commands. Prepared sources are discovered
automatically by both multisource and source-balanced training, so adding a new
source does not require a Python change.

## Commands

Train the model, save the best checkpoint, and run the robustness benchmark:

```powershell
python main.py train
```

Train the staged EfficientNet-B0 experiment:

```powershell
python main.py train-staged
```

Staged training freezes the full feature extractor for two head-only epochs at
`1e-3`, then unfreezes feature blocks 6–8 for five epochs by default. During
Stage 2, the classifier uses `1e-4`, the unfrozen backbone uses `1e-5`, and both
learning rates follow cosine annealing. Change only the Stage 2 duration with
`--stage2-epochs`.

Train the same staged workflow with combined CIFAKE and WildFake sources:

```powershell
python main.py train-multisource
```

The multisource command recursively loads `data/raw/cifake` and every prepared
source under `data/raw/WildFake` with explicit `FAKE=0` and `REAL=1` labels.
Training is shuffled across sources; internal validation is not shuffled. Its best
checkpoint is saved to
`checkpoints/efficientnet_staged_multisource_best.pt`, then that exact
checkpoint is evaluated with the ByteDance validation workflow.

Select a deterministic fraction from each training source independently:

```powershell
python main.py train-multisource --train-fraction 0.5
```

Sampling uses seed 42 and only changes the file paths included in training.
Internal and ByteDance validation always use their complete datasets. The
default fraction is `1.0`.

### Source-balanced held-out-generator training

Train the final source-balanced model using CIFAKE plus every prepared WildFake
FAKE and REAL source:

```powershell
python main.py train-source-balanced
```

This uses the fixed source-balanced epoch length and selects
`checkpoints/efficientnet_balanced_all_sources_best.pt` using pooled internal
validation ROC-AUC. Add `--samples-per-epoch` or `--seed` as needed.

For diagnostic unseen-generator experiments, provide an optional holdout.

Train while completely excluding DDPM from training and using DDPM TEST as the
unseen FAKE validation source:

```powershell
python main.py train-source-balanced --holdout DDPM
```

Run the corresponding experiment with ADM held out instead:

```powershell
python main.py train-source-balanced --holdout ADM
```

Any prepared WildFake FAKE source can be held out, matched case-insensitively:

```powershell
python main.py train-source-balanced --holdout Midjourney
```

This workflow keeps the existing two-stage EfficientNet training strategy but
replaces ordinary random sampling with source-balanced batches. Each batch is
approximately half FAKE and half REAL, and the active sources within each class
contribute approximately equally. Smaller sources can be reused; no image files
are copied, moved, or changed.

Each epoch uses a fixed 100,000 training samples by default. Change the epoch
size or deterministic sampler seed with:

```powershell
python main.py train-source-balanced --holdout DDPM `
    --samples-per-epoch 100000 `
    --seed 42
```

The validation set contains only the held-out FAKE generator plus the complete
CIFAKE REAL test source and every prepared WildFake REAL test source.
Checkpoints are selected using the held-out generator's FAKE-positive ROC-AUC
against those combined REAL images, not pooled source-matched validation AUC.
Per-source recall is printed after every epoch.

The default checkpoints are:

- `checkpoints/efficientnet_balanced_holdout_ddpm_best.pt`
- `checkpoints/efficientnet_balanced_holdout_adm_best.pt`
- `checkpoints/efficientnet_balanced_all_sources_best.pt`

Future holdouts use the same lowercase naming convention, for example
`checkpoints/efficientnet_balanced_holdout_midjourney_best.pt`. Use
`--checkpoint` to override it.

This command deliberately does not run robustness or ByteDance validation.
After selecting an approach using held-out validation, run ByteDance manually:

```powershell
python main.py validate-bytedance `
    --checkpoint checkpoints/efficientnet_balanced_holdout_ddpm_best.pt
```

### Progressive deeper EfficientNet fine-tuning

Run the additive three-stage EfficientNet-B0 experiment with one FAKE source
reserved for held-out-generator validation:

```powershell
python main.py train-efficientnet-deeper `
    --data-dir data/raw/cifake `
    --wildfake-dir data/raw/WildFake `
    --holdout-fake-source ddpm `
    --samples-per-epoch 100000 `
    --batch-size 32 `
    --stage1-epochs 2 `
    --stage2-epochs 2 `
    --stage3-epochs 3 `
    --seed 42 `
    --run-name ddpm_progressive
```

The same ImageNet-pretrained EfficientNet-B0 continues through all stages:

- Stage 1 trains only the classifier at `1e-4`.
- Stage 2 adds blocks 6-8 at `1e-5` and starts a stage-local cosine schedule.
- Stage 3 adds blocks 4-5 at `3e-6`, restores all configured base rates, and
  starts a fresh cosine schedule.

AdamW state remains continuous for parameters already being trained. Frozen
blocks stay in evaluation mode so their BatchNorm statistics cannot drift. The
folder named `TEST` for the excluded generator is used as held-out-generator
validation during checkpoint selection; it is not described as an untouched
final test set.

The command creates
`checkpoints/efficientnet_deeper_ddpm_progressive_best.pt`. It refuses any
existing automatic or explicit output path, never reads an older trained
checkpoint, and writes its copied best state exactly once after the run.

### Hybrid spatial-frequency model

The optional hybrid detector keeps EfficientNet-B0 as a 1,280-dimensional
spatial branch and adds a lightweight FFT branch. The same robustness-augmented
RGB tensor feeds both branches: EfficientNet applies ImageNet normalization
internally, while the frequency branch computes a float32 luminance FFT,
log-magnitude spectrum, and a small CNN with 256 output features. Their
concatenated features pass through a 256-unit fusion MLP and one output logit.
Labels and probabilities are unchanged: `FAKE=0`, `REAL=1`, and
`sigmoid(logit)=P(REAL)`.

Train the hybrid on CIFAKE plus every prepared WildFake source with the existing
source-balanced sampler:

```powershell
python main.py train-hybrid `
    --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt `
    --samples-per-epoch 100000 `
    --stage1-epochs 2 `
    --stage2-epochs 5
```

The spatial checkpoint is loaded strictly into every EfficientNet feature
entry; incompatible or partial checkpoints fail with a clear diagnostic. If
`--spatial-checkpoint` is omitted, the spatial branch starts from ImageNet
weights. Stage 1 freezes EfficientNet and trains FFT plus fusion at `1e-4`.
Stage 2 unfreezes EfficientNet blocks 6–8 at `1e-5` while FFT plus fusion
continue at `1e-4`. FFT preprocessing always runs in float32, including inside
an autocast context.

Arbitrary prepared FAKE sources can also be held out without changing the
sampler or data layout:

```powershell
python main.py train-hybrid --holdout Midjourney `
    --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt
```

The default hybrid checkpoints are
`checkpoints/hybrid_balanced_all_sources_best.pt` and
`checkpoints/hybrid_balanced_holdout_<source>_best.pt`. Prediction, evaluation,
robustness, and ByteDance validation recognize hybrid checkpoint metadata and
select the matching raw-input transform automatically.

### Hybrid V2: controlled FFT residual

Hybrid V1 concatenates 1,280 EfficientNet features with 256 FFT features and
learns a fusion MLP. Hybrid V2 instead keeps the spatial detector as the main
decision path and adds a smaller 128-feature FFT prediction as a controlled
correction:

```text
final_logit = spatial_logit + frequency_scale * frequency_logit
```

The default `frequency_scale` is `0.25`. Training-only frequency branch dropout
defaults to `0.20`, so some samples must be classified from the spatial path
alone. Optional spectrum masking is available but disabled by default. These
constraints are intended to reduce dependence on generator-specific frequency
cues; improved ByteDance performance must be confirmed experimentally.

Run a short source-balanced V2 experiment from the trained EfficientNet:

```powershell
python main.py train-hybrid-v2 --data-dir data/raw/cifake --wildfake-dir data/raw/WildFake --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt --samples-per-epoch 5000 --stage1-epochs 1 --stage2-epochs 2 --frequency-scale 0.25 --frequency-branch-dropout 0.20 --frequency-mask-prob 0.0 --run-name smoke_alpha025
```

For a longer experiment, use 25,000 samples, two Stage 1 epochs, and five Stage
2 epochs. Automatically named V2 checkpoints refuse to overwrite an existing
file; choose a new `--run-name`, or use an explicit `--checkpoint` when an
overwrite is intentional. Examples:

```text
checkpoints/hybrid_v2_balanced_all_sources_best.pt
checkpoints/hybrid_v2_alpha025_all_sources_best.pt
checkpoints/hybrid_v2_alpha025_holdout_ddpm_best.pt
```

EfficientNet, Hybrid V1, and Hybrid V2 checkpoints remain independently
loadable by prediction, evaluation, robustness, and ByteDance validation.

### Hybrid V3: magnitude and phase residual

Hybrid V3 keeps the V2 spatial prediction as the primary decision and derives
two small forensic corrections from one shared float32 luminance FFT. The
magnitude path models normalized frequency energy, while the phase path uses
bounded sine/cosine channels to avoid the `-pi`/`+pi` angle discontinuity:

```text
final_logit = spatial_logit + frequency_scale * (
    normalized_magnitude_weight * magnitude_logit
    + normalized_phase_weight * phase_logit
)
```

The supplied magnitude and phase weights are normalized to sum to one, so the
default `0.5/0.5` mixture keeps the total residual scale at `0.25` rather than
doubling it. Combined branch dropout remains `0.20`; optional masking affects
only magnitude and remains disabled by default. V3 reuses the prepared dataset,
source-balanced sampler, robustness augmentation, staged trainer, and existing
evaluation commands. Performance must be established experimentally.

Run a phase-only smoke experiment:

```powershell
python main.py train-hybrid-v3 --data-dir data/raw/cifake --wildfake-dir data/raw/WildFake --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt --samples-per-epoch 5000 --stage1-epochs 1 --stage2-epochs 2 --frequency-scale 0.25 --magnitude-weight 0 --phase-weight 1 --frequency-branch-dropout 0.20 --frequency-mask-prob 0.0 --batch-size 32 --seed 42 --run-name phase_only_smoke
```

Run a 50/50 magnitude-plus-phase smoke experiment:

```powershell
python main.py train-hybrid-v3 --data-dir data/raw/cifake --wildfake-dir data/raw/WildFake --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt --samples-per-epoch 5000 --stage1-epochs 1 --stage2-epochs 2 --frequency-scale 0.25 --magnitude-weight 1 --phase-weight 1 --frequency-branch-dropout 0.20 --frequency-mask-prob 0.0 --batch-size 32 --seed 42 --run-name dual_spectrum_smoke
```

Alternatively, `--v2-checkpoint PATH` strictly warm-starts the spatial and
magnitude paths from a complete Hybrid V2 checkpoint while initializing phase
randomly. It cannot be combined with `--spatial-checkpoint`. Automatically
named V3 checkpoints use `checkpoints/hybrid_v3_<run>_all_sources_best.pt` (or
the corresponding holdout name) and refuse accidental overwrites. V3
checkpoints are supported by prediction, evaluation, robustness, and ByteDance
validation through the existing model-aware loader.

### Hybrid V3.1: radial frequency and learned fusion

Hybrid V3.1 keeps V3 unchanged and adds a 32-bin radial profile that summarizes
how normalized log-magnitude energy changes from low to high spatial
frequencies. Magnitude, phase, and radial representations all come from one
shared float32 FFT. A small radial MLP produces a third frequency logit, and
three learned scalar parameters are normalized with softmax:

```text
frequency_logit =
    w_magnitude * magnitude_logit
    + w_phase * phase_logit
    + w_radial * radial_logit

final_logit = spatial_logit + frequency_scale * frequency_logit
```

The weights start at one third each and always remain nonnegative with a sum of
one. The overall frequency scale remains fixed at `0.25` by default. After each
validation epoch, training reports the learned weights and the mean absolute
spatial, magnitude, phase, and radial logits from the same validation forward.
These values are diagnostic only and do not alter predictions.

Run a smoke experiment:

```powershell
python main.py train-hybrid-v31 --data-dir data/raw/cifake --wildfake-dir data/raw/WildFake --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_best.pt --samples-per-epoch 10000 --stage1-epochs 1 --stage2-epochs 2 --frequency-scale 0.25 --frequency-branch-dropout 0.20 --frequency-mask-prob 0.0 --radial-bins 32 --batch-size 32 --seed 42 --run-name radial32_smoke
```

For the full comparison, use `--samples-per-epoch 100000`,
`--stage1-epochs 2`, `--stage2-epochs 5`, and a new run name. Automatically
named V3.1 checkpoints refuse accidental overwrites. V3.1 is experimental; no
performance improvement is assumed until it is measured.

Evaluate the saved model:

```powershell
python main.py evaluate
```

Evaluate the staged checkpoint explicitly:

```powershell
python main.py evaluate --checkpoint checkpoints/efficientnet_staged_best.pt
```

Evaluate an existing checkpoint on the external ByteDance validation set:

```powershell
python main.py validate-bytedance
```

The command reads the CIFAKE and ByteDance ImageFolder class mappings, reports
dataset counts, and calculates ROC-AUC with FAKE/AIGC as the positive class. It
uses `validation/` and `checkpoints/efficientnet_staged_best.pt` by default.
Compare another compatible checkpoint without retraining:

```powershell
python main.py validate-bytedance --checkpoint checkpoints/best_model.pt
```

Run only the robustness benchmark:

```powershell
python main.py robustness
```

Evaluate a checkpoint against the clean ByteDance validation set and apply all
19 conditions deterministically in memory. No distorted image files are saved:

```powershell
python main.py robustness-matrix --checkpoint checkpoints/model.pt --validation-dir validation --probability-threshold 0.63 --batch-size 32 --run-name final_v31
```

The command uses the same clean images, checkpoint, preprocessing, and fixed
threshold for every condition. To run clean plus one condition:

```powershell
python main.py robustness-matrix --checkpoint checkpoints/model.pt --validation-dir validation --probability-threshold 0.63 --batch-size 32 --run-name jpeg30_check --only jpeg_30
```

Existing prepared files remain supported by explicitly adding
`--distorted-dir validation_distorted`; that mode requires its
`distortions.csv` manifest and performs the full manifest audit.

Classify one image:

```powershell
python main.py predict --image path\to\image.jpg
```

For a faster CPU experiment, reduce the image size, epochs, and worker count:

```powershell
python main.py train --image-size 64 --epochs 1 --num-workers 0
```

## Outputs

- `data/raw/` contains the downloaded dataset.
- `checkpoints/best_model.pt` contains the baseline checkpoint selected by validation accuracy.
- `checkpoints/efficientnet_staged_best.pt` contains the staged checkpoint selected by validation ROC-AUC.
- `checkpoints/efficientnet_staged_multisource_best.pt` contains the staged CIFAKE + WildFake checkpoint.
- `checkpoints/efficientnet_balanced_holdout_ddpm_best.pt` contains the best DDPM-held-out experiment.
- `checkpoints/efficientnet_balanced_holdout_adm_best.pt` contains the best ADM-held-out experiment.
- `checkpoints/efficientnet_balanced_all_sources_best.pt` contains the all-source-balanced model.
- `checkpoints/hybrid_balanced_all_sources_best.pt` contains the all-source hybrid model.
- `checkpoints/hybrid_balanced_holdout_<source>_best.pt` contains a hybrid held-out experiment.
- `checkpoints/hybrid_v2_balanced_all_sources_best.pt` contains the default Hybrid V2 experiment.
- `checkpoints/hybrid_v2_<run>_all_sources_best.pt` contains a named Hybrid V2 experiment.
- `checkpoints/hybrid_v3_<run>_all_sources_best.pt` contains a named Hybrid V3 experiment.
- `checkpoints/hybrid_v31_<run>_all_sources_best.pt` contains a named Hybrid V3.1 experiment.
- `results/robustness.json` contains robustness metrics.
- `results/robustness/<run>/` contains manifest-audited matrix summaries and
  per-image predictions for a named robustness run.

Run `python main.py --help` to see all options.
