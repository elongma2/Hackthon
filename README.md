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
- `results/robustness.json` contains robustness metrics.

Run `python main.py --help` to see all options.
