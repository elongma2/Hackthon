# TikTok TechJam — CIFAKE Detector

This project fine-tunes EfficientNet-B0 to distinguish real images from AI-generated images using the CIFAKE dataset. It includes training, evaluation, robustness testing, checkpointing, and single-image prediction.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The dataset and pretrained model weights are downloaded automatically on first use.

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

The multisource command recursively loads `data/raw/cifake` and
`data/raw/WildFake` with explicit `FAKE=0` and `REAL=1` labels. Training is
shuffled across sources; internal validation is not shuffled. Its best
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

Train while completely excluding DDPM from training and using DDPM TEST as the
unseen FAKE validation source:

```powershell
python main.py train-source-balanced --holdout DDPM
```

Run the corresponding experiment with ADM held out instead:

```powershell
python main.py train-source-balanced --holdout ADM
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
CIFAKE, COCO, and LAION-5B REAL test sources. Checkpoints are selected using the
held-out generator's FAKE-positive ROC-AUC against those combined REAL images,
not pooled source-matched validation AUC. Per-source recall is printed after
every epoch.

The default checkpoints are:

- `checkpoints/efficientnet_balanced_holdout_ddpm_best.pt`
- `checkpoints/efficientnet_balanced_holdout_adm_best.pt`

This command deliberately does not run robustness or ByteDance validation.
After selecting an approach using held-out validation, run ByteDance manually:

```powershell
python main.py validate-bytedance `
    --checkpoint checkpoints/efficientnet_balanced_holdout_ddpm_best.pt
```

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
- `results/robustness.json` contains robustness metrics.

Run `python main.py --help` to see all options.
