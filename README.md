# Robust AI-Generated Image Detection

This ByteDance/TikTok TechJam Track 5 solution trains a source-balanced
EfficientNet-B0 and then Hybrid V3.1, which adds lightweight Fourier magnitude,
phase, and radial-frequency evidence.

The project uses `FAKE = 0`, `REAL = 1`, `sigmoid(logit) = P(REAL)`, and
`P(AIGC) = 1 - P(REAL)`.

## Quick Start: Run on Your Own Images

Put arbitrary images in an unlabeled folder; nested folders are supported.
Checkpoints trained by the team are already provided and can be immediately be use for predicting images.
Alternatively, you can try replicating the checkpoints by downloading and training on the image folders the team
has used.

```powershell
uv run --no-sync python main.py predict --input-dir .\my_images --checkpoint .\checkpoints\hybrid_v31_midjourney_50k_7epoch_all_sources_best.pt --output .\predictions.json
```

`pred` near `1.0` means more likely AI-generated; `pred` near `0.0` means more
likely real. No threshold is applied.

## Installation

Python 3.12 is required. Install [Astral uv](https://docs.astral.sh/uv/getting-started/installation/).

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the verified environment:

```powershell
uv sync
uv pip install -r requirements.txt
```

Use `uv run --no-sync` below so uv retains packages installed from the
requirements file.

## Final Architecture

```text
RGB image [0,1]
├── EfficientNet-B0 spatial features → spatial logit
└── one float32 luminance FFT
    ├── normalized log-magnitude → magnitude logit
    ├── sin/cos phase → phase logit
    └── radial-frequency profile → radial logit
             ↓
      learned softmax frequency fusion
             ↓
final = spatial logit + fixed frequency scale × frequency residual
```

The same augmented tensor reaches every branch. The FFT runs once in float32.
The final model is `src/hybrid_v31_model.py`.

## Dataset Setup

### CIFAKE

CIFAKE is downloaded automatically through KaggleHub using
`birdy654/cifake-real-and-ai-generated-synthetic-images`. Existing files are
reused. No manual arrangement is normally required.

```text
data/raw/cifake/
├── train/{FAKE,REAL}/
└── test/{FAKE,REAL}/
```

### WildFake

WildFake is too large for Git. To reproduce the final source mix, manually obtain
only these ModelScope folders:

```text
Diffusion_based/ADM/
Diffusion_based/DALLE2/
Diffusion_based/DDPM/
Diffusion_based/Midjourney/part_4/
Real/coco/
Real/laion5b/
```

Copy them into:

```text
data/raw/WildFake/
├── FAKE/
│   ├── ADM/
│   ├── DALLE2/
│   ├── DDPM/
│   └── part_4/        # Midjourney subset used in final training
└── REAL/
    ├── coco/
    └── laion5b/
```

Keep official `TRAIN` and `TEST` folders. Do not remove TEST images: training
reads TRAIN while validation reads TEST. Nested folders may remain unchanged.

## Prepare WildFake

```powershell
uv run --no-sync python main.py prepare-data --data-dir data/raw --wildfake-dir data/raw/WildFake
```

This recursively scans every source, preserves a valid existing TRAIN/TEST
split, and otherwise creates the deterministic project split. The result is:

```text
data/raw/WildFake/
├── FAKE/{ADM,DALLE2,DDPM,part_4}/{TRAIN,TEST}/...
└── REAL/{coco,laion5b}/{TRAIN,TEST}/...
```

Training combines FAKE sources CIFAKE, ADM, DALLE2, DDPM, and Midjourney
`part_4`, plus REAL sources CIFAKE, COCO, and LAION-5B. Sampling stays roughly
50/50 FAKE/REAL and roughly uniform across sources within each class.

## Train From Scratch

### Train EfficientNet-B0

The final spatial trainer is `src/train.py`. This command uses the successful
50,000-samples-per-epoch, two-stage (2+5 epoch) recipe:

```powershell
uv run --no-sync python main.py train-source-balanced --data-dir data/raw --wildfake-dir data/raw/WildFake --samples-per-epoch 50000 --stage2-epochs 5 --batch-size 32 --seed 42 --checkpoint checkpoints/efficientnet_balanced_all_sources_50k_7epochs_best.pt
```

### Train Hybrid V3.1

```powershell
uv run --no-sync python main.py train-hybrid-v31 --data-dir data/raw --wildfake-dir data/raw/WildFake --spatial-checkpoint checkpoints/efficientnet_balanced_all_sources_50k_7epochs_best.pt --samples-per-epoch 50000 --stage1-epochs 2 --stage2-epochs 5 --frequency-scale 0.25 --frequency-branch-dropout 0.20 --frequency-mask-prob 0.0 --radial-bins 32 --batch-size 32 --seed 42 --run-name midjourney_50k_7epoch
```

Expected output:

```text
checkpoints/hybrid_v31_midjourney_50k_7epoch_all_sources_best.pt
```

Automatic V3.1 paths refuse to overwrite an existing checkpoint.

## Evaluation

ByteDance validation is optional evaluation data, never training data:

```powershell
uv run --no-sync python main.py validate-bytedance --checkpoint checkpoints/hybrid_v31_midjourney_50k_7epoch_all_sources_best.pt --validation-dir validation --data-dir data/raw --batch-size 32
```

Optional robustness matrix:

```powershell
uv run --no-sync python main.py robustness-matrix --checkpoint checkpoints/hybrid_v31_midjourney_50k_7epoch_all_sources_best.pt --validation-dir validation --batch-size 32 --run-name final_v31
```

## Prediction JSON Format

```json
[
  {"image_path": "picture1.jpg", "pred": 0.923481},
  {"image_path": "nested/picture3.webp", "pred": 0.071928}
]
```
Every supported image produces one record in deterministic order. `pred` is a
numeric AIGC confidence in `[0, 1]`.

## Limitations and future improvements
Our main limitation is that the model’s ability to generalize to unseen AI generators still depends heavily on the diversity of generators represented in the training data. Real-world transformations such as compression, resizing, and blur can also weaken the spatial and frequency-domain artifacts that the model relies on. Although source-balanced sampling reduces dataset bias, the model may still learn source-specific shortcuts. Given more time, we would expand the training set with a wider range of AI generators, strengthen transformation-based augmentation, and conduct more extensive held-out-generator testing to improve and verify generalization to completely unseen AI-generated images.

## Team Contributions
Team member contributions
1. Weng Jia Lin
      Developed the initial baseline code and fine-tuning pipeline, and contributed to the model’s training and refinement throughout the project.
      Identified the issue with the model’s probability threshold and prediction bias, and introduced dataset shuffling to improve the training process.
Do DevPost write-up.
2. Men Xuanmo
      Contributed to brainstorming the code implementation throughout the development process.
      Attended the relevant technical workshop and documented key insights and notes for the team.
      Contributed to the video production, including preparation and editing of the project presentation.
3. Goh Joshua
      Identified the limitation of relying on only a single dataset and helped drive the move toward a more diverse training setup.
      Contributed to model fine-tuning and experimentation.
      Introduced the hybrid model approach and contributed to the development of robustness evaluation metrics.
4. William Edward Sugiharto
      Contributed to brainstorming and developing the code architecture and implementation strategy.
      Participated in model fine-tuning and experimentation.
      Introduced and contributed to the development of the hybrid model architecture and robustness evaluation metrics.
5. Koh Fong Jun Damien
      Contributed to video production and the preparation of the project presentation.
      Ran experiments and was involved in the training, validation, and testing of the models.
      Assisted with executing experiments and evaluating model performance across different configurations.
