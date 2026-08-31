# Explaining Our AI-Generated Image Detector

This document explains the model, experiments, failures, and lessons from our TikTok/ByteDance Track 5 project: **Robust Detection of AI-Generated Images Under Real-World Transformations**. It is written for readers with introductory programming knowledge but little machine-learning experience.

## Evidence note

The current repository at commit `aeb7948` contains the training, evaluation, augmentation, source-balancing, and Hybrid V1–V3.1 implementations. It does **not** contain the datasets, trained checkpoints, ByteDance prediction files, or full training logs. The only local numerical result artifact is [`results/robustness.json`](../results/robustness.json), and it does not identify the checkpoint that produced it.

For that reason, this document uses three evidence labels:

- **Locally verified:** supported by current code or a result file in the repository.
- **Team-recorded:** supplied by our team from experiments, but not independently attributable to a local checkpoint or log in this repository.
- **Experimental:** implemented in code, but no local evidence proves that it was trained or evaluated.

This distinction prevents us from accidentally presenting remembered results as if they were reproducible from the current checkout.

## 1. The problem

We want to classify one image as either AI-generated or real. This is **binary classification**, meaning there are exactly two possible classes:

```text
FAKE / AIGC = 0
REAL        = 1
```

Our code uses one consistent probability convention:

```text
sigmoid(logit) = P(REAL)
P(AIGC)        = 1 - sigmoid(logit)
```

### From an image to a decision

The network produces a **logit**, which is an unrestricted number such as `-3.2`, `0.4`, or `5.1`. A logit is not yet a probability. Positive logits favour `REAL=1`, while negative logits favour `FAKE=0`.

The **sigmoid** function converts the logit into a number between 0 and 1:

```text
logit = -3.0  → sigmoid ≈ 0.047 → about 4.7% probability of REAL
logit =  0.0  → sigmoid = 0.500 → 50% probability of REAL
logit =  3.0  → sigmoid ≈ 0.953 → about 95.3% probability of REAL
```

A **decision threshold** converts this probability into a class. With the default threshold of `0.5`:

```text
P(REAL) >= 0.5 → predict REAL
P(REAL) <  0.5 → predict FAKE
```

The threshold controls the final decision, not what the score means. The prediction path preserves this convention in [`src/predict.py`](../src/predict.py), and the ByteDance evaluator correctly converts it to `P(AIGC) = 1 - P(REAL)` in [`src/bytedance_validation.py`](../src/bytedance_validation.py).

### How training works

The model is trained with `BCEWithLogitsLoss`, short for **binary cross-entropy with logits**. It combines sigmoid and the binary classification loss in one numerically stable calculation. For a real image (`label=1`), the loss encourages a larger positive logit. For a fake image (`label=0`), it encourages a more negative logit.

**Backpropagation** calculates how much each trainable model parameter contributed to the error. It works backwards through the network and produces gradients. An optimiser then uses those gradients to update the parameters.

An **epoch** is one training cycle. In ordinary training, it often means one pass through the dataset. In our source-balanced training, an epoch instead contains a fixed number of sampled images, controlled by `--samples-per-epoch`; it does not guarantee that every image is seen exactly once.

The **learning rate** controls the size of each parameter update. A rate that is too large can make training unstable. A rate that is too small can make learning extremely slow.

We use **AdamW**, an optimiser that adapts updates for different parameters and applies weight decay to discourage unnecessarily large weights. In Stage 2, the code uses cosine annealing to gradually reduce the learning rates.

**Frozen layers** have `requires_grad=False`, so training does not update them. **Fine-tuning** means starting from a pretrained model and carefully updating some or all of its weights for our new task.

## 2. Accuracy and ROC-AUC

The correct term is **AUC**, not “AUG.”

### Accuracy

Accuracy is the fraction of predictions that are correct:

```text
accuracy = correct predictions / total predictions
```

Suppose a model scores an image with `P(REAL)=0.47`. At threshold `0.5`, it predicts FAKE. At threshold `0.4`, it predicts REAL. The model score has not changed, but the final class has. Accuracy therefore depends on the chosen threshold.

Accuracy can also be misleading on an imbalanced dataset. If 90% of images are fake, a model that always predicts fake obtains 90% accuracy while learning nothing useful about real images.

### ROC-AUC

**ROC-AUC** measures how well the model ranks the two classes apart across all possible thresholds. An intuitive question is:

> If we randomly choose one REAL image and one FAKE image, how often does the model give the REAL image a higher `P(REAL)` score?

An AUC near `0.5` is approximately random ranking. An AUC closer to `1.0` means better separation. Changing only the decision threshold does not reorder scores, so it can change accuracy while leaving AUC unchanged.

Our ordinary evaluator computes AUC with REAL as the positive class using `P(REAL)`. The ByteDance report describes AIGC as positive, so it uses `P(AIGC)=1-P(REAL)` together with AIGC labels. Complementing both the labels and scores preserves the same ranking quality.

### Other useful metrics

A **confusion matrix** counts four outcomes:

| Actual class | Predicted correctly | Predicted incorrectly |
|---|---|---|
| FAKE | true FAKE | FAKE predicted as REAL |
| REAL | true REAL | REAL predicted as FAKE |

**FAKE recall** asks: out of all truly fake images, what fraction did we detect as fake? **REAL recall** asks the equivalent question for real images.

**Balanced accuracy** is the average of FAKE recall and REAL recall. It gives equal importance to both classes even when their image counts differ.

## 3. Overfitting and domain shift

Our first CIFAKE-based EfficientNet result looked excellent on CIFAKE:

| Dataset | Accuracy | ROC-AUC | Evidence status |
|---|---:|---:|---|
| CIFAKE test | ≈97.65% | ≈0.9973 | Team-recorded |
| ByteDance external validation | ≈61.59% | ≈0.5764 | Team-recorded |

This gap was one of our most important findings.

**Overfitting** happens when a model learns patterns that work extremely well on its training environment but do not transfer to new data. The first model learned CIFAKE very well, but CIFAKE's fake images, real images, resolutions, compression histories, and visual content did not fully represent the ByteDance data.

This is also **domain shift**: the data distribution changes between training and evaluation. CIFAKE evaluation was **in-domain**, because its train and test sets came from the same dataset design. ByteDance evaluation was **out-of-domain**, with a different real-image collection and a different AI generator.

An **unseen generator** is an AI image generator that did not contribute images to training. Detecting it requires transferable evidence of AI generation rather than memorising the style of a known generator.

A **dataset shortcut** is an easier but misleading pattern. For example, if many fake training images are PNG files while many real images originated as JPEGs, a model might learn compression differences. It can then answer “which dataset does this resemble?” instead of “was this AI-generated?”

## 4. The naive multi-source failure

We next added more fake and real sources. The current repository dynamically discovers every prepared directory under `WildFake/FAKE` and `WildFake/REAL` in [`src/multisource_dataset.py`](../src/multisource_dataset.py). The repository documentation explicitly shows these example sources:

| Class | Sources verified in current repository documentation/code |
|---|---|
| FAKE | CIFAKE, ADM, DDPM, Midjourney |
| REAL | CIFAKE, COCO, LAION-5B, Flickr |

DALL-E 2 is part of the team-recorded project history, but its use cannot be verified from a local dataset, checkpoint, or fixed source declaration. The dynamic loader would accept it if it existed as a prepared source directory.

The first multi-source approach shuffled all images together. This sounds reasonable, but it sampled sources roughly according to their size. A source with hundreds of thousands of images could dominate one with only thousands.

The team-recorded outcome was:

| Evaluation | ROC-AUC |
|---|---:|
| Internal multi-source validation | ≈0.9723 |
| ByteDance external validation | ≈0.3660 |

An external AUC below 0.5 means the ranking was not merely weak; it was substantially reversed on that domain. Adding more data had made the external result worse.

The likely lesson was that the model could associate source identity with the label:

```text
wrong question learned by the model:
“Which dataset or generator does this image resemble?”

desired question:
“Does this image contain evidence of AI generation?”
```

More data is not automatically better. The composition and sampling of that data determine which patterns are easiest for the model to learn.

## 5. Source-balanced training

We addressed source-size imbalance with [`SourceBalancedBatchSampler`](../src/source_balanced.py). Each batch is approximately half fake and half real. Within each class, slots are distributed approximately evenly across active sources.

Conceptually:

```text
choose class balance
      ↓
approximately 50% FAKE and 50% REAL
      ↓
spread each class across its available sources
      ↓
choose images from each source
```

The actual sampler builds separate index groups for every `(class, source)` pair. Small sources are reshuffled and reused when exhausted. This prevents a large source from automatically receiving most training slots.

The `--samples-per-epoch` option sets the number of samples produced in one epoch. Its default is 100,000. Because small sources may be reused and large sources may be only partly sampled, an epoch is a controlled training budget rather than one complete pass through every unique file.

### Held-out-generator validation

For a held-out experiment, one fake generator is completely excluded from training. Validation then contains that generator's test images plus the available real test sources. The best checkpoint is selected using the held-out generator's FAKE-positive AUC rather than a pooled same-source score.

The team-recorded results included:

| Held-out fake generator | ROC-AUC |
|---|---:|
| DDPM | ≈0.7725 |
| ADM | ≈0.7537 |

These scores are lower than near-perfect same-dataset AUC, but they are more trustworthy for our goal. They ask whether the model can recognise a generator it never saw during training. A difficult validation test can be more useful than an easy test with an impressive number.

## 6. Adding Midjourney

Midjourney is explicitly supported and documented as a fake source in the current repository. A recent team-recorded experiment produced approximately:

```text
ByteDance accuracy ≈ 81.9%
ByteDance ROC-AUC  ≈ 0.8905
```

No local checkpoint, log, or result file identifies the exact model, run name, training duration, or data configuration that produced these figures. They must therefore remain **team-recorded**, not locally verified.

The result is consistent with the idea that generator diversity improves generalisation: Midjourney may have added visual and forensic patterns that were missing from the previous fake sources. However, we cannot claim that Midjourney alone caused the gain. Training duration, initial checkpoint, random sampling, augmentation, model version, and other sources may also have changed.

## 7. The EfficientNet baseline

EfficientNet-B0 is a convolutional neural network designed to extract useful image features efficiently. Instead of training it from scratch, our code can initialise it with weights learned from ImageNet, a large general image dataset. This is **transfer learning**.

```text
input image
    ↓
ImageNet-pretrained EfficientNet-B0
    ↓
1,280 learned image features
    ↓
one output logit
    ↓
P(REAL) and P(AIGC)
```

Pretraining gives the network a useful starting vocabulary of edges, textures, shapes, and objects. We then adapt those features to real-versus-AI classification. This is much easier and cheaper than asking a randomly initialised network to learn both general vision and AI-image forensics from our datasets alone.

### Staged fine-tuning

The shared trainer in [`src/train.py`](../src/train.py) uses two stages:

1. **Stage 1 — head only:** freeze the EfficientNet feature extractor and train the classifier or new frequency branches. The default EfficientNet-only classifier learning rate is `1e-3` for two epochs.
2. **Stage 2 — partial unfreezing:** keep early blocks frozen, unfreeze the final three top-level EfficientNet feature blocks (blocks 6–8), and fine-tune them slowly. The default classifier rate is `1e-4`, while the backbone uses `1e-5` for five epochs.

This approach protects useful pretrained features at the start and then lets later, more task-specific features adapt carefully.

## 8. FFT theory

An image normally lives in the **spatial domain**: a grid of pixels showing objects, shapes, colours, edges, and textures.

The **frequency domain** describes how quickly image values change across space:

- Low frequencies represent smooth gradients and broad structure.
- High frequencies represent edges, fine texture, small details, and noise.

The **Fast Fourier Transform (FFT)** is an efficient algorithm for converting the image from the spatial domain into a frequency representation.

```text
spatial image                         frequency view
pixels and local structure   → FFT → smooth-to-detailed frequency components
```

FFT is not a classifier. It is a mathematical transformation. We still need a neural network to learn which frequency patterns are useful for separating real and AI-generated images.

AI generators can leave frequency clues through upsampling, denoising, synthesis, or post-processing. Real cameras also have their own sensor, demosaicing, sharpening, and compression patterns. These clues are useful but dangerous: a model can also learn generator-specific or dataset-specific frequency shortcuts.

## 9. Hybrid model evolution

All hybrid versions receive one augmented RGB tensor in the `[0,1]` range. The EfficientNet path applies ImageNet normalisation internally, while FFT operates on the unnormalised image. This ensures that every branch sees the same crop and corruption without feeding ImageNet-normalised values into FFT.

### Hybrid V1: free feature fusion

Hybrid V1 is implemented in [`src/hybrid_model.py`](../src/hybrid_model.py).

```text
image ──→ EfficientNet ──→ 1,280 spatial features ──┐
  │                                                 ├─→ concatenate
  └──→ luminance FFT magnitude ─→ CNN ─→ 256 features ┘
                                                     ↓
                                          256-unit fusion MLP
                                                     ↓
                                                final logit
```

The FFT path computes luminance, performs a centred orthonormal FFT, takes `log(1 + magnitude)`, and standardises each spectrum. Spatial and frequency features are concatenated and mixed freely by a fusion network.

This flexibility can discover useful interactions, but it also provides no direct limit on how strongly frequency features affect the final prediction. V1 could therefore depend too heavily on generator- or dataset-specific frequency shortcuts.

V1 is implemented and tested structurally, but the repository contains no V1 checkpoint or result file that verifies its quantitative performance.

### Hybrid V2: controlled residual fusion

Hybrid V2 is implemented in [`src/hybrid_v2_model.py`](../src/hybrid_v2_model.py).

```text
image ─→ EfficientNet ─────────────→ spatial_logit
  │
  └──→ FFT magnitude ─→ CNN ──────→ frequency_logit

final_logit = spatial_logit + frequency_scale × frequency_logit
```

The default frequency scale is `0.25`. EfficientNet makes the main prediction, while FFT provides a smaller correction. This is **controlled residual fusion**.

V2 also implements training-only **frequency branch dropout**, defaulting to 20%. For selected training samples, the frequency correction is set to zero, forcing the spatial branch to solve the example alone. Optional frequency masking exists but defaults to zero probability.

The team recorded several V2-era external AUC values:

```text
≈0.739
≈0.769
later best ranking ≈0.7869
```

No local checkpoint or result file maps these values to exact V2 run names, so their precise checkpoint attribution is unverified.

### Hybrid V3: magnitude and phase

Hybrid V3 is implemented in [`src/hybrid_v3_model.py`](../src/hybrid_v3_model.py). It performs one shared FFT, then creates two representations:

```text
image ─→ EfficientNet ─────────────────→ spatial_logit
  │
  └──→ one FFT ─┬─→ magnitude ─→ CNN ─→ magnitude_logit
                └─→ phase ─────→ CNN ─→ phase_logit
```

**Magnitude** describes how much energy exists at each frequency. **Phase** describes how frequency components align spatially and is strongly connected to image structure and position.

Raw phase angles wrap from `+π` to `-π`, even though those directions are nearly identical. V3 avoids this artificial discontinuity by representing phase as two bounded channels:

```text
sin(phase)
cos(phase)
```

The supplied magnitude and phase weights are normalised to sum to one. With the default 0.5/0.5 weights:

```text
frequency_logit = 0.5 × magnitude_logit + 0.5 × phase_logit
final_logit     = spatial_logit + 0.25 × frequency_logit
```

V3 is implemented and covered by architecture tests. No local checkpoint or result file proves that it was trained or evaluated, so it is **experimental in this repository snapshot**.

### Hybrid V3.1: radial profile and learned fusion

Hybrid V3.1 is implemented in [`src/hybrid_v31_model.py`](../src/hybrid_v31_model.py). It adds a radial summary of the same normalised FFT magnitude.

Imagine drawing rings around the centre of the shifted FFT. Inner rings represent lower frequencies; outer rings represent higher frequencies. The radial extractor averages the magnitude values inside each ring.

```text
one shared FFT
├── magnitude image ─→ CNN ─────────────→ magnitude_logit
├── phase sin/cos ───→ CNN ─────────────→ phase_logit
└── 32 radial bins ──→ 64 → ReLU
                       → Dropout(0.3)
                       → radial_logit
```

V3.1 learns three raw fusion parameters and applies softmax:

```text
[w_mag, w_phase, w_radial] = softmax(raw weights)

w_mag + w_phase + w_radial = 1
each weight is non-negative
```

The weights start at one third each. The final calculation is:

```text
frequency_logit =
    w_mag    × magnitude_logit
  + w_phase  × phase_logit
  + w_radial × radial_logit

final_logit = spatial_logit + 0.25 × frequency_logit
```

The outer `0.25` keeps the entire frequency mixture as a controlled correction, even if the learned internal weights shift heavily toward one branch.

V3.1 reports its learned weights and average branch-logit magnitudes after validation epochs. It is implemented and structurally tested, but no local training checkpoint or evaluation artifact exists. It is therefore **experimental**.

## 10. Training augmentation

The challenge includes realistic transformations such as JPEG recompression, blur, resizing, noise, colour changes, and cropping. A detector should continue recognising an AI image after these changes:

```text
AI-generated image
        ↓
JPEG / blur / resize / noise / colour / crop
        ↓
still detected as AI-generated
```

The current training pipeline is defined in [`src/transforms.py`](../src/transforms.py):

```text
PIL RGB image
→ RandomResizedCrop to model size, retaining 80–100% area
→ horizontal flip with probability 0.5
→ downscale/upscale with probability 0.3
   └─ scale 0.25 or 0.5
→ real JPEG encode/decode with probability 0.4
   └─ integer quality uniformly from 30 to 90
→ Gaussian blur with probability 0.3
   └─ sigma uniformly from 0.5 to 2.0
→ brightness/contrast/saturation jitter of ±20% on every image
→ convert to tensor in [0,1]
→ Gaussian noise with probability 0.3
   └─ sigma uniformly from 0.02 to 0.10, then clamp to [0,1]
→ ImageNet normalisation for EfficientNet-only models
```

JPEG, resize, noise, and colour jitter broadly match the challenge ranges. The downscale implementation truly shrinks and restores the image. Noise is applied on the correct `[0,1]` scale. The blur requests sigma up to 2.0, but its fixed 3×3 kernel truncates severe blur, so exposure to a true sigma-2 blur is weaker than the name suggests. Random resized crop teaches crop invariance but is not identical to a deterministic centre crop retaining 80%.

Augmentation can improve robustness, but stronger is not always better. The optional resize, JPEG, blur, and noise gates are independent, so about 39.24% of images receive at least two of those four corruptions. Colour jitter and random crop/resizing always occur. A heavily corrupted image may lose the subtle texture, phase, or spectrum evidence we wanted the model to learn.

It is important to distinguish two kinds of diversity:

```text
source diversity = different generators and datasets
augmentation     = different transformations of existing images
```

Augmentation cannot replace missing generators, and generator diversity cannot by itself teach robustness to severe post-processing. We need both.

The locally verified [`results/robustness.json`](../results/robustness.json) reports:

| Internal test condition | Accuracy | ROC-AUC |
|---|---:|---:|
| Clean | 95.43% | 0.99499 |
| JPEG quality 50 | 96.14% | 0.99436 |
| Gaussian blur labelled sigma 2 | 95.76% | 0.99485 |
| Downscale 0.25 then upscale | 96.19% | 0.99391 |
| Gaussian noise sigma 0.10 | 94.10% | 0.98718 |

These are strong same-source results. However, the file contains no checkpoint identity, omits many official levels, and uses the undersized blur kernel. It should not be treated as complete proof of Track 5 robustness or attributed to a named model.

## 11. What worked and what failed

| Experiment | Result | Main lesson | Evidence status |
|---|---|---|---|
| CIFAKE only | Excellent CIFAKE performance; weak ByteDance performance | Near-perfect in-domain results can hide domain shift | Team-recorded metrics |
| Staged fine-tuning | Better external result than the earliest baseline | Careful transfer learning matters | Team-recorded qualitative outcome; implementation verified |
| Naive multi-source | Internal AUC ≈0.9723; ByteDance AUC ≈0.3660 | More data can amplify shortcuts when source sizes are imbalanced | Team-recorded metrics |
| Source-balanced training | External and held-out behaviour improved | Sampling strategy matters, not only dataset size | Implementation verified; external outcome team-recorded |
| Held-out DDPM/ADM | ≈0.7725 / ≈0.7537 AUC | Harder unseen-generator validation is more realistic | Team-recorded metrics |
| Hybrid V1 | Spatial plus FFT-magnitude feature fusion | Frequency information is complementary but can dominate freely | Architecture verified; quantitative result unavailable |
| Hybrid V2 | Controlled FFT residual; team-recorded external AUCs ≈0.739–0.7869 | Keep spatial evidence primary and constrain frequency shortcuts | Architecture verified; exact result attribution unavailable |
| Add Midjourney | ≈81.9% accuracy and ≈0.8905 ByteDance AUC | Generator diversity appears highly important | Team-recorded; exact checkpoint unavailable |
| Hybrid V3 | Added magnitude and phase residuals | Phase may add structural frequency evidence | Implemented; no local training/evaluation evidence |
| Hybrid V3.1 | Added radial profile and learned softmax fusion | Let the model balance complementary frequency views under a fixed outer scale | Implemented; no local training/evaluation evidence |

## 12. Main lessons from our experience

1. **High internal accuracy does not guarantee generalisation.** A model can excel on the dataset it knows and fail on a new generator.
2. **A model can learn dataset shortcuts instead of AI-generation evidence.** File format, source style, resolution, or compression history can be easier to learn than general forensic clues.
3. **More data can make performance worse.** If one source dominates, the added data changes the shortcut rather than solving it.
4. **Source balancing matters.** Giving each class and source a controlled role reduced the influence of dataset size.
5. **Held-out-generator validation is more realistic.** Lower but honest scores provide better guidance than near-perfect same-source validation.
6. **Generator diversity matters enormously.** The Midjourney-era result is consistent with broader fake-source coverage improving transfer, although it does not prove a single cause.
7. **Data design can matter as much as architecture.** Sampling and validation changes produced major insights before adding more model branches.
8. **Spatial and frequency information are complementary.** EfficientNet models objects and textures, while FFT branches can model spectral evidence.
9. **Failed experiments were useful.** The naive multi-source collapse revealed what the model was actually learning and directly motivated source balancing.

## 13. Our Development Experience

We began the project believing that the central problem was straightforward: take a strong convolutional neural network, fine-tune it on real and AI-generated images, and obtain a reliable detector. EfficientNet-B0 was a practical starting point because it already contained useful ImageNet features and could be adapted with a small binary classifier. Our first CIFAKE experiment appeared to confirm this plan. It achieved approximately 97.65% accuracy and 0.9973 AUC on the CIFAKE test set. At first glance, those numbers suggested that the detector was nearly finished.

The ByteDance validation result changed our understanding of the problem. Accuracy fell to roughly 61.59%, and AUC fell to about 0.5764. The network had learned CIFAKE extremely well, but it had not learned a general definition of AI generation. CIFAKE and ByteDance differed in generators, real-image sources, content, compression, and probably many other details. This was our first clear experience of overfitting and domain shift. A near-perfect internal score was not proof of a robust detector.

Our next response was to add more datasets. This also seemed obvious: if one source was too narrow, combining ADM, DDPM, DALL-E 2 in the team history, CIFAKE, COCO, LAION, and later other sources should create broader coverage. However, the naive multi-source experiment became our largest failure. Internal AUC remained high at approximately 0.9723, while ByteDance AUC collapsed to around 0.3660. More data had not solved the problem; it had made the external ranking substantially worse.

That failure forced us to examine how the data was sampled. Different sources had very different sizes. When all images were shuffled together, large sources appeared much more often. Labels could also become correlated with dataset properties. The model might identify a JPEG-heavy real source or the visual style of one fake generator instead of detecting general AI-generation evidence. We realised that “how much data?” was the wrong question unless we also asked “which source contributes each training example?”

We built a source-balanced sampler to address this. Each batch became approximately half fake and half real, and sources within each class contributed approximately equally. Small sources could be reused rather than disappearing beside much larger datasets. We also introduced held-out-generator validation. By removing DDPM or ADM completely from training and validating on that unseen generator, we obtained AUCs around 0.7725 and 0.7537. These values were lower than our same-dataset scores, but we trusted them more because the task was closer to the real challenge.

After improving the data pipeline, we explored frequency-domain forensic evidence. Hybrid V1 combined EfficientNet's spatial features with FFT-magnitude features. This gave the network access to two views of the image: visible content and spectral structure. It also revealed an architectural risk. If frequency features were freely mixed with spatial features, the network might allow a generator-specific frequency shortcut to dominate.

Hybrid V2 responded with controlled residual fusion. EfficientNet produced the main spatial logit, and the FFT branch supplied a correction scaled by 0.25. Frequency-branch dropout sometimes removed that correction during training, forcing the spatial path to remain useful. Team records include V2-era external AUCs around 0.739, 0.769, and later 0.7869, although the current repository does not contain the checkpoints needed to attribute each score exactly.

We then implemented Hybrid V3 to add phase alongside magnitude. Magnitude describes how much frequency energy is present, while phase helps describe how structures align. We represented phase with sine and cosine so that the wrap between `-π` and `+π` would not create a false discontinuity. Hybrid V3.1 extended the idea with a 32-bin radial spectrum and learned softmax weights for magnitude, phase, and radial logits. The overall frequency contribution remained scaled by 0.25. These versions are implemented, but without local checkpoints or result files we must describe them as experimental rather than successful.

Adding Midjourney was another major point in the journey. The team recorded a ByteDance result of about 81.9% accuracy and 0.8905 AUC. This strongly suggested that generator diversity was important. Still, we should not claim that Midjourney alone caused the improvement because other training details may have changed at the same time.

We originally thought the challenge was mainly about training a better CNN. Our experiments showed that performance depends on the entire system: dataset design, generator diversity, source sampling, validation strategy, augmentation, and architecture. The failed experiments were not wasted work. They exposed shortcuts, corrected our evaluation strategy, and changed what we considered a trustworthy result. Our most important lesson was that robust machine learning is not only about making a model more complex. It is about designing evidence that encourages the model to learn the right problem.
