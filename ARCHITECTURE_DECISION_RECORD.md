# Architecture Decision Record (ADR)

> This document logs all significant architectural and design decisions made throughout the project. Each entry includes the context, decision, rationale (with mathematical/logical justification where applicable), and consequences.

---

## ADR-001: GroupKFold by Writer ID for Cross-Validation

**Date:** 2026-03-06  
**Status:** Accepted  
**Category:** Data Splitting

### Context

The CircleID dataset contains 40,250 images from 51 writers using 8 pen types. A single writer contributes multiple samples across multiple pens. Standard KFold would allow the same writer's images to appear in both training and validation sets, causing the model to learn writer-specific traits (pressure, speed) rather than pen-specific features.

### Decision

Use `sklearn.model_selection.GroupKFold(n_splits=5)` with `groups=writer_ids`.

### Mathematical Justification

Let $\mathcal{W}_{\text{train}}^{(k)}$ and $\mathcal{W}_{\text{val}}^{(k)}$ denote the writer sets for fold $k$. We enforce:

$$\forall k \in \{1,\ldots,5\}: \quad \mathcal{W}_{\text{train}}^{(k)} \cap \mathcal{W}_{\text{val}}^{(k)} = \emptyset$$

This eliminates information leakage through writer identity. The model is forced to generalize pen features across unseen handwriting styles, which mirrors the test-time distribution where unknown writers may appear.

### Consequences

- **Positive:** Validation accuracy faithfully estimates test-time performance on unseen writers.
- **Negative:** Fold sizes may be slightly unequal due to varying number of samples per writer. Acceptable given 51 writers across 5 folds (~10 writers per val fold).

---

## ADR-002: Ink-Safe Augmentation Policy

**Date:** 2026-03-06  
**Status:** Accepted  
**Category:** Data Augmentation

### Context

The discriminative signal for pen classification resides in **ink-level features**: stroke texture, ink deposition density, line width consistency, and granularity. Aggressive augmentations (heavy color jitter, strong blur, elastic transforms) can destroy these subtle features, making different pen types indistinguishable.

### Decision

Adopt a restricted augmentation policy:
- **Allowed:** Geometric transforms (flip, rotate — circles are rotationally invariant), light color jitter (brightness ≤ 0.1, hue ≤ 0.02), Gaussian noise (σ ≤ 25), coarse dropout.
- **Prohibited:** Heavy color jitter (AP-07), elastic/grid distortion on stroke regions, aggressive blur.

### Logical Justification

Let $\phi(x)$ represent the ink texture feature of image $x$, and $T$ an augmentation transform. We require:

$$d(\phi(T(x)), \phi(x)) < \epsilon \quad \forall T \in \mathcal{T}_{\text{allowed}}$$

i.e., the augmentation must not shift the ink feature representation beyond a tolerance $\epsilon$ in feature space. Geometric transforms (flips, rotations) do not alter pixel intensities, preserving $\phi$. Light color jitter maintains relative intensity ordering.

### Consequences

- **Positive:** Model learns genuine pen-discriminative features.
- **Negative:** Smaller augmentation space may reduce overall regularization effect. Mitigated by dropout, EMA, and label smoothing.

---

## ADR-003: Aspect-Ratio-Preserving Resize with Padding (AP-09)

**Date:** 2026-03-06  
**Status:** Accepted  
**Category:** Preprocessing

### Context

Input images vary in size and aspect ratio. Naive resizing (e.g., `cv2.resize(img, (224, 224))`) distorts circles into ellipses and uses bilinear interpolation by default, which blurs stroke edges and destroys ink granularity — the very features needed for pen classification.

### Decision

Implement `resize_with_pad()`:
1. Compute scale factor: `scale = target_size / max(H, W)`
2. Use `INTER_AREA` for downscaling, `INTER_CUBIC` for upscaling
3. Pad to square with white (`255`) background
4. Result: (224, 224, 3) with preserved aspect ratio

### Mathematical Justification

**Interpolation quality:** For downscaling by factor $s < 1$, `INTER_AREA` computes the average over the source region mapped to each destination pixel:

$$I_{\text{dst}}(x, y) = \frac{1}{|R_{xy}|} \sum_{(u,v) \in R_{xy}} I_{\text{src}}(u, v)$$

This is a proper anti-aliased downsample, unlike `INTER_LINEAR` which samples a single point and aliases high-frequency ink texture.

**Aspect ratio preservation:** Padding preserves the original width-to-height ratio, avoiding shape distortion of the drawn circles.

### Consequences

- **Positive:** Ink texture features are preserved through scale changes.
- **Negative:** Padding introduces non-informative border regions. The model must learn to ignore padding, which is straightforward with sufficient data.

---

## ADR-004: Backbone Selection — ConvNeXt-Tiny (Provisional)

**Date:** 2026-03-06  
**Status:** Proposed — Pending Experimental Validation  
**Category:** Model Architecture

### Context

The pen classification task requires extracting **fine-grained texture features** from hand-drawn circle images. The backbone must balance capacity (to capture subtle ink characteristics) with efficiency (to enable rapid experimentation across 5 folds).

### Decision

Start with **ConvNeXt-Tiny** (28.6M params, ImageNet Top-1: 82.1%) as the initial backbone. Evaluate **EfficientNet-B3** and **SwinV2-Tiny** as alternatives in ablation experiments.

### Logical Justification

| Factor | ConvNeXt-Tiny | EfficientNet-B3 | SwinV2-Tiny |
|---|---|---|---|
| Texture inductive bias | ✅ Strong (convolution kernels) | ✅ Strong | ⚠️ Weaker (attention is global) |
| Pretrained quality | ✅ 82.1% IN-1K | ✅ 81.6% IN-1K | ✅ 81.8% IN-1K |
| Training speed (V100) | ~45 img/s | ~60 img/s | ~35 img/s |
| Fine-tuning stability | ✅ LayerNorm + residual | ✅ BN + SE blocks | ⚠️ Requires careful lr tuning |

ConvNeXt's full-resolution feature maps in early stages are well-suited for capturing pixel-level ink texture, unlike ViT-based architectures that tokenize and may lose local detail.

### Consequences

- **Positive:** Strong baseline with proven texture extraction capability.
- **Negative:** Must validate empirically. If ConvNeXt-Tiny underfits, upgrade to ConvNeXt-Small. If attention-based global reasoning proves important, switch to SwinV2.

---

*New entries should be appended below this line.*

---
