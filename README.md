# Do Masked Autoencoders Learn Less Texture-Biased Representations?

A controlled PyTorch study of whether masked autoencoders (MAEs) learn a different *kind* of representation from standard autoencoders, not just a more accurate one.

All models share the same small Vision Transformer encoder, data, optimiser and training schedule. The only thing that changes is the reconstruction objective:

| Model | What the encoder sees during pretraining |
|---|---|
| **AE** | Every patch of the image |
| **DAE** | Every patch, with Gaussian noise added (σ = 0.25) |
| **MAE-25 / 50 / 75** | Only the visible patches, with 25%, 50% or 75% masked out |

The denoising AE is the key control. It makes reconstruction harder *without* removing spatial context, so it separates "masking helps" from "any harder task helps".

## Key results

The encoders are frozen after pretraining and evaluated on STL-10 (5,000 labelled training images, 8,000 test images).

| Model | Linear probe | kNN | Local shuffle | High-pass retention ↓ | Cue-conflict: follows global cue |
|---|---|---|---|---|---|
| AE | 35.3% | 27.0% | 23.6% | 0.300 | 33.2% |
| DAE | 38.7% | 28.4% | 26.6% | 0.293 | 36.8% |
| MAE-25 | 43.3% | 35.2% | 35.6% | 0.231 | 42.1% |
| MAE-50 | 47.1% | 37.3% | 37.7% | 0.212 | 43.9% |
| **MAE-75** | **48.5%** | **38.7%** | **41.5%** | **0.210** | **46.0%** |

- **Transfer improves steadily with mask ratio.** MAE-75 beats the AE by +13.2 pp on linear probing (95% paired bootstrap CI [+11.9, +14.5]; McNemar p ≈ 7 × 10⁻⁹²).
- **Less reliance on local texture.** MAE-75 loses 7.0 pp under local patch shuffling, against 11.7 pp for the AE, and gets proportionally less of its accuracy from high-frequency-only images.
- **More reliance on global cues.** In hybrid images (the low frequencies of one image combined with the high frequencies of another), MAE-75 follows the low-frequency source 46% of the time, against 33% for the AE.
- **It's not just a harder task.** The DAE improves only slightly over the AE and stays far behind even MAE-25. CKA shows the MAEs form their own representational family.
- **Better with few labels.** With only 10% of the labels, MAE-75 reaches 37.5%, against 24.8% for the AE.

**Conclusion:** masking doesn't produce a purely shape-based model, but it reduces dependence on local texture-like shortcuts and strengthens globally integrated cues.

## Evaluation battery

| Probe | What it tests |
|---|---|
| Linear probe and kNN (k = 5) | How useful and well structured the frozen features are |
| Low-pass / high-pass Fourier filtering | Dependence on coarse layout vs fine detail |
| Local patch shuffle (8×8 blocks) | Whether short-range texture continuity is needed |
| Gaussian blur | Robustness to losing fine texture |
| Hybrid cue-conflict images | Whether predictions follow global or local cues |
| Sample efficiency (10 to 100% of labels) | Feature quality under limited supervision |
| Paired bootstrap and McNemar tests | Whether gains over the AE are statistically reliable |
| RSA and linear CKA | Representation geometry and similarity between models |

## Setup

| Setting | Value |
|---|---|
| Encoder | ViT: 96×96 input, 16×16 patches, 192-d embeddings, depth 4, 4 heads (~2.26M params) |
| Decoder | 2 transformer layers, discarded after pretraining |
| Pretraining data | 25,000 unlabelled STL-10 images |
| Training | 40 epochs, AdamW (lr 1.5e-4, weight decay 0.05), cosine schedule with 5 warm-up epochs, batch 128, mixed precision |
| Loss | Per-patch normalised MSE (computed on masked patches only for the MAEs) |
| Hardware | Single Tesla T4 (Kaggle), about 17 minutes per model |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook mae_texture_bias.ipynb
```

Set `RUN_PRESET` in the configuration cell:

| Preset | Use |
|---|---|
| `SMOKE` | Quick end-to-end check on fake data, with no download |
| `BALANCED` | The configuration used for the reported results |
| `FULL_HONOURS` / `HD_PLUS` | Longer, multi-seed runs if you have the GPU time |

STL-10 (about 2.6 GB) downloads automatically. Checkpoints are cached and reused unless `force_retrain = True`. Outputs are written to `./mae_texture_bias/`:

```
mae_texture_bias/
├── checkpoints/   trained encoders
├── figures/       loss curves, reconstructions, probe and ablation plots
└── tables/        CSV results, including final_results.csv
```

## Limitations

- **Single pretraining seed.** The confidence intervals cover evaluation-set uncertainty, not variation between training runs.
- **Small scale.** The encoder is tiny and pretraining is short, so absolute accuracies are low. The study is about *relative* differences under matched conditions, not state-of-the-art performance.
- **Imperfect probes.** Texture, edges and shape are entangled in natural images, so no single probe measures shape bias. The conclusions rest on agreement across all of them.

## Repository contents

| File | Purpose |
|---|---|
| `mae_texture_bias.ipynb` | Full pipeline: data, models, pretraining, evaluation and figures |
| `paper.pdf` | The full write-up |
| `requirements.txt` | Python dependencies |

---

Group project for COMP4329 (Deep Learning), University of Sydney, 2026.
