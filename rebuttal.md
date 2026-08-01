| Method | R@Top 1 | R@Top 2 | R@Top 3 | FID | CLIP Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Real motions | 0.940 ±.001 | 0.976 ±.001 | 0.985 ±.001 | 0.001 ±.000 | 0.837±.000 |
| MDM  | 0.503 ±.002 | 0.653 ±.002 | 0.727 ±.002 | 57.783 ±.092 | 0.481±.001
| StableMoFusion | 0.679 ±.002 | 0.823 ±.002 | 0.888 ±.002 | 27.801 ±.063 | 0.605 ±.001
| MARDM  | 0.659 ±.002 | 0.812 ±.002 | 0.860 ±.002 | 26.878 ±.131 | 0.602±.001
| MoMask | 0.777 ±.002 | 0.888 ±.002 | 0.927 ±.002 | 17.404 ±.051 | 0.664±.001
| MoMask++ in | 0.805 ±.002 | 0.904 ±.002 | 0.938 ±.001 | 15.56 ±.071 | 0.684±.001
| MoMask++ cra | 0.802 ±.001 | 0.905 ±.002 | 0.938 ±.001 | 15.06 ±.065 | 0.685±.001
| | | |  | | |
| MDM+CLIP | xxx | xxx | xxx | xxx | xxx |
| MDM+BERT | xxx | xxx | xxx | xxx | xxx |
| LeGO  | xxx | xxx | xxx | xxx | xxx |
| LeGO without tuning CLIP  | xxx | xxx | xxx | xxx | xxx |
| MDM+LeGO-CLIP  | xxx | xxx | xxx | xxx | xxx |

---

**Response to Reviewer: Fine-grained Analysis across Prompt Types**

We thank the reviewer for this constructive suggestion. To verify that LeGO-CLIP learns better text representations robustly across different prompt types, we conducted a stratified evaluation on the HumanML3D test set by partitioning prompts into three categories based on token count:

- **Short**: ≤ 8 words (1,548 samples, 35.3%)
- **Medium**: 9–13 words (1,360 samples, 31.0%)
- **Long**: ≥ 14 words (1,477 samples, 33.7%)

The boundaries were determined by the 33% and 67% percentiles of the text-length distribution to ensure roughly balanced splits. Both MDM (baseline, without CLIP LoRA) and LeGO-CLIP (ours, with CLIP LoRA fine-tuning) were evaluated under identical conditions: 50 diffusion steps, classifier-free guidance scale 2.5, and the same GRU-based evaluator for FID and R_precision.

**Results (R_precision & FID):**

```
| Category | Model     | R@1    | R@2    | R@3    | FID    |
|----------|-----------|--------|--------|--------|--------|
| Short    | MDM       | 0.4145 | 0.6161 | 0.7207 | 0.4539 |
| Short    | LeGO-CLIP | 0.5408 | 0.7270 | 0.8157 | 0.1356 |
| Medium   | MDM       | 0.3990 | 0.5589 | 0.6810 | 0.5218 |
| Medium   | LeGO-CLIP | 0.5283 | 0.7289 | 0.8190 | 0.1447 |
| Long     | MDM       | 0.3883 | 0.5565 | 0.6656 | 0.5507 |
| Long     | LeGO-CLIP | 0.4854 | 0.6915 | 0.7879 | 0.1708 |
```

**Relative improvement of LeGO-CLIP over MDM:**

```
| Category | ΔR@1    | ΔR@2   | ΔR@3   | ΔFID      |
|----------|---------|--------|--------|-----------|
| Short    | +30.5%  | +18.0% | +13.2% | -70.1%    |
| Medium   | +32.4%  | +30.4% | +20.3% | -72.3%    |
| Long     | +25.0%  | +24.3% | +18.4% | -69.0%    |
```

**Key observations:**

1. **R_precision**: LeGO-CLIP consistently and substantially outperforms MDM across all three length categories. The improvement is most pronounced at R@1, with relative gains of +25% to +32%, indicating that CLIP LoRA fine-tuning strengthens fine-grained text-motion alignment regardless of prompt length.

2. **FID**: LeGO-CLIP achieves markedly lower FID (better fidelity) than MDM in every category, with a ~70% relative reduction. This demonstrates that the improved text representations translate into higher-quality generated motions.

3. **Robustness**: The consistent improvement from short (≤8 words) to long (≥14 words) prompts confirms that LeGO-CLIP is robust to varying description lengths. Even for concise prompts with limited textual context, the fine-tuned CLIP encoder extracts more discriminative features than the frozen counterpart.

4. **Length effect**: Both models exhibit an expected trend where R_precision declines slightly as prompts grow longer (harder to precisely match richer descriptions), while generated motion fidelity remains stable.

Regarding the reviewer's additional suggestion to stratify by **number of action verbs** as a measure of action complexity — this is an excellent direction that we plan to incorporate. We will parse each caption with a POS tagger, count the number of action verbs, and report stratified results across varying levels of action complexity in the revised manuscript. This analysis will further elucidate whether LeGO-CLIP's advantage stems primarily from better encoding of static attributes or also from improved representation of dynamic action semantics.