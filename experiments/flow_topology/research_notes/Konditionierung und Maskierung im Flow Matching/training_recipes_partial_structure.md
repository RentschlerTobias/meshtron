# Training recipes for masked conditioning (continuous tokens) and AR/pointer generation with partially known structure

Scope note: 17 tool calls were used. Several primary PDFs (Palette, Cascaded Diffusion, SD-inpainting model card) could not be fetched, so some well-known details are listed under Gaps rather than stated as facts. Peer-review status is given as [PR: venue] or [preprint]; "status unverified" means I could not confirm it in this session.

## Q1. Mask distributions in practice and their measured effect ("train on the masks you will see")

### Takeaway
No single best ratio: the generative models that ablate this use a broad random ratio, weighted toward harder (higher) masking for generation (MaskGIT cosine, MAR [0.7,1.0]). An auxiliary masking loss in a diffusion transformer works best at a low ratio (MDT 30%). The clearest "match inference masks" evidence comes from the motif-scaffolding work, which samples random multi-segment motifs covering 5-50% of residues to cover the test-time patterns.

### Cited Findings
- MaskGIT [PR: CVPR 2022] samples the mask ratio from a schedule function and ablated seven of them on ImageNet 256 (FID): cosine 6.06 (IS 181.5), square 6.35, cubic 7.26, linear 7.51 (IS 113.2), exponential 7.89, sqrt 12.33, log 29.17. Concave schedules, which give more high-mask training cases, win, but over-prioritising them hurts (cubic < square, exponential much worse). — [MaskGIT](https://arxiv.org/pdf/2202.04200); summary of the ablation via [search result for CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf)
- MAR [PR: NeurIPS 2024], continuous tokens with a per-token diffusion loss: during training the masking ratio is sampled uniformly in [0.7, 1.0] (0.7 = 70% unknown). At inference the ratio goes from 1.0 to 0 on a cosine schedule. The encoder is always padded with 64 [cls] tokens "to improve stability and capacity". Mask tokens [m] plus positional embedding are inserted in the decoder. — [MAR](https://arxiv.org/html/2406.11838v2)
- MDT [PR: ICCV 2023] adds mask latent modeling to a DiT (an asymmetric encoder/side-interpolator predicts masked latent tokens while still denoising). The best mask ratio for MDT-S/2 is **30%**, much lower than MAE's 75%, because generation needs detail from more patches. A model trained at 30% still handles much larger masks (e.g. 70%) at inference. — [MDT/MDTv2](https://arxiv.org/pdf/2303.14389); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf)
- FrameFlow motif amortization [preprint, arXiv 2401.04082, status unverified] builds training motifs by random augmentation of unlabeled PDB structures: motif fraction between γ_min = 0.05 and γ_max = 0.5 of residues, a random number of motif segments (random start residues, then extended by appending residues). This gives multi-segment masks resembling the benchmark's discontiguous motifs. — [Yim et al., Improved motif-scaffolding with SE(3) flow matching](https://arxiv.org/html/2401.04082)
- Diffusion Forcing [PR: NeurIPS 2024] trains with **independent per-token noise levels**. As a result, at inference any subset of tokens can be held at zero noise (fully known) while the others start from full noise, so arbitrary mask patterns are covered by construction. — [Diffusion Forcing](https://arxiv.org/pdf/2407.01392v3); [NeurIPS version](https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf)
- Language-model analogue [preprint 2022]: in Fill-in-the-Middle, a large fraction of transformed (middle-span-moved-to-end) training documents does **not** degrade left-to-right perplexity or sample quality ("FIM-for-free"). The paper ablates FIM rate, span selection (character vs line level), PSM vs SPM ordering, and pretraining vs finetuning, and recommends FIM by default. — [Bavarian et al., FIM](https://arxiv.org/abs/2207.14255)

### Inferences
- For a set/flow transformer with a known subset (e.g. fixed core blocks or blade-adjacent tokens): sample a per-example known-fraction from a broad distribution, plus a structured component that mimics the inference pattern (contiguous/spatially coherent groups, like FrameFlow's multi-segment motifs). Diffusion Forcing's per-token noise gives coverage by construction, but it spends capacity on mixed-noise patterns that may never occur at inference.
- The FIM and MDT results suggest that mixing a large share of masked/conditional training with unconditional training costs little on the unconditional task. Mixing in an unconditional share (known set empty) is cheap and also enables CFG-style guidance.
- If the inference-time known fraction is known (e.g. always ~X% of blocks), weight the ratio distribution toward it but keep tails. MaskGIT shows that badly skewed schedules (log/sqrt) are very costly.

### Gaps
- BERT's 15% / 80-10-10 rule and MAE's ablation curve were not re-fetched (the MAE 75% figure is cited only via MDT).
- I found no paper that directly ablates "structured masks matching inference" vs "uniform random masks" for continuous-token flow matching. LaMa/SD-inpainting mask-generator ablations were not retrieved.

## Q2. Representing known tokens: clean values + mask flag, time embedding, loss masking, boundary consistency

### Takeaway
The best-documented recipe for flow matching (FrameFlow-amortization):
- keep known tokens **clean (un-noised)**;
- add a **binary 1-D mask channel** (1 = known);
- compute the flow-matching loss **only on unknown tokens**;
- remove trivial cues that leak the answer, e.g. zero-center the motif and noise so the model cannot read the scaffold position from the motif's absolute offset.

Per-token time (Diffusion Forcing) generalises this: known tokens are simply tokens at t = clean.

### Cited Findings
- FrameFlow-amortization: "a 1D mask as additional input with a 1 at the location of the motif and 0 elsewhere". During training "only the scaffold is corrupted with noise". The loss is the SE(3) flow-matching loss on scaffold tokens only, conditioned on the motif. Motif and initial noise are zero-centered, which "prevents the model from using the motif offset from the origin to memorize scaffold locations". — [FrameFlow motif paper](https://arxiv.org/html/2401.04082)
- Diffusion Forcing gives every token its own noise level, so known tokens are represented as noise level 0 under the same per-token time embedding. There is no separate token type: the per-token time acts as the mask signal. — [Diffusion Forcing](https://arxiv.org/pdf/2407.01392v3)
- MAR represents unknown tokens with a learned [m] token plus positional embedding in the decoder (MAE style). Known tokens pass through the encoder as clean values. — [MAR](https://arxiv.org/html/2406.11838v2)
- RFdiffusion [PR: Nature 2023; bioRxiv preprint cited]: ablating self-conditioning, pretraining, or MSE loss (vs FAPE) "dramatically decrease[s]" performance. Motif amortization (training a conditional model that takes the motif as input) was introduced there. Fine-tuned variants for small active-site motifs "hold smaller motifs better". — [RFdiffusion bioRxiv](https://www.biorxiv.org/content/10.1101/2022.12.09.519842.full.pdf); [repo](https://github.com/RosettaCommons/RFdiffusion)

### Inferences
- Recommended defaults for a continuous-token flow transformer with known tokens:
  - x_known clean (or t_known = 1 / data end in the FM convention);
  - a learned "known" embedding or mask bit added to the token;
  - loss masked to unknown tokens.
- Give known tokens a per-token time = clean (Diffusion-Forcing style) rather than the global t, so the model never confuses a known token with a noisy one.
- Watch for leakage cues analogous to FrameFlow's origin offset: absolute positions or canonical orderings of known blocks that let the model memorise where unknown blocks go. Center/normalise jointly and randomise ordering.
- Boundary consistency: with loss only on unknown tokens, the model never pays for changing a known token, so known tokens must be hard-overwritten at every integration step. They are clean by construction, so this is automatic if inputs are re-clamped.

### Gaps
- No quantitative ablation found for "loss on unknown only vs on all tokens", or for "known tokens receive global t vs t = clean", in flow-matching transformers.
- Stable Diffusion inpainting details (5 extra input channels for mask + masked-image latent, zero-initialised, LaMa-style masks plus a fraction fully masked) are from memory: the model card returned 401 and the repo URL 404, so treat them as unverified.
- Palette's mask mix (free-form + box) and its ablation on loss region were not retrievable from the abstract page.

## Q3. Amortized (trained) conditioning vs inference-time inpainting/guidance: accuracy and speed

### Takeaway
On the motif-scaffolding benchmark, amortized conditioning clearly beats guidance on the same base model:
- same number of motifs solved (20/24);
- 353 vs 192 unique designable clusters (about 1.8x more);
- same 18 s runtime.

Both FrameFlow variants are 2.8-6.5x faster than RFdiffusion (50 s) and TDS (117 s). In image inpainting, training-free RePaint is roughly competitive with trained Palette on LPIPS but worse on FID. That last comparison comes from a secondary table, so treat it with caution.

### Cited Findings
- FrameFlow table (24 motifs): amortization 20/24 solved, 353 unique designable clusters, 18 s; guidance (reconstruction guidance from the unconditional model) 20/24, 192 clusters, 18 s; RFdiffusion 20/24, 141 clusters, 50 s; TDS (twisted SMC) 19/24, 217 clusters, 117 s. Authors: "Amortization outperforms guidance but requires conditional training while guidance can use unconditional flow models without further training." They blame the guidance gap on approximation error in estimating ∇log p_t(motif | x_t) from denoised predictions. — [FrameFlow motif paper](https://arxiv.org/html/2401.04082)
- RePaint [PR: CVPR 2022] conditions an **unconditional** DDPM only by replacing unmasked regions during reverse steps (plus resampling). It generalises to any mask without retraining. — [RePaint](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html)
- Palette [PR: SIGGRAPH 2022 per my knowledge, status unverified here] trains conditional diffusion models for inpainting/uncropping and reports beating GAN/regression baselines. — [Palette](https://arxiv.org/abs/2111.05826)
- One comparison table reported via search (apparently from GradPaint, arXiv 2309.09614): FFHQ, Palette LPIPS 0.096 / FID 7.28 vs RePaint LPIPS 0.090 / FID 9.09. — [GradPaint](https://arxiv.org/pdf/2309.09614). Caveat: I could not verify these numbers in the source; treat as indicative only.
- DEFT (arXiv 2406.01781) frames conditional fine-tuning as learning the generalised h-transform, a middle ground between full amortization and training-free guidance. — [DEFT](https://arxiv.org/pdf/2406.01781). Its numbers were not retrieved.

### Inferences
- When the conditioning pattern is fixed and known (e.g. given blocks), amortized training is the default choice: equal speed, better diversity/accuracy.
- Replacement-style inpainting (RePaint) is a cheap baseline and a useful diagnostic, but it needs resampling loops (slower) to fix boundary inconsistency.

### Gaps
- No direct FM-transformer benchmark on geometric/mesh tokens comparing amortized conditioning vs replacement/guidance was found.

## Q4. AR / pointer generation of graphs, meshes, blocks with given partial connectivity

### Takeaway
Three practical patterns:
1. **Prefix conditioning**: put the given faces/subgraph first in the canonical order and continue (MeshGPT and MeshXL completion).
2. **History-free state-based decoding from an arbitrary seed subgraph**, trained with randomised generation order (MoLeR), which works for any given subgraph, not just canonical prefixes.
3. **Hard logit masking at inference** for validity constraints (PolyGen). PolyGen found that masking during *training* slightly hurts.

### Cited Findings
- PolyGen [PR: ICML 2020]:
  - Face model is a pointer network, p(f_n = k | f_<n, V) = softmax_k(p_nᵀ e_k), over transformer-encoded vertex embeddings.
  - Hard constraints at sampling time via logit masking: non-decreasing z, stop only after x, no duplicate vertex index within a face, no repeated new-face token, indices only to existing vertices.
  - "Masking during training [worsens] performance to a small degree, so we always train without masking". Masking at evaluation redistributes probability mass to valid values.
  - Uses nucleus sampling p = 0.9 to reduce sampling degradation.
  - Failure modes: disconnected components, premature stop token (incomplete objects), formless meshes.
  - [PolyGen](https://ar5iv.labs.arxiv.org/html/2002.10880)
- MeshGPT [PR: CVPR 2024]: VQ-tokenised face sequence plus decoder-only transformer. Shape completion is done by feeding the partial mesh as prefix and sampling multiple completions. — [MeshGPT](https://arxiv.org/pdf/2311.15475)
- MeshXL [PR: NeurIPS 2024]: NeurCF coordinate sequences with a pre-defined ordering. Supports unconditional generation and **partial mesh completion** (prefix continuation); 125M/350M/1.3B models. — [MeshXL](https://arxiv.org/html/2405.20853v2)
- MoLeR [PR: ICLR 2022]:
  - The decoder is **not conditioned on generation history**, only on the current partial graph (GNN) plus latent, so any scaffold can be the initial state of the decoding loop.
  - Because generation order is randomised during training, it "implicitly learns to complete arbitrary subgraphs".
  - Outperforms baselines on scaffold-based tasks, about 10x faster to train and sample.
  - [MoLeR](https://arxiv.org/abs/2103.03864v3); [MSR blog](https://www.microsoft.com/en-us/research/blog/moler-creating-a-path-to-more-efficient-drug-design/)
- Orderless Regularization [preprint 2024, status unverified] penalises differences in the RNN hidden state across valid orderings of the same graph. It helps sequential graph generators, especially with little data. No numbers were retrieved. — [Overcoming Order in AR Graph Generation](https://arxiv.org/abs/2402.03387)
- TreeMeshGPT (2503.11629) and ARMesh (2509.20824) are newer AR mesh orderings (tree sequencing, next-level-of-detail). They are relevant if the given part is not a canonical prefix. Details were not fetched. — [TreeMeshGPT](https://arxiv.org/pdf/2503.11629); [ARMesh](https://arxiv.org/pdf/2509.20824)

### Inferences
- Encoding given blocks/edges, three options:
  - **(a) prefix tokens**: simple, but only valid if the given part is a prefix of the canonical order. Otherwise there is a train/test mismatch, since training saw given parts only as canonical prefixes.
  - **(b) MoLeR-style state encoding**: encode the current partial structure (given + generated) with a GNN/transformer at every step, with no history. Works for arbitrary given subsets, given randomised orders in training.
  - **(c) FIM-style**: move the given part to a conditioning segment with a separator, training with random given subsets at a high rate. FIM evidence suggests this costs little on unconditional generation.
- Validity: implement given-connectivity and topological constraints (e.g. "no duplicate blocks", "pointer only to existing/allowed nodes", "do not re-emit given blocks") as inference-time logit masks. Following PolyGen, train without the masks, or at least ablate this. Combine with beam search or nucleus sampling.

### Gaps
- No quantitative result found comparing prefix vs attention-bias vs mask encodings of given edges.
- No MeshAnything/BPT partial-mesh completion numbers were retrieved.
- Constrained beam search with hard masks in mesh generators: only PolyGen's qualitative statement was found.

## Q5. Failure modes and diagnostics

### Takeaway
The main documented risks:
- compounding error when known/conditioning inputs at inference come from an imperfect upstream model rather than ground truth (fix: noise or conditioning augmentation during training);
- leakage shortcuts in known tokens (fix: centering/randomisation);
- training/inference mask-pattern mismatch;
- AR-specific early stopping, disconnected components and sampling degradation.

### Cited Findings
- Cascaded diffusion [PR: JMLR 2022]: "conditioning augmentation prevents compounding error during sampling in a cascaded model". This is data augmentation (noise/blur) of the conditioning inputs during training. Final FIDs: 1.48 (64²), 3.52 (128²), 4.88 (256²). — [Ho et al., Cascaded Diffusion](https://arxiv.org/abs/2106.15282). Ablation numbers not retrieved (the PDF exceeded the fetch limit).
- Shortcut/leakage: FrameFlow zero-centers motif and noise so the model cannot memorise scaffold location from the motif offset. — [FrameFlow motif paper](https://arxiv.org/html/2401.04082)
- Guidance-based conditioning suffers approximation error (FrameFlow-guidance 192 vs 353 clusters). — [FrameFlow motif paper](https://arxiv.org/html/2401.04082)
- AR failure modes (PolyGen): disconnected components, premature stop token, formless outputs. Sampling degradation is mitigated by nucleus p = 0.9 and by inference-time validity masks. — [PolyGen](https://ar5iv.labs.arxiv.org/html/2002.10880)
- Masked-diffusion LMs show a "training-inference divide": MDPO (arXiv 2508.13148) targets the mismatch between random training masks and the progressive inference unmasking schedule. — [MDPO](https://arxiv.org/pdf/2508.13148). Numbers were not fetched.
- Order dependence in AR graph models (OLR) is a separate source of overfitting with small data. — [OLR](https://arxiv.org/abs/2402.03387)

### Inferences
- If known tokens at inference are estimated (e.g. upstream-predicted blocks) rather than exact (e.g. exact dtOO blade geometry), train with noise-augmented known tokens and optionally feed the augmentation level as a feature (cascaded-diffusion recipe). If they are exact, clean GT is fine.
- Diagnostics to run:
  - accuracy vs known-fraction curve (does the model degrade at fractions outside the training distribution?);
  - shuffled/permuted known tokens (does the model use them or ignore them?);
  - zeroed mask flag (over-reliance on the flag vs content);
  - for AR: rate of premature stops, disconnected components, and invalid pointer attempts before masking.

### Gaps
- No quantitative study found on exposure bias specifically with given prefixes in mesh/graph AR models.
- No quantitative "over-reliance on conditioning features" ablation found for geometric generators.
