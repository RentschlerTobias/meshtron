# Masked / conditional flow matching and diffusion inpainting with fixed known tokens

Scope: how a flow-matching (FM / rectified flow / CFM) or diffusion model generates the unknown part of a sample when part of it is known and must stay fixed. Notation (FM convention, t=0 noise, t=1 data unless stated): x_1 data, x_0 ~ N(0,I), m = binary mask (1 = unknown / to generate, 0 = known), x_t = (1-t) x_0 + t x_1, target velocity u_t = x_1 - x_0. Peer-review status is given per source where it could be verified.

## 1. Training-time masked conditioning (amortized / "any-subset" conditioning)

### Takeaway
The standard recipe is the same everywhere (speech, proteins, molecules, images): give the known tokens to the network **clean** (either as a separate context channel or by holding them at "t = data / noise level 0"), add a binary mask flag, noise only the unknown tokens, and compute the loss only on the unknown tokens. Ablations show the masked loss helps (Voicebox), and in protein motif scaffolding amortized training beats inference-time guidance on diversity at equal success rate (353 vs 192 unique designable clusters).

### Cited Findings

**Voicebox (Le et al., Meta; NeurIPS 2023, peer-reviewed) – the cleanest masked-FM recipe**
- Problem setup: x_mis = m ⊙ x, x_ctx = (1 - m) ⊙ x; model learns p(x_mis | y, x_ctx). — [Voicebox arXiv 2306.15687](https://arxiv.org/pdf/2306.15687)
- Path: OT path, x_t = (1 - (1 - σ_min) t) x_0 + t x, target u_t(x_t | x) = x - (1 - σ_min) x_0. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Embedding of known tokens: the three sequences (x_t, x_ctx, z_emb) are **concatenated frame-by-frame** and projected with W_p ∈ R^{(2F+H)×D}; flow time t is a sinusoidal embedding prepended as an extra token. So every frame sees both its noisy state and its clean context value (zero where masked); the mask is implicit in x_ctx. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Loss: full loss L = E ||u_t(x_t|x) - v_t(x_t, x_ctx, z)||² "computes the loss on all frames, including those that are not masked"; masked version L_m = E ||m ⊙ (u_t - v_t)||² "is only computed on masked frames" and "leads to better results". — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Ablation (Table B2, 1K h subset, zero-shot TTS): FM masked loss WER 2.1 / SIM-r 0.597 / FSD 242.5 vs FM all-frame loss WER 2.0 / SIM-r 0.528 / FSD 243.1; regression (non-generative) masked 2.0 / 0.520 / 278.8. I.e. masked loss improves similarity to the context (+0.069 SIM-r), negligible effect on intelligibility. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Mask sampling: audio sequence fully masked with p_drop = 0.3, otherwise **one contiguous segment** of r% of the length is masked with r ~ U[70, 100] (duration model: p_drop 0.2, r ~ U[10,100]). Contiguous masking was chosen because "infilling of which is a more challenging task compared to infilling multiple smaller segments". — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Inference: midpoint solver, step 0.0625 (NFE = 32) default; good quality with < 10 NFE; FM-OT needs 8 NFE vs ≥16 for FM-diffusion path and >64 for score matching. — [Voicebox](https://arxiv.org/pdf/2306.15687)

**FrameFlow motif scaffolding (Yim et al., TMLR 2024, peer-reviewed) – direct amortization vs guidance comparison on a geometric task**
- Amortization: motif enters as a "1D mask as additional input with a 1 at the location of the motif and 0 elsewhere"; motif frames are held at their clean values (t = 1, data end) while only scaffold residues are noised; loss restricted to scaffold residues (conditional field u(T_t^S, t | T_1^S, T^M)). — [Yim et al. 2401.04082](https://arxiv.org/html/2401.04082)
- Training mask sampling: pseudo-motifs = random residue segments from unlabeled PDB structures, multiple non-contiguous segments allowed, total motif fraction between γ_min = 0.05 and γ_max = 0.5 of residues. — [Yim et al.](https://arxiv.org/html/2401.04082)
- Guidance variant: reconstruction guidance on the unconditional model with weight ω_t² = (1-t)²/(t² + (1-t)²) (same form as OT-ODE's r_t², see §2). — [Yim et al.](https://arxiv.org/html/2401.04082)
- Results on 24-motif benchmark: amortization 20/24 solved, 353 unique designable clusters; guidance 20/24, 192 clusters; RFdiffusion 20/24, 141 clusters; inference 18 s vs 50 s (RFdiffusion). — [Yim et al.](https://arxiv.org/html/2401.04082)

**DiffLinker (Igashov et al.; NeurIPS 2022 workshop version, later journal version not verified here) – point-cloud analogue**
- Fragment (context) atoms "remain unchanged throughout the diffusion and denoising processes"; only linker atoms are noised; EGNN runs on the fully connected graph of linker + context atoms but displacement outputs for context nodes are discarded; loss only on linker atoms; a "fragment flag" feature marks known atoms. — [DiffLinker ar5iv 2210.05274](https://ar5iv.labs.arxiv.org/html/2210.05274)
- Equivariance with fixed part: data centred at the context's centre of mass (anchor or full context) before sampling noise N(0, I). — [DiffLinker](https://ar5iv.labs.arxiv.org/html/2210.05274)

**Stable Diffusion inpainting checkpoint (industry model card, not peer-reviewed)**
- UNet gets 5 extra input channels (4 = VAE latent of masked image, 1 = mask), **zero-initialised**, fine-tuned 440k steps from the base model; synthetic masks, and "in 25% mask everything"; 10% text-conditioning dropout. — [SD-1.5-inpainting model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/raw/main/README.md)

**Per-token time / noise level as the mask (TD-Paint, Diffusion Forcing)**
- TD-Paint replaces scalar t with a per-pixel tensor τ ∈ {0..T}^{h×w}; known pixels get τ = 0 (clean), unknown pixels τ = t. Loss L = ||ε - ε_θ(x_τ, τ)||². Training masks: random patch size from {2^i ≤ w}, random known fraction in [0,1], at least one unknown patch. Inference: x_τ = x_t^unknown ⊙ (1-m) + x_0^known ⊙ m (clean known values, no forward re-noising). ~50 steps single pass vs ~5,000 for RePaint-20; LPIPS ~20% better than RePaint-20 on CelebA-HQ wide masks, ~30% on narrow; ~6x speedup over pixel-space baselines. Status: arXiv preprint (v2 June 2025); venue not confirmed in fetched page. — [TD-Paint 2410.09306](https://arxiv.org/html/2410.09306)
- TD-Paint's motivation: standard diffusion inpainting models "make minimal use of the input condition during the initial sampling steps". — [TD-Paint abs](https://arxiv.org/abs/2410.09306)
- Diffusion Forcing (Chen et al., NeurIPS 2024): independent per-token noise level k_t ~ U{0..K}, loss Σ_t ||ε_t - ε_θ(z_{t-1}, x_t^{k_t}, k_t)||²; "noising tokens is a form of partial masking—zero noise means a token is unmasked"; known tokens represented by k = 0; conditioning on slightly noisy (k ≪ K) instead of clean tokens stabilises long rollouts. — [Diffusion Forcing ar5iv](https://ar5iv.labs.arxiv.org/html/2407.01392); [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf)

**Shortcut risk in masked CFM**
- A 2026 preprint on spatial transcriptomics reports that conditional FM with masking "can suffer from a shortcut issue where networks copy near-target values rather than learning dynamics", suppressed by coupling the masking ratio to the timestep (heavy masking at large t). Only search-snippet level evidence; preprint. — [Correlation-Guided FM with Annealed Masking 2609.22187](https://arxiv.org/pdf/2609.22187)

### Inferences
- Two equivalent ways to feed known tokens, both used successfully: (a) **context channel** (Voicebox, SD-inpainting: [x_t, x_known⊙(1-m), m] concatenated per token) or (b) **in-place clean value + per-token time t_i = 1 (FM) / noise level 0 (diffusion)** (FrameFlow, DiffLinker, TD-Paint, Diffusion Forcing). For a set transformer over point coordinates / control-point offsets, (b) with a per-token time embedding plus a learned "known" flag embedding is the natural choice; (a) is the lowest-risk fine-tuning path for an existing unconditional model (zero-init the extra input projection, as SD did).
- Always restrict the loss to unknown tokens (velocity on known tokens is undefined/zero when they are pinned). Overwrite known tokens with their exact values after every ODE step at inference so they stay bit-exact.
- Mask sampling should match the test-time mask structure: Voicebox uses large contiguous spans (70–100%), FrameFlow 5–50% in several segments, SD includes 25% "everything masked" to retain unconditional ability. For geometry, sample masks that mimic real known/unknown patterns (e.g. whole curves/blocks) plus a fraction of fully-unconditional and random-subset masks.
- If the known part is in-place at t = 1 while others are at shared t, the network sees mixed times; a per-token time embedding (TD-Paint/Diffusion Forcing) makes this explicit rather than relying on the flag.

### Gaps
- MaskDiT (Zheng et al.) uses masking for training efficiency, not conditioning; not researched further. ACE / "arbitrary conditional generation" specific papers for FM not found in this pass.
- No source found that ablates "known tokens via context channel" vs "known tokens in place with t=1" head-to-head.
- Voicebox does not report how strongly results depend on the mask-ratio distribution.

## 2. Inference-only methods (pretrained unconditional prior)

### Takeaway
Training-free methods range from cheap replacement (RePaint-style, OT-ODE, PnP-Flow, FlowChef: 1–5 s/image, ~1x model cost per step) to expensive optimisation through the ODE (D-Flow: ~30–80 s, 6 GB). They work, but they are consistently weaker on large masks / boundary consistency than amortized models (TD-Paint vs RePaint; FrameFlow amortization vs guidance), produce artefacts or over-smoothing, and need hyper-parameter tuning (step sizes, start time, guidance scale).

### Cited Findings

**RePaint (Lugmayr et al., CVPR 2022, peer-reviewed)**
- Uses a pretrained unconditional DDPM; known region at step t-1 is obtained by forward-noising the known image, unknown region by the reverse step: x_{t-1} = m_known ⊙ x_{t-1}^known + (1 - m_known) ⊙ x_{t-1}^unknown. Plain replacement "doesn't harmonize the inpainted region well"; fix = resampling: diffuse back to x_t via the forward process with jump length j, repeat. — [RePaint CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf); [summary review](https://liner.com/ko/review/repaint-inpainting-using-denoising-diffusion-probabilistic-models)
- Cost: e.g. 3 resample iterations with jump length 10 gives 961 steps vs 200 plain steps. — [review](https://liner.com/ko/review/repaint-inpainting-using-denoising-diffusion-probabilistic-models); RePaint-20 needs ~5,000 steps per TD-Paint. — [TD-Paint](https://arxiv.org/html/2410.09306)
- Theoretical critique: RePaint "has a bias due to misalignment that hampers sample recovery even in a two-state diffusion process". — [A Theoretical Justification for Image Inpainting using DDPMs (2302.01217)](https://arxiv.org/abs/2302.01217v1)

**OT-ODE / "Training-free Linear Image Inverses via Flows" (Pokle et al., CMU/Meta; arXiv 2310.04432 v2 Mar 2024 – peer-review venue not confirmed in fetched page)**
- Initialise at t_0 = 0.2: x_{t0} = t_0 y + (1 - t_0) ε; integrate to t = 1. Convert denoiser to velocity v̂ = (x̂_1 - x_t)/(1 - t); ΠGDM-style correction v̂ ← v̂ + ((1-t)/t) γ_t g with g = (y - A x̂_1)^T (r_t² A A^T + σ_y² I)^{-1} A ∂x̂_1/∂x_t; for OT path γ_t = 1, r_t² = (1-t)²/(t² + (1-t)²) (no tuning). Requires a vector-Jacobian product (backprop through network, not through ODE). — [OT-ODE html](https://arxiv.org/html/2310.04432v2)
- NFE 70–100 (ΠGDM 100, RED-Diff 1000). Centred-mask inpainting, σ_y = 0.05: ImageNet-128 OT-ODE FID 6.39 / LPIPS 0.120 / PSNR 35.04 vs ΠGDM 9.75 / 0.130 / 34.45 vs RED-Diff 14.63 / 0.171 / 32.42; AFHQ-256 OT-ODE FID 9.93 vs ΠGDM 57.46. — [OT-ODE](https://arxiv.org/html/2310.04432v2)
- Failure mode: "OT-ODE and VP-ODE inpainting occasionally produces artifacts in the inpainted region as the resolution of image increases", more pronounced for noiseless cases. — [OT-ODE](https://arxiv.org/html/2310.04432v2)

**PnP-Flow (Martin, Gagneux, Hagemann, Steidl; ICLR 2025, peer-reviewed)**
- Per iteration: (1) gradient step z_n = x_n - γ_n ∇F(x_n) on data fidelity; (2) reprojection onto the FM path z̃_n = (1 - t_n) ε + t_n z_n; (3) denoise x_{n+1} = D_{t_n}(z̃_n) with D_t = Id + (1 - t) v_θ(·, t). t_n = n/N (N = 100), γ_n = (1 - t_n)^α, α ∈ (0,1]. No backprop through the ODE, no trace computation. — [PnP-Flow html](https://arxiv.org/html/2410.02423v3); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30965)
- Cost (CelebA 128, deblurring): OT-ODE 1.50 s / 0.65 GB, Flow-Priors 16.01 s / 2.96 GB, D-Flow 32.19 s / 5.91 GB, PnP-Flow 3.40 s / 0.10 GB. — [PnP-Flow](https://arxiv.org/html/2410.02423v3)
- CelebA 128 box inpainting (40×40) PSNR/SSIM: OT-ODE 28.84/0.914, D-Flow 29.70/0.893, Flow-Priors 29.40/0.858, PnP-Flow 30.59/0.943. Random inpainting (70% missing): PnP-Diff 31.43/0.917, OT-ODE 28.36/0.865, D-Flow 33.07/0.938, Flow-Priors 32.33/0.945, PnP-Flow 33.54/0.953. — [PnP-Flow](https://arxiv.org/html/2410.02423v3)
- Limitation: "reconstructions seem to be more on the smooth side" (MMSE denoiser). — [PnP-Flow](https://arxiv.org/html/2410.02423v3)
- PnP methods in general "face inherent limitations on more generative tasks like inpainting" (the motivation for PnP-Flow). — [ICLR poster](https://iclr.cc/virtual/2025/poster/30965)

**D-Flow (Ben-Hamu, …, Lipman; ICML 2024, peer-reviewed)**
- Optimises the source noise x_0 by differentiating the loss through the whole ODE solve; for Gaussian-path FM/diffusion this projects the gradient onto the data manifold, implicitly injecting the prior. Applied to image/audio inpainting and conditional molecule generation. — [D-Flow PMLR](https://proceedings.mlr.press/v235/ben-hamu24a.html); [arXiv](https://arxiv.org/html/2402.14017v2)
- Cost: slowest of the family (32 s / 5.9 GB in PnP-Flow's benchmark; 80 s / 6.4 GB in FlowChef's). — [PnP-Flow](https://arxiv.org/html/2410.02423v3); [FlowChef](https://arxiv.org/html/2412.00100v1)

**FlowChef (Patel et al.; ICCV 2025, peer-reviewed)**
- Gradient-free w.r.t. network: x̂ = x_t + t u_θ(x_t, t) (one-step clean prediction; their time convention), step x_{t-Δt} = x_t + Δt u_θ(x_t, t) - s' ∇_{x̂} L, where for inpainting/editing L = ||(x̂ - x_ref) ⊙ M||²; relies on straight trajectories / slowly varying Jacobian. — [FlowChef html](https://arxiv.org/html/2412.00100v1); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Patel_FlowChef_Steering_of_Rectified_Flow_Models_for_Controlled_Generations_ICCV_2025_paper.html)
- Reported cost: D-Flow 80.42 s / 6.44 GB, OT-ODE 10.39 s / 0.70 GB, PnP-Flow 5.23 s / 0.40 GB, FlowChef 4.31 s / 0.43 GB. Box inpainting (pixel space) PSNR/SSIM/LPIPS: D-Flow 20.37/70.06/13.67, PnP-Flow 22.12/68.02/14.70, FlowChef 26.32/87.70/3.36. — [FlowChef](https://arxiv.org/html/2412.00100v1)
- Conflict: these numbers are much lower for PnP-Flow/D-Flow than PnP-Flow's own (30.59 / 29.70 PSNR box inpainting), i.e. benchmark settings differ; cross-paper numbers should not be compared directly. — [PnP-Flow](https://arxiv.org/html/2410.02423v3) vs [FlowChef](https://arxiv.org/html/2412.00100v1)
- Limitations stated: assumes locally linear vector field; degradation on Flux due to trajectory nonlinearity/crossovers; sensitive to guidance scale s' and NFE. — [FlowChef](https://arxiv.org/html/2412.00100v1)

**DPS (Chung et al., ICLR 2023)**
- Step size ζ_t must be tuned because the likelihood gradient grows with the number of observations, causing instability; DPS is computationally expensive (gradient through network at every step). — [Online Posterior Sampling with a Diffusion Prior 2410.03919](https://arxiv.org/pdf/2410.03919); [search summary](https://arxiv.org/html/2506.13614v1)

### Inferences
- For FM with exact (noise-free) known tokens, the cheapest training-free baseline is FM-RePaint: at each step set x_t[known] = t·x_known + (1-t)·ε_fixed (or the OT-ODE init with t_0 ≈ 0.2), optionally with back-and-forth resampling. Expect boundary inconsistency (known tokens never influenced the trajectory at early t) – the same reason TD-Paint and the amortized FrameFlow beat their training-free counterparts.
- Guidance-style methods (OT-ODE, DPS, FlowChef) only softly enforce the known part; for hard geometric constraints one must still overwrite known tokens at the end, which can create a seam if the generated neighbours were not consistent.
- Given an amortized model is affordable to train (the same data, random masks), the evidence favours training-time masking for geometric accuracy and diversity; inference-only methods are mainly useful as baselines or when retraining is impossible.

### Gaps
- No quantitative study found of boundary-seam error specifically (most metrics are global PSNR/LPIPS/FID).
- Flow-Priors (Zhang et al. 2024) and DDNM/CoPaint not fetched in detail.
- No point-cloud/set-token benchmark comparing these inference-time methods was found.

## 3. Classifier-free guidance combined with masked conditioning

### Takeaway
CFG is applied by dropping the whole condition (known context + other conditioning) with probability ~0.1–0.2 during training and extrapolating at inference: ṽ = (1+α) v(x_t, ctx) - α v(x_t). Voicebox shows guidance strength trades similarity-to-context and stability against diversity, and its effect depends on NFE; no source quantifies its effect on geometric accuracy directly.

### Cited Findings
- Voicebox CFG: conditioner c = (z, x_ctx) dropped with p_uncond = 0.2 during training; inference ṽ_t(w, x_mis, z) = (1 + α) v_t(w, x_ctx, z) - α v_t(w), α "selected based on empirical results"; α = 0.7 used for main zero-shot TTS results with NFE 32. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Note the two dropout mechanisms in Voicebox: p_drop = 0.3 masks the whole audio (target = everything unknown) vs p_uncond = 0.2 drops the conditioner for CFG. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- Effect of α: WER stable (mean 2.0, variance 0.005) across α/NFE; for speaker similarity, low α (0 or 0.3) is better at NFE ≤ 4, but from NFE = 8 a higher α improves similarity; lower α gives lower FSD / more diverse samples; for α = 0, WER rises slightly 2.8 → 3.1 as NFE 2 → 32, for larger α WER stays more stable. — [Voicebox](https://arxiv.org/pdf/2306.15687)
- SD inpainting: CFG over text with 10% text dropout; masked-image channels kept in both branches; 25% fully masked samples keep the model able to generate unconditionally. — [SD inpainting model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/raw/main/README.md)

### Inferences
- For geometry: the "unconditional" branch could be either (i) everything masked (equivalent to Voicebox's p_drop path) or (ii) known tokens present but other conditioning dropped. Option (i) pushes generated tokens toward closer agreement with the known ones (higher fidelity to context) – analogous to Voicebox's speaker-similarity gain – at the cost of diversity; α should be tuned jointly with NFE, as Voicebox shows the sign of the effect can flip at very low NFE.
- High α extrapolates velocities and can overshoot for low-dimensional coordinate targets; evaluate geometric error (e.g. Chamfer/corner error) over an α sweep rather than assuming α > 0 helps.

### Gaps
- No source found measuring CFG strength vs geometric/positional accuracy (point coordinates, CAD, meshes) under masked conditioning.
- No source found comparing "drop known context" vs "drop only global condition" as the unconditional branch.

## 4. OT coupling when part of the sample is fixed

### Takeaway
Plain minibatch-OT coupling ignores conditions, so in conditional training it produces a condition-dependent (skewed) prior that is unavailable at test time and can hurt; C²OT fixes this by adding a condition-mismatch penalty to the cost. For pinned known tokens the analogous, simple fix is to couple only the free tokens within samples (or only across samples with the same known part).

### Cited Findings
- Minibatch OT straightens paths in unconditional FM, but "in the conditional setting, minibatch optimal transport falls short because the default optimal transport mapping disregards conditions, resulting in a conditionally skewed prior distribution during training"; at test time samples come from the unskewed prior → train/test gap. — [C²OT, Cheng & Schwing, ICCV 2025](https://arxiv.org/pdf/2503.10636); [ICCV page](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_The_Curse_of_Conditions_Analyzing_and_Improving_Optimal_Transport_for_Conditional_Flow-Based_Generation_ICCV_2025_paper.html)
- C²OT objective: min_P ||X_0 - P X_1||² + Σ_i f(C_i, [P C]_i); discrete: f = ∞ if classes differ, else 0; continuous: f_c = w (1 - cos(c_1, c_2)), w set adaptively via target ratio r_tar ≈ 0.01. — [C²OT html](https://arxiv.org/html/2503.10636v3)
- FID results: CIFAR-10 Euler-5: FM 21.97 / OT 18.19 / C²OT 17.57; Euler-25: 5.64 / 6.82 / 5.64; adaptive: 2.92 / 4.14 / 2.90. ImageNet-32 (captions) Euler-5: 23.02 / 20.76 / 21.97; Euler-25: 7.17 / 7.85 / 7.07; adaptive: 5.36 / 6.22 / 5.35. I.e. naive OT helps only at very few steps and hurts at higher NFE; C²OT ≈ FM at high NFE and better than FM at low NFE. — [C²OT html](https://arxiv.org/html/2503.10636v3)
- DiffLinker handles a fixed context in an equivariant point-cloud model by centring on the context CoM before sampling noise for the free atoms. — [DiffLinker](https://ar5iv.labs.arxiv.org/html/2210.05274)

### Inferences
- With known tokens pinned in-sample, OT/assignment should only involve free tokens: (a) within-sample permutation OT between noise points and free target tokens (for set data, matching noise to targets inside one sample is condition-safe because the condition is identical), and (b) any cross-sample minibatch OT must penalise differing known parts (C²OT-style with f based on a distance between known parts), otherwise the prior becomes condition-skewed.
- Given C²OT's numbers, the gain from OT coupling is mainly at low NFE (≤5 Euler steps); at 25+ steps plain independent coupling is as good.
- Centre/normalise the free tokens relative to the known part (CoM of known tokens, as DiffLinker) so the noise prior is defined in a frame consistent with the context.

### Gaps
- TorchCFM (Tong et al.) conditional variants and "coupling only the free part" were not found as a published, evaluated method; the within-sample suggestion above is an inference, not a cited result.
- No source on OT coupling specifically for partially observed samples (inpainting) was found.
