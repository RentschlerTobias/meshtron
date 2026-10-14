# Constraint handling in flow matching / diffusion: points on given curves and surfaces (manifold constraints, intrinsic coordinates)

Scope: making generated 3D points satisfy equality constraints "point i lies on curve C_i or surface S_i" (a different curve or surface for each point, possibly a mix of 1D, 2D and free 3D tokens). Four families compared: (a) projection during sampling, (b) Riemannian / manifold FM, (c) reparametrisation into intrinsic coordinates (t, (u,v)), (d) penalty / guidance. Venue status is marked as [peer-reviewed] or [preprint].

## Q1: PCFM and follow-ups (CCFM, SNAP-FM, constraint-aware training): how projection is done, how often, effect on the distribution

### Takeaway
PCFM [NeurIPS 2025] gives exact equality satisfaction without retraining. At every step it extrapolates to the predicted clean sample u1, applies one Gauss-Newton projection u_proj = u1 - J^T (J J^T)^{-1} h(u1), maps that back to the current time, and applies a relaxed penalty correction. A final hard projection follows if the residual is still large. Overhead is reported at 1-3 % of sampling time when m = dim h is much smaller than n. Follow-ups target (i) distribution distortion from projecting noisy intermediates (CCFM: project in clean-sample space), (ii) the cost of the nonlinear projection (SNAP-FM: sparse interior-point solves on GPU), and (iii) training-time penalties instead of inference-time projection (Constraint-aware FM, TMLR 2026).

### Cited Findings
- PCFM is a zero-shot inference method: it "continuously guides the sampling process through physics-based corrections applied to intermediate solution states" and ensures "exact constraint satisfaction at the final solution". Accepted to NeurIPS 2025 [peer-reviewed]. — [arXiv 2506.04171](https://arxiv.org/abs/2506.04171)
- PCFM projection formula: `u_proj = u1 − J^T (J J^T)^{-1} h(u1)`, which is "a single Gauss–Newton projection to softly align u1 to the constraint manifold". Here h is the constraint residual and J its Jacobian. — [PCFM HTML](https://arxiv.org/html/2506.04171)
- PCFM relaxed correction at intermediate time τ': `u_τ' = argmin_u ||u − û_τ'||² + λ ||h(u + γ v_θ(u,τ'))||²` with γ = 1−τ'. It enforces the constraint at the extrapolated clean point while preserving local flow alignment. — [PCFM HTML](https://arxiv.org/html/2506.04171)
- PCFM step structure: integrate forward to τ=1 (extrapolate), project, then go back to the current time along the OT path with "a single Euler step". This repeats for N user-chosen steps, with a final projection "if constraint residuals remain high at τ=1". — [PCFM HTML](https://arxiv.org/html/2506.04171)
- PCFM accuracy: constraint errors of 0 (machine precision) for initial/boundary conditions and mass conservation. Baselines range from about 6e-4 to more than 1e-1. Solver cost is O(m³) per step with m = dim h ≪ n, and the overhead is "only 1–3% of total time". — [PCFM HTML](https://arxiv.org/html/2506.04171)
- PCFM limitations: equality constraints only (inequalities left as future work), needs a pretrained model, and nonlinear constraints need constraint Jacobians. — [PCFM HTML](https://arxiv.org/html/2506.04171)
- CCFM (chance-constrained FM) is training-free and puts stochastic optimisation into sampling. It works on intermediate states yet is "theoretically equivalent to projecting onto the feasible set defined by clean samples". It argues that repeatedly projecting noisy intermediates "can distort the learned distribution", and reports higher feasibility and fidelity than SOTA constrained generative models on PDEs and molecular docking (no numbers in the abstract). arXiv preprint 2509.25157, Sept 2025 [preprint, no venue found]. — [arXiv 2509.25157](https://arxiv.org/abs/2509.25157)
- SNAP-FM builds on PCFM and targets "repeated nonlinear projection during sampling" as the main bottleneck. It compiles the structured projection NLP symbolically with ExaModels.jl, solves it with the MadNLP.jl interior-point solver, and uses GPU sparse factorisation of the KKT systems. It keeps hard constraint satisfaction on PDE benchmarks with linear, nonlinear, 1D and 2D constraints. It has an ICML 2026 virtual page, so it is likely peer-reviewed; v2 is dated 3 Sept 2026. — [arXiv 2607.00095](https://arxiv.org/abs/2607.00095); [ICML 2026 page](https://icml.cc/virtual/2026/72648)
- Constraint-Aware Flow Matching via Randomized Exploration (Huan et al.): for differentiable constraints it adds a training penalty on the distance between generated samples and the constraint set. For oracle-only constraints (membership queries) it uses randomisation to learn a mean flow with high probability of satisfaction. A two-stage variant is cheaper than one-stage. Published in TMLR 2026, with an ICLR 2026 listing [peer-reviewed]. — [TMLR 2026](https://mlanthology.org/tmlr/2026/huan2026tmlr-constraintaware/); [ICLR 2026 page](https://iclr.cc/virtual/2026/10020411)
- Projected Diffusion Models (PDM, Christopher, Baek, Fioretto) iteratively project the diffusion sampling process onto arbitrary constraint sets, both convex and non-convex, recasting sampling as constrained optimisation. NeurIPS 2024 [peer-reviewed]. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a28af221f2f70be183afc16797a56b91-Abstract.html)
- Reflected Diffusion Models (Lou & Ermon) note that thresholding (projecting to the domain after each step) "leads to a mismatch between training and generative processes". They show such schemes are discretisations of reflected SDEs. — [ICML 2023 / PMLR](https://proceedings.mlr.press/v202/lou23a.html)

### Inferences
- For "point i lies on curve/surface X_i", h is block-diagonal: each point has 1 (surface) or 2 (curve) scalar residuals. The PCFM Gauss-Newton step therefore splits into independent per-point closest-point problems. For a triangulated surface the exact Euclidean projection (closest point on mesh, e.g. BVH or AABB query) replaces the GN step and costs O(log F) per point. The O(m³) dense cost in PCFM does not arise.
- With the PCFM-style "extrapolate to x̂1, project, re-noise" loop, the network only ever sees off-manifold inputs drawn from its own training distribution. This avoids the train/test mismatch that Lou & Ermon point out for naive per-step thresholding. Projecting only the clean-sample estimate (CCFM idea) is the less distorting option.
- A projection-based method is exact by construction at the end, but it does not repair a wrong discrete choice (wrong curve or face). It only snaps the point to the nearest location on the given target.

### Gaps
- SNAP-FM speedup numbers and CCFM quantitative results were not on the landing pages, and I did not fetch the full PDFs.
- I found no PCFM-family results for many-point geometric snapping (closest-point onto meshes). All benchmarks are PDE fields.

## Q2: Riemannian FM, general geometries, manifolds with boundaries, mixed-dimensional / product manifolds, discrete choice of manifold

### Takeaway
Riemannian FM (Chen & Lipman) defines conditional fields through a premetric. On meshes it uses spectral (biharmonic) distances with about 200 Laplace-Beltrami eigenfunctions, and handles boundaries through Neumann eigenfunctions. Products of manifolds are standard (FlowMM: torus × Euclidean × discrete). I found no paper on "mixed-dimensional" sets where each token picks its own curve or surface. The closest tools are product manifolds (fixed assignment), discrete/continuous multimodal flows (Campbell et al. 2024: discrete label + continuous state), and trans-dimensional jump diffusion.

### Cited Findings
- Conditional vector field in RFM: `u_t(x|x1) = (d log κ(t)/dt) · d(x,x1) · ∇d(x,x1) / ||∇d(x,x1)||²_g`, with κ(t) = 1 − t giving a linear decrease in distance. On simple geometries (sphere, torus, hyperbolic) it uses exp/log maps and is simulation-free. On general geometries (meshes, boundaries) the conditional ODE must be integrated, but without backprop through the solver. — [arXiv 2302.03660](https://arxiv.org/html/2302.03660)
- Spectral distances on meshes: `d_w(x,y)² = Σ_i w(λ_i)(φ_i(x) − φ_i(y))²`, with biharmonic w(λ) = λ⁻² and diffusion w(λ) = exp(−2τλ). Eigenfunctions are computed once and each query costs O(k), versus O(n log n) for fast geodesic approximations. Biharmonic beats diffusion distance, and k = 200 eigenfunctions suffice on the Stanford Bunny and Spot the Cow. Manifolds with boundary (mazes) use Neumann boundary conditions so the conditional field does not cross the boundary. — [arXiv 2302.03660](https://arxiv.org/html/2302.03660)
- Venue: the fetched summary said "ICLR 2023", but the paper is commonly cited as ICLR 2024 (v3 is Feb 2024). Treat it as peer-reviewed at ICLR 2024; the fetched page's wording is inconsistent. — [arXiv 2302.03660](https://arxiv.org/abs/2302.03660)
- FlowMM extends RFM to a product of periodic fractional coordinates (torus), lattice parameters and binary atom types. It uses "mean-free" RFM for translation symmetry and reports about 3x fewer integration steps to find stable materials. ICML 2024 [peer-reviewed]. — [FlowMM](https://arxiv.org/html/2406.04713v1)
- Discrete Flow Models (Campbell, Yim, Barzilay, Rainforth, Jaakkola) give a CTMC analogue of flow matching. They allow joint multimodal flows over discrete and continuous data, used for protein structure + sequence co-design. ICML 2024 [peer-reviewed]. — [PMLR v235](https://proceedings.mlr.press/v235/campbell24a.html)
- Trans-dimensional jump diffusion (Campbell, De Bortoli, Harvey, Rainforth, Weilbach, Doucet) jointly models state and dimension. A dimension-destroying forward process has a learned dimension-creating reverse process with an ELBO objective. NeurIPS 2023 spotlight [peer-reviewed]. — [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70230)
- Generic search for "stratified space / mixed-dimension manifold flow matching" turned up only theory on FM adapting to a single manifold's intrinsic dimension (arXiv 2602.22486) and on discontinuities of optimal velocity fields under topology mismatch (arXiv 2512.12821). — [arXiv 2602.22486](https://arxiv.org/abs/2602.22486); [arXiv 2512.12821](https://arxiv.org/html/2512.12821v1)
- Background (well-known, not re-fetched): Riemannian Score-Based Generative Models (De Bortoli et al., NeurIPS 2022) and Riemannian Diffusion Models (Huang et al., NeurIPS 2022) are the diffusion predecessors of RFM. — [arXiv 2202.02763](https://arxiv.org/abs/2202.02763); [arXiv 2208.07949](https://arxiv.org/abs/2208.07949)

### Inferences
- A per-token assignment "token i lives on X_i" with X_i known as conditioning makes the state space a fixed product M = Π_i X_i (curves, surfaces, R³). RFM on product manifolds factorises per token, so the "mixed dimension" is not a problem once the assignment is given. Each token just uses its own geodesic or projection rule. Within one batch this means heterogeneous per-token operators (curve: 1D param, surface: closest-point or UV, free: identity), implemented with masks.
- If the assignment (which curve or face) must itself be generated, the natural formulation is Campbell-style multimodal flow: a discrete CTMC over the label plus a continuous flow conditioned on the current label. Points then jump between manifolds as labels change, which is essentially what jump diffusion formalises for dimension changes. No paper found applies this to CAD curves or surfaces.
- Spectral-distance RFM on a mesh needs a Laplace-Beltrami eigen-decomposition per geometry. That is cheap with about 200 eigenpairs, but it must be redone for each new CAD geometry, which matters for a dataset of many geometries. A Euclidean-interpolation-plus-closest-point-projection scheme avoids this.

### Gaps
- I found no published work on flows over unions of manifolds of different dimension with a learned discrete choice of component for 3D point sets.
- I did not verify how RFM on meshes keeps ODE iterates exactly on triangles (projection after each step vs. intrinsic stepping); the fetched summary did not say.

## Q3: Flows / diffusion in intrinsic coordinates (UV, curve parameter); seams, periodicity, charts

### Takeaway
Intrinsic-coordinate generation is well established when the coordinate domain is a simple manifold (torus of torsion angles, fractional crystal coordinates). Periodic parameters are handled with wrapped distributions (wrapped normal, take mod 2π) or Riemannian FM on the torus. CAD generators (BrepGen, BrepDiff) sample faces on UV grids and edges along U as a representation, but they generate 3D point grids or latents, not points constrained to a given surface. Multi-chart atlases for generative models exist mainly as theory and early work.

### Cited Findings
- Torsional diffusion: the state is torsion angles on the hypertorus T^m. The perturbation kernel is a wrapped normal ("sample from the corresponding unwrapped isotropic normal and take elementwise mod 2π"), and its score is precomputed numerically. It needs 5-20 steps versus 5,000 for Euclidean GeoDiff, reduces average minimum recall RMSD by 30 % on GEOM-DRUGS, and costs about 1.76-4.90 core-seconds per conformer versus 305. NeurIPS 2022 [peer-reviewed]. — [arXiv 2206.01729](https://ar5iv.labs.arxiv.org/html/2206.01729)
- FlowMM treats periodic fractional coordinates with RFM on the flat torus, the standard chart-free way to handle a periodic parameter. — [FlowMM](https://arxiv.org/html/2406.04713v1)
- BrepGen samples face geometry on a uniform UV grid of the parametric surface ("fingerprint") and edges along U, then compresses these with a VAE into latents for diffusion. — [BrepGen review](https://www.themoonlight.io/en/review/brepgen-a-b-rep-generative-diffusion-model-with-structured-latent-geometry)
- BrepDiff uses a masked UV-grid representation with a diffusion transformer and an asynchronous, shifted noise schedule. — [BrepGen/BrepDiff search context](https://www.themoonlight.io/en/review/brepgen-a-b-rep-generative-diffusion-model-with-structured-latent-geometry)
- Further 2025-26 B-rep generators: BrepGPT (Voronoi half-patch, autoregressive) and TG-Diff (discrete topology diffusion coupled with topology-conditioned geometry diffusion). — [BrepGPT arXiv 2511.22171](https://arxiv.org/pdf/2511.22171); [TG-Diff arXiv 2607.21928](https://arxiv.org/pdf/2607.21928)
- Atlas flow imposes gluing compatibility on overlapping local charts so that local structures can be glued into a global generative flow. — [arXiv 2210.14149](https://arxiv.org/pdf/2210.14149)

### Inferences
- For a curve with known parametrisation, flowing t ∈ [0,1] (or S¹ for closed curves) is the cheapest exact method: 1 scalar per token, and the constraint holds by construction. However, the network's loss lives in t, so a non-arclength parametrisation distorts distances. Reparametrise by arc length.
- For a triangulated surface with no global UV, options are: (i) per-face barycentric coordinates plus a discrete face index (discrete + continuous, Campbell-style), which is exact but has non-smooth transitions across edges; (ii) a global UV atlas from a parametriser, which requires seam handling (periodic wrap for cylinders and annuli, chart transitions otherwise); (iii) ambient 3D flow plus closest-point projection, which is smooth in R³, needs no charts and works with any triangulation. Option (iii) generalises best across geometries.
- Periodic CAD parameters (closed curves, revolved or cylindrical faces) should be treated as S¹ factors with wrapped interpolation (geodesic on the torus), not as clamped intervals. Otherwise flows cross the seam the long way round.

### Gaps
- I found no paper doing flow matching of free point sets directly in the UV coordinates of given CAD/mesh surfaces with multiple charts and seam transitions.
- I did not access details of how BrepGen or BrepDiff handle periodic faces.

## Q4: Reflected / mirror diffusion and bounded domains

### Takeaway
For inequality-bounded domains (e.g. a curve parameter t in [0,1], or a face's barycentric simplex) the options are: reflected SDEs (principled, and they explain thresholding), log-barrier metrics, Metropolis rejection (simple, 7-10x faster training, O(d)), and mirror maps (convex sets, tractable). None of these handle equality constraints directly.

### Cited Findings
- Reflected Diffusion Models reverse a reflected SDE on the data support. They learn the score with a generalised score-matching loss, keep guidance, likelihood training and ODE sampling, and are more faithful under high CFG weight. ICML 2023 [peer-reviewed]. — [PMLR v202](https://proceedings.mlr.press/v202/lou23a.html)
- Diffusion Models for Constrained Domains (Fishman, Klarner, De Bortoli, Mathieu, Hutchinson) propose a log-barrier metric process and reflected Brownian motion for inequality-constrained manifolds, applied to robotics and protein design. TMLR 2023 [peer-reviewed]. — [arXiv 2304.05364](https://arxiv.org/abs/2304.05364)
- Metropolis Sampling for Constrained Diffusion Models: steps that would leave the set are rejected. It needs only a membership oracle, scales O(d) versus O(d²) for reflection, and trains about 7-10x faster with comparable or better results. It explicitly covers only inequality sets {f_i(x) < 0}, not equality constraints. NeurIPS 2023 [peer-reviewed]. — [NeurIPS 2023 text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2023/metropolis_sampling_for_constrained_diffusion_models__39fe78ed.txt)
- Mirror Diffusion Models (Liu, Chen, Theodorou, Tao) generate on convex constrained sets via a mirror map to an unconstrained dual space without losing tractability. NeurIPS 2023 [peer-reviewed]. — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/85f5c7372625d1e0df0e3996f85062d6-Abstract.html)

### Inferences
- In an intrinsic-coordinate design, open curve parameters t ∈ [0,1] and barycentric coordinates are bounded convex sets. A mirror map (e.g. logit for t, softmax for barycentrics) or simple reflection at the ends fits there. Closed curves need the periodic (torus) treatment instead.

### Gaps
- No quantitative comparison found of reflection vs. mirror vs. clipping for flow matching (rather than diffusion) on bounded intervals.

## Q5: Practical trade-offs (exactness, smoothness, compute, failure modes)

### Takeaway
For points on known curves and surfaces of a triangulated geometry, the most transferable recipe is: an ambient-space Euclidean flow, conditioning on the target curve or surface per token, PCFM/CCFM-style projection of the predicted clean sample x̂1 onto that target at each step, and a hard final projection. Intrinsic coordinates are best for curves with good parametrisations and periodic parameters. Full RFM is principled but needs per-geometry spectral precomputation. Penalties alone do not give exactness.

### Cited Findings
- Projection-type methods give exact final satisfaction: PCFM reaches machine-precision residuals with 1-3 % overhead. — [PCFM HTML](https://arxiv.org/html/2506.04171)
- Naive per-step projection (thresholding) creates a train/generation mismatch. — [Lou & Ermon](https://proceedings.mlr.press/v202/lou23a.html)
- Projecting noisy intermediates distorts the distribution, and projecting in clean-sample space mitigates this. — [CCFM](https://arxiv.org/abs/2509.25157)
- Penalty-based training (distance to constraint set) improves satisfaction while matching the target distribution, but gives no hard guarantee. — [Constraint-aware FM, TMLR 2026](https://mlanthology.org/tmlr/2026/huan2026tmlr-constraintaware/)
- Intrinsic coordinates on a simple manifold can cut steps by orders of magnitude (torsional diffusion: 5-20 vs 5,000 steps). — [Torsional diffusion](https://ar5iv.labs.arxiv.org/html/2206.01729)
- Spectral RFM on meshes: one eigen-decomposition per mesh, then O(k) per distance query with k ≈ 200. — [Chen & Lipman](https://arxiv.org/html/2302.03660)
- A topology mismatch between prior and target can make optimal FM velocity fields spatially discontinuous. — [arXiv 2512.12821](https://arxiv.org/html/2512.12821v1)

### Inferences
- Comparison for this use case (3D points, each tied to a known curve, surface or free):
  - (a) Projection during sampling: exact; the field stays smooth (the network is unchanged); compute is about one closest-point query per token per step. Failure mode: snapping to the wrong branch near high curvature or close sheets of the same surface, and distribution distortion if noisy states are projected.
  - (b) RFM: exact if integrated intrinsically. Each geometry needs its own precomputation (eigenfunctions or geodesics), and the conditional field is non-smooth at cut loci. Heterogeneous per-token manifolds are fine as a product.
  - (c) Intrinsic coordinates: exact by construction and lowest dimension. Seams, periodicity, non-arclength distortion and chart switches are the failure modes, and the network loses 3D context unless 3D positions are fed back as features.
  - (d) Penalty/guidance: cheap and flexible, but only approximate. The guidance weight trades fidelity against satisfaction, so it is best combined with a final hard projection.
- A combined approach seems promising: train in ambient 3D with per-token target conditioning (and optionally a distance-to-target penalty), then sample with x̂1-projection per step (PCFM/CCFM-like) and finish with a hard projection. This needs no charts, generalises to new triangulated geometries, and its exactness rests on a cheap closest-point query.
- Points must also respect topology (curve endpoints, shared vertices of curve and surface). Tokens on a vertex can be treated as 0-dimensional constraints, a fixed point, which the per-token framework covers trivially.

### Gaps
- No head-to-head benchmark found comparing (a)-(d) on the same geometric point-on-surface task.
- No reported numbers on how often projection-based snapping picks the wrong surface sheet. This would need in-house experiments.
