# Learning-based hex / block-structured mesh generation and ML/RL mesh-quality optimisation (2018–2026)

Status legend: **[PR]** = peer-reviewed venue confirmed in this session; **[PP]** = arXiv preprint only, or venue not confirmed in this session. Research date: 2026-10-12. About 20 search and fetch calls, so coverage is broad but not exhaustive.

## Which works generate block-structured or hex meshes with neural networks, and how do they represent topology and geometry (tokens, point sets, curves)?

### Takeaway
No published neural method directly generates a *block-structured hex topology plus curved block edges* the way the planned "Polytron" pipeline would. I found no work called "Polytron". Learning-based hex work splits into five families:
- (a) RL/GNN agents that choose **cut operations** on CAD (block decomposition).
- (b) **Polycube** predictors (classification, diffusion on point clouds), followed by classical polycube-to-hex mapping.
- (c) **Neural frame and cross fields** (self-supervised implicit nets), followed by classical field-guided meshing.
- (d) **Unsupervised neural mappings** for structured meshes (PDE-residual losses, TFI/elliptic-like).
- (e) **Autoregressive token generators** for surface tri/quad meshes (MeshGPT lineage: Meshtron, QuadGPT, DeepMesh, Mesh-RFT, Mesh-Pro). None of these is volumetric.

### Cited Findings
**Block decomposition by RL / GNN (topology = sequence of cut actions on a B-rep)**
- DiPrete et al., "Reinforcement learning for block decomposition of planar CAD models", Engineering with Computers 40:2807–2817 (2024) **[PR]**. A SAC agent learns location and direction of cuts on the CAD model and "quickly learns an effective strategy"; it combines imitation learning with RL. Preprint arXiv:2302.11066. — [Springer PDF](https://link.springer.com/content/pdf/10.1007/s00366-023-01940-6.pdf); [arXiv](https://arxiv.org/abs/2302.11066); [OSTI](https://www.osti.gov/pages/biblio/2305429)
- The Owen et al. survey reports for DiPrete: "100% quadrilateral decompositions" on axis-aligned datasets, and mostly-quad results on arbitrary polygons. — [Survey arXiv:2512.23719 (html)](https://arxiv.org/html/2512.23719)
- "Reinforcement learning based automatic block decomposition of solid models for hexahedral meshing", Computer-Aided Design (2025), S0010448525000120 **[PR]**. 3D SAC with sequential decomposition into sweepable sub-volumes. I could not open the full text (HTTP 403), and authors were not retrieved. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448525000120)
- The survey (Owen, Brown, Chrisochoides, Garimella, Gu, Ledoux, Lei, Quadros, Ray, Winovich, Zhang; Dec 2025, rev. Jan 2026) **[PP]** lists the following:
  - Patel et al. 2021, "Auto-Hex": an RL webcut policy conditioned on T-junctions, sweep directions and the chordal-axis skeleton.
  - Zhang et al. 2025a: SAC for 3D sweepable decomposition. Reward shaping and mesh quality are "open".
  - Zhang et al. 2025b: graph-based RL for block-decomposition scripting, with ">95% hex-meshable volume ratios".
  - Quadros et al. 2025/26: a supervised heterogeneous GNN on a chordal-axis bipartite graph that predicts WebCut operations. Accuracy is 98.72% per action on synthetic CAD and 85% on out-of-distribution industrial models.
  - Narayanan et al. 2024: actor-only PPO / self-play that turns unstructured quad meshes into block-structured ones.
  - Bourmaud et al. 2025: MCTS for quad decomposition.

  — [Survey html](https://arxiv.org/html/2512.23719); [abstract](https://arxiv.org/abs/2512.23719)
- Dmsh (Kalyan, Anitescu, Zhuang, Rabczuk, Goswami, Natarajan; arXiv:2606.10601, June 2026) **[PP]**. Three cooperating agents handle topology simplification, geometric regularisation and mesh generation. It uses parametric SAC with decoupled critics over a hybrid discrete-continuous action space, plus curriculum learning and recursive decomposition. It produces conforming all-quad meshes "without post-processing". The quantitative claims are only qualitative in the abstract. — [arXiv](https://arxiv.org/abs/2606.10601)

**Polycube-based hex (topology = polycube labelling / polycube point cloud; geometry = classical mapping)**
- DL-Polycube (Yu et al.), Engineering with Computers 2025 **[PR]**. A DNN classifies the surface triangle mesh into a polycube structure. Unsupervised segmentation follows, then octree subdivision and mapping, giving an all-hex mesh and an IGA volumetric spline. — [Springer](https://link.springer.com/article/10.1007/s00366-025-02161-9); [survey](https://arxiv.org/html/2512.23719)
- DDPM-Polycube (Yu et al. 2025a) generates the polycube layout with a DDPM. Scalable DDPM-Polycube is arXiv:2604.17266 **[PP]**. — [arXiv 2604.17266](https://arxiv.org/pdf/2604.17266); [survey](https://arxiv.org/html/2512.23719)
- PolycubeNet (He, …, Na Lei; arXiv:2605.20274, May 2026) **[PP]**. A dual-latent conditional diffusion model maps the input point cloud to a polycube point cloud. Self-attention runs only in a fixed-size latent, so cost does not grow quadratically with the number of points. Rigid and non-rigid registration follow, then polycube-to-hex conversion. It handles "arbitrary genus" in seconds and releases a paired CAD/polycube dataset. — [arXiv](https://arxiv.org/abs/2605.20274)
- Also in the survey: DeepShape (CNN voxel-polycube correspondence, 2025), Evocube (evolutionary labelling, 2022), and Neural Integral Parameterization (Zhan et al. 2025; an NN inverse map that targets bijectivity and distortion). — [survey](https://arxiv.org/html/2512.23719)

**Neural frame and cross fields (topology emerges from field singularities)**
- NeurCross (arXiv:2405.13745) **[PP; venue not verified]**. A self-supervised network has two heads, one for the SDF and one for the cross field. The loss covers surface approximation, alignment with principal directions and smoothness. It beats the state of the art on ShapeNet and Thingi10K in area and angle distortion, Chamfer distance and Jacobian ratio. — [arXiv](https://arxiv.org/html/2405.13745v2)
- CrossGen (arXiv:2506.07020) learns and generates cross fields for quad meshing **[PP]**. Only the title was confirmed. — [arXiv](https://arxiv.org/pdf/2506.07020)
- NeurFrame (Yu, Huang, Chen, Cao; arXiv:2603.12820, Mar 2026) **[PP]**. One network represents a continuous 3D frame field, evaluable at "infinite resolution" without dense tet meshes. It yields quad surface and hex volume meshes with fewer, better-distributed singularities. — [arXiv](https://arxiv.org/abs/2603.12820)
- NeuFrameQ (Liu et al. 2025b) predicts neural frame fields for anisotropic quadrangulation (per the survey). — [survey](https://arxiv.org/html/2512.23719)

**Unsupervised neural structured-mesh mapping (closest to TFI / elliptic grid generation)**
- 3DMeshNet (Peng, Chen, Liu; arXiv:2407.01560) **[PP]**. A feed-forward network learns the parametric-to-physical map, with meshing PDEs embedded in the loss. Training is stabilised by loss reweighting, gradient projection and finite-difference derivatives. Output is a structured quad or hex mesh with user-chosen cell counts. It claims 85% less training time than other neural approaches and 4–8x faster meshing than traditional methods. — [arXiv](https://arxiv.org/abs/2407.01560)
- PINN-MG (arXiv:2503.00814) is a physics-informed NN for mesh generation **[PP]**. Only the title was confirmed. — [arXiv html](https://arxiv.org/html/2503.00814v1)

**Autoregressive mesh generators (tokens = quantised vertex coordinates per face; surface only)**
- Meshtron (Hao, Romero, Lin, Liu; NVIDIA; arXiv:2412.09548) **[PP as found]**. It generates up to 64K faces at 1024-level coordinate quantisation. Its four ingredients are an hourglass transformer, truncated-sequence training, sliding-window inference, and order-enforcing sampling. It reports more than 50% less training memory and 2.5x throughput. — [arXiv](https://arxiv.org/html/2412.09548v1); [project](https://research.nvidia.com/labs/cosmos-lab/meshtron)
- QuadGPT (ICLR 2026) **[PR]**. Unified tokenisation for mixed tri/quad faces. It uses a 36,864-token context, about 230 tok/s on an A100, and 1.3M quad-dominant training meshes (ShapeNetV2, 3D-FUTURE, Objaverse(-XL), proprietary). It adds RL fine-tuning with "tDPO". — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/255a6f2248df64fbd37267cf93a25b7c-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2509.21420)
- QuadLink (arXiv:2605.16813) **[PP]** does autoregressive quad-dominant generation via "point-relation learning", i.e. it predicts point sets plus relations instead of per-face tokens. Only the title was confirmed. — [arXiv](https://arxiv.org/pdf/2605.16813)
- DeepMesh combines an efficient tokeniser (72% compression) with DPO. — [alphaxiv overview](https://alphaxiv.org/overview/2503.15265v1); [Mesh-RFT paper](https://arxiv.org/abs/2505.16761v2)

**Structured curve / control-point generation (CAD, not meshes)**
- BrepGen (arXiv:2401.15563) **[PP; venue not verified here]**. It is a tree-structured latent: solid → faces → edges → vertices, each node holding a bounding box plus a latent of sampled curve or surface points. Transformer diffusion denoises the nodes, and **duplicate nodes are detected and merged to recover B-rep topology**. Topology is therefore implicit, recovered by deduplication. — [arXiv](https://arxiv.org/html/2401.15563v3)

### Inferences
- The "corner set + Bezier edges" representation in the target pipeline is structurally closest to **BrepGen**: geometry is a set of nodes, and topology is recovered from coincident or duplicated entities. It is also close to **PolycubeNet**: diffusion produces a point set, and a classical step makes it a valid hex structure. Both precedents support "generate geometry as a set, then fix topology and geometry deterministically".
- Industrial block-decomposition ML (Sandia/Cubit line: DiPrete, Owen, Quadros) represents topology as **actions on CAD** rather than generated tokens. Its labels and rewards come from meshability (fraction of hex-meshable volume), not element quality.
- The AR mesh models (Meshtron, QuadGPT) supply engineering tricks that carry over to any token-based block generator: hourglass architecture, truncated training and sliding windows. Their representation is surface-only, though.

### Gaps
- I found no paper named "Polytron", and no NN that directly outputs multi-block hex topology with curved edges for turbomachinery or other CFD blocking.
- Not confirmed here: the final venues of NeurCross, BrepGen and Meshtron, the author list of the CAD-2025 RL solid decomposition paper, and the details of QuadLink and CrossGen.
- MeshGPT, PolyGen and MeshAnything were not fetched in this session (time budget). They are well known but uncited here.

## How is mesh quality (Jacobian, orthogonality, smoothness, feature alignment) incorporated: as loss, as reward, via differentiable meshing, via post-processing?

### Takeaway
I found all four modes:
- **Loss**: PDE residuals and metric losses in 3DMeshNet, GMSNet, NeurCross and NeurFrame.
- **RL reward**: element quality in Pan/SAC and Dmsh; meshability in block-decomposition RL.
- **Preference / RL fine-tuning of AR generators**: DPO, Masked-DPO, tDPO and ARPO, with topology or geometry scores. GRPO is a baseline in these papers but is reported as sample-inefficient.
- **Classical post-processing**: the dominant pattern in the polycube and frame-field families.

### Cited Findings
- Pan, Huang, Cheng, Zeng, "Reinforcement learning for automatic quadrilateral mesh generation: a soft actor-critic approach", *Neural Networks* 157:288–304 (2023) **[PR]**. Meshing is an MDP on the advancing front, solved with SAC. They compare against two commercial packages on scalability, generalisability and effectiveness. — [arXiv](https://arxiv.org/abs/2203.11203v2); [venue via search summary / dblp](https://dblp.uni-trier.de/pid/77/6147)
- FreeMesh-S trains with actor-critic and distributed PPO (DPPO) **[PP]**. The source context is "Generate plane quad mesh with neural networks and tree search" (arXiv:2111.07613), which combines NN and tree search. — [arXiv 2111.07613](https://arxiv.org/pdf/2111.07613)
- SRL-assisted AFM (Tong et al. 2023) uses supervised pretraining, then RL fine-tuning, to improve element quality near holes and constraints. — [deepai](https://deepai.org/publication/srl-assisted-afm-generating-planar-unstructured-quadrilateral-meshes-with-supervised-and-reinforcement-learning-assisted-advancing-front-method); [survey](https://arxiv.org/html/2512.23719)
- GMSNet, "An intelligent mesh-smoothing method with graph neural networks", *FITEE* (Frontiers of Information Technology & Electronic Engineering) **[PR]** (arXiv:2311.12815):
  - A GNN takes each node's neighbourhood and outputs the smoothed position directly.
  - **MetricLoss** is a quality-metric loss that needs no high-quality target meshes.
  - A fault-tolerance mechanism prevents negative-volume elements.
  - It uses 5% of the parameters of the prior model and runs 13.56x faster than optimisation-based smoothing.

  — [arXiv](https://arxiv.org/abs/2311.12815v2); [FITEE](https://journal.hep.com.cn/fitee/EN/10.1631/FITEE.2300878)
- GNNRL-Smoothing (arXiv:2410.19834) **[PP]** is a "prior-free" RL model for mesh smoothing, so mesh quality is used as the reward. Only the title was confirmed. — [arXiv](https://arxiv.org/pdf/2410.19834)
- 3DMeshNet puts **meshing PDEs into the loss**, making it unsupervised with no target meshes. — [arXiv](https://arxiv.org/abs/2407.01560)
- NeurCross losses cover SDF fit, principal-direction alignment and cross-field smoothness. It is evaluated by Jacobian ratio and angle/area distortion after classical quad extraction. — [arXiv](https://arxiv.org/html/2405.13745v2)
- Mesh-RFT, NeurIPS 2025 **[PR]**:
  - Pipeline: an hourglass AR transformer is pretrained, then candidate meshes are scored with an objective topology-aware system, Boundary Edge Ratio (BER) and Topology Score (TS), at object and face level.
  - **Masked DPO** masks the preference loss to low-quality faces.
  - Results: −24.6% Hausdorff and +3.8% TS over the pretrained model; −17.4% HD and +4.9% TS over global DPO.
  - It argues that object-level rewards (e.g. DeepMesh's DPO) over-optimise some regions and under-optimise others.

  — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e4ca3675c3569a7c222c8351e4db8b9e-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2505.16761)
- Mesh-Pro (Zhou et al., arXiv:2603.00526, accepted CVPR 2026 per the abstract) **[PR – acceptance stated by authors]**. It uses asynchronous online RL with **ARPO** (advantage-guided ranking preference optimisation). The authors say this trades off efficiency and generalisation better than DPO and **GRPO**. Other points: a ray-based geometric-integrity reward, 3.75x faster than synchronous RL, and SOTA on artistic and dense quad meshes. Search snippets around it say GRPO "exhibits low exploration–exploitation efficiency and slow convergence" when limited by the pretrained model and complex reward distributions. — [arXiv](https://arxiv.org/abs/2603.00526)
- QuadGPT uses tDPO, an RL fine-tune for topological quality. — [arXiv](https://arxiv.org/pdf/2509.21420)
- MeshingNet (Zhang, Wang, Jimack, Wang; ICCS 2020) **[PR]**. An NN predicts the local mesh-density field that guides a standard mesher. It is trained from a-posteriori error estimates, so quality is defined by solution error, not element shape. — [arXiv](https://arxiv.org/abs/2004.07016v1)
- Quality *prediction*: Random-forest models on B-rep features (Owen et al. 2019–2024; Parry & Owen 2023) are deployed in Cubit to flag low-quality pre-mesh regions. The survey notes that predicting quantiles works better than single scores. — [survey](https://arxiv.org/html/2512.23719)
- The survey lists open problems: reward shaping, sparse long-horizon rewards, constraint enforcement and injectivity in RL policies, and the fact that "mesh quality is proxy, not ground truth". — [survey](https://arxiv.org/html/2512.23719)

### Inferences
- For a quality reward on a block generator, Mesh-RFT's result carries over directly: **localised (per-block or per-edge) advantages or masking beat a single global scalar**. A per-block minimum scaled Jacobian or orthogonality could be masked to the offending tokens or points.
- GMSNet's MetricLoss and 3DMeshNet's PDE loss show that quality metrics work as *differentiable training losses without targets*. That is an alternative or complement to "retrain on optimised targets".
- Mesh-Pro's evidence against GRPO concerns large AR mesh models with sparse, complex rewards. Whether it applies to small set-based flow models is untested.

### Gaps
- I found no paper that differentiates *through* a TFI or elliptic block-meshing step to train a generator end-to-end. 3DMeshNet is the closest, but it learns the map itself rather than training a topology generator.
- I found no work that explicitly uses hex scaled-Jacobian or orthogonality as an RL or preference reward for a *generative* block model.

## Are there flow-matching / diffusion approaches for structured mesh or curve/control-point generation, and how do they handle constraints (points on surfaces/curves)?

### Takeaway
Diffusion exists for polycubes (DDPM-Polycube, PolycubeNet) and B-rep curves and surfaces (BrepGen). In each case the generative model output is **post-hoc projected, registered or deduplicated** by classical steps rather than constrained during sampling. General flow-matching literature offers inference-time hard-constraint projection (PCFM and successors), which could enforce "corner on surface / point on curve". I found no flow-matching hex or block generator.

### Cited Findings
- PolycubeNet handles constraints by registration: after diffusion it applies rigid and non-rigid point-cloud registration to the input surface, then polycube-to-hex. — [arXiv](https://arxiv.org/abs/2605.20274)
- DDPM-Polycube / Scalable DDPM-Polycube use DDPM for polycube layout, with classical hex and volumetric-spline construction after. — [arXiv 2604.17266](https://arxiv.org/pdf/2604.17266); [survey](https://arxiv.org/html/2512.23719)
- BrepGen: edges are curves sampled as 1D arrays of 3D points, and transformer diffusion denoises node features. Topology comes from merging duplicate nodes, and parametric curves are fitted afterwards. — [arXiv](https://arxiv.org/html/2401.15563v3)
- Physics-Constrained Flow Matching (PCFM, arXiv:2506.04171) **[PP]**. A pretrained flow model gets zero-shot enforcement of arbitrary nonlinear equality constraints "up to machine precision" by projecting intermediate flow states onto the constraint manifold during sampling. — [arXiv](https://arxiv.org/html/2506.04171v1)
- Repeated projection can distort the learned distribution. Chance-constrained FM (arXiv:2509.25157) proves equivalence to projecting clean samples. Constraint-aware FM puts the projection into training. SNAP-FM (arXiv:2607.00095) speeds up the nonlinear projection. — [CCFM](https://arxiv.org/pdf/2509.25157); [SNAP-FM](https://arxiv.org/pdf/2607.00095)

### Inferences
- For "corners on dtOO surfaces / edge control points on curves", a PCFM-style projection after each ODE step (or the last few) is a direct, published recipe. Its listed downside is distribution distortion; constraint-aware training is the mitigation.
- The BrepGen and PolycubeNet "generate then repair" pattern is consistent with the planned deterministic feature/TFI optimisation stage.

### Gaps
- I found no flow-matching or diffusion paper that generates *block corners plus edge control points for hex blocking*.
- I found no evaluation of constrained flow matching on meshing tasks specifically.

## What results and datasets exist, and which ideas transfer to a pipeline "flow-matching generator of block corners + Bezier edges -> deterministic feature/TFI optimisation -> retraining on optimised targets / quality reward"?

### Takeaway
The datasets are mostly surface meshes (ShapeNet, Thingi10K, Objaverse; 1.3M quad meshes for QuadGPT), synthetic CAD for decomposition, and one new paired CAD/polycube set (PolycubeNet). I found no public block-structured hex dataset. Five ideas transfer:
1. Generate, then deterministic repair (polycube, BrepGen).
2. Constraint projection during flow sampling (PCFM).
3. Localised preference or advantage on quality (Mesh-RFT, Mesh-Pro).
4. Target-free quality losses (GMSNet MetricLoss, 3DMeshNet PDE loss).
5. Curriculum from simple to complex geometry (Dmsh).

### Cited Findings
- Datasets and benchmarks:
  - NeurCross evaluates on ShapeNet and Thingi10K. — [arXiv](https://arxiv.org/html/2405.13745v2)
  - QuadGPT trains on 1.3M quad-dominant models. — [arXiv](https://arxiv.org/pdf/2509.21420)
  - PolycubeNet releases a paired CAD-polycube dataset. — [arXiv](https://arxiv.org/abs/2605.20274)
  - Quadros et al.'s GNN uses randomised synthetic CAD with multi-action labels. — [survey](https://arxiv.org/html/2512.23719)
- The survey names as an open challenge the need for "curated datasets, shareable benchmarks, and robustness across CAD sources", and the domain shift from synthetic to industrial parts (98.7% → 85% accuracy). — [survey](https://arxiv.org/html/2512.23719)
- Dmsh: curriculum learning plus recursive decomposition (parallel sub-region meshing). — [arXiv](https://arxiv.org/abs/2606.10601)
- Mesh-RFT shows a three-stage pattern of pretrain, score candidates with an objective metric, then masked preference fine-tune. It reports measurable gains over both the pretrained model and global DPO. — [arXiv](https://arxiv.org/pdf/2505.16761)
- The survey states that "AI complements established geometry and meshing algorithms rather than replacing them". Learning reduces manual effort, while the downstream mapping "remains classical geometry processing" (DL-Polycube pattern). — [survey](https://arxiv.org/html/2512.23719)

### Inferences
These map onto the user's pipeline as follows; they are inferences, not cited facts.
- **"Retrain on optimised targets"** is a form of expert iteration / self-distillation, analogous to the Mesh-RFT pipeline, where a scorer defines the better sample. With a deterministic optimiser, the optimised output is a "chosen" sample and the raw output a "rejected" one. That gives DPO-style pairs for free, or simply supervised regression targets for a flow model.
- **Quality reward for a flow-matching model**: RL fine-tuning for flow and diffusion models was not covered in this session. Two options carry over from the cited work: reward-weighted regression on optimised targets, and localised (per-block) credit assignment, which follows from Mesh-RFT.
- **Constraints**: projecting corners onto surfaces and curves at sampling time (PCFM) shrinks the residual the deterministic optimiser must fix. Training with the projection in the loop reduces the distribution shift.
- **Topology**: BrepGen's dedup-merge and PolycubeNet's registration suggest the set-based generator can stay simple, as long as a robust deterministic topology-recovery and repair step follows.
- **Generalisation**: Dmsh's curriculum and the survey's domain-shift numbers suggest training from simple to complex geometries and holding out geometry families when evaluating.

### Gaps
- I found no public benchmark of block-structured hex meshes (e.g. turbomachinery O/H-grids) with quality labels.
- Not researched in this session: reward fine-tuning methods specific to flow matching (e.g. DDPO / flow-GRPO style). This should be checked separately if a quality reward on the flow generator is pursued.
- Uncited due to budget: MeshGPT, PolyGen, MeshAnything (v1/v2), "learning to smooth meshes" variants other than GMSNet, and Narayanan / Bourmaud primary sources.
