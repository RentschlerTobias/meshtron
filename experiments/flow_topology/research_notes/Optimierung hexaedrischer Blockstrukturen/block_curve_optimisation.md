# Geometry optimisation of multi-block structured hex meshes: block vertices, curved block edges, TFI-based volume meshes

Scope: generic algorithms that move block corners / block-edge control points (e.g. one cubic Bezier per edge) and build the volume mesh with transfinite interpolation (TFI), with objectives on scaled Jacobian, orthogonality, G1 grid-line smoothness and cell-size jumps. 22 tool calls; primary PDFs were read in full text for Sauer 2018, Dobrev et al. 2019 (TMOP), Toulorge et al. 2013, and Hinz/Möller/Vuik.

## Q1: Which papers/theses optimise block-edge control points or block corners of a block-structured mesh for quality?

### Takeaway
The closest published match to "optimise a reduced block-level control description, then discretise" is Marcel Sauer's (DLR) optimisation-based multi-block grid generator. It uses a quadratic B-spline "abstraction layer" (a coarse control grid plus shared per-edge parameter vectors, the "u-vectors"). Analytic gradients go into Levenberg-Marquardt, and the final grid comes from TFI plus a projection step. The IGA "multi-patch parameterisation" literature solves the same mathematical problem: it optimises inner spline control points with fixed boundary under a Jacobian constraint. I found no paper that does exactly "one cubic Bezier per block edge + Coons/TFI + gradient through the TFI" in 3D for generic geometries.

### Cited Findings
- **Sauer 2018 (ECCM6/ECFD7, DLR Institute of Propulsion Technology), "An optimization based approach to multi-block structured grid generation".** Instead of elliptic or algebraic generation, it "optimizes grid quality criteria such as cell expansion ratios, inner cell angles, and grid line curvatures". These are "evaluated approximately on a B-spline based abstract representation of the multi-block structured grid, which drastically reduces the degrees of freedom". The paper formulates and demonstrates 2D only (blade-to-blade slices of a compressor, a turbine and a tandem) — [Sauer 2018, DLR elib](https://elib.dlr.de/120879/1/p1246.pdf)
  - Representation: 2nd-degree B-splines with equally spaced knots on a coarse control grid. The "u-vectors" (per-edge sampling parameters) are "shared over edges", so their parameter count scales linearly with resolution while the control grid does not scale at all. The examples need fewer than 1,000 parameters, against about 30,000 in 2D for a native per-node parameterisation. The 3D estimate is fewer than 6,000 parameters against 3,500,000 native — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
  - Topology preprocessing: each block edge has "either no connectivity or exactly one full connectivity to another block" (blocks are split where needed). Near a block-to-block border a merged control grid is used, so the border becomes a single smooth grid line. This is how smoothness across block interfaces is built in — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
  - Singularities (a block corner with edge neighbours but no corner neighbour) get a special blended surface. Three surface definitions are combined: S = f_mid·S_mid + f_ver·S_ver + f_hor·S_hor. The weights are quadratic: f_mid = (1−u)²(1−f_d) + (1−v)² f_d, f_hor = (1−(1−v)²) f_d, f_ver = (1−(1−u)²)(1−f_d). f_d is a diagonal-distance blend: ½(2−(1−d)²) for v > u, and 1 − ½(2−(1−d)²) otherwise — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
  - Objective (least squares): c_sqsum = Σ(w_a c_a)² + Σ(w_c c_c)² + Σ(w_sa c_sa)² + Σ(w_sd c_sd)². The four criteria are:
    - Angle: c_a = arccos((P2−P1)·(P3−P2)/(|P2−P1||P3−P2|)) − α_target (in degrees). α_target defaults to 90° and may differ near singularities.
    - Curvature: c_c = |(P2−P1)/|P2−P1| − (P3−P2)/|P3−P2|| / (min(|P2−P1|,|P3−P2|)·L_ref), taken over three consecutive control nodes along a grid line.
    - Control-grid stretching: c_sa = max(|P2−P1|,|P3−P2|)/min(·) − 1.
    - c_sd: the same stretching ratio on estimated final-grid curves, with an optional prescribed expansion ratio.

    The weights are the product of a node weight, a criterion weight and a multiplicity factor that compensates for the resolution difference between control grid and final grid — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
    (Note: the extracted PDF text prints |P2−P1|·|P2−P1| in the denominator of c_a, which looks like a typo for |P2−P1||P3−P2|.)
  - Constraints are set per node: fixed, on a curve, circumferential, or on a surface of revolution. **Wall orthogonality is a hard constraint**: the first control point lies on the geometry and the next one follows the geometry normal. First-cell distances are imposed by fixing the second control point's distance and the u-vector entry — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
  - Optimiser: Levenberg-Marquardt. "By the usage of the chain rule all partial derivatives of the objective function regarding the parameters are calculated analytically." All parameters are optimised simultaneously. The linear system is preconditioned and solved with CG. An iteration that folds the control grid is rated as not improving — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
  - Pipeline: coarse optimisation → edge-refinement optimisations → surface refinement (interior initialised by TFI from the refined edges, then optimised) → u-vector re-optimisation → discretisation → final projection onto geometry, with the projection shift decaying within one block. One common weight set was used for all 5 compressor and 3 turbine geometries, each at 3 span positions — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
- A follow-up journal paper exists: Sauer, "An optimization based multi-block-structured grid generation method", IJNME 2023, doi:10.1002/nme.7308. The search snippet says the B-spline intermediate layer can represent "continuous surfaces or volumes". Each control point "can be assigned to follow curves or surfaces, be completely fixed in space"; blocks are meshed "using transfinite interpolations and elliptic smoothing" — [Wiley IJNME 2023](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7308) (full text was 403 for me; content only from the search snippet) — also [ResearchGate entry](https://www.researchgate.net/publication/371751654_An_optimization_based_multi-block-structured_grid_generation_method)
- Sauer gave a talk on the method (RPTU Kaiserslautern, 6 Oct 2022): the grid is generated with "minimal parameters for block dimensions and distance specifications", the problem is formulated as optimisation over a B-spline reduced description, and results are compared against algebraic grid generation inside design optimisations — [RPTU news](https://math.rptu.de/en/aktuelles/newsdetail/news/marcel-sauer-development-of-an-optimization-based-automatic-block-structured-grid-generation-method)
- IGA planar parameterisation (Gravesen, Evgrafov, Nguyen et al., 2014) works by "constrained minimization of a quality functional over the inner control points". To avoid self-intersection it adds "a nonlinear and nonconvex sufficient condition for the Jacobian determinant as a constraint". Typical functionals are area-orthogonality, Liao, Winslow, uniformity and harmonic energy — [Hinz/Möller/Vuik summary of the literature, arXiv 1904.03009](https://www.arxiv.org/pdf/1904.03009); [search listing for Gravesen et al.](https://worldwidescience.org/topicpages/p/parametric+analysis.html)
- Hinz, Möller, Vuik, "An IGA framework for PDE-based planar parameterization on convex multipatch domains" (Springer, IGA 2018, publ. 2021): elliptic grid generation (EGG) with C0 multipatch support via a mixed-FEM formulation and Newton-Krylov. It does **not** need a bijective initial guess (a folded initial guess converged in 6 Newton iterations) — [arXiv 1904.03009](https://www.arxiv.org/pdf/1904.03009)
- Zauner & Sandham (ICCFD10, 2018) build multiblock DNS airfoil grids with polynomial functions. High-order methods "require continuous metric terms up to the second order of derivatives or higher", so they control 1st–3rd derivatives of the point distribution at interfaces (e.g. parameters prof_deriv1/2/3_cont for ∂ⁿn/∂ηⁿ at the near-field interface). This is a hand-crafted, analytic approach, not an optimiser — [ICCFD10-129](https://iccfd.org/iccfd10/papers/ICCFD10-075-Paper.pdf)
- A Robust-TFI multiblock paper exists that modifies TFI "to include improved orthogonality and spacing control" and adds an aspect-ratio-based smoothing that "removes grid crossover" (search snippet only) — [ResearchGate: Automatic Structured Multiblock Mesh Generation Using Robust TFI](https://www.researchgate.net/publication/229806466_Automatic_Structured_Multiblock_Mesh_Generation_Using_Robust_Transfinite_Interpolation)
- The Pietroni et al. hex survey (ACM TOG 2022) states the core challenge: "to obtain a function and a corresponding optimization method such that the optimum obtained has no inverted elements and maximizes as much as possible the mesh Jacobian". Known objective functions are "highly non-linear and do not allow for robust global optimum computation" — "Hex-Mesh Generation and Processing: a Survey" (Pietroni, Campen, Sheffer, Cherchi, Bommes, Gao, Scateni, Ledoux, Remacle, Livesu), [arXiv 2202.12670](https://arxiv.org/abs/2202.12670) (I read it from a locally cached PDF; I inferred the arXiv ID from the title and did not re-verify it)

### Inferences
- Sauer's pipeline is the closest template for a "block vertices + curved block edges" optimiser. Its key design choices:
  - least-squares criteria evaluated on the control polygon (cheap), not on the full mesh;
  - analytic chain-rule gradients with Levenberg-Marquardt;
  - hard constraints via parameterising nodes on curves/surfaces;
  - wall orthogonality built in structurally (second control point along the normal) rather than penalised.
- With one cubic Bezier per block edge and Coons/TFI interior, every mesh node is an affine function of corner and edge control points: x_node = Σ_k c_k(ξ,η,ζ)·P_k. The coefficients c_k come from the TFI blending functions times the Bernstein polynomials and are fixed for fixed topology and resolution. So the mesh-node Jacobian w.r.t. the control points is a constant sparse matrix (precomputable). Any node-level objective (TMOP, Toulorge barrier, scaled Jacobian) can then be pulled back with one sparse mat-vec per gradient. This is exactly the structure TMOP exploits element-wise (A = Σ x_j ∇φ_j). [inference]

### Gaps
- I could not read the full IJNME 2023 Sauer paper (HTTP 403), so its 3D formulation, any volume criteria and the final optimiser are unverified.
- I found no paper that explicitly optimises one cubic Bezier per block edge with gradients through Coons/Gordon-Hall TFI in 3D for arbitrary block topologies.
- I found no source text for the NASA/NTRS multiblock reports or the Thompson/Soni/Weatherill handbook within the budget.

## Q2: How are objectives formulated, and how are constraints handled?

### Takeaway
There are three families. (a) Least-squares geometric criteria (angle, stretching, curvature), as in Sauer. (b) Jacobian-matrix metrics in the target-matrix (TMOP / Knupp) framework, using shape, size and alignment metrics with barrier properties. (c) Barrier or regularised-determinant functionals for untangling (Toulorge log-barrier; Escobar regularised σ). Surface and curve constraints are handled almost universally by optimising parametric (t or u,v) coordinates and chaining with CAD tangents, or by tangential relaxation. Fixed topology is implicit in all of them.

### Cited Findings
- **TMOP (Dobrev, Knupp, Kolev, Mittal, Tomov, "The target-matrix optimization paradigm for high-order meshes", SIAM J. Sci. Comput. 2019 / arXiv 1807.09807)**
  - Objective: F_V(x) = Σ_E ∫_{E_t} μ(T(x)) dx_t = Σ_E Σ_q w_q det(W(x_q)) μ(T(x_q)), with T = A W⁻¹. A is the physical Jacobian and W is the target Jacobian; W decomposes as [volume][orientation][skew][aspect ratio] — [arXiv 1807.09807](https://arxiv.org/pdf/1807.09807)
  - Metrics, with τ = det T and |T|² = tr(TᵀT):
    - μ1 = |T|² (no barrier)
    - μ2 = |T|²/(2τ) − 1 (2D shape, ideal barrier)
    - μ9 = τ|T − T⁻ᵗ|² (shape+size)
    - μ77 = ½(τ − 1/τ)² (size)
    - μ303 = |T|²/(3τ^{2/3}) − 1 (3D shape)
    - Alignment: μ30 = |a1||w1| − a1·w1 + |a2||w2| − a2·w2
    - μ7 = |T − T⁻ᵗ|²; μ14 = |T − I|² (shape+size+alignment)

    — [arXiv 1807.09807](https://arxiv.org/pdf/1807.09807)
  - Gradient: ∂F_V/∂x_i = Σ_E ∫ (∇̃φ_i W⁻¹) : ∂μ/∂T dx_t, using A(x) = Σ_j x_j ∇̃φ_j. The Hessian is derived the same way, and the problem is solved with Newton plus MINRES. The step α is reduced until det(T) > 0 at all quadrature points, F(x_{n+1}) < 1.2 F(x_n) and the residual norm grows by less than 20% — [arXiv 1807.09807](https://arxiv.org/pdf/1807.09807)
  - Quadrature: at least (map order + 1) points per direction. Uniform, Gauss-Lobatto and Gauss-Legendre showed no systematic differences — [arXiv 1807.09807](https://arxiv.org/pdf/1807.09807)
  - Boundary: either fixed nodes, or "tangential relaxation", i.e. optimising boundary nodes in CAD parametric coordinates when the boundary is parametric. There is a variant for implicit boundaries. A "limiting" term restricts deviation from the initial mesh: μ(T) + μ_lim with a replication barrier using a second target W0 — [arXiv 1807.09807](https://arxiv.org/pdf/1807.09807)
  - Follow-on work drives TMOP from simulation data (ALE hydrodynamics: shock positions, material regions, error estimates) — [arXiv 2001.11536](https://arxiv.org/pdf/2001.11536)
- **Toulorge, Geuzaine, Remacle, Lambrechts, "Robust untangling of curvilinear meshes", JCP 254 (2013) 8–26**
  - Objective: f = E + F.
  - E is a displacement penalty: E = ½ Σ_e Σ_ij (x_i^e − X_i^e) K_ij^e (x_j^e − X_j^e), with K diagonal with entries w_i^e/L². Typical weights: w = 10⁵ for boundary nodes, w = 100 for interior nodes.
  - F is the barrier: F_l^e = [log((B_l^e − εJ_0^e)/(J_0^e − εJ_0^e))]² + (B_l^e/J_0^e − 1)². The B_l^e are the Bezier coefficients of the Jacobian, which give provable bounds min_ξ J ≥ min_i B_i.
  - The 2D version adds a second (upper) barrier: f1 = E + F(ε̄_min) + F(ε̄_max).

  — [Toulorge et al. preprint](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf)
  - Constraints: a node on a CAD curve is optimised in its curve parameter t, with df/dt = ∂f/∂x · dx/dt. A node on a CAD face is optimised in (u,v), with tangents taken from CAD. Nodes on model vertices are fixed. A diagonal preconditioner uses the tangent norms — [Toulorge et al.](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf)
  - Strategy: local "blobs" of N element layers around invalid elements, with the blob boundary fixed. A "moving barrier" sequence raises ε step by step (ε < 0 when starting tangled). Conjugate gradients (ALGLIB) beat IPOPT and L-BFGS on efficiency — [Toulorge et al.](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf)
- Toulorge, Lambrechts, Remacle, "Optimizing the geometrical accuracy of curvilinear meshes", JCP 310 (2016) 361–380. It adds a fast geometric-accuracy estimate (Taylor expansion of curves) to the untangling objective, inside one unconstrained optimisation — [arXiv 1507.04052](https://arxiv.org/pdf/1507.04052)
- **Roca / Gargallo-Peiró / Ruiz-Gironés / Sarrate (UPC)**
  - Their distortion measure on CAD surfaces is independent of the surface parameterisation. Smoothing and untangling minimise a regularisation of that measure "in terms of the parametric coordinates of the nodes to enforce that nodes slide on the surfaces" — Gargallo-Peiró, Roca, Peraire, Sarrate, IJNME 106(13):1100–1130 (2016) — [LaCàN abstract](https://www.lacan.upc.edu/publications/abstract/?id=734); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/107144)
  - Distortion and quality measures for high-order hexahedra/quads, with simultaneous smoothing and untangling by minimising mesh distortion — Gargallo-Peiró, Ruiz-Gironés, Roca, Sarrate, "On curving high-order hexahedral meshes", IMR24 (2014/15) — [UPCommons](https://upcommons.upc.edu/entities/publication/342b76ae-3a09-4cc5-9752-ed003c243519/full)
  - 3D variant: minimises volume distortion w.r.t. inner node coordinates plus the parametric coordinates of curve and surface nodes. A later variant adds a geometric "disparity" term to the distortion in a single objective — [UPCommons 2117/86072](https://upcommons.upc.edu/handle/2117/86072); [UPCommons: Optimal quality and geometric accuracy](https://upcommons.upc.edu/entities/publication/d955977a-87cd-4f2d-816b-9c519e3aff0a/full)
- **Escobar et al. (CMAME 192, 2003), simultaneous untangling and smoothing.** The barrier in shape metrics is replaced by a regularised determinant, defined and smooth on all of R³, so meshes with inverted elements can be optimised directly without a separate untangling pass. The SUS code uses h(σ) = ½(σ + √((σ−2ε)² + 4δ²)) (a modification of the original ½(σ + √(σ² + 4δ²))) — [ULPGC SUS tech note](https://www.dca.iusiani.ulpgc.es/SUScode/TechNote.pdf); [ULPGC paper PDF](https://www.dca.iusiani.ulpgc.es/proyecto0204/pdf/sustm.pdf)
- **Knupp, "Algebraic Mesh Quality Metrics", SIAM J. Sci. Comput. 23(1):193–218 (2001).** Metrics are built algebraically on the Jacobian and related matrices. Condition number and mean ratio are proved equivalent, and the condition number measures distance to degenerate elements. The framework extends to non-simplicial (hex) elements with skew, length-ratio, shape, volume and orientation measures — [SIAM](https://epubs.siam.org/doi/10.1137/S1064827500371499); [OSTI report](https://www.osti.gov/biblio/754328)
- **Winslow functional** (as used for spline parameterisations): W(x) = ∫ (g11 + g22)/det J dξ, with g11 = x_ξ·x_ξ and g22 = x_η·x_η. Its global minimiser approximates the inverse-harmonic (elliptic) map. Because det J sits in the denominator, direct minimisation needs a bijective initial guess — [Hinz et al., arXiv 1904.03009](https://www.arxiv.org/pdf/1904.03009)
- The elliptic form is L(x) = g22 x_ξξ − 2 g12 x_ξη + g11 x_ηη = 0, with x on ∂Ω̂ equal to ∂Ω — [Hinz et al.](https://www.arxiv.org/pdf/1904.03009)

### Inferences
- For a TFI-from-edges setup, TMOP's variational objective transfers almost directly. Evaluate T = A W⁻¹ at Gauss points of each fine hex (or of each block treated as a high-order element, if the block is regarded as a single tensor-product map). Then chain ∂F/∂x_nodes through the constant TFI/Bernstein matrix to get ∂F/∂(control points). Targets W can encode wall-normal spacing and aspect ratio (boundary layer), which subsumes "cell size jump" control. [inference]
- Starting from a possibly invalid block geometry favours barrier-free or moving-barrier formulations: the Toulorge moving ε, Escobar's regularised σ, or the Garanzha-style foldover-free maps found in other research lines. A pure Winslow or μ2 barrier needs a valid start. [inference]
- Wall orthogonality has two treatments. It can be a hard parameterisation constraint (Sauer: second control point along the surface normal; for a cubic Bezier edge, this means fixing the first interior control point on the normal ray), or a soft alignment metric (TMOP μ30-type with W aligned to the wall normal). [inference]

### Gaps
- I did not retrieve the explicit hex distortion formula of Roca et al.; it is commonly η = |S|²/(d·σ^{2/d}) with regularised σ, but I did not verify it from source.
- I found no published comparison of these objectives specifically on block-level (very coarse, very high "order") maps.

## Q3: How do high-order / curved-mesh optimisation methods transfer to block-level edges?

### Takeaway
A block with Bezier edges and a TFI interior is mathematically a single high-order curved element (or a coarse-control "abstraction layer", as Sauer calls it). The high-order toolkits therefore carry over:
- Jacobian bounds via Bezier coefficients (Toulorge/Johnen);
- target-matrix metrics evaluated at quadrature points (TMOP);
- CAD-parametric sliding of boundary DOFs (Toulorge, Roca);
- blob/local optimisation with moving barriers.

Two adaptations are needed. Edge DOFs are shared between all blocks around an edge. And the evaluation points must cover the fine mesh (or a sufficiently fine quadrature), because a block-level map is far higher "order" than typical p = 2–4 elements.

### Cited Findings
- Validity bounds: the element Jacobian J(ξ) is polynomial, so it can be written exactly in a Bezier basis, J = Σ B_i^(q)(ξ) B_i, which gives min J ≥ min_i B_i and max J ≤ max_i B_i. Adaptive subdivision tightens the bounds — [Toulorge et al. 2013](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf)
- TMOP's "fully algebraic" formulation relies only on local Jacobian matrices, which makes it applicable to all element types in 2D and 3D and any mesh order. It is implemented open-source in MFEM. The number of quadrature points must be at least map order + 1 — [arXiv 1807.09807](https://arxiv.org/abs/1807.09807); [arXiv PDF](https://arxiv.org/pdf/1807.09807)
- Toulorge's method is applicable "to any kind of polynomial element, for surface, volume, hybrid or boundary layer meshes" — [search summary of Toulorge et al. JCP 2013](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf)
- Sauer evaluates quality directly on the control polygon because quadratic B-splines "follow its control grid closely". That allows inner cell angles and grid-line curvatures to be estimated "directly from it". The discretised grid is smoother than the control grid (convex-hull property) — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
- The convex-hull smoothing of B-splines leaves a geometric gap at walls. Sauer closes it with control-grid refinement near the geometry and then a final projection whose shift decays within one block — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)

### Inferences
- Practical transfer recipe for Bezier edges plus TFI [inference]:
  1. **DOFs:** block corners (free, on surface (u,v), on curve (t), or fixed) and the 2 interior Bezier control points per edge (constrained like the edge's classification). Face and volume interiors come from Coons/Gordon-Hall, so they add no DOFs.
  2. **Linear map:** X_fine = M·p, with M constant (blending functions × Bernstein × distribution functions).
  3. **Objective:** Σ_hex Σ_q w_q det W μ(A_q W_q⁻¹), plus a log-barrier on the scaled Jacobian at fine-mesh corners (or Bezier-bounded Jacobian), plus smoothness terms (Q4), plus a regularising displacement term à la Toulorge E.
  4. **Gradient:** gradient = Mᵀ ∂F/∂X. Optimise with L-BFGS/CG (Toulorge) or Newton/LM (TMOP/Sauer), with a line search that rejects det ≤ 0.
- Because M is linear, the Hessian of any convex-in-X term stays convex in p. The non-convexity comes only from μ, and barrier validity can be checked cheaply. [inference]
- Gmsh's built-in HighOrderMeshOptimizer (the implementation of Toulorge et al.) and MFEM TMOP could be reused as metric/gradient kernels on a fine mesh, with an external chain rule to the block DOFs. [inference; Gmsh/MFEM do not provide block-level DOFs natively as far as found]

### Gaps
- I found no published experiments on how coarse (block-level) parameterisations interact with fine-mesh validity in 3D; Sauer reports 2D only.
- MFEM documentation pages for TMOP options were not fetched.

## Q4: What is known on G1 / smoothness of grid lines across block boundaries and at singular vertices?

### Takeaway
Classical multiblock practice distinguishes interface types: C0 (points match), and C∞ or periodic (metric derivatives continuous). There are three main ways to get grid-line smoothness across block interfaces:
1. Sauer: optimise a shared ("merged") B-spline control grid across interfaces, plus special blended surface definitions at singular block corners, plus curvature/angle penalties on consecutive control points.
2. Elliptic/Winslow smoothing over patches. This gives only C0 at patch interfaces when patches are pulled back independently (Hinz et al.).
3. The IGA G1-multipatch theory (gluing functions α, β), which gives exact algebraic G1 conditions across an interface that are linear in the control points for fixed gluing data.

At extraordinary (singular) vertices, smoothness is generally only achievable in a weakened sense.

### Cited Findings
- FMG3D-type multiblock grids support interface types periodic, C-zero and C-infinity. Grid points must match for all three; for periodic and C-infinity, "derivatives of grid metrics must be continuous at interfaces" (search snippet) — [Science.gov topic page: multi-block structured grid](https://www.science.gov/topicpages/m/multi-block+structured+grid.html)
- Structured multi-block grids "may or may not exhibit grid line continuity across the block interface"; relaxing that continuity eases grid generation (search snippet) — [Science.gov topic page](https://www.science.gov/topicpages/m/multi-block+structured+grid.html)
- Sauer: the curvature criterion c_c on three consecutive control points along a grid line (crossing block borders via the merged control grid) penalises kinks, i.e. a discrete G1 measure. Stretching criteria c_sa/c_sd penalise size jumps, with an extra weight on stretching "across an edge that leads to a singularity". At singularities the target angle α_target may differ from 90°. The blended S_mid/S_hor/S_ver construction "construct[s] a smooth interface to the standard B-spline region" — [Sauer 2018](https://elib.dlr.de/120879/1/p1246.pdf)
- Hinz/Möller/Vuik, multipatch EGG: "the isolines make steep angles by the internal patch interfaces", because the patchwise affine pull-back "generally introduces a C0-continuity in the composite mapping". "Higher-order smoothness across patch interfaces is generally difficult to achieve and usually done by constructing bases whose elements possess higher-order continuity sufficiently far away from the extraordinary vertices" — [arXiv 1904.03009](https://www.arxiv.org/pdf/1904.03009)
- **G1 gluing condition (Collin, Sangalli, Takacs, arXiv 1509.07619, 2015).** Across an interface between patches F^(L) and F^(R) with common boundary F0: α^(R)(v) D_u F^(L)(0,v) − α^(L)(v) D_u F^(R)(0,v) + β(v) D_v F0(v) = 0, with α^(L) α^(R) > 0.
  - "Analysis-suitable G1" requires α^(L), α^(R) and β^(L), β^(R) to be linear polynomials. All bilinear multipatch parameterisations satisfy it automatically.
  - The paper gives existence results and examples, not an optimisation algorithm.

  — [ar5iv 1509.07619](https://ar5iv.arxiv.org/html/1509.07619)
- Zauner & Sandham enforce continuity of 1st–3rd wall-normal distribution derivatives at block interfaces by analytic construction, motivated by high-order solvers that need continuous metric terms — [ICCFD10](https://iccfd.org/iccfd10/papers/ICCFD10-075-Paper.pdf)
- A paper on reconstructing singular connections in multi-block CFD exists (arXiv 1711.00110). I saw it only in search results and did not read it — [arXiv 1711.00110](https://arxiv.org/pdf/1711.00110)

### Inferences
- For one cubic Bezier per block edge, G1 of a grid line through a regular block vertex shared by two collinear edges (valence 4 in 2D / regular in 3D) needs two things: the two adjacent edge control points and the vertex must be collinear, P_{1}^{(a)} − V ∥ V − P_{1}^{(b)}, and a ratio constraint for C1. This can be imposed exactly by parameterising, or penalised as (1 − t_a·t_b)², which is Sauer's c_c with t the unit tangents. Inside faces, Coons/TFI between G1-joined edges does not by itself guarantee G1 across a block face: the cross-derivative of TFI at the face depends on the opposite edges. The Collin-Sangalli-Takacs condition, evaluated at fine-grid lines, gives the appropriate linear residual to penalise. [inference]
- At singular (valence ≠ 4) block vertices, G1 of all grid lines is impossible in general. Sauer's approach (target angles 360°/valence, weighted stretching across singular edges) and "smoothness away from extraordinary vertices" (Hinz) are the documented pragmatic choices. [inference based on the cited statements]
- Cell-size jumps across block interfaces can be penalised as in Sauer's c_sa/c_sd (ratio of consecutive spacings across the interface minus 1). Alternatively, use a TMOP size metric (μ77) with a smooth target-size field that is continuous across blocks. [inference]

### Gaps
- I found no paper giving an optimisation algorithm that enforces G1 of grid lines across 3D block faces for arbitrary block topologies with TFI interiors.
- I found no source for classic smoothness measures at singular vertices in structured multiblock CFD (e.g. handbook chapters by Thompson/Soni/Weatherill); not retrieved.
- Eriksson's 1982 TFI paper (AIAA J.) and Gordon & Hall 1973 are well-known foundations. Search confirmed only Gordon & Hall: "Construction of curvilinear co-ordinate systems and applications to mesh generation", IJNME 7:461–477 (1973), and "Transfinite element methods: blending-function interpolation over arbitrary curved element domains", Numer. Math. 21:109–129 (1973) — [doi 10.1007/BF01436298](https://www.doi.org/10.1007/BF01436298). I found no verifiable link for the Eriksson citation.
