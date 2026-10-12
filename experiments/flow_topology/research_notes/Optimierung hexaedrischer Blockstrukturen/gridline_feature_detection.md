# Grid-line continuity, feature curves and singularities in hexahedral block structures

Scope: generic (geometry-independent) theory and algorithms that, given a hex block decomposition / hex mesh plus a boundary with patch labels, decide (a) which edges continue as one grid line through a vertex, (b) which edges lie on feature curves / patch boundaries, (c) how singularities are treated. Primary sources were read in full text where possible (Pietroni et al. 2022 survey, Gao et al. 2017, Brückler et al. 2022, Lyon et al. 2016 HexEx, Liu et al. 2018 SCOF).

## Q1: What is the base complex, how is it extracted, and how are grid lines / sheets / chords defined combinatorially?

### Takeaway
Everything is built on purely combinatorial "opposite" relations. At a regular interior vertex (6 edges) each edge has exactly one opposite edge (the one sharing no face with it). At a regular interior edge (4 faces) each face has exactly one opposite face. Grid lines are maximal chains of opposite edges. Sheets and chords are the equivalence classes of the dual opposite relations. The base complex (the coarsest conforming block decomposition) is found by flooding from all faces touching singular edges through opposite faces across regular edges until the flood stops.

### Cited Findings
- **Regularity (counts).** In a Cartesian grid every vertex, edge and face touches 8, 4 and 2 hexes. An interior edge is singular if its valence k (number of incident hexes) is not 4. A boundary edge is singular if k is not 2. Interior faces can never be irregular, and vertex singularities are never isolated, so the singular edges, which form the "singularity graph", carry the structure. — [Pietroni et al. 2022 survey §2.1.1](https://arxiv.org/pdf/2202.12670)
- **Edge index (Liu et al. 2018).** idx(e) = (4 − val(e))/4 for interior edges and (2 − val(e))/4 for boundary edges. The practically relevant boundary edge valences are 1, 2, 3 and 4 (convex, flat, concave, turning). — [Liu et al. 2018, SCOF](https://cgg.unibe.ch/media/papers/SCOF_SG2018.pdf)
- **Vertex types.** Hex vertex configurations correspond 1:1 to triangulations of the sphere (Nieser et al. 2011). With edge valences {3,4,5} there are only 11 interior vertex types. An interior vertex cannot have exactly one singular edge, and if it has two they must be of the same type. Singular arcs are therefore chains of singular edges of the same valence that end on the boundary or at singular vertices. — [Pietroni et al. 2022 §2.1.1](https://arxiv.org/pdf/2202.12670)
- **Boundary vertex types.** Boundary vertices correspond to triangulations of a disc (a hemisphere cut). Limiting boundary edge valences to 1–4, interior edge valences to 3–5, and to fewer than 9 incident hexes gives 237 topologically different singular boundary vertex types. — [Liu et al. 2018](https://cgg.unibe.ch/media/papers/SCOF_SG2018.pdf)
- **Opposite elements (the G1 / grid-line rule).** "At each regular interior vertex v, there are 6 incident edges. For each edge e1 of these, there is exactly one edge e2 among these 6 that does not share a facet with e1; the edges e1 and e2 are called opposite at v." Faces at a regular edge work the same way: of 4 faces, the opposite face is the one that shares no cell. The survey states that the opposite-edge relation is what algorithms use "that trace internal arcs in the mesh", and the opposite-face relation is what they use to "flood internal facet sheets bounded by singular arcs". — [Pietroni et al. 2022 §2.2](https://arxiv.org/pdf/2202.12670)
- **Sheets and chords (dual definitions).** Sheets are the equivalence classes of the transitive closure of the dual-face opposite relation. They are also called twist planes or (pseudo-)hyperplanes. They are 2-manifolds, possibly self-intersecting, of arbitrary genus. Chords (or polychords) are the classes of the dual-edge opposite relation. Dual vertices are where 3 sheets cross, and chords are where 2 sheets cross. A sheet is one layer of hexes, and a chord is one column of hexes. — [Pietroni et al. 2022 §2.2, §5.1](https://arxiv.org/pdf/2202.12670)
- **Base complex: definition and extraction.** The base complex is the coarsest conforming block decomposition (Bommes et al. 2011; Gao et al. 2015). Its defining property: "a facet is part of a block side if and only if it is transitively incident to a singular edge via opposite facets". Extraction works by "starting from all facets incident to any singular edge, iteratively expanding through opposite facets across regular edges until termination". — [Pietroni et al. 2022 §2.3](https://arxiv.org/pdf/2202.12670)
- **Base complex as separation surfaces (Gao et al.).** "From a singularity with valence n, n separation surfaces can be traced out and either meet with other singularities or terminate at the surface boundary." These surfaces divide the volume into components that are topologically cubes, or in degenerate cases tori. The faces of a component are called "patches". — [Gao et al. 2017, Robust structure simplification](https://cims.nyu.edu/gcl/papers/Robust-Hex-2017.pdf)
- **Base-complex sheets and chords (block-level).** Two base-complex edges in one patch are parallel if they share no vertex. The parallel edge set is the transitive closure of this relation. A base-complex sheet is all components that have an edge in the same parallel edge set. Likewise, two patches of the same component are parallel if they share no vertex, and a base-complex chord is all components that have a patch in the same parallel patch set. — [Gao et al. 2017 §3.1](https://cims.nyu.edu/gcl/papers/Robust-Hex-2017.pdf)
- **Base complex vs. motorcycle complex.** In 2D, particles start from each singularity along each iso-line direction. A particle stops at (i) an existing trace, (ii) a singularity, or (iii) the boundary; the result is the motorcycle graph. If rule (i) is ignored, the result is the base complex, so the motorcycle graph is a subgraph of the base complex. In 3D the authors model this as a "brush fire": it starts on the singular curves, spreads within iso-surfaces, and cannot cross burnt regions or the boundary. The result is a non-conforming (T-joint) partition into regular cuboid blocks. Example: 79 instead of 1352 blocks (5.8%). — [Brückler, Gupta, Mandad, Campen 2022, 3D Motorcycle Complex](https://arxiv.org/pdf/2112.05793)
- **Singularity in a parametrization.** An edge is regular if its incident parametric dihedral angles sum to 2π (interior) or π (boundary). Otherwise it is singular. In an integer-grid parametrization, sheets of quads coincide with iso-surfaces. — [Brückler et al. 2022 §3.1](https://arxiv.org/pdf/2112.05793)
- **Block-level simplification.** Removing a single hex sheet may leave the block layout unchanged, and collapsing a hex chord may even make it more complex. Removing a base-complex sheet or chord, in contrast, reduces the component count monotonically. Each collapse is validated against five rules:
  1. Euler characteristic of both the surface and the base complex is preserved.
  2. The result stays manifold.
  3. Edge valence never drops: τ(e′) ≥ min(τ(e1), τ(e2)), predicted from the neighbouring components.
  4. Boundary conformity: no collapse that merges two corners, two different feature curves or two different boundary patches.
  5. A Hausdorff bound (1% of the bounding box).

  Candidates are ranked by width (thinnest first). — [Gao et al. 2017 §3.3–3.4](https://cims.nyu.edu/gcl/papers/Robust-Hex-2017.pdf); [Pietroni et al. 2022 §5.5](https://arxiv.org/pdf/2202.12670)
- **Follow-up work.** Xu et al. 2021 improve the sheet/chord ranking. Gao et al. 2015 correct misaligned singularities by sheet removal, since the same singularities can produce very different base complexes. Shen et al. 2021 edit singularities by collapsing and inserting chords. — [Pietroni et al. 2022 §5.5](https://arxiv.org/pdf/2202.12670)
- **Fun-sheets.** Ledoux & Shepherd 2010 (Eng. w. Computers 26(4)) classify the sheets that capture geometry as "fundamental sheets" (fun-sheets). These link the global layered structure to the captured geometric surfaces. — [search summary of Ledoux & Shepherd 2010 / survey](https://arxiv.org/pdf/2202.12670)

### Inferences
- For a block structure (not a fine mesh) the same rules apply to block vertices and edges. A block edge continues "straight" through a block vertex into its opposite edge if and only if the vertex is regular in the relevant sense: 6 incident block edges with the opposite relation defined via shared block faces (interior), or the boundary equivalent (see Q3). A "grid line" is a maximal chain under this relation. It ends at singular vertices or at the boundary.
- Gao's block-level parallel-edge sets are exactly the block-structure counterpart of sheets. They are computed purely from face/edge incidences (no geometry, no labels), which matches the "generic method" requirement.

### Gaps
- I did not obtain the full text of Gao et al. "Structured volume decomposition via generalized sweeping" (TVCG 2016) or Gao et al. 2015 "Hexahedral mesh re-parameterization from aligned base-complex". Their details are known only through the survey.
- I did not read the original text of Ledoux & Shepherd 2010; its description comes from search snippets.

## Q2: How do frame-field / integer-grid-map methods (HexEx, AlgoHex) encode feature alignment and singularities, and what continuity guarantees result?

### Takeaway
An integer-grid map (IGM) stores, per tet, an affine map to R³. Neighbouring charts are related by transitions that are a cube rotation (one of 24) plus an integer translation. Singular edges must map onto integer lines, and singular vertices onto integer points. Grid lines are integer iso-lines, so they continue through every regular point automatically. Feature alignment is a linear constraint (an edge or face stays constant in 2 or 1 coordinates). Guarantees are conditional: a flip-free, meshable map yields aligned and conforming grid lines, but nothing guarantees such a map exists or will be found.

### Cited Findings
- **IGM conditions.** (i) Grid-preserving transitions: integer translations plus cube symmetries. (ii) Local injectivity. (iii) Singularities and boundaries map to integer-grid entities. IGMs can describe every hex mesh. — [Pietroni et al. 2022 §2.5](https://arxiv.org/pdf/2202.12670)
- **HexEx constraints A1–A4.** A1: g_ij(u) = Π_ij·u + t_ij with Π in the chiral cubical group of 24 and t in Z³. A2: singular edges map onto integer lines. A3: "points incident to other than two singular edges" map to integer points. A4: every tet image has positive volume. An edge is singular if the transition accumulated around it is not the identity. HexEx sanitizes the parameters, extracts vertices and "darts" at grid intersections, and connects darts by tracing in the parametrization, which yields the hex topology. Pairs of darts in flipped regions cancel. — [Lyon, Bommes, Kobbelt 2016, HexEx](https://www.graphics.rwth-aachen.de/media/papers/HexEx_lowres.pdf)
- **Pipeline.**
  1. Boundary-aligned smooth frame field (24-fold octahedral symmetry, often a spherical-harmonics representation).
  2. IGM fitted to the field (Nieser et al. 2011).
  3. Integer level-set extraction (HexEx).

  The singularities of the field define the singularity structure of the mesh. — [Pietroni et al. 2022 §4.8](https://arxiv.org/pdf/2202.12670)
- **Meshability.** Smooth frame fields often contain singularities that cannot occur in a hex mesh. A typical example is 3-5 singularities. Liu et al. 2018 generate fields with a prescribed hex-meshable singularity graph. Common invalid nodes are "turning points" where an arc switches from valence 3 to 5 and returns. — [Pietroni et al. 2022 §4.8.3](https://arxiv.org/pdf/2202.12670); [Liu et al. 2018](https://cgg.unibe.ch/media/papers/SCOF_SG2018.pdf)
- **Boundary sector constraint (SCOF).** At a boundary singular node, the surface triangles around it are split into sectors by the singular boundary edges. Each sector is convex, flat, concave or a turning point, which corresponds to a parametric interior angle of k·π/2 with k = 1..4. This is the IGM-level encoding of "which boundary edges continue and with what angle". — [Liu et al. 2018](https://cgg.unibe.ch/media/papers/SCOF_SG2018.pdf)
- **Alignment.** A face (edge) is aligned if its image is constant in one (two) coordinates. Seamless parametrization uses "alignment constraints along the boundary and feature edges". Singular edges must be aligned too. — [Brückler et al. 2022](https://arxiv.org/pdf/2112.05793)
- **No guarantees.** Field-guided IGMs are not guaranteed to be flip-free. HexEx's heuristics recover some invalid maps, but "no general guarantees are available". — [Pietroni et al. 2022 §4.8.4](https://arxiv.org/pdf/2202.12670)
- **AlgoHex.** Four stages:
  1. A feature-aligned smooth frame field.
  2. A locally meshable frame field (Liu & Bommes 2023).
  3. Parameterization (seamless map plus quantization).
  4. HexEx extraction.

  The input is a tet mesh "with user-specified feature tags" (VTK or OVM). Features are not detected by AlgoHex itself. The output "may be an incomplete hexahedral mesh". — [AlgoHex GitHub](https://github.com/cgg-bern/AlgoHex)
- **Locally Meshable Frame Fields.** Liu & Bommes 2023 (TOG 42(4)) raised the success rate of producing a valid IGM on HexMe from 2% to 58%. — [AlgoHex search summary / cgg.unibe.ch](https://boris.unibe.ch/194871)
- **Feature fidelity in practice.** HexMe has 189 tet meshes with features tagged from CAD B-rep via Gmsh physical groups. In the frame-field/IGM pipeline only 19/189 cases had all features correct. Average success was 70.9% for points, 48.5% for curves and 34.6% for surfaces. A curve counts as correct if a chain of hex edges connects its two feature-point endpoints, stays within τ = h, and follows the tangents. — [Beaufort et al. 2022, Hex Me If You Can](https://pmc.ncbi.nlm.nih.gov/articles/PMC9828569/)
- **Polycube maps.** Sharp creases are preserved only if they map to integer iso-lines. "A convex vertex with more than three incoming feature lines cannot be correctly meshed". Labelling methods (Livesu et al. 2013; Hu & Zhang 2016) promote features implicitly but often fail to keep them. — [Pietroni et al. 2022 §4.7](https://arxiv.org/pdf/2202.12670)

### Inferences
- In IGM methods, "G1 continuity of grid lines" is not decided after the fact. It holds by construction: wherever the transition around an edge is the identity, iso-lines continue. Singular edges are exactly the places where the holonomy is not the identity.
- For a block structure that comes from AlgoHex and has been coarsened, the block-edge continuity rule from Q1 (opposite edges) is the discrete equivalent of iso-line continuation.

### Gaps
- I did not read the full text of Liu & Bommes 2023 (exact local meshability conditions) or of the AlgoHex feature-tag file format.
- I found no formal statement of a G1 (tangent-continuity) guarantee for the geometric embedding. The guarantees are combinatorial or parametric (straight in the φ-metric), not geometric G1 in physical space.

## Q3: Which topological rules decide G1 continuation at a vertex, and how are ambiguous or singular vertices treated?

### Takeaway
The rule is opposite-edge continuation at regular vertices. An edge is regular at valence 4 (interior) or 2 (boundary). At singular vertices or edges, continuation is undefined: lines (arcs) end there, and the singular vertex becomes a block corner. On the boundary, the parametric sector angle (k·π/2), or equivalently the boundary edge valence, decides whether a line turns or runs straight.

### Cited Findings
- **Opposite edge.** Defined only at regular interior vertices, as the unique edge sharing no face. In the dual mesh every vertex and edge is regular, which is why sheet and chord tracing always works there. — [Pietroni et al. 2022 §2.2](https://arxiv.org/pdf/2202.12670)
- **Singular arcs.** Chains of same-valence singular edges. They end at the boundary or at singular vertices, defined as "incident to other than two singular edges" in HexEx. — [Pietroni et al. 2022](https://arxiv.org/pdf/2202.12670); [Lyon et al. 2016](https://www.graphics.rwth-aachen.de/media/papers/HexEx_lowres.pdf)
- **Boundary edge valence and geometry.** A boundary edge of valence 1 is a convex crease (about 90°), 2 is flat, 3 is concave (about 270°), and 4 is a turning point. — [Liu et al. 2018](https://cgg.unibe.ch/media/papers/SCOF_SG2018.pdf)
- **Valence couples to angle.** "The number of hexahedra incident to a singular edge directly bounds the inner angles." — [Pietroni et al. 2022 Fig. 7](https://arxiv.org/pdf/2202.12670)
- **Motorcycle tracing rules.** Traces start at singularities, follow iso-lines and stop at traces, singularities or the boundary. In 3D, the T-joint configurations (three walls, and two interacting T-joints) are the non-conforming cases. — [Brückler et al. 2022](https://arxiv.org/pdf/2112.05793)
- **Collapses must not create degenerate edges.** Simplification must not create valence-1 or valence-2 interior edges, because these lead to negative Jacobians. — [Gao et al. 2017 §3.3](https://cims.nyu.edu/gcl/papers/Robust-Hex-2017.pdf)
- **Surface vertex types (Cubit).** End (0–135°, 1 element), Side (136–224°, 2 elements), Corner (225–314°, 3 elements), Reversal (315–360°, 4 elements). Map, submap and trimesh use these to place corners and to group curves into logical i-j sides. — [Cubit 15.8 docs, surface vertex types](https://sandia.gov/files/cubit/15.8/help_manual/WebHelp/mesh_generation/meshing_schemes/traditional/surface_vertex_types.htm); [Cubit submap](https://www.sandia.gov/files/cubit/16.02/help_manual/WebHelp/mesh_generation/meshing_schemes/traditional/submap.htm)

### Inferences
- **Practical rule set for a block structure.** Each item below follows from the cited definitions and uses topology only.
  - Interior block vertex with 6 block edges and 8 incident blocks, where all incident edges are regular (valence 4): each edge continues into its opposite.
  - Boundary vertex: regular if it has 4 incident blocks and 5 edges, one of which leaves the boundary. Its 4 boundary edges pair up as opposites, and the edge leaving the boundary continues nowhere.
  - Any other count: singular. No continuation; the vertex is a block corner and an end of a grid line.
- Applying the Cubit angle bins on the boundary turns geometry into an expected boundary valence. This makes it possible to check whether the topological valence matches the geometric angle.
- **Ambiguous cases.** Self-adjacent blocks, and tori in Gao's base complex, break uniqueness. These need detection rather than a rule.

### Gaps
- I found no source with an explicit algorithm for the combined geometric+topological "G1 test" on coarse block structures that compares a curve tangent with the opposite edge. In the literature this is implicit in the parametrization (straightness in the φ-metric).

## Q4: How is feature-curve detection (dihedral angle, patch boundaries, corners) handled in hex meshing pipelines?

### Takeaway
Two strategies are used. (1) Take features from CAD/B-rep or label boundaries: Gmsh physical groups, HexMe, AlgoHex input tags. (2) On triangle soups, threshold the dihedral angle, then classify vertices into corner / feature-line / surface by the number of incident feature edges. Transferring features onto an existing hex topology is a heuristic, order-dependent shortest-path problem.

### Cited Findings
- **Gao et al. 2017.** Sharp features are all edges with dihedral angle < 140°. Following Livesu et al. 2015, boundary vertices are classified as corner (fixed), feature line (slides along its tangent) or regular (stays on its tangent plane). Collapses are forbidden if they would merge two corners, vertices on two different feature curves, or vertices on two different boundary patches. — [Gao et al. 2017 §3.2–3.3](https://cims.nyu.edu/gcl/papers/Robust-Hex-2017.pdf)
- **Gmsh classifySurfaces(angle, includeBoundary, forReparametrization, curveAngle).** Splits a discrete surface along edges whose triangle-to-triangle angle exceeds `angle`, creating discrete surfaces, curves and points. `curveAngle` additionally splits curves at corners, and forReparametrization forces patches that can be parametrized. — [Gmsh search summary, t13 / API](https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_8_4/tutorial/t13.geo); [Gmsh mailing list](https://onelab.info/pipermail/gmsh/2020/013543.html)
- **HexMe.** Features come from CAD B-rep entities, not from sharp-edge detection. They become 0D, 1D and 2D tags. — [Beaufort et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9828569/)
- **Feature transfer onto a hex topology.** Feature transfer is "primarily a topological problem": each feature line must become a chain of edges without spurious overlaps. Methods insert one curve at a time, project its endpoints to the nearest vertices, and run Dijkstra on a distance-to-feature field (Gao et al. 2019). Greedy order "may lead to catastrophic results" (Born et al. 2021), and conflicts can leave a feature without any path. — [Pietroni et al. 2022 §4.6, §8](https://arxiv.org/pdf/2202.12670)
- **Padding.** Padding ensures that each element has at most one feature edge. — [Pietroni et al. 2022 §4.6](https://arxiv.org/pdf/2202.12670)
- **Requirement statement.** "Feature alignment requires that sequences of edges of the hex-mesh conform to feature curves, otherwise some deviation is inevitable, regardless of resolution." — [Pietroni et al. 2022 §4.2](https://arxiv.org/pdf/2202.12670)
- **Dual sheet-based methods.** These capture geometry via fun-sheets (Ledoux & Shepherd). Sheet insertion is used "to capture analytic features" (Merkley et al. 2007). — [Pietroni et al. 2022 §4.4, §5.1](https://arxiv.org/pdf/2202.12670)

### Inferences
- When patch labels are given (as with labelled CAD/STL patches), feature curves follow from the labels alone, with no thresholds: a curve is an edge between two different patch labels, and a corner is a vertex touching 3 or more labels or 3 or more feature edges. This is the generic, threshold-free route that HexMe/AlgoHex/Gmsh physical groups use. A dihedral threshold (around 140°) is only a fallback for unlabelled creases inside one patch.
- An edge of a hex or block structure "lies on a feature curve" if both of its boundary vertices are classified on the same curve and the edge's boundary face(s) border two different patch labels. Gao's rule (never merge vertices of different curves or patches) is the matching validity check.

### Gaps
- I did not check the exact Cubit "feature angle" defaults for automatic curve detection, beyond the vertex-type angle bins.
- I did not retrieve the original Livesu et al. 2015 or Gao et al. 2019 feature-mapping texts. Their content is known only via the survey and Gao 2017.
- I did not cover advances in 2024–2026 (e.g. newer feature-aligned polycube or neural hex methods). Time limits allowed only a survey-level search, and I found no source newer than 2023 in this pass.
