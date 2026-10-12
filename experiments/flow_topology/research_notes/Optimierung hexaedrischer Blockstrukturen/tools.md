# Tools for block-structured hex meshing with curved block edges (turbomachinery focus)

Scope note: about 25 tool calls (web search/fetch plus inspection of the local dtOO checkout at `/dtOO`, last commit 2026-09-15 "Merge pull request #98 from ihs-ustutt/improve_firstDer"). Several vendor manuals (Ansys help, Cadence/Pointwise docs) only returned table-of-contents pages or 403s, so some vendor details are listed under Gaps. Statements cited to "local source" are from reading `/dtOO/src` headers in this container.

## Q1: How does each tool represent block edges, and how does it make edges on CAD features follow them exactly?

### Takeaway
There are three families. (a) Association/projection tools (ICEM Hexa, Pointwise, blockMesh/classy_blocks, HexBox): a block edge is a topological entity, either projected onto a CAD curve or surface or given as an explicit curve (arc/spline/polyLine). Edges on features are kept exactly by association. (b) Topology-plus-dynamic-boundary-conforming tools (GridPro): only corner-to-surface assignments are given and the grid is driven onto the boundaries. (c) Template/parametric tools (TurboGrid ATM, AutoGrid5, dtOO): block edges are generated from the parametric blade geometry, e.g. dtOO's analytic maps and transfinite volumes, so feature fidelity comes from construction, not from association.

### Cited Findings
**ICEM CFD Hexa (Ansys, commercial)**
- Block topology and mesh can both be smoothed. "The geometry and its associative surfaces, curves, and points are all constraints when smoothing the block topology model". Block smoothing happens before mesh generation, mesh smoothing after edge bunching is set — [Ansys ICEM CFD User Guide, Smoothing Techniques (v251)](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/icm_user/iuser_smoothtech.html)
- Practitioner report: smoothers rely on edge-to-curve association to know which features to keep. Unassociated edges let nodes slide over surfaces and cause problems — [CFD-Online ICEM thread](https://www.cfd-online.com/Forums/ansys-meshing/86593-how-generate-mesh-operating-plain-flap-next-thread.html) (forum, secondary)

**OpenFOAM blockMesh (open source, GPL)**
- Edge types: `arc` (1 interpolation point), `simpleSpline`, `polyLine`, `polySpline`, `line` — [OpenFOAM.com user guide 4.3](https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility)
- Geometry projection: `searchableSurface` entries under `geometry`, and `project (v0 v1 v2 v3) surface` for faces (vertices and edges can also be projected). Without face projection, "block edges conform to the sphere but block faces lie somewhere between the original cube and the spherical surface", because the block interior is an edge-based transfinite interpolation. Labelled "experimental" in the OpenFOAM-dev sphere tutorial (OpenFOAM-5.x era) — [OpenFOAM-5.x mirror, blockMesh/sphere tutorial](https://1gnis-git.duckdns.org/research/OpenFOAM-5.x/src/commit/80123f59cea5e8b18201e55b0c4825bae7e099cb/tutorials/mesh/blockMesh/sphere) (unofficial git mirror)

**classy_blocks (MIT, Python, damogranlabs)**
- Python library that writes blockMeshDict. Edge classes: Arc/Origin/Angle/Spline/PolyLine/Project. Vertices, edges and faces can be projected to STL or searchable surfaces, or to intersections of several surfaces — [GitHub damogranlabs/classy_blocks](https://github.com/damogranlabs/classy_blocks)
- Building operations: Loft, Extrude, Revolve, Wedge, Connector, plus "automatic helpers for building and optimization of block-structured hexahedral meshes" — [PyPI classy-blocks](https://pypi.org/project/classy-blocks/1.6.4)

**GridPro (Program Development Co., commercial)**
- The topology is a wireframe of block corners with their connectivity, surface assignments and initial corner positions. "The action of assignment does not move any of the wireframe corners". Dynamic boundary conforming (DBC) moves the non-conforming initial grid until it conforms to all region boundaries — [ResearchGate: Multi-block grid generation around airfoils](https://www.researchgate.net/publication/323005088_MULTI-BLOCK_GRID_GENERATION_AROUND_AIRFOILS); [GridPro product page](https://gridpro.com/product/gridpro)
- Users work in the Az-Manager, which combines a scripting language with a GUI. Founded 1989 by Peter Eiseman for NASA Glenn; used on a turbine blade with several hundred cooling holes — [NASA Spinoff](https://spinoff.nasa.gov/node/10258); [ARL HPC software description](https://arl.hpc.mil/software/description.html?sw=GridPro)

**TurboGrid (Ansys, commercial)**
- "The topology is a structure of blocks that acts as a framework for positioning mesh elements". ATM Optimized and ATM3D topologies are offered alongside traditional topologies — [TurboGrid User Guide 10.4 Topology](https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/tg_user/i1101982.html)
- ATM Optimized "enables you to create high-quality meshes with minimal effort; there is no need for control point adjustment". Workflow: load blade, set Topology Set method, Apply. TurboGrid computes "master topology, topology, and refined mesh" for the hub and shroud layers — [TurboGrid User Guide 10.4.2 ATM Basic Usage (v261)](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/tg_user/tg_atm_chapt.html)

**AutoGrid5 (NUMECA → Cadence Fidelity, commercial)**
- Blade-row topologies include "Default" and "HOH", suited to axial machines. With Default the tip-gap patches usually stay fully matched; with HOH the tip-gap blocks may not match — [CFD-Online Fidelity/NUMECA forum](https://cfd-online.com/Forums/fidelity-cfd/212857-full-matching-mesh.html) (forum, secondary)

**Pointwise / Cadence Fidelity Pointwise (commercial)**
- Elliptic solver boundary conditions: "Floating" lets boundary points move "as though they were interior grid points". It only applies when Pointwise finds a unique neighbouring grid among the selected ones, otherwise it falls back to Fixed. Thomas-Middlecoff control mainly carries boundary clustering into the interior and gives little orthogonality control. Steger-Sorenson-type control is used for boundary orthogonality — [Pointwise User Manual, boundary conditions](https://pointwise.com/doc/user-manual/grid/solve/structured-domains-and-blocks/edge-attributes-tab/boundary-conditions.html) (from search snippet; the page now redirects to cadence.com, which returned 403)

**dtOO (University of Stuttgart IHS)**: see Q4. Block edges and faces are analytic maps (`map1dTo3d`, `map2dTo3d`), and volumes are transfinite interpolations of six bounding surfaces (`trans6SidedCube`, `transIntCube`, "Transfinite interpolation of a cube bounded by six surface mappings") — local source `/dtOO/src/geometryKernel/analyticFunctionHeaven/transIntCube.h`, `/dtOO/src/geometryKernel/analyticGeometryHeaven/trans6SidedCube.h`

**HexBox (open source, research)**
- Interactive "box modelling" of hex meshes: subdivision, extrusion, removal, cloning, conformal or non-conformal grids. Vertices can be placed manually, smoothed automatically or projected onto an external target surface. Uses a hex mesh plus a DAG of operations. Released open source — [HexBox project page (Univ. Cagliari)](https://cg3hci.dmi.unica.it/lab/tools/hexbox/); [CGF 2023 paper, EG diglib](https://diglib.eg.org/handle/10.1111/cgf14899)

### Inferences
- Association-based tools (ICEM, blockMesh/classy_blocks) give exact feature following only for the edges explicitly associated or projected. The interior is TFI from the edges, so faces need projection too (blockMesh sphere example). This bears directly on blade reinsertion: blade-cut edges have to be bound to the blade curves explicitly.
- GridPro differs in kind. Its edges are not specified; corners float and the grid is pulled onto boundaries. This resembles giving only the topology plus surface labels, as a learned topology generator would produce.

### Gaps
- Exact ICEM edge-spline representation ("convert edge to spline", "link shape") could not be fetched from Ansys docs (only the summary page loaded).
- TurboGrid ATM internals (how the O-grid is built, what is optimised) are not described on the publicly fetched pages; subsections 10.4.x were not reachable.
- AutoGrid5 user-guide details (smoothing steps, orthogonality controls) were only found on Scribd copies, which were not used.
- Construct2D and Glyph scripts were not researched within the call budget.

## Q2: What block-level smoothing or optimisation does each tool provide, and with which objectives?

### Takeaway
Block-level optimisation, i.e. moving block corners or edges before meshing, exists in ICEM (block topology smoothing constrained by associations), GridPro (smoothing that drives floating corners and the whole grid to be smooth and orthogonal), TurboGrid ATM (automatic, no control points) and classy_blocks (treating blocks as single cells and optimising vertices against OpenFOAM cell-quality criteria with 1, 2 or 3 DoF). Pointwise, Gmsh, Cubit and dtOO's gmsh operators smooth at mesh level (elliptic PDE, Laplacian, mean ratio, condition number, HO untangling).

### Cited Findings
- **ICEM**: block topology smoothing before meshing, with associated surfaces, curves and points as constraints. Quality criteria are those of the "Pre-Mesh Quality" section — [Ansys ICEM Smoothing Techniques](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/icm_user/iuser_smoothtech.html)
- **GridPro**: the grid is "optimized to be smooth and orthogonal throughout the volumetric region and to be clustered about the locations of concave and convex boundary curvature". Corners are mapped onto boundaries iteratively — [ResearchGate: Multi-block grid generation around airfoils](https://www.researchgate.net/publication/323005088_MULTI-BLOCK_GRID_GENERATION_AROUND_AIRFOILS); [GridPro](https://gridpro.com/product/gridpro). GridPro v9.0 added an "advanced block topology smoother", a topology macro for axisymmetric geometries and a 3D block-topology array — [Digital Engineering 24/7](https://www.digitalengineering247.com/article/program-development-unveils-gridpro-v9.0/cfd); [develop3d](https://develop3d.com/simulation/gridpro-v9-0-enhances-mesh-quality-for-cfd/)
- **TurboGrid**: ATM Optimized removes the need for control-point adjustment and produces the master topology, the topology and the refined mesh per layer automatically — [TurboGrid 10.4.2](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/tg_user/tg_atm_chapt.html)
- **Pointwise**: elliptic PDE smoothing with Floating, Fixed and other boundary conditions. Thomas-Middlecoff gives interior clustering, Steger-Sorenson gives boundary orthogonality — [Pointwise manual](https://pointwise.com/doc/user-manual/grid/solve/structured-domains-and-blocks/edge-attributes-tab/boundary-conditions.html)
- **classy_blocks**: an "Optimizer" treats each block as a single cell. Vertices move freely (3 DoF), along curves (1 DoF) or on surfaces (2 DoF), and the objective is "OpenFOAM's cell quality criteria". Sketch points can use Laplacian smoothing or quality optimisation. Invalid or inverted blocks show up in debug.vtk — [GitHub classy_blocks](https://github.com/damogranlabs/classy_blocks)
- **Cubit / Coreform Cubit**: sweep smoothing options are auto, smart affine, linear, residual and winslow (winslow keeps biased source meshes). Volume smoothers are condition number and mean ratio. Smart Laplacian uses Mean Ratio and guarantees the minimum Mean Ratio does not degrade, while the Mean Ratio smoother raises the average — [Cubit 16.06 sweep docs](https://www.sandia.gov/files/cubit/16.06/help_manual/WebHelp/mesh_generation/meshing_schemes/traditional/sweep.htm); [Cubit 16.02 Smart Laplacian](https://www.sandia.gov/files/cubit/16.02/help_manual/WebHelp/mesh_generation/mesh_modification/mesh_smoothing/smart_laplacian.htm)
- **Gmsh**: HO untangling by unconstrained optimisation in which element Jacobians are kept in a prescribed range via moving log-barriers. It works for surface, volume, hybrid and boundary-layer meshes and is implemented in Gmsh. Provable Jacobian-determinant bounds check validity — [Gmsh untangling preprint](https://gmsh.info/doc/preprints/gmsh_untangling_preprint.pdf); [HAL 01299472](https://hal-mines-paristech.archives-ouvertes.fr/hal-01299472)
- **pyHyp**: hyperbolic marching (theory from Chan and Steger) that extrudes structured surface meshes, typically from ICEM, into volume meshes for ADflow — [pyHyp docs](https://mdolab-pyhyp.readthedocs-hosted.com/); [MACH-Aero tutorial](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/machAeroTutorials/aero_pyhyp.html)
- **HexBox**: automatic smoothing plus projection onto a target surface — [HexBox](https://cg3hci.dmi.unica.it/lab/tools/hexbox/)
- **dtOO**: the elliptic generator for block-structured CFD meshes is described in the CSME-CFDSC-CSR 2025 paper abstract and summary (see Q4). The source includes face operators `dtMeshLaplacianGFace` (a transfinite-2D operator with Laplacian smoothing), `dtMeshTransfiniteGFace`, `dtMeshGFaceWithTransfiniteLayer`, `dtMeshCustom2x3TransfiniteGFace`, `dtOptimizeMeshGRegion` and `dtHybridOptMeshGRegion` — local source `/dtOO/src/libraryKernel/meshEngine/`

### Inferences
- Objectives fall into three groups. Smoothness plus orthogonality is the elliptic or variational group (GridPro, Pointwise, Winslow). Shape quality is the algebraic group (mean ratio, condition number, OpenFOAM non-orthogonality and skewness in classy_blocks). Validity is the Jacobian-bound group (Gmsh HO).
- classy_blocks' block-as-cell optimiser with curve and surface DoF constraints is the closest open-source analogue to GridPro corner floating or ICEM block smoothing. It is the best reference for a "block corner optimisation" step after a learned topology.

### Gaps
- GridPro's exact energy functional is proprietary and not documented publicly. The GridPro HO presentation at MIT HOW5 was too large to fetch.
- The objective function of TurboGrid ATM is not publicly documented.
- ICEM block-smoothing algorithm names (e.g. orthogonality, Laplace) and the Pre-Mesh Quality metrics could not be confirmed from fetched pages.

## Q3: Which tools are scriptable or open source and usable as a reference or back-end in a Python/Linux pipeline?

### Takeaway
Open source and Python-friendly: dtOO (C++ with a Python interface, scientific licence), classy_blocks (MIT) driving OpenFOAM blockMesh (GPL), Gmsh, pyHyp (MDO Lab) and HexBox (open source, interactive). Commercial but scriptable: ICEM (Tcl replay), Pointwise (Glyph/Tcl), GridPro (Az-Manager scripting language), TurboGrid (CCL/batch, not verified here), AutoGrid5 (Python, not verified) and Cubit (Python/journal; Coreform Cubit is commercial).

### Cited Findings
- dtOO is a scientific C++ library with a Python interface, coupled to single- and multi-objective evolutionary algorithms and running on large clusters "free of license costs" — [Tismer & Riedelbauch, CSME-CFDSC-CSR 2025](https://espace2.etsmtl.ca/id/eprint/32512/); [dtOO docs](https://ihs-ustutt.github.io/dtOO/)
- The dtOO licence is a "Scientific Licence under dual licensing model" for research and teaching — local `/dtOO/LICENSE.txt`; [GitHub ihs-ustutt/dtOO](https://github.com/ihs-ustutt/dtOO) (README: "custom license")
- classy_blocks is MIT-licensed and installed with `pip install classy_blocks` — [GitHub](https://github.com/damogranlabs/classy_blocks)
- pyHyp is part of the MDO Lab MACH-Aero framework — [pyHyp docs](https://mdolab-pyhyp.readthedocs-hosted.com/)
- HexBox is "released as open source" — [HexBox](https://cg3hci.dmi.unica.it/lab/tools/hexbox/)
- GridPro has the Az-Manager scripting language and GUI. GridPro's automation page promotes it for parametric shape optimisation — [NASA Spinoff](https://spinoff.nasa.gov/node/10258); [GridPro parametric shape optimization](https://gridpro.com/solutions/automation/meshing-for-parametric-shape-optimization)
- Cubit sweep and smoothing schemes are documented in the Sandia manual — [Cubit docs](https://www.sandia.gov/files/cubit/16.06/help_manual/WebHelp/mesh_generation/meshing_schemes/traditional/sweep.htm)

### Inferences
- For an automated Linux pipeline that already uses dtOO and Gmsh, two back-ends need no new licences. (1) dtOO/gmsh transfinite operators keep exact dtOO geometry for blade reinsertion. (2) classy_blocks' optimiser, adapted, gives block-vertex optimisation with curve and surface constraints. OpenFOAM blockMesh's `project` edges and faces are a usable exact-on-surface fallback.
- The commercial tools (GridPro, TurboGrid ATM, ICEM block smoothing) are best used as conceptual references, especially GridPro's floating corners.

### Gaps
- Licensing and scripting details for TurboGrid (CCL/session files), AutoGrid5 (Python scripting) and Pointwise Glyph/Python were not confirmed from primary sources in this pass.
- Construct2D (2D airfoil O/C-grid, GPL per common knowledge) is not verified.

## Q4: What is known about dtOO specifically?

### Takeaway
dtOO ("(d)esign (t)ool (O)bject-(O)riented") is the IHS Stuttgart hydraulic-machinery design framework by Alexander Tismer, with Stefan Riedelbauch. It is C++ with Python, built on OpenCASCADE geometry and Gmsh meshing, with a direct OpenFOAM interface. It builds block-structured meshes from analytic maps through transfinite interpolation and an elliptic generator. It is on GitHub (ihs-ustutt/dtOO) and actively developed as of Sept 2026.

### Cited Findings
- "(d)esign (t)ool (O)bject-(O)riented (dtOO), a framework to design hydraulic machines". The repository has about 2,721 commits, a dtOO-ThirdParty submodule, `scripts/python` and a custom licence — [GitHub ihs-ustutt/dtOO](https://github.com/ihs-ustutt/dtOO); docs at [ihs-ustutt.github.io/dtOO](https://ihs-ustutt.github.io/dtOO/)
- It is a toolbox for geometry generation, mesh generation and simulation-setup generation for hydraulic machinery. It defines components as free-form curves, surfaces and regions via an underlying CAD library, and is designed for hybrid meshes. OpenFOAM solves the case "by a direct interface to dtOO" — [ResearchGate figure: dtOO framework overview](https://www.researchgate.net/figure/Overview-of-the-gray-framed-dtOO-framework-including-dependencies-shown-as-a-filled-or_fig1_388655506) (search snippet)
- Tismer and Riedelbauch, "Current development status of a design framework for hydraulic machines", CSME-CFDSC-CSR 2025 Congress, Montreal, 25–28 May 2025. Examples cover kinetic and axial turbines, and "AI-based multi-objective optimization of an axial runner shows a substantial reduction of the required computing resources" — [ÉTS espace2 eprint 32512](https://espace2.etsmtl.ca/id/eprint/32512/)
- An elliptic mesh generator creates the block-structured flow meshes. Unstructured tet meshes are used for CSM. The framework is coupled to EAs through Python — [search summary of the ÉTS paper and the IHS pages](https://espace2.etsmtl.ca/32512/1/266%20-%20Current%20development%20status%20of%20a%20de.pdf) (PDF not text-extractable here; the claim comes from search snippets)
- Dissertation: A. Tismer, "Development of a software framework to design hydraulic machinery using the island model", Univ. Stuttgart, 2020, listed by OpenCASCADE — [OpenCASCADE publications](https://dev.opencascade.org/publication/development-software-framework-design-hydraulic-machinery-using-island-model); [IHS Tismer team page](https://www.ihs.uni-stuttgart.de/en/institute/team/Tismer)
- Local source structure (`/dtOO/src`, commit of 2026-09-15):
  - Geometry: `transIntCube` and `trans6SidedCube` (TFI volumes from six `map2dTo3d` faces), `trans4SidedFace`, `vec3dTransVolThreeD_skinBSplineSurfaces`, `trans6SidedCube_splitTrans6SidedCube`, `map1dTo3d_normalOffsetMap1dTo3dInMap2dTo3d` (curve offset on a surface, relevant to O-grids).
  - Meshing (Gmsh-based): `dtGmshModel/Edge/Face/Region`, `dtMeshTransfiniteGFace`, `dtMeshLaplacianGFace`, `dtMeshGFaceWithTransfiniteLayer`, `dtMeshGRegionWithBoundaryLayer`, `dtMeshGRegionTetgen`, `dtOptimizeMeshGRegion`, `dtHybridOptMeshGRegion`, and OpenMesh/OpenVolumeMesh/MOAB wrappers.
  - Bounded-volume operators: `bVOTransfiniteFaces`, `bVOTransfiniteRegions`, `bVORecombine`, `bVOSetGrading`, `bVOSetNElements`, `bVOSetRotationalPeriodicity`, `bVOWriteMSH/SU2/INP` — local `/dtOO/src/{geometryKernel,libraryKernel/meshEngine,meshKernel}`

### Inferences
- In dtOO a "block edge" is an exact analytic curve, an edge of a `map2dTo3d` face bounding a `map3dTo3d`, and block interiors are TFI (Coons) volumes. Inserting the blade at block level with exact dtOO geometry therefore means producing `map2dTo3d` faces and `trans6SidedCube` blocks whose blade-adjacent faces are the blade surface maps. No association step is needed because the geometry is the edge.
- dtOO has no block-corner optimiser comparable to GridPro or ICEM (none found in the source listing). Smoothing happens at the gmsh face and region level (Laplacian, optimise). A block-level smoother would have to be added, e.g. modelled on classy_blocks.

### Gaps
- Names given in the assignment (Tiede, Schmidt) could not be linked to dtOO publications. The verified authors are Tismer and Riedelbauch. IHS publication lists for 2016, 2021 and 2025 exist ([IHS 2025](https://www.ihs.uni-stuttgart.de/en/publications/2025/)) but were not fetched.
- The full text of the 2025 ÉTS PDF could not be extracted (no pdftotext in the container), so details of the elliptic generator (control functions, boundary orthogonality) are unverified.
