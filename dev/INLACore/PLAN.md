# INLACore Development Plan
**Scope:** Purely mathematical engine. Agnostic to statistics or geometry.
**Goal:** Calculate the Newton-Raphson conditional mode of latent fields and approximate marginals using sparse Hessians.

**Key Steps:**
1. Setup Newton-Raphson solvers via `Optimization.jl`.
2. Implement matrix coloring via `SparseDiffTools.jl` for sparse Hessian evaluation.
3. Code Takahashi equations for marginal variances from Cholesky factors.
4. Integrate `ExperimentalDesign.jl` for Central Composite Design (CCD) evaluation nodes.
