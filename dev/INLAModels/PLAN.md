# INLAModels Development Plan
**Scope:** Bridge standard probability theory and INLA matrices.
**Goal:** Define likelihoods, link functions, and GMRF precision matrix constructors.

**Key Steps:**
1. Define probability density functions leveraging `Distributions.jl`.
2. Build sparse precision matrix generators for RW1, RW2, AR1, and IID structures.
3. Establish Penalized Complexity (PC) prior structs for regularizing hyperparameters.
