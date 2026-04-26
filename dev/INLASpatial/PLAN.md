# INLASpatial Development Plan
**Scope:** Spatial mapping, mesh generation, and environmental data.
**Goal:** Create Stochastic Partial Differential Equation (SPDE) discretizations.

**Key Steps:**
1. Generate 2D Delaunay triangulations using `Meshes.jl`.
2. Compute Galerkin mass and stiffness matrices for the Matérn covariance.
3. Use `GeoInterface.jl` to ingest polygons and boundaries from external sources without rigid type constraints.
4. Integrate `Rasters.jl` for lazy loading and extraction of environmental covariates.
