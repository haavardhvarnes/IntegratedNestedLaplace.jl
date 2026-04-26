# Strategic Layer: IntegratedNestedLaplace.jl Meta-Package
**Mission:** Replicate and extend the R-INLA methodology natively in Julia using a meta-package architecture.

**Architectural Rules:**
1.  **Monorepo Pattern:** The root package `IntegratedNestedLaplace.jl` acts solely as the user-facing API and formula parser.
2.  **No Monoliths:** Do not mix core numerical linear algebra with spatial geometries. Use the isolated packages in `/dev`.
3.  **Strict Dispatch:** Use `StatsModels.jl` multiple dispatch for the `@formula` macro.

**Phase Execution Plan:**
- Phase 1: Foundational Mathematics (Target: `dev/INLACore`)
- Phase 2: Statistical Infrastructure (Target: `dev/INLAModels`)
- Phase 3: Spatial & SPDEs (Target: `dev/INLASpatial`)
- Phase 4: API integration (Target: `src/IntegratedNestedLaplace.jl`)
