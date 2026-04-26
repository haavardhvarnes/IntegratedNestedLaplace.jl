using IntegratedNestedLaplace
using DataFrames
using SparseArrays
using LinearAlgebra
using Distributions
using Plots

# 1. Mock Swiss Rainfall Data
function get_swiss_data(n=100)
    coords = [(rand(), rand()) for _ in 1:n]
    # Smooth spatial signal
    y = [sin(c[1]*3) + cos(c[2]*3) for c in coords]
    df = DataFrame(y = y, loc_id = 1:n)
    return df, coords
end

println("--- Tutorial 3: Swiss Rainfall (SPDE) ---")
data, coords = get_swiss_data(100)

# 2. Build Mesh
mesh = build_mesh(coords)
C, G = spde_matrices(mesh)
spde = SPDEModel(C, G)

# 3. Run INLA with a fixed hyper-starting point for stability
println("Running IntegratedNestedLaplace.jl...")
# Start at more reasonable log-parameters [log(kappa), log(tau)]
res = inla(@formula(y ~ f(loc_id, SPDE)), data, latent=spde, theta0=[1.0, 2.0])

println(res)

# 4. Final Verification: Latent Field Plot
# We plot the predicted mode at coordinate locations
println("Generating plot...")
p = scatter([c[1] for c in coords], [c[2] for c in coords], 
            marker_z = res.mode_latent, 
            title="Swiss Rainfall Mode (Julia INLA)",
            label="", color=:viridis, markersize=5)
savefig(p, "swiss_rainfall_mode.png")
println("Saved plot to swiss_rainfall_mode.png")
