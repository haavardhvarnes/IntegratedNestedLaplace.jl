using IntegratedNestedLaplace
using DataFrames
using Random
using SparseArrays
using LinearAlgebra
using Distributions
using KernelAbstractions
using Metal

function get_large_data(n=250; rng = MersenneTwister(20260426))
    coords = [(rand(rng), rand(rng)) for _ in 1:n]
    y = [sin(c[1]*3) + cos(c[2]*3) + rand(rng, Normal(0, 0.1)) for c in coords]
    df = DataFrame(y = y, loc_id = 1:n)
    return df, coords
end

println("--- Performance Benchmark (Implicit Diff: CPU vs Metal) ---")
n_obs = 250
data, coords = get_large_data(n_obs)

for i in 1:2
    println("\n--- Round $i ---")
    
    # --- CPU ---
    println("[CPU] Running...")
    start_cpu = time()
    res_cpu = inla(@formula(y ~ 1), data, theta0=[1.0], backend=CPU())
    t_cpu = time() - start_cpu
    println("  Time: ", round(t_cpu, digits=4), "s")

    # --- GPU (Metal) ---
    println("[GPU] Running...")
    try
        backend_gpu = MetalBackend()
        start_gpu = time()
        res_gpu = inla(@formula(y ~ 1), data, theta0=[1.0], backend=backend_gpu)
        t_gpu = time() - start_gpu
        println("  Time: ", round(t_gpu, digits=4), "s")
    catch e
        @warn "Metal Backend failed."
        println(e)
    end
end
