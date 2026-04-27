using IntegratedNestedLaplace
using DataFrames
using Random
using Statistics

println("--- Example 05: Bivariate Meta-Analysis (Correlated Effects) ---")

function get_meta_data(n_studies=20; rng = MersenneTwister(20260426))
    studies = Int[]
    y = Float64[]
    type = Int[]
    for i in 1:n_studies
        u = [randn(rng) * 0.5, randn(rng) * 0.7]
        push!(studies, i); push!(y, u[1] + randn(rng)*0.1); push!(type, 1)
        push!(studies, i); push!(y, u[2] + randn(rng)*0.1); push!(type, 2)
    end
    df = DataFrame(y = y, study = studies, type = type)
    return df
end

df = get_meta_data(10) # 10 studies for fast demo

# 2. Run Model: y ~ study_effect (BivariateIID).
# Gaussian likelihood contributes one hyperparameter (log τ_y), then the
# BivariateIID block adds three (log τ₁, log τ₂, atanh ρ) — four total.
println("Running IntegratedNestedLaplace.jl...")
start_time = time()

res = inla(
    @formula(y ~ f(study, BivariateIID)),
    df,
    family = GaussianLikelihood(),
    theta0 = [4.0, 1.0, 1.0, 0.5],   # log τ_y, log τ₁, log τ₂, atanh ρ
)

end_time = time()

# 3. Results
println(res)
println("Total Execution Time: ", round(end_time - start_time, digits=4), "s")

# theta layout: [log τ_y, log τ₁, log τ₂, atanh ρ]. Last entry is atanh(ρ).
rho_est = tanh(res.mode_hyper[end])
println("Estimated correlation (rho): ", round(rho_est, digits=4))
