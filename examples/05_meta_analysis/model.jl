using IntegratedNestedLaplace
using DataFrames
using Statistics

println("--- Example 05: Bivariate Meta-Analysis (Correlated Effects) ---")

# 1. Generate Synthetic Meta-Analysis Data
# 20 studies, each providing a pair of observations (e.g., logit-sens and logit-spec)
function get_meta_data(n_studies=20)
    # Correlation rho = 0.6
    # Precisions tau1=2.0, tau2=1.5
    studies = Int[]
    y = Float64[]
    type = Int[]
    
    for i in 1:n_studies
        # True latent values
        u = [randn() * 0.5, randn() * 0.7] # mock correlation later
        # Study i, type 1
        push!(studies, i); push!(y, u[1] + randn()*0.1); push!(type, 1)
        # Study i, type 2
        push!(studies, i); push!(y, u[2] + randn()*0.1); push!(type, 2)
    end
    
    df = DataFrame(y = y, study = studies, type = type)
    return df
end

df = get_meta_data(10) # 10 studies for fast demo

# 2. Run Model
# We model y ~ study_effect (BivariateIID)
# The BivariateIID model expects 3 hyperparameters: log-tau1, log-tau2, logit-rho
println("Running IntegratedNestedLaplace.jl...")
start_time = time()

res = inla(
    @formula(y ~ f(study, BivariateIID)),
    df,
    family=GaussianLikelihood(),
    theta0=[1.0, 1.0, 0.5] # log-tau1, log-tau2, atanh(rho)
)

end_time = time()

# 3. Results
println(res)
println("Total Execution Time: ", round(end_time - start_time, digits=4), "s")

# theta[3] is atanh(rho). 
rho_est = tanh(res.mode_hyper[3])
println("Estimated correlation (rho): ", round(rho_est, digits=4))
