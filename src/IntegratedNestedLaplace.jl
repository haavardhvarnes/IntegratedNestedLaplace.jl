module IntegratedNestedLaplace

using Reexport
@reexport using INLACore
@reexport using INLAModels
@reexport using INLASpatial
using StatsModels
using LinearAlgebra
using SparseArrays
import ForwardDiff
using ADTypes
using DifferentiationInterface
using OptimizationOptimJL
using Enzyme
import Meshes
using Printf
using RecipesBase
using KernelAbstractions

export inla, @formula, f

f(args...) = nothing

struct INLAResult{T}
    mode_latent::Vector{T}
    mode_hyper::Vector{T}
    marginals_latent::Vector{T}
    nodes_hyper::Vector{Vector{T}}
    weights_hyper::Vector{T}
    formula::FormulaTerm
    latent_model::Any
end

# --- Result Summary ---

function Base.show(io::IO, mime::MIME"text/plain", res::INLAResult)
    println(io, "INLA Approximation Result")
    println(io, "-------------------------")
    println(io, "Formula: ", res.formula)
    println(io, "Latent Model: ", res.latent_model === nothing ? "IID" : typeof(res.latent_model))
    println(io, "")

    println(io, "Hyperparameters (Mode):")
    for i in 1:length(res.mode_hyper)
        @printf(io, "  theta[%d]: %10.4f\n", i, res.mode_hyper[i])
    end
    println(io, "")

    n = length(res.mode_latent)
    println(io, "Latent Field Summary (n=$n):")
    limit = min(n, 10)
    for i in 1:limit
        var_val = res.marginals_latent[i]
        sd = sqrt(max(zero(res.mode_latent[1]), var_val))
        @printf(io, "  [%3d] %10.4f  %10.4f\n", i, res.mode_latent[i], sd)
    end
    n > 10 && println(io, "  ...")
end

@recipe function f(res::INLAResult, mesh::Meshes.SimpleMesh; type=:mode)
    values = if type == :mode
        res.mode_latent
    elseif type == :sd
        sqrt.(max.(zero(res.mode_latent[1]), res.marginals_latent))
    else
        error("Unknown plot type: $type")
    end
    title := "Latent Field ($type)"
    colorbar := true
    return mesh, values
end

"""
    inla(formula, data; family=GaussianLikelihood(), latent=nothing, theta0=[1.0], backend=CPU())
"""
function inla(form::FormulaTerm, data; family=GaussianLikelihood(), latent=nothing, theta0=[1.0], backend=CPU())
    # 1. Setup
    sch = schema(form, data)
    f_applied = apply_schema(form, sch)
    y_raw, X = modelcols(f_applied, data)
    n = length(y_raw)
    
    actual_latent = latent === nothing ? IIDModel() : latent
    
    # Determine default type from backend
    # Metal only supports Float32. CPU, CUDA, and ROCm support Float64.
    backend_name = string(typeof(backend))
    T_data = (contains(backend_name, "MetalBackend")) ? Float32 : Float64

    # 3. Hyperparameter Objective
    function hyper_objective(theta, p_opt)
        TT = eltype(theta)
        tau = exp(theta[1])
        Q = precision_matrix(actual_latent, n, tau)
        
        # Inner optimization (stable nested version)
        function latent_obj(x, p)
            ll = sum(INLAModels.log_pdf(family, y_raw[i], x[i], one(TT)) for i in 1:n)
            lp = -TT(0.5) * dot(x, Q * x)
            return -(ll + lp)
        end
        
        # Newton or LBFGS
        res_x = find_mode(latent_obj, zeros(TT, n); solver=LBFGS(), adtype=Optimization.AutoForwardDiff())
        x_star = res_x.u
        
        log_joint = -latent_obj(x_star, nothing)
        log_det_H = INLACore.sparse_logdet(Q + sparse(TT(1.0) * I, n, n))
        
        return -(log_joint - TT(0.5) * log_det_H - TT(0.5) * theta[1]^2)
    end

    # 5. Run Optimization for theta
    res_theta = find_mode(hyper_objective, T_data.(theta0); solver=NelderMead())
    theta_star = res_theta.u

    # 6. Compute Marginals & Integration
    H_theta = DifferentiationInterface.hessian(t -> hyper_objective(t, nothing), AutoFiniteDiff(), theta_star)
    theta_nodes = integration_nodes(theta_star, H_theta)
    
    total_marginals = zeros(T_data, n)
    weights = fill(T_data(1.0 / length(theta_nodes)), length(theta_nodes))

    for (i, node) in enumerate(theta_nodes)
        tau_node = exp(node[1])
        Q_node = precision_matrix(actual_latent, n, tau_node)
        H_x = Q_node + sparse(T_data(1.0) * I, n, n)
        total_marginals .+= weights[i] .* takahashi_marginals(sparse(H_x))
    end

    # Final latent mode
    tau_star = exp(theta_star[1])
    Q_star = precision_matrix(actual_latent, n, tau_star)
    res_x_final = find_mode((x,p) -> (T_data(0.5) * dot(x, Q_star * x) - sum(INLAModels.log_pdf(family, T_data(y_raw[i]), x[i], T_data(1.0)) for i in 1:n)), zeros(T_data, n); solver=LBFGS())

    return INLAResult(res_x_final.u, theta_star, total_marginals, theta_nodes, weights, form, actual_latent)
end

end # module
