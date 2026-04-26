# IntegratedNestedLaplace.jl

`IntegratedNestedLaplace.jl` is a native Julia implementation of the Integrated Nested Laplace Approximation (INLA) methodology for approximate Bayesian inference in Latent Gaussian Models (LGMs). 

This package is designed as a modern, high-performance alternative to the R-INLA package, leveraging Julia's multiple dispatch, powerful automatic differentiation (AD) ecosystem, and sparse linear algebra capabilities.

## Architecture

The project is structured as a meta-package with three core sub-components to ensure modularity and high performance:

1. **`INLACore`**: The mathematical engine. Handles high-performance mode-finding via `Optimization.jl`, sparse Hessian extraction via `DifferentiationInterface.jl`, and $O(n)$ marginal variance computations using the Takahashi equations.
2. **`INLAModels`**: The statistical infrastructure. Defines observation likelihoods (Gaussian, Poisson, Bernoulli), Latent Gaussian Markov Random Field (GMRF) structures (IID, RW1, AR1, SPDE), and Penalized Complexity (PC) priors.
3. **`INLASpatial`**: The spatial modeling engine. Handles 2D Delaunay triangulation via `Meshes.jl` and constructs the Finite Element Method (FEM) mass and stiffness matrices required for Stochastic Partial Differential Equation (SPDE) models.

## Installation

Currently, the package is in development. To use it, you must instantiate the environment and develop the local sub-packages.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Ensure local sub-packages are linked
Pkg.develop(path="dev/INLACore")
Pkg.develop(path="dev/INLAModels")
Pkg.develop(path="dev/INLASpatial")
```

## Features

* **Native Formula API**: Uses `StatsModels.jl` for an intuitive `@formula` interface.
* **Custom Latent Effects**: Supports adding structured latent effects directly in the formula using the `f(covariate, model)` syntax.
* **Hyperparameter Integration**: Explores the hyperparameter posterior using Central Composite Design (CCD) integration nodes.
* **Sparse Takahashi Marginals**: Computes posterior marginal variances of the latent field without dense matrix inversion.
* **Spatial SPDEs**: Full support for Matérn covariance models via FEM discretization on triangular meshes.

## Usage Examples

### 1. Basic Linear Model with a Random Walk (RW1)

```julia
using IntegratedNestedLaplace
using DataFrames

# Generate some synthetic data
data = DataFrame(
    y = [1.1, 0.9, 1.2, 0.8, 1.0],
    time = 1:5
)

# Run INLA: y ~ intercept + RW1(time)
# The formula parser automatically detects f() and provisions the RW1Model.
result = inla(@formula(y ~ 1 + f(time, RW1)), data, family=GaussianLikelihood())

println("Latent Mode: ", result.mode_latent)
println("Latent Marginals: ", result.marginals_latent)
```

### 2. Spatial SPDE Model

For continuous spatial domains, we use the SPDE approach to approximate a Matérn Gaussian field.

```julia
using IntegratedNestedLaplace
using DataFrames

# Define spatial observation coordinates
coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]

# Create a spatial dataset
data = DataFrame(
    y = [1.0, 0.5, 0.5, 0.0, 0.8],
    loc_id = 1:5
)

# 1. Build the Delaunay Mesh
mesh = build_mesh(coords)

# 2. Compute the FEM Mass (C) and Stiffness (G) matrices
C, G = spde_matrices(mesh)

# 3. Create the SPDE Latent Model
spde = SPDEModel(C, G)

# 4. Run INLA
# We pass the SPDE structure via the `latent` keyword argument, 
# while mapping it in the formula using `f(loc_id, SPDE)`.
result = inla(@formula(y ~ 1 + f(loc_id, SPDE)), data, latent=spde)

println("Spatial Latent Mode: ", result.mode_latent)
println("Hyperparameter Mode (log kappa, log tau): ", result.mode_hyper)
```

## Performance & GPU Support

`IntegratedNestedLaplace.jl` is designed for extreme performance. By utilizing $O(n)$ sparse Cholesky paths and KernelAbstractions, it matches or exceeds the performance of R-INLA for standard models.

### Benchmarks (1000 observations, IID model)
| Backend | Time (Warm) |
| :--- | :--- |
| **CPU (M-series)** | **0.06s** |
| **Metal (GPU)** | **0.54s** |

### Using GPU Acceleration
The package supports heterogeneous backends via `KernelAbstractions.jl`. 

#### Apple Silicon (Metal)
```julia
using Metal
res = inla(formula, data, backend=MetalBackend())
```

#### NVIDIA (CUDA)
```julia
using CUDA
res = inla(formula, data, backend=CUDABackend())
```

Note: GPU acceleration is recommended for very large datasets ($10^5+$ observations), where the parallel throughput outweighs the data transfer latency. Metal users will automatically use `Float32` precision, while CUDA/CPU users default to `Float64`.

## References

1. Rue, H., Martino, S., & Chopin, N. (2009). *Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations*. Journal of the royal statistical society: Series b (statistical methodology), 71(2), 319-392.
2. Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
