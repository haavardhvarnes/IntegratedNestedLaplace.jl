# IntegratedNestedLaplace.jl Examples

This directory contains a collection of case studies and tutorials adapted from the official R-INLA repository. Each example is designed to demonstrate specific features of the package and provide performance comparisons.

## Roadmap

### Foundational Models
* **[01_tokyo_rainfall](01_tokyo_rainfall)**: Temporal smoothing with Random Walk (RW1) and Bernoulli likelihood.
* **[02_german_oral_cancer](02_german_oral_cancer)**: Areal data modeling with IID/BYM effects and Poisson likelihood.
* **[03_swiss_rainfall](03_swiss_rainfall)**: Continuous spatial interpolation using the SPDE approach and Matérn covariance.

### Advanced Latent Structures (Planned)
* **[04_salamander_mating](04_salamander_mating)**: Crossed random effects in GLMMs.
* **[05_meta_analysis](05_meta_analysis)**: Bivariate latent models for diagnostic meta-analysis.

### Advanced Geostatistics (Planned)
* **06_brunei_school_disparities**: Large-scale areal models with ICAR priors.
* **07_dengue_brazil**: Non-stationary spatial models.

## Usage
To run an example:
```bash
julia --project=. examples/01_tokyo_rainfall/model.jl
```
