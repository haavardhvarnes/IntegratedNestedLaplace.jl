#!/usr/bin/env Rscript
# Bivariate-meta-analysis-style R-INLA reference fit.
#
# Synthetic bivariate Gaussian data: 30 studies, each contributing two
# observations (y_(i,1), y_(i,2)) with a correlated bivariate latent
# (u_i, v_i) ~ N(0, Σ). Same data on both sides; the only thing being
# tested is that Julia and R-INLA find the same posterior under the
# *same* prior on (log τ₁, log τ₂, atanh ρ).
#
# Layout: R-INLA's `2diid` with `n = 2 * n_studies` expects pairs at
# *interleaved* positions: study i has u at index `2(i-1)+1` and v at index
# `2(i-1)+2`. Julia's `BivariateIIDModel` uses the same convention. The R
# fixture builds `diid` from `(study, type)` accordingly.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
})

example_id <- "05_meta_analysis"
this_script <- normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
root <- normalizePath(file.path(dirname(this_script), "..", ".."))
data_path <- file.path(root, "examples", example_id, "data", "bivariate_synthetic.csv")
out_dir <- file.path(root, "test", "fixtures", example_id)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "rinla_reference.json")

stopifnot(file.exists(data_path))
df <- read.csv(data_path)
n_studies <- max(df$study)
# Interleaved layout: pair i lives at indices 2(i-1)+1 and 2(i-1)+2.
df$diid <- 2L * (df$study - 1L) + df$type

# Match Julia's prior: a=0.25, b=0.025 on each precision; rho_precision=0.4.
formula <- y ~ -1 +
  f(diid, model = "2diid", n = 2 * n_studies,
    param = c(0.25, 0.025, 0.25, 0.025, 0, 0.4))

run <- function() {
  t0 <- Sys.time()
  res <- inla(formula, family = "gaussian", data = df,
              control.compute = list(return.marginals = FALSE),
              control.predictor = list(compute = TRUE),
              silent = 2L)
  t1 <- Sys.time()
  list(res = res, wall = as.numeric(t1 - t0, units = "secs"))
}

invisible(run())
fit <- run()
cat(sprintf("inla wall (warm)    = %.4f s\n", fit$wall))
cat(sprintf("inla cpu.used Total = %.4f s\n", as.numeric(fit$res$cpu.used["Total"])))

named_summary <- function(s) {
  cols <- c("mean", "sd", "0.025quant", "0.5quant", "0.975quant")
  out <- list()
  for (i in seq_len(nrow(s))) {
    out[[rownames(s)[i]]] <- as.list(s[i, cols, drop = TRUE])
    names(out[[rownames(s)[i]]]) <- cols
  }
  out
}

ref <- list(
  example_id   = example_id,
  formula      = deparse(formula),
  inla_version = as.character(INLA::inla.version("version")),
  n            = nrow(df),
  n_studies    = n_studies,
  prior        = list(a1 = 0.25, b1 = 0.025, a2 = 0.25, b2 = 0.025, rho_precision = 0.4),
  hyper        = named_summary(fit$res$summary.hyperpar),
  random_diid  = list(
    id   = fit$res$summary.random$diid$ID,
    mean = fit$res$summary.random$diid$mean,
    sd   = fit$res$summary.random$diid$sd
  ),
  cpu          = as.list(fit$res$cpu.used),
  wall_seconds = fit$wall
)

write_json(ref, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat(sprintf("wrote %s\n", out_path))
