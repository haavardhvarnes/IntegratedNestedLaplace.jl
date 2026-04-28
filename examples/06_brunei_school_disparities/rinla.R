#!/usr/bin/env Rscript
# Brunei-style areal Poisson model: R-INLA reference fit.
#
# Uses the simpler `besag` (rather than `bym2`) so the prior matches what the
# Julia BesagModel implements: log-Gamma(1, 5e-5) on log-precision, no
# unstructured component, scale.model = TRUE so τ is comparable across graphs.
# BYM2 with the (τ, φ) mixing parameter remains a future deliverable.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
})

example_id <- "06_brunei_school_disparities"
this_script <- normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
root <- normalizePath(file.path(dirname(this_script), "..", ".."))
data_path <- file.path(root, "examples", example_id, "data", "areas.csv")
graph_path <- file.path(root, "examples", example_id, "data", "areas.graph")
out_dir <- file.path(root, "test", "fixtures", example_id)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "rinla_reference.json")

stopifnot(file.exists(data_path), file.exists(graph_path))
df <- read.csv(data_path)
g  <- inla.read.graph(graph_path)

formula <- y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE)

run <- function(verbose = FALSE) {
  t0 <- Sys.time()
  res <- inla(formula, family = "poisson", data = df, E = df$E,
              control.compute = list(return.marginals = FALSE),
              control.predictor = list(compute = TRUE),
              verbose = verbose, silent = 2L)
  t1 <- Sys.time()
  list(res = res, wall = as.numeric(t1 - t0, units = "secs"))
}

invisible(run())
fit <- run()

cat(sprintf("inla wall (warm)    = %.4f s\n", fit$wall))
cat(sprintf("inla cpu.used Total = %.4f s\n", as.numeric(fit$res$cpu.used["Total"])))

named_summary <- function(s) {
  cols <- intersect(c("mean", "sd", "0.025quant", "0.5quant", "0.975quant", "mode"),
                    colnames(s))
  out <- list()
  for (i in seq_len(nrow(s))) {
    out[[rownames(s)[i]]] <- as.list(s[i, cols, drop = TRUE])
    names(out[[rownames(s)[i]]]) <- cols
  }
  out
}

lp <- fit$res$summary.linear.predictor[, c("mean", "sd")]
ru <- fit$res$summary.random$area[, c("ID", "mean", "sd")]

ref <- list(
  example_id        = example_id,
  formula           = deparse(formula),
  inla_version      = as.character(INLA::inla.version("version")),
  n                 = nrow(df),
  fixed             = named_summary(fit$res$summary.fixed),
  hyper             = named_summary(fit$res$summary.hyperpar),
  cpu               = as.list(fit$res$cpu.used),
  wall_seconds      = fit$wall,
  linear_predictor  = list(mean = lp$mean, sd = lp$sd),
  random_area       = list(id = ru$ID, mean = ru$mean, sd = ru$sd)
)

write_json(ref, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat(sprintf("wrote %s\n", out_path))
