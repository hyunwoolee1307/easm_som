#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[1]) else ""
script_dir <- if (nzchar(script_path)) dirname(script_path) else getwd()
base_dir <- normalizePath(file.path(script_dir, ".."))
lib_dir <- file.path(base_dir, ".R_libs")
if (!dir.exists(lib_dir)) dir.create(lib_dir, recursive = TRUE)
.libPaths(c(lib_dir, .libPaths()))

options(repos = c(CRAN = "https://cloud.r-project.org"))
suppressWarnings(suppressMessages({
  if (!requireNamespace("kohonen", quietly = TRUE)) install.packages("kohonen", lib = lib_dir)
  if (!requireNamespace("zoo", quietly = TRUE)) install.packages("zoo", lib = lib_dir)
}))

library(kohonen)
library(zoo)

results_dir <- file.path(base_dir, "Results")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
input_path <- file.path(results_dir, "som_input_jja.csv.gz")
if (!file.exists(input_path)) {
  py_cmd <- sprintf("conda run -n nwp_som python %s", file.path(base_dir, "Scripts", "export_som_input_for_kohonen.py"))
  message("Generating SOM input matrix via Python: ", py_cmd)
  status <- system(py_cmd)
  if (status != 0) stop("Failed to generate som_input_jja.csv.gz")
}

df <- read.csv(gzfile(input_path))
time_jja <- as.POSIXct(df$time, tz = "UTC")
mat <- as.matrix(df[, -1, drop = FALSE])
mat_scaled <- scale(mat)

# Kohonen SOM
set.seed(42)
grid <- somgrid(xdim = 3, ydim = 3, topo = "rectangular")
model <- som(
  X = mat_scaled,
  grid = grid,
  rlen = 4000,
  alpha = c(0.5, 0.05),
  radius = c(2.0, 0.5),
  keep.data = TRUE
)

# BMU output
bmu <- model$unit.classif
bmu_df <- data.frame(time = format(time_jja, "%Y-%m-%d"), node_id = bmu)
write.csv(bmu_df, file.path(results_dir, "kohonen_bmu.csv"), row.names = FALSE)

# Codebook output
codes <- model$codes[[1]]
code_df <- as.data.frame(codes)
code_df <- cbind(node_id = 1:nrow(code_df), code_df)
write.csv(code_df, file.path(results_dir, "kohonen_codes.csv"), row.names = FALSE)

summary_path <- file.path(results_dir, "kohonen_summary.txt")
writeLines(c(
  "kohonen SOM run summary",
  paste("n_samples:", nrow(mat_scaled)),
  paste("n_features:", ncol(mat_scaled)),
  paste("grid:", "3x3"),
  paste("alpha:", "0.5 -> 0.05"),
  paste("radius:", "2.0 -> 0.5"),
  paste("rlen:", "4000")
), con = summary_path)
