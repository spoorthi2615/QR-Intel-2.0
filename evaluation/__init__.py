"""QRIntel evaluation package.

Provides scripts for feature extraction, hyperparameter optimization,
model evaluation, ablation studies, baseline comparisons, latency
benchmarking, and result visualisation for the QRIntel two-stage
quishing-detection framework.

Submodules
----------
feature_extraction
    Parallel feature-extraction pipeline with SQLite caching.
optimize
    Optuna-based TPE weight / threshold search.
evaluate
    Stratified cross-validation and held-out test evaluation.
ablation
    Frozen-weight ablation study for each intelligence module.
baselines
    Naive, equal-weight, and ML baseline comparisons.
latency_benchmark
    Microsecond-resolution latency profiling of the lexical module.
generate_plots
    Publication-quality matplotlib / seaborn visualisations.
"""
