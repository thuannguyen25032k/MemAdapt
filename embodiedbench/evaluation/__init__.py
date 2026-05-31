"""
evaluation/

End-to-end benchmark evaluation harness for the Memory Adapter.

Sub-modules
-----------
schemas        — ExperimentConfig, EpisodeResult, ExperimentResult, AggregateMetrics
metrics        — task metrics incl. stale-memory recovery rate
runner         — run_experiment(config) orchestrator
aggregators    — aggregate_results, compare_modes, cross_seed_summary
reporting      — JSON / CSV / Markdown report generation
visualization  — matplotlib plots
utils          — JSON / config helpers
"""
