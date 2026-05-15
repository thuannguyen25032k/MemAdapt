"""
evaluation/

Step-29 end-to-end benchmark evaluation harness for the Memory Adapter.

Sub-modules
-----------
schemas        — dataclasses: ExperimentConfig, EpisodeResult, ExperimentResult,
                 AggregateMetrics
metrics        — embodied-task metric functions incl. stale-memory recovery rate
runner         — run_experiment(config) orchestrator
aggregators    — aggregate_results, compare_modes, cross_seed_summary
reporting      — JSON / CSV / Markdown report generation
visualization  — matplotlib plots (success, replans, stale-recovery, …)
utils          — I/O helpers, config patching
"""
