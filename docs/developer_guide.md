# Developer Guide

## Environment Setup

```bash
git clone https://github.com/thuannguyen25032k/MemAdapt.git
cd MemAdapt
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e ".[dev]"
```

## Repository Layout

All production code lives under `embodiedbench/`.  The top-level also contains:

| Path | Purpose |
|---|---|
| `tests/` | pytest test suite |
| `docs/` | documentation (this directory) |
| `conda_envs/` | conda environment specs |
| `Docker/` | Docker build files |
| `setup.py` / `pyproject.toml` | packaging |
| `requirements.txt` | pip requirements |

## Running Tests

```bash
pytest tests/ -q               # fast summary
pytest tests/ -v               # verbose
pytest tests/memory/ -v        # single module
pytest tests/ --cov=embodiedbench --cov-report=html
```

## Adding a New Module

1. Create `embodiedbench/<your_module>/`.
2. Add `__init__.py` with public API.
3. Add tests to `tests/<your_module>/`.
4. Register in `setup.py` if it needs separate extras.

## Adding a New Reward Component

1. Implement `score_<name>(output, episode) -> float` in
   `embodiedbench/memory_adapter_rl/rewards.py`.
2. Add `w_<name>: float` to `RLRewardWeights` in
   `embodiedbench/memory_adapter_rl/config.py`.
3. Wire it into `compute_reward()` in `rewards.py`.
4. Add a weight entry in `qwen_grpo.yaml` under `reward_weights:`.
5. Add a test in `tests/memory_adapter_rl/test_rl_pipeline.py`.

## Code Style

- Black formatting (`black embodiedbench/ tests/`)
- isort imports (`isort embodiedbench/ tests/`)
- Type annotations required for all public functions
- Docstrings in NumPy style

## Commit Convention

```
feat: short description        # new feature
fix: short description         # bug fix
docs: short description        # documentation only
test: short description        # test only
refactor: short description    # no behaviour change
```

## CI (local)

```bash
black --check embodiedbench/ tests/
isort --check-only embodiedbench/ tests/
pytest tests/ -q --tb=short
```
