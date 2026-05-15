# Troubleshooting

## Installation Issues

### `conda env create` fails with solver timeout

```bash
conda env create -f conda_envs/environment.yaml --solver=libmamba
```

Install `conda-libmamba-solver` first if needed:
```bash
conda install -n base conda-libmamba-solver
```

### `pip install -e .` fails with build errors

Make sure you are inside the conda env:
```bash
conda activate embench
pip install -e .
```

---

## Import Errors

### `ModuleNotFoundError: No module named 'embodiedbench'`

You need to install the package in editable mode:
```bash
pip install -e .
```

### `ModuleNotFoundError: No module named 'bitsandbytes'`

Install the optional QLoRA dependencies:
```bash
pip install bitsandbytes>=0.41.0
```

### `ModuleNotFoundError: No module named 'wandb'`

Install the optional tracking dependency:
```bash
pip install wandb>=0.16.0
```

---

## Training Issues

### CUDA OOM during GRPO training

Reduce the effective batch size:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

Or use the debug config to verify your setup:
```bash
python -m embodiedbench.memory_adapter_training.trainer \
    --config embodiedbench/configs/memory_adapter_training/debug_tiny.yaml \
    --output_dir /tmp/debug_sft
```

For GRPO OOM, use the debug GRPO config:
```bash
python embodiedbench/scripts/train_memory_adapter_grpo.py \
    --config embodiedbench/configs/memory_adapter_rl/debug_grpo_tiny.yaml \
    --output_dir /tmp/debug_grpo
```

### Loss is NaN from the first step

Check that `bnb_4bit_compute_dtype` is set to `"bfloat16"` (not `"float16"`) for A100.

---

## Evaluation Issues

### `FileNotFoundError: data/episodes/...`

Run episode recording first:
```bash
python embodiedbench/main.py \
    --config embodiedbench/configs/eb-alf.yaml \
    --record_memory \
    --episodes_output_dir data/episodes/eb_alfred
```

### ALFRED simulator not found

Run the install script:
```bash
bash install.sh
```

---

## Test Suite

### Running all tests

```bash
conda activate embench
pytest tests/ -q
```

### Running a specific module

```bash
pytest tests/memory_adapter_rl/ -v
pytest tests/experiments/ -v
```

### 8 skipped tests

The 8 skipped tests require optional GPU-only dependencies (`bitsandbytes`, CUDA).
They are skipped automatically on CPU-only machines — this is expected.

---

## Getting Help

Open an issue on GitHub with:
1. Your OS and Python version (`python --version`)
2. The full traceback
3. The command you ran
4. The contents of `conda list | grep -E "torch|transformers|peft|trl"`
