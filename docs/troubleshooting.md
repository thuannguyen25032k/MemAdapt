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

### CUDA OOM during SFT training

Reduce the effective batch size and keep gradient checkpointing on:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true
```

> GRPO (Stage 2) is planned and not yet implemented; the
> `embodiedbench/memory_adapter_rl/` configs are a forward-looking reference only.

### Loss is NaN from the first step

Check that `bnb_4bit_compute_dtype` is set to `"bfloat16"` (not `"float16"`) for A100.

---

## Evaluation Issues

### `FileNotFoundError` for training records / logs

Enable training-record logging when you run the benchmark:
```bash
python embodiedbench/main.py \
    env=eb-alf \
    memory_experiment.mode=adapted_planner_critic \
    memory_experiment.save_training_records=true \
    memory_experiment.log_dir=./alfred_memory_logs
```

### ALFRED simulator not found

Run the install script:
```bash
bash install.sh
```

---

## Getting Help

Open an issue on GitHub with:
1. Your OS and Python version (`python --version`)
2. The full traceback
3. The command you ran
4. The contents of `conda list | grep -E "torch|transformers|peft|trl"`
