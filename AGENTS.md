# HeartcareGPT Agent Instructions

## Environment & Commands
- **Package Manager**: Use `uv` (Python 3.13). Torch is pinned to CUDA 12.4.
  - Sync environment: `uv sync`
  - Download dataset: `uv run python -m utils.download_data`
  - View metrics: `uv run tensorboard --logdir runs`
- **Secrets**: Provide `.env` with `HF_TOKEN` (based on `.env_example`) before running stage 2/3 notebooks.

## Execution Flow & Entrypoints
- **Working Directory**: All scripts and notebooks MUST be run from the repository root. Paths (`data/...`, `beat/...`) are hardcoded relative to the root.
- **Entrypoints**: `main.py` is an unused stub. The workflow is driven entirely via notebooks:
  1. `train_step_1.ipynb`: Trains the "Beat" tokenizer from scratch on PTB-XL. Final output is `beat/<cfg>/tokenizer.pth`.
  2. `train_step_2_3.ipynb`: Performs Stages 2 (modal pretraining) and 3 (LoRA joint finetuning).
  3. `process/` dir: Contains standalone evaluation/generation scripts hitting OpenAI APIs.

## Config & Quirks
- **Config Derivation Trap**: `utils/my_config.py` derives directory paths (like `beat_dir`) from hyperparameter values. Changing hyperparameters (e.g., `residual_levels`, `seq_seconds`) silently points the pipeline to non-existent directories, causing stage-1 to process from scratch.
- **Stage 2/3 Config Split**: `train_step_2_3.ipynb` ignores model path configurations in `my_config.py`. To change paths for Stages 2/3 (e.g. `HF_PHI3`, `OUTPUT_DIR`), you must edit the constants in the first cell of `train_step_2_3.ipynb`.
- **Dataloader Quirk**: `train_step_2_3.ipynb` bypasses standard DataLoaders and iterates over the dataset directly one sample at a time (`BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=16`) due to variable image/signal prefix lengths.
- **Caching**: The QA dataset split is cached in `data/qa_dataset/{train,valid}`. To regenerate the split, you must delete these directories first.
- **Notebook Editing**: Notebooks contain large embedded plot outputs. Be cautious when reading or editing them directly; parse them correctly rather than using plain text manipulation.
