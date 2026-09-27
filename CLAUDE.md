# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reimplementation/training harness for **Heartcare Suite / HeartcareGPT** (arXiv 2506.05831): an ECG multimodal LLM that consumes ECG *signals* and ECG *images* in a shared language space. Two pieces:

- **Beat** — a structure-aware discrete ECG tokenizer (transformer encoder + residual vector quantization) trained from scratch on PTB-XL.
- **DSPA** — dual-stream projection alignment: Beat quantized codes and SigLIP image features are each projected into Phi-3's embedding space, wrapped in special tokens, and prepended to the prompt embeddings.

## Environment & commands

`uv`-managed, Python 3.13, torch pinned to the CUDA 12.4 wheel index (`[[tool.uv.index]] pytorch-cu124` in `pyproject.toml`).

```bash
uv sync                                    # install deps into .venv
uv run python -m utils.download_data       # PTB-XL records500 via wfdb (default)
uv run python -m utils.download_data --source kaggle   # same data, faster, needs kaggle creds
uv run tensorboard --logdir runs           # training curves (Beat: runs/beat/<cfg>, DSPA: runs/phi3_siglip_beat_<stage>)
```

`transformers` and `accelerate` are installed transitively via `peft`; they are **not** listed in `pyproject.toml` dependencies even though the notebooks import them directly.

There is no test suite, linter config, or CI. `main.py` is an unused uv scaffold stub. Training is driven entirely from notebooks.

`.env` holds `HF_TOKEN` (see `.env_example`); `train_step_2_3.ipynb` loads it via `dotenv`.

## Critical: paths are relative to the repo root

Every path in `utils/my_config.py` and in the notebooks (`data/...`, `model/...`, `beat/...`) is **relative**. Notebooks and `python -m utils.*` must be run with the working directory set to the repo root, not from `data/` or `process/`.

## utils/my_config.py is the single source of truth

It is `import *`-ed by both training notebooks. Hyperparameters there are not just values — several **derive directory names**:

```
beat_config = f"level_{residual_levels}_code_{codebook_size}_len_{seq_length}_ratio_{latent_ratio}"
```

which feeds `beat_dir`, `beat_ckpt_dir`, `tokenizer_path`, `processed_data_path`, `my_model_dir`. Changing `residual_levels`, `codebook_size`, `seq_seconds`/`target_rate` (→ `seq_length`), or `latent_ratio` silently points everything at a **new, empty** set of paths, so preprocessing and stage-1 training will re-run from scratch. Existing artifacts on disk are for `level_2_code_512_len_1000_ratio_0.5`.

Note `original_model_dir`, `my_model_dir` and `QA_TYPES` are declared in the config but `train_step_2_3.ipynb` ignores them and defines its own `HF_PHI3` / `HF_SIGLIP` / `OUTPUT_DIR` constants in its first cell. Edit the notebook cell, not the config, for stage 2/3 paths.

## Pipeline (run in this order)

### `train_step_1.ipynb` — train Beat
1. `ecg.load_raw_data` reads PTB-XL `filename_hr` (500 Hz) records, resamples 500→250 Hz and `nk.ecg_clean`s each of the 12 leads → `data/records250/records250.npy` (cell 2, slow, run once).
2. `ecg.process_loaded_data` picks the best-quality `total_length` window per record (`nk.ecg_quality` + sliding-window max, falling back to a center crop when quality estimation throws) and normalizes → `records250_len_1000_ratio_0.5.npy`. `total_length = seq_length * (1 + latent_ratio)`: the first `seq_length` samples are the reconstruction target, the tail is the forecasting target.
3. Trains `Tokenizer(**signal_cfg)` with `loss = 1.0*recon + 0.5*pred + 0.25*vq`, logging per-level codebook utilization. Checkpoints land in `beat/<cfg>/ckpt/epoch_N.pth` every 5 epochs and **auto-resume** from the highest-numbered one on re-run; the final weights are saved to `beat/<cfg>/tokenizer.pth`, which is what stage 2/3 loads.

### `train_step_2_3.ipynb` — stages 2 and 3 in one loop
Set `STAGE = 2` for the full run (`for stg in range(STAGE, 4)`), or `STAGE = 3` to skip modality pretraining.

- Cell 1 splits the three flat `data/*QA*.json` / `ReportGeneration.json` files into `data/qa_dataset/{train,valid}/<type>.json` (90/10, seed 42). It **no-ops if both dirs already exist** — delete them to re-split.
- Cell 5 runs every processed signal through the frozen Beat and keeps `quant` (the quantized latents, `[num_latent_tokens, 128]`) in the in-memory `tokens` list. This list is indexed positionally: a sample's `filename` like `ptbxl_img/04332` maps to `tokens[4332 - 1]`, so the order of `records250.npy` must match `ptbxl_database.csv` row order.
- Stage 2 freezes all of Phi-3 and trains only `visual_proj` / `signal_proj` at lr 2e-4, on 1/5 of the data. Stage 3 unfreezes the LoRA adapters (`r=64`, `target_modules=["qkv_proj","o_proj"]`) and trains everything at 2e-5.
- Training iterates the `Dataset` **directly, one sample at a time** (no DataLoader, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=16`) because each sample has a variable number of image/signal prefixes. bf16 `autocast` + `GradScaler`.
- Last cell runs greedy generation over `data/qa_dataset/valid/` and writes each item back with a `gen_answer` field into `data/qa_dataset/valid_output/`, skipping files that already exist.

### `process/` — dataset construction (not part of training)
`generate_prompt.ipynb` holds the GPT-4o system prompts that turn PTB-XL SCP labels into closed/open QA pairs per dimension (diagnosis / waveform / rhythm / report+forecast); `report_score_prompt.ipynb` holds the 13-criterion LLM-judge rubric for scoring generated reports. Both need an OpenAI `base_url`/`api_key` filled in by hand.

## How the prefix is assembled (the DSPA detail that matters)

Each QA item carries parallel `filename` and `label` lists; `label[j]` is `"img"` or `"signal"` and selects which encoder handles it. `get_feats` returns two lists, and the prefix is built as:

```
[<image_start>] + visual_proj(siglip_feats) + [<image_end>]
[<signal_start>] + signal_proj(beat_quant)  + [<signal_end>]
```

concatenated ahead of the tokenized `<|user|> ... <|assistant|>` prompt embeddings, with `labels` set to `-100` over the entire prefix + question so loss is computed on the answer only. The `<image_start>`/`<signal_start>`/`<image_end>`/`<signal_end>`/`<prediction>` tokens are added via `add_special_tokens` + `resize_token_embeddings`, and their embeddings are snapshotted **before** LoRA wrapping, so they act as frozen constants.

`visual_proj` expands one SigLIP vector into `VISUAL_PREFIX_LEN` (2) token slots. `signal_proj` is a plain `Linear(128 → hidden) + GELU + LayerNorm`.

SigLIP features are cached per image as `siglip_cache*/<basename>.pt`. `cut_12 = False` embeds the whole ECG image once (cache dir `siglip_cache_no12cut/`); `cut_12 = True` crops the sheet into a 6×2 lead grid and embeds 12 tiles (cache dir `siglip_cache/`). The two caches are deliberately separate — flipping the flag must not reuse the other cache.

Only signal-labelled samples exist in the checked-in QA JSONs; `IMAGE_ROOT = data/img` is not present locally, so the image branch is currently dead in practice.

## Beat internals (`utils/my_tokenizer.py`)

- `sequence_to_tokens`: `[b, 12, 1000] → [b, 40, 128]` via `Rearrange` patching at `patch_size=25` (12 leads × 25 samples flattened per patch) + linear.
- Learned `latents` (`num_latent_tokens = num_tokens * latent_ratio = 20`) are appended to the patch tokens, run through the encoder, then **only the latent slice is quantized** — this is the core idea, quantizing a compressed latent set rather than per-patch tokens.
- The decoder is fed positional embeddings as mask tokens packed with the quantized latents and emits the full reconstruction; `tokens[:, num_tokens:]` from the encoder doubles as the forecast.
- `forward` chops an arbitrary-length input into left-padded `seq_length` segments, runs `forward_segment` per segment, and concatenates — so it handles inputs longer than `seq_length` transparently.
- `utils/my_residual_vector_quantize.py` is a vendored/adapted `vector-quantize-pytorch` with a `residual_levels` argument; `indices` come back shaped `[b, n, residual_levels]`, which is why codebook-utilization is computed per level.

## Conventions

- `.gitignore` excludes all of `model/`, `beat/`, `runs/`, `data/records*/`, `data/qa_dataset/`, `*.npy`, `*.pth` — those directories exist locally but are untracked. Do not add them.
- `train_step_1.ipynb` is ~20 MB because it stores plot outputs. Read it with `json.load` + cell `source` extraction rather than `cat`, and prefer `NotebookEdit` over rewriting the file.
- Some comments and print strings are in Portuguese (the maintainer's language); English is used for model-facing prompts and dataset content.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
rtk uv run <cmd>        # Compact uv project command output
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->