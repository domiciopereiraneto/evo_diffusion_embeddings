# Inference-Time Optimization of Prompt Embeddings for Text-to-Image Diffusion Models Using Evolutionary Algorithms

Code accompanying the paper "Inference-Time Optimization of Prompt Embeddings for Text-to-Image Diffusion Models Using Evolutionary Algorithms".

The implementation optimizes continuous prompt embeddings at inference time while keeping the diffusion and scoring models frozen. The paper compares GA, CoSyNE, SNES, and sep-CMA-ES against zero-order search and AdamW, independently guided by ImageReward, CLIPScore, or HPSv2. The shared implementation is the `EmbeddingOptimizer` class in [embedding_optimizer.py](embedding_optimizer.py).

This guide describes how to configure tuning, DrawBench evaluation, and result processing for the supplied manuscript. The configs in [algorithms/config/reproducible_configs](algorithms/config/reproducible_configs) provide the paper's Optuna studies and evaluation settings; the base YAML files are editable working examples. Read the reproduction limitations below before launching the full study.

## Installation

Run commands from the repository root. The recommended environment uses Python 3.10:

```bash
conda env create -f environment.yml
conda activate evo_embs
```

Alternatively, inside an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

The first run downloads diffusion checkpoints, reward models, and datasets. Set `cuda` to your GPU index in each configuration. The paper used an NVIDIA RTX A6000; runtime and VRAM comparisons depend on hardware, precision, batch size, and memory settings. Preserve those settings across compared runs.

## Paper protocol

| Component | Manuscript protocol |
| --- | --- |
| Search variable | Continuous prompt embeddings; `optimization_target: prompt_embeddings` |
| Tuning data | The same 10 predefined DiffusionDB prompts for every trial |
| Tuning budget | 30 Optuna trials per algorithm–objective combination |
| Evaluation data | All 200 DrawBench prompts |
| Objectives | ImageReward, CLIPScore, and HPSv2, optimized separately |
| Regular optimization budget | 240 generated-image evaluations per prompt; 15 generations × 16 candidates for evolutionary methods |
| Repetitions | One seed; reported score variation is across prompts |
| Selection | Highest guiding-objective score observed during each prompt run |
| Extended comparison | SNES given approximately AdamW's mean wall-clock budget for the corresponding model |

Use a separate study and evaluation folder for each model, optimizer, and guiding objective. Six optimization methods × three objectives × three diffusion models gives 54 regular optimization configurations, before extended runs and sampling baselines.

| Model | `model_backend` | Denoising steps | Guidance scale | Paper's nominal NFE budget |
| --- | --- | --- | --- | --- |
| Stable Diffusion 1.5 | `sd` | 50 | 7.5 | 12,000 |
| SDXL Turbo | `sdxl` | 1 | 0 | 240 |
| SANA-Sprint | `sana_sprint` | 2 | 4.5 | 480 |

The paper uses the following `model_id` values:

- Stable Diffusion 1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- SDXL Turbo: `stabilityai/sdxl-turbo`
- SANA-Sprint (0.6B): `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`

For SANA-Sprint, set `model_backend: sana_sprint`. Use the image resolution, precision, and run seed recorded in the supplied reproducibility configuration for each experiment.

### Choose one guiding objective

Set exactly one of the following weights to a nonzero value for each paper experiment:

| Guiding objective | `image_reward_score_weight` | `clip_score_weight` | `hpsv2_score_weight` |
| --- | --- | --- | --- |
| ImageReward | 1.0 | 0.0 | 0.0 |
| CLIPScore | 0.0 | 1.0 | 0.0 |
| HPSv2 | 0.0 | 0.0 | 1.0 |

Set `aesthetic_score_weight: 0.0` and `pickscore_score_weight: 0.0`. Preserve the original normalization constants and scorer checkpoints when available. In particular, objective scaling changes AdamW's gradients. The manuscript's CLIPScore values are displayed on a percentage-like scale; the implementation records raw CLIP similarity. Keep any reporting conversion separate from the optimization weights.

## Reproduce with the supplied configurations

[algorithms/config/reproducible_configs](algorithms/config/reproducible_configs) contains the Optuna and batch-experiment configuration files for reproducing the paper's results. The `config_run_experiments_*.yaml` files **already include the hyperparameters selected by Optuna**, so you can run the evaluation directly. Running Optuna is optional unless you want to reproduce the complete tuning-and-evaluation pipeline.

For example, run the supplied SNES/SANA-Sprint/CLIPScore evaluation:

```bash
python algorithms/run_experiments.py --config algorithms/config/reproducible_configs/config_run_experiments_snes_sanasprint_clip.yaml
```

To repeat hyperparameter selection as well, use its corresponding Optuna config:

```bash
python algorithms/run_optuna_search.py --config algorithms/config/reproducible_configs/config_run_optuna_search_snes_sanasprint_clip.yaml
```

The Optuna configs enable a follow-up experiment using the matching evaluation YAML and the newly selected trial parameters. Set `optuna.follow_up_experiment.enabled: false` to tune only. Before running, review the GPU index, output folders, and study storage paths for your machine; use separate folders to preserve existing results. Files ending in `_extended.yaml` configure the extended-budget runs.

## Rebuild the tables and figures

Update each processing YAML's input folders to your completed runs; checked-in paths refer to example local results.

| Paper artifact | Script | Configuration |
| --- | --- | --- |
| Best-of-run scores, runtime, VRAM, paired comparisons | `process_quantitative_results.py` | `config_process_quantitative_results.yaml` |
| Best-so-far objective vs NFE and wall-clock time | `process_runchart_results.py` | `config_process_runchart_results.yaml` |
| Initial/best image comparisons across algorithms | `create_prompt_algorithm_grid.py` | `config_prompt_algorithm_grid.yaml` |

```bash
python algorithms/process_quantitative_results.py --config algorithms/config/config_process_quantitative_results.yaml
python algorithms/process_runchart_results.py --config algorithms/config/config_process_runchart_results.yaml
python algorithms/create_prompt_algorithm_grid.py --config algorithms/config/config_prompt_algorithm_grid.yaml
python algorithms/create_optimization_step_grid.py --config algorithms/config/config_optimization_step_grid.yaml
```

Quantitative outputs include an Excel workbook. Scores are aggregated across prompts; they are not repeated-seed confidence estimates. The script uses population standard deviation (`ddof=0`). Preserve the same family of pairwise tests when reproducing Holm-adjusted significance marks.

## Artifacts

Each prompt run stores its effective `config.yaml`, `checkpoint.pt`, baseline `it_0.jpg`, and generated images. AdamW writes `score_results.csv`; population methods, zero-order, and random sampling write `fitness_results.csv`. Best-image outputs include `best_all.jpg`. Retain the full run directories for resuming and processing.

For a single-prompt check, edit [config_run_single_prompt.yaml](algorithms/config/config_run_single_prompt.yaml) and run `python algorithms/run_single_prompt.py`. The repository also retains latent-noise optimization, other CMA-ES variants, and aesthetic/PickScore objectives; these are additional capabilities outside the main paper protocol described here.

## License

See [LICENSE](LICENSE): Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0).
