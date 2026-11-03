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

For SD 1.5, the working config uses `stable-diffusion-v1-5/stable-diffusion-v1-5`; SDXL Turbo can be selected with `stabilityai/sdxl-turbo`. Set `model_id` to the exact SANA-Sprint checkpoint used for your experiment. The manuscript does not specify its checkpoint size/revision, image resolution, precision, or numeric run seed; use the values in the supplied reproducibility configurations for each experiment. Do not substitute another SANA-Sprint size and assume equivalent results.

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

## 1. Tune hyperparameters on DiffusionDB

Start with [config_run_optuna_search.yaml](algorithms/config/config_run_optuna_search.yaml). Its `optuna.search_space` contains the ranges listed in the manuscript appendix, including conditional GA crossover parameters. Keep generation counts and population sizes fixed during tuning.

Enable **only one** `test_*` optimizer flag per study. The supplied file enables zero-order search; enabling another flag without disabling it would make the optimizer itself a trial parameter, which differs from the paper's separate studies.

| Method | Flag | Runtime configuration |
| --- | --- | --- |
| AdamW | `test_adam` | `optimization_method: adam` |
| GA | `test_ga` | `optimization_method: ga` |
| sep-CMA-ES | `test_cmaes` | `optimization_method: cmaes`, `cmaes_variant: sep` |
| SNES | `test_snes` | `optimization_method: snes` |
| CoSyNE | `test_cosyne` | `optimization_method: cosyne` |
| Zero-order | `test_zero_order` | `optimization_method: zero_order` |

Apply the model and objective settings above, then use these budget settings in both tuning and final evaluation:

```yaml
optimization_target: prompt_embeddings
num_generations: 15
pop_size: 16
snes_num_generations: 15
snes_pop_size: 16
cosyne_num_generations: 15
cosyne_pop_size: 16
zero_order_num_generations: 15
zero_order_pop_size: 16
num_iterations: 240
save_gens: true
time_limit_seconds: null
```

`save_gens: true` retains population images for subsequent canonical rescoring and qualitative analysis, at a substantial storage cost. It does not change the nominal population budget. The current code also evaluates an initial baseline; see budget accounting below.

To sample tuning prompts, configure the following nested settings inside the existing `optuna` section:

```yaml
optuna:
  n_trials: 30
  direction: maximize
  metric: auto
  study_name: sd15_snes_imagereward
  storage: sqlite:///results_optuna_sd15_snes_imagereward/optuna.db
  load_if_exists: true
  prompt_source:
    type: diffusiondb
    dataset: poloclub/diffusiondb
    subset: 2m_random_1k
    split: train
    count: 10
    seed: 42
    streaming: false
    require_clip_tokenizable: true
    clip_context_length: 77
  follow_up_experiment:
    enabled: false
```

This is an example sampling setup, not a claim about the original paper's unspecified sampling seed/subset. Alternatively, put the exact ten prompts in `selected_prompts` and use `optuna.prompt_source.type: config`. The supplied config already contains ten explicit prompts; verify their provenance against the original experiment records. Resolve the tuning set once and reuse it across all studies, keeping DrawBench out of tuning.

Use distinct `results_folder`, `optuna.study_name`, and `optuna.storage` values for every study. With persistent storage, `n_trials` adds trials on each invocation; use a new study to start a fresh 30-trial search.

```bash
python algorithms/run_optuna_search.py --config algorithms/config/config_run_optuna_search.yaml --dry-run
python algorithms/run_optuna_search.py --config algorithms/config/config_run_optuna_search.yaml
```

The results folder contains `optuna_search_summary.yaml`, including the best trial's sampled parameters and objective value, along with trial artifacts. Keep the SQLite study and resolved configurations. The implementation's current final-row trial selection differs from the manuscript's best-of-run definition; see the limitations below.

For sequential studies, edit [optuna_schedule.yaml](algorithms/config/optuna_schedule.yaml), then run:

```bash
python algorithms/optuna_schedule.py \
  --base-config algorithms/config/config_run_optuna_search.yaml \
  --schedule algorithms/config/optuna_schedule.yaml
```

Schedule entries merge recursively into the base configuration. Explicitly reset all optimizer flags and objective weights in each entry to avoid inheriting another active method or reward.

## 2. Evaluate the tuned configurations on DrawBench

Use [config_run_experiments.yaml](algorithms/config/config_run_experiments.yaml), set the chosen optimizer and best trial's parameters, and retain the same model, reward, precision, and budget used during tuning. Configure the full evaluation set:

```yaml
prompt_dataset: drawbench
prompt_dataset_path: null
prompt_dataset_config: null
prompt_dataset_split: null
use_entire_dataset: true
prompt_index_range: [0, 200]
prompt_per_category: null
single_prompt_per_seed: false
seed_path: null
seed: 42
resume_experiment: false
results_folder: results_snes_sd15_imagereward
```

Here `42` is an example; replace it with the original experiment seed when available. `seed_path: null` gives one run seed rather than repeated seeds from a file. Use all 200 prompts for final results. A small range such as `[0, 2]` is useful for a smoke run in a separate output folder.

```bash
python algorithms/run_experiments.py --config algorithms/config/config_run_experiments.yaml
```

To apply a study's best trial automatically, enable `optuna.follow_up_experiment` and point `config_path` to an existing experiment YAML. The launcher writes `best_trial_experiment_config.yaml` and runs it after tuning. Only the chosen optimizer and sampled hyperparameters override the experiment YAML; dataset, model, objective weights, budgets, seed, and output folder must already be correct there. Replace the supplied Optuna config's example `algorithms/config/tmp/...` path before enabling this feature.

For multiple tuned configurations, edit [experiments_schedule.yaml](algorithms/config/experiments_schedule.yaml):

```bash
python algorithms/experiments_schedule.py \
  --base-config algorithms/config/config_run_experiments.yaml \
  --schedule algorithms/config/experiments_schedule.yaml
```

The supplied schedules are partial working examples. Review inherited weights, GPU indices, and budgets before using them as a paper experiment matrix.

### Extended SNES runs

Keep the tuned SNES settings fixed. Use a separate output folder with a larger `snes_num_generations` and set `time_limit_seconds` to the corresponding AdamW mean runtime. Alternatively, extend an existing checkpoint with `resume_experiment: true` and a larger generation limit. Time limits are checked between generations, so matching is approximate; resumed runs use cumulative elapsed time.

For retrospective comparisons, `time_limit` in the quantitative-processing config truncates at steps whose **mean elapsed time across prompts** is within the specified budget. It is not a per-prompt cutoff. Keep regular-budget and extended SNES results separate, and exclude extended SNES from the paper's regular-budget significance comparisons.

### Sampling baseline

The manuscript's SD 1.5 table also includes Best-of-N. Use `optimization_method: random_sampler`, `optimization_target: prompt_embeddings`, and `num_images_to_generate: 240` to keep embeddings fixed while sampling generation seeds. Select the highest guiding-reward image. This is an additional baseline, not one of the six tuned embedding optimizers.

## 3. Rebuild the tables and figures

Update each processing YAML's input folders to your completed runs; checked-in paths refer to example local results.

| Paper artifact | Script | Configuration |
| --- | --- | --- |
| Best-of-run scores, runtime, VRAM, paired comparisons | `process_quantitative_results.py` | `config_process_quantitative_results.yaml` |
| Best-so-far objective vs NFE and wall-clock time | `process_runchart_results.py` | `config_process_runchart_results.yaml` |
| Initial/best image comparisons across algorithms | `create_prompt_algorithm_grid.py` | `config_prompt_algorithm_grid.yaml` |
| Images across optimization steps | `create_optimization_step_grid.py` | `config_optimization_step_grid.yaml` |
| Per-experiment aggregate reports | `aggregate_results.py` | `config_aggregate_results.yaml` |

For the regular-budget table, set `row_selector: best`, `time_limit: null`, and `metrics: canonical` in the quantitative config. Its current non-null time limit is an example truncation and must be cleared for full-budget reporting. Select only regular-budget folders when calculating significance. The script performs two-sided paired Wilcoxon tests on matching prompt texts, with Holm correction across valid tests in that processing invocation. Set `statistics.enabled: true` and `statistics.alpha: 0.05`.

```bash
python algorithms/process_quantitative_results.py --config algorithms/config/config_process_quantitative_results.yaml
python algorithms/process_runchart_results.py --config algorithms/config/config_process_runchart_results.yaml
python algorithms/create_prompt_algorithm_grid.py --config algorithms/config/config_prompt_algorithm_grid.yaml
python algorithms/create_optimization_step_grid.py --config algorithms/config/config_optimization_step_grid.yaml
```

Quantitative outputs include `quantitative_summary_long.csv`, `quantitative_summary_wide.csv`, `quantitative_prompt_values.csv`, `quantitative_pairwise_tests.csv` when enabled, and an Excel workbook. Scores are aggregated across prompts; they are not repeated-seed confidence estimates. The script uses population standard deviation (`ddof=0`). Preserve the same family of pairwise tests when reproducing Holm-adjusted significance marks.

For runcharts, select one model/objective group, set `best_so_far: true`, and assign matching `experiment_folders` and `experiment_labels`. The script exports curves against NFE and time. For image comparisons, use `layout: algorithm_x_prompt` to put algorithms in columns, and configure the selected prompt indices and guiding-score captions. Use the same prompt subset for all algorithms.

If an additional metric was skipped during generation, [fill_zero_weight_metrics.py](algorithms/fill_zero_weight_metrics.py) can evaluate saved images using [config_zero_weight_metrics.yaml](algorithms/config/config_zero_weight_metrics.yaml). This does not turn an experiment guided by one metric into an experiment optimized for another. The batch runner can also defer extra scoring via `evaluate_zero_weight_metrics` and `zero_weight_metrics`.

## Artifacts

New prompt-run folders use `results_<seed>_<prompt_number>` (for example, `results_42_1`), or `results_<seed>` for single-prompt runs without a prompt number. Result-processing scripts also accept the legacy `results_<aesthetic_model>_<seed>_<prompt_number>` format, so existing results can be processed without renaming their folders.

Each prompt run stores its effective `config.yaml`, `checkpoint.pt`, baseline `it_0.jpg`, and generated images. AdamW writes `score_results.csv`; population methods, zero-order, and random sampling write `fitness_results.csv`. Best-image outputs include `best_all.jpg`. Retain the full run directories for resuming and processing.

For a single-prompt check, edit [config_run_single_prompt.yaml](algorithms/config/config_run_single_prompt.yaml) and run `python algorithms/run_single_prompt.py`. The repository also retains latent-noise optimization, other CMA-ES variants, and aesthetic/PickScore objectives; these are additional capabilities outside the main paper protocol described here.

## License

See [LICENSE](LICENSE): Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0).
