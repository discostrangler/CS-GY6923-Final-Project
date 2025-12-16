# Scaling Laws for Symbolic Music Language Models (ABC)

This repository contains a complete pipeline for studying scaling laws in symbolic music generation using ABC notation. It covers corpus cleaning, character level tokenization, training matched size Transformer and RNN (LSTM) language models at multiple scales, analyzing validation loss scaling with power law fits, and generating plus evaluating ABC samples.

**Report:** `ML1.pdf`

## What is included

* Data cleaning and corpus statistics
* Character vocabulary builder and token dataset preparation
* Transformer scaling sweep (tiny, small, medium, large, xl)
* RNN scaling sweep (tiny, small, medium, large, xl)
* Scaling analysis and plots for both architectures
* Combined comparison plots and tables
* Sample generation and evaluation utilities, including test set perplexity

## Repository layout

### Top level folders

* `transformer/`
  * Transformer run logs, scaling artifacts, and evaluation scripts
* `rnn/`
  * RNN run logs, scaling artifacts, and summaries
* `compare/`
  * Combined comparison plots, per size loss curves, and tables

### Top level scripts

* `data_cleaning.py`
  * Cleans the ABC corpus and applies quality filters
* `data_stats.py`
  * Computes dataset statistics such as file counts and length distributions
* `build_tokenizer.py`
  * Builds the character vocabulary
* `prepare_tokens.py`
  * Converts the cleaned corpus into token IDs and writes train, val, test splits
* `train_transformers.py`
  * Trains Transformer models across sizes
* `train_rnns.py`
  * Trains RNN (LSTM) models across sizes
* `trans_analyze_scaling.py`
  * Parses Transformer logs and produces scaling summaries and plots
* `compare_models.py`
  * Produces combined comparison plots and a summary table

### HPC scripts and logs

* `tokenize_abc.sbatch`, `prepare_tokens.sbatch`
  * Slurm templates for preprocessing
* `*.out`
  * Logs captured from preprocessing and training runs

## Folder details

### `transformer/`

Files typically found in `transformer/`:

* Evaluation utilities
  * `generate_samples.py`
  * `evaluate_abc.py`
  * `compute_test_perplexity.py`
* Scaling artifacts
  * `trans_scaling_plot.png`
  * `trans_scaling_summary.csv`
  * `trans_train_curve.png`
* Logs
  * `trans_tiny.out`, `trans_small.out`, `trans_medium.out`, `trans_large.out`, `trans_xl.out`
  * `trans_analyze_scaling.out`

### `rnn/`

Files typically found in `rnn/`:

* `rnn_analyze_scaling.py`
* `rnn_scaling_plot.png`
* `rnn_scaling_summary.csv`
* `rnn_train_curves.png`
* `scaling_results.csv`
* Logs
  * `rnn_tiny.out`, `rnn_small.out`, `rnn_medium.out`, `rnn_large.out`, `rnn_xl.out`

### `compare/`

Files typically found in `compare/`:

* `compare_scaling_plot.png`
* `compare_table.csv`
* Per size curve comparisons
  * `compare_curve_tiny.png`
  * `compare_curve_small.png`
  * `compare_curve_medium.png`
  * `compare_curve_large.png`
  * `compare_curve_xl.png`

## Quickstart

### 1) Environment

Recommended:

* Python 3.10+
* PyTorch
* numpy
* matplotlib

If you do not have a pinned environment file, install the core dependencies:

```bash
pip install torch numpy matplotlib
```

### 2) Configure paths

Most scripts assume file paths are set near the top of the file. Before running, open each script and update:

* input ABC directory or corpus file
* output directories
* checkpoint paths (for sample generation and perplexity)

## Data pipeline

### Step 1: Clean the corpus

```bash
python data_cleaning.py
```

This stage removes unusable outputs (empty or malformed files) and applies the project filters described in the report.

### Step 2: Dataset stats

```bash
python data_stats.py
```

This produces corpus statistics such as counts and length distribution.

### Step 3: Build vocabulary

```bash
python build_tokenizer.py
```

This builds a character vocabulary from the cleaned corpus.

### Step 4: Tokenize and write dataset splits

```bash
python prepare_tokens.py
```

This converts text into token IDs and writes train, validation, and test arrays (for example `train.npy`, `val.npy`, `test.npy`).

### HPC option

If you are running preprocessing on Slurm:

```bash
sbatch tokenize_abc.sbatch
sbatch prepare_tokens.sbatch
```

## Training

The scaling sweeps train 5 sizes: tiny, small, medium, large, xl. The scaling experiments are run under a controlled one epoch budget for comparable results.

### Train Transformers

```bash
python train_transformers.py
```

Transformer logs are saved as `trans_*.out` and artifacts are written under `transformer/`.

### Train RNNs

```bash
python train_rnns.py
```

RNN logs are saved as `rnn_*.out` and artifacts are written under `rnn/`.

## Scaling analysis

### Transformer scaling analysis

```bash
python trans_analyze_scaling.py
```

Expected outputs:

* `transformer/trans_scaling_summary.csv`
* `transformer/trans_scaling_plot.png`
* `transformer/trans_train_curve.png`

### RNN scaling analysis

```bash
python rnn/rnn_analyze_scaling.py
```

Expected outputs:

* `rnn/rnn_scaling_summary.csv`
* `rnn/rnn_scaling_plot.png`
* `rnn/rnn_train_curves.png`

## Combined comparison

```bash
python compare_models.py
```

Expected outputs under `compare/`:

* `compare_scaling_plot.png`
* `compare_table.csv`
* `compare_curve_*.png`

## Sample generation and evaluation

Sample generation and evaluation scripts live in `transformer/`.

### Generate ABC samples

```bash
python transformer/generate_samples.py
```

### Evaluate ABC validity

```bash
python transformer/evaluate_abc.py
```

This checks syntactic validity and other lightweight formatting constraints. If ABC to MIDI conversion is configured in your environment, this script can also report conversion success.

### Compute test set perplexity

```bash
python transformer/compute_test_perplexity.py
```

This evaluates perplexity on the held out test split built during `prepare_tokens.py`.

## Results summary (from the report)

* After filtering, the corpus contains 120,000,000 tokens with a 103 character vocabulary
* Under the fixed one epoch scaling setup, the XL Transformer achieved the best validation loss among Transformer sizes
* Power law fits show stronger scaling for Transformers than for RNNs in this setup
* Sample evaluation reports test perplexity and the fraction of generated ABC that successfully converts to MIDI

For exact numbers and plots, see `ML1.pdf` and the generated artifacts under `transformer/`, `rnn/`, and `compare/`.

## Common issues

* If a scaling analysis script does not parse logs, ensure the log format matches what the parser expects and that the `.out` file paths are correct
* If perplexity cannot be computed, confirm that `test.npy` exists and that the vocabulary used for evaluation matches the training vocabulary

## License

Add a license if you plan to keep this repository public.
