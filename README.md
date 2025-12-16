Scaling Laws for Symbolic Music Language Models (ABC)

This repository contains an end to end project studying scaling behavior for language models trained on symbolic music represented in ABC notation. The workflow covers data cleaning and tokenization, matched size scaling sweeps for Transformer and RNN (LSTM) language models, scaling curve analysis with power law fits, and sample generation plus evaluation.

The full report is included in this repo as ML1.pdf.

What is included
	•	Data cleaning and statistics for an ABC corpus
	•	Character level tokenizer and dataset preparation to NumPy arrays
	•	Transformer scaling sweep (tiny to XL)
	•	RNN scaling sweep (tiny to XL)
	•	Scaling analysis scripts that produce summary CSVs and plots
	•	A combined comparison module producing side by side plots and a comparison table
	•	Sample generation plus evaluation utilities, including test set perplexity

Repository layout

Top level:
	•	data_cleaning.py
Cleans the raw ABC corpus and applies quality filters.
	•	data_stats.py
Computes dataset statistics such as file counts and length distributions.
	•	build_tokenizer.py
Builds a character vocabulary from the cleaned corpus.
	•	prepare_tokens.py
Converts the cleaned corpus into token IDs and writes dataset splits.
	•	compare_models.py
Produces combined plots and tables comparing Transformer and RNN scaling.
	•	prepare_tokens.sbatch, tokenize_abc.sbatch
Slurm batch scripts used for preprocessing on HPC.
	•	*.out
Captured logs from preprocessing and training runs.

Folders:
	•	transformer/
Transformer training, scaling analysis, plots, summaries, and sample evaluation:
	•	Training: train_transformers.py, train_trans_xl.py, train_trans_xl1.py
	•	Scaling analysis: trans_analyze_scaling.py, trans_scaling_summary.csv, trans_scaling_plot.png, trans_train_curve.png
	•	Sampling and evaluation: generate_samples.py, evaluate_abc.py, compute_test_perplexity.py
	•	Logs: trans_*.out, train_trans_xl.out
	•	rnn/
RNN training logs, scaling analysis, plots, and summaries:
	•	Training logs: rnn_tiny.out, rnn_small.out, rnn_medium.out, rnn_large.out, rnn_xl.out
	•	Scaling analysis: rnn_analyze_scaling.py, rnn_scaling_summary.csv, rnn_scaling_plot.png, rnn_train_curves.png
	•	compare/
Combined comparison artifacts:
	•	compare_scaling_plot.png
	•	compare_curve_tiny.png, compare_curve_small.png, compare_curve_medium.png, compare_curve_large.png, compare_curve_xl.png
	•	compare_table.csv

Data pipeline

Overview

The pipeline assumes you start with an ABC corpus generated from a symbolic music dataset (for example Lakh MIDI converted to ABC). The project then:
	1.	Cleans the corpus and removes unusable files
	2.	Computes summary statistics (counts and length distribution)
	3.	Builds a character vocabulary
	4.	Tokenizes and writes splits as NumPy arrays for fast training

Run preprocessing

Run these in order:

python data_cleaning.py
python data_stats.py
python build_tokenizer.py
python prepare_tokens.py

If you are running on Slurm, use the provided scripts as templates:

sbatch tokenize_abc.sbatch
sbatch prepare_tokens.sbatch

Outputs produced by preprocessing typically include:
	•	vocabulary JSON (character to id mapping)
	•	train.npy, val.npy, test.npy
	•	stats logs from data_stats.py

Modeling and experiments

This project trains two architecture families under a controlled setup:
	•	Transformer: decoder only causal self attention language model
	•	RNN: stacked LSTM language model

Both are evaluated at multiple sizes: tiny, small, medium, large, xl.

Training configuration

Training is designed to be comparable across model sizes and architectures.

Common settings:
	•	Optimizer: AdamW
	•	Learning rate: 3e-4
	•	Weight decay: 0.1
	•	Dropout: 0.1
	•	Context length: 256
	•	Batch size: 64 sequences
	•	Gradient clipping: 1.0
	•	LR schedule: linear warmup then cosine decay
	•	Budget: 1 epoch for scaling sweeps

Transformer scaling sweep

Run the Transformer sweep:

python transformer/train_transformers.py

If you trained XL separately, use:

python transformer/train_trans_xl.py

Analyze Transformer scaling:

python transformer/trans_analyze_scaling.py

Expected outputs:
	•	transformer/trans_scaling_summary.csv
	•	transformer/trans_scaling_plot.png
	•	transformer/trans_train_curve.png

RNN scaling sweep

Run the RNN workflow:

python train_rnns.py

Analyze RNN scaling:

python rnn/rnn_analyze_scaling.py

Expected outputs:
	•	rnn/rnn_scaling_summary.csv
	•	rnn/rnn_scaling_plot.png
	•	rnn/rnn_train_curves.png

Combined comparison

Once both scaling summaries exist, generate combined plots and tables:

python compare_models.py
python compare_models.py

Primary artifacts are written under compare/:
	•	compare_scaling_plot.png
	•	compare_table.csv
	•	per size loss curve comparisons

Best model selection and sample generation

Based on the scaling study, the XL Transformer achieved the best validation loss under the fixed one epoch budget, so it is used for sample generation and evaluation.

Generate samples

python transformer/generate_samples.py

Evaluate samples

This repo includes three evaluation utilities:
	•	ABC syntactic validity and formatting checks
	•	ABC to MIDI conversion success (if MIDI tooling is available)
	•	Test set perplexity

Run evaluation:

python transformer/evaluate_abc.py
python transformer/compute_test_perplexity.py

Outputs and where to look
	•	Transformer scaling artifacts: transformer/
	•	RNN scaling artifacts: rnn/
	•	Combined comparison: compare/
	•	Raw logs: *.out and transformer/trans_*.out, rnn/rnn_*.out

If you are grading or reviewing quickly, the most important files are:
	•	transformer/trans_scaling_plot.png
	•	rnn/rnn_scaling_plot.png
	•	compare/compare_scaling_plot.png
	•	compare/compare_table.csv
	•	ML1.pdf

Notes and limitations
	•	Scaling experiments use a fixed one epoch training budget to keep runs comparable. This is useful for controlled scaling analysis but does not measure fully converged performance.
	•	ABC is format sensitive, so sample evaluation includes MIDI conversion success as a strict end to end validity check.
	•	Qualitative evaluation depends on listening to converted MIDI outputs and manual inspection of ABC structure.

Requirements
	•	Python 3.10+
	•	PyTorch
	•	numpy
	•	matplotlib

If you have a requirements.txt or environment.yml, install from that. Otherwise, install the main dependencies with pip.

License

Add a license if you plan to keep this repository public.

Citation

If you reuse this code or results, cite the report in ML1.pdf and reference the source symbolic music dataset used to produce the ABC corpus.
