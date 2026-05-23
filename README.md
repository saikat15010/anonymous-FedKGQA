# FedKGQA Artifact

This repository contains the source code and dataset files for **FedKGQA**, a federated learning framework for multi-hop Knowledge Graph Question Answering (KGQA). The artifact supports experiments on distributed KGQA settings, including 2-hop and 3-hop reasoning, differential privacy analysis, ablation studies, and baseline comparisons.

## Repository Structure

```text
.
├── 2-hop-experiments/       # Code for 2-hop KGQA experiments
├── 3-hop-experiments/       # Code for 3-hop KGQA experiments
├── ablation-study/          # Code for centralized, no-aggregation, and aggregation ablations
├── baseline-comparison/     # Code for adapted baseline comparisons
├── dataset/                 # Dataset files used in the experiments
└── dp-experiments/          # Code for differential privacy experiments
```

## Description

The repository is organized by experiment type. Each experiment folder contains the corresponding training, evaluation, model, dataloader, and inference scripts. The dataset folder contains the processed dataset files used by the experiments.

The artifact covers:

- 2-hop federated KGQA experiments
- 3-hop federated KGQA experiments
- Differential privacy experiments
- Ablation studies
- Baseline comparison experiments
- Processed dataset files for MetaQA and PathQuestion

## Datasets

The `dataset/` directory contains the processed data used in the experiments. It includes dataset files for MetaQA and PathQuestion under 2-hop and 3-hop settings.

Expected structure:

```text
dataset/
├── MetaQA/
└── PathQuestion/
```

Depending on the experiment, the scripts read data from the corresponding subdirectory under `dataset/`.

## How to Run

Each experiment directory contains a `.txt` instruction file that explains how to run that specific experiment. Please check the relevant `.txt` file before running a script.

Examples of instruction files include:

```text
CODE_RUN_INSTRUCTIONS.txt
commands_train_all.txt
commands_eval_all.txt
commands_train_all_distmult.txt
commands_eval_all_distmult.txt
commands_train_all_rotate.txt
commands_eval_all_rotate.txt
run.txt
```

To reproduce a specific experiment:

1. Open the relevant experiment folder.
2. Read the `.txt` instruction file inside that folder.
3. Follow the listed training and evaluation commands.

For example:

```bash
cd 2-hop-experiments/PathQuestion/ComplEx/Client3
cat commands_train_all.txt
cat commands_eval_all.txt
```

Then run the commands shown in those files.

## Environment Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages according to your local environment. The main dependencies are commonly used Python machine learning libraries, including:

```bash
pip install torch transformers numpy pandas scikit-learn tqdm
```

Depending on the specific experiment and system configuration, additional packages may be required.

## Experiment Groups

### 2-hop Experiments

The `2-hop-experiments/` directory contains scripts for evaluating FedKGQA on 2-hop KGQA tasks. Experiments are organized by dataset, encoder, KGE model, and number of clients.

### 3-hop Experiments

The `3-hop-experiments/` directory contains scripts for evaluating FedKGQA on 3-hop KGQA tasks for MetaQA and PathQuestion.

### Differential Privacy Experiments

The `dp-experiments/` directory contains scripts for evaluating FedKGQA under differential privacy settings.

### Ablation Study

The `ablation-study/` directory contains scripts for comparing centralized training, federated training without aggregation, and federated training with aggregation.

### Baseline Comparison

The `baseline-comparison/` directory contains adapted baseline implementations used for comparison with FedKGQA.

## Notes

- The repository is organized to match the experimental settings used in the paper.
- Some folders contain separate scripts for different encoders, KGE models, client counts, and datasets.
- The `.txt` files inside each folder provide the most reliable commands for running that specific experiment.
- Large intermediate outputs, trained checkpoints, and generated logs may need to be created locally by running the corresponding scripts.

## License

This artifact is provided for research and reproducibility purposes.
