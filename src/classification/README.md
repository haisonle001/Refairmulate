# Classification

This directory contains a self-contained module for training and using a text classification model, likely for query classification tasks.

## Overview

The module uses a BERT-based model for sequence classification. It provides scripts for training the model on a custom dataset and for performing inference on new data.

## Directory Structure

```
classification/
├── data/                      # Data files for training and testing
├── model/                     # Saved model artifacts
├── src/                       # Source code for training and inference
│   ├── train.py               # Main training script
│   ├── infer.py               # Single-instance inference script
│   └── infer_batch.py         # Batch inference script
├── scripts/                   # Shell scripts for running tasks
│   ├── train.sh               # Script to run training
│   ├── infer.sh               # Script for single inference
│   └── infer_batch.sh         # Script for batch inference
├── data_loader/               # Data loading utilities
│   ├── dataloader.py
│   └── dataset.py
└── requirements.txt           # Python dependencies
```

## Dataset Format

The training and inference scripts expect tab-separated values (TSV) files with the following format:

- **Training data (`.tsv`)**:
  - `sequence`: The text sequence to classify
  - `label`: The ground truth label
  
  **Example**:
  ```tsv
  q1	"is this a neutral query?"	n
  q2	"man query text"	m
  q3  "female job"  f
  ```

- **Inference data (`.tsv`)**:
  - `id`: Unique identifier
  - `sequence`: The text sequence to classify

  **Example**:
  ```tsv
  q3	"how to classify this?"
  q4	"another sequence"
  ```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   Place your training data (e.g., `train.tsv`) and testing data in the `data/` directory.

## Training

- **Using Python command**:
  ```bash
  python src/train.py --train_file data/your_train_data.tsv
  ```
  The trained model will be saved to a `bert_model` directory by default.

## Inference

### Single Inference

To perform inference on a single text sequence:

- **Using shell script**:
  ```bash
  # The script takes the model directory and text as arguments
  bash scripts/infer.sh path/to/your/model "your text to classify"
  ```

- **Using Python command**:
  ```bash
  python src/infer.py --model_dir path/to/your/model --text "your text to classify"
  ```

### Batch Inference

To perform inference on a TSV file of text sequences:

- **Using shell script**:
  ```bash
  # The script takes the model path, input file, and output file
  bash scripts/infer_batch.sh path/to/model data/input.tsv results.tsv
  ```

- **Using Python command**:
  ```bash
  python src/infer_batch.py \
      --model_dir path/to/your/model \
      --input_file data/your_input.tsv \
      --output_file results.tsv
  ```

The output file will be a TSV with the ID, sequence, and predicted label.

## Model

The training script uses `bert-base-uncased` by default, but this can be changed in `src/train.py`. The trained model, including the tokenizer and configuration, is saved to the specified output directory. 