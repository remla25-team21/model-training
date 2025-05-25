# model-training

<!-- PYLINT_BADGE_START -->
![Pylint Score](https://img.shields.io/badge/pylint-10%2E00%2F10-brightgreen)
<!-- PYLINT_BADGE_END -->

<!-- COVERAGE_BADGE_START -->
![Coverage](https://codecov.io/gh/remla25-team21/model-training/graph/badge.svg)
<!-- COVERAGE_BADGE_END -->

<!-- ML_SCORE_BADGE_START -->
![ML Test Score](https://img.shields.io/badge/ML%20Test%20Score-12%2F12-brightgreen)
<!-- ML_SCORE_BADGE_END -->

This repository contains the training pipeline for the sentiment analysis model used in our REMLA project. 

- It uses the [lib-ml](https://github.com/remla25-team21/lib-ml) library for data preprocessing and saves the trained model (`sentiment_model_*.pkl`) as a release artifact.
- The training dataset can be found in `data/raw/a1_RestaurantReviews_HistoricDump.tsv`.
- The project now uses [DVC](https://dvc.org/) (Data Version Control) to track data, models, and metrics.

> [!NOTE]
> TL;DR:
>
> 1. Clone the repository
>
> ```bash
> git clone https://github.com/remla25-team21/model-training.git
> ```
>
> 2. Install the required dependencies
>
> ```bash
> pip install -r requirements.txt
> ```
>
> 3. (Optional) Configure DVC remote storage (only needed if you want to push changes to the remote storage or if `dvc pull` doesn't work without authentication)
>
> ```bash
> dvc remote modify storage --local gdrive_use_service_account true
> dvc remote modify storage --local gdrive_service_account_json_file_path <path/to/file.json>  # Replace with your Google Drive service account JSON file path
> ```
>
> 4. Pull the data from remote storage or download it directly (see [Troubleshooting](#troubleshooting) section if facing issues)
>
> ```bash
> dvc pull
> ```
>
> 5. Run the pipeline
>
> ```bash
> dvc repro
> ```
>
> 6. Run the test 
>
> ```bash
> pytest
> ```

## Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## DVC Pipeline

The training process is now divided into three stages using DVC:

1. **Preprocessing**: Data preparation and feature extraction
2. **Training**: Model training with hyperparameter tuning
3. **Evaluation**: Model evaluation and metrics generation

To configure the DVC pipeline, run:

```bash
dvc remote modify storage --local gdrive_use_service_account true
dvc remote modify storage --local gdrive_service_account_json_file_path <path/to/file.json>  # Replace with your Google Drive service account JSON file path
```

To pull the data from the remote storage:

```bash
dvc pull
```

To run the complete pipeline:

```bash
dvc repro
```

To run a specific stage:

```bash
dvc repro <stage_name>  # e.g., dvc repro preprocess
```

To view metrics:

```bash
dvc metrics show
```

To view all experiments:

```bash
dvc exp show
```

For more details on collaborating with DVC, refer to [./docs/dvc-ref.md](./docs/dvc-ref.md).

## Troubleshooting

### Google Authentication Issues

If you encounter "This app is blocked" error during Google authentication when using DVC with Google Drive, you can download the dataset directly using one of these methods:

#### Linux/macOS

```bash
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1mrWUgJlRCf_n_TbxPuuthJ9YsTBwGuRh' -O ./data/raw/a1_RestaurantReviews_HistoricDump.tsv
```

#### Windows (PowerShell)

```powershell
Invoke-WebRequest -Uri "https://drive.google.com/uc?export=download&id=1mrWUgJlRCf_n_TbxPuuthJ9YsTBwGuRh" -OutFile "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
```

After downloading the dataset directly, you can proceed with the pipeline by running:

```bash
dvc repro
```

## Manual Training

If you prefer to run each stage manually:

```bash
# Preprocessing
python src/preprocess.py

# Training
python src/train.py

# Evaluation
python src/evaluate.py
```

## Pipeline Outputs

The pipeline produces the following artifacts:

* `preprocessed_data_*.pkl`: Preprocessed data (features and labels)
* `c1_BoW_Sentiment_Model_*.pkl`: Text vectorizer model
* `trained_model_*.pkl`: Trained ML model before evaluation
* `sentiment_model_*.pkl`: Final ML model after evaluation
* `metrics_*.json`: Model performance metrics

# Linters

Linters help improve code quality by identifying errors, enforcing style rules, and spotting security issues without running the code.

## Linters Used

* **Pylint**: Checks for coding errors and enforces standards.
* **Flake8**: Checks code style and complexity.
* **Bandit**: Scans for security vulnerabilities in Python code.

## How to Run

To run all linters and generate reports:

### For Mac/Linux

```bash
bash lint.sh
```

### For Windows

Use Git Bash as your terminal:

```bash
1. chmod +x lint.sh
```

```bash
2. ./lint.sh
```

## ML Test Score

<!-- ML_TEST_SCORE_START -->

<!-- ML_TEST_SCORE_END -->