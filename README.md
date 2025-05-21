# model-training

This repository contains the training pipeline for the sentiment analysis model used in our REMLA project. 

- It uses the [lib-ml](https://github.com/remla25-team21/lib-ml) library for data preprocessing and saves the trained model (`sentiment_model_*.pkl`) as a release artifact. 
- The training dataset can be found in `data/raw/a1_RestaurantReviews_HistoricDump.tsv`.
- The project now uses [DVC](https://dvc.org/) (Data Version Control) to track data, models, and metrics.

> [!NOTE]
> TL;DR: 
> 1. Clone the repository
> ```bash
> git clone https://github.com/remla25-team21/model-training.git
> ```
> 2. Install the required dependencies
> ```bash
> pip install -r requirements.txt
> ```
> 3. Configure DVC remote storage
> ```bash
> dvc remote modify storage gdrive_client_id <xxx> --local  # Replace <xxx> with your Google Drive client ID
> dvc remote modify storage gdrive_client_secret <xxx> --local  # Replace <xxx> with your Google Drive client secret
> ```
> 4. Pull the data from remote storage
> ```bash
> dvc pull
> ```
> 5. Run the pipeline
> ```bash
> dvc repro
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
dvc remote modify storage gdrive_client_id <xxx> --local  # Replace <xxx> with your Google Drive client ID
dvc remote modify storage gdrive_client_secret <xxx> --local  # Replace <xxx> with your Google Drive client secret
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

The old training script is still available for backward compatibility:

```bash
python -m src.train_model 
```

## Pipeline Outputs

The pipeline produces the following artifacts:

- `preprocessed_data_*.pkl`: Preprocessed data (features and labels)
- `c1_BoW_Sentiment_Model_*.pkl`: Text vectorizer model
- `trained_model_*.pkl`: Trained ML model before evaluation
- `sentiment_model_*.pkl`: Final ML model after evaluation
- `metrics_*.json`: Model performance metrics
