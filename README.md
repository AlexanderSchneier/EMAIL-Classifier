# Email Classifier

**Pattern Recognition Final Project** — Upi Shanker & Alexander Schneier

Multi-class email classifier that goes beyond binary spam detection. Classifies emails into four categories — **spam, phishing, personal, and work** — using a combination of word-level TF-IDF, character n-grams, and sender metadata features across five classifiers.

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK data (stopwords, wordnet, punkt) downloads automatically on first run.

---

## Data

The pipeline uses two datasets:
- **CMU-annotated Enron corpus** (`data/raw/enron/brianray-enron-email-dataset/`) — provides work/personal/spam labels via human annotations
- **Nazario phishing corpus** (`data/raw/phishing/*.mbox`) — provides ground-truth phishing examples

```bash
# Download phishing corpus (no account needed)
python scripts/download_data.py --phishing

# Download Enron dataset (requires Kaggle API credentials)
python scripts/download_data.py --enron
```

---

## Usage

```bash
# Full pipeline — all datasets, all models
python run_pipeline.py

# Enron only
python run_pipeline.py --dataset enron

# Specific models (nb, lr, svm, rf, nn)
python run_pipeline.py --models nb,lr,svm

# Force re-preprocessing from raw data (otherwise cached CSV is reused)
python run_pipeline.py --force-preprocess

# Quick test run on a subset
python run_pipeline.py --max-samples 5000
```

---

## Models

| Key | Model | Features |
|-----|-------|----------|
| `nb` | Naive Bayes | TF-IDF only |
| `lr` | Logistic Regression | TF-IDF + char n-grams + metadata |
| `svm` | Linear SVM | TF-IDF + char n-grams + metadata |
| `rf` | Random Forest | SVD(300) of full features |
| `nn` | Neural Network (MLP 512→256→128) | SVD(300) of full features |

All models are evaluated on a stratified 80/20 train/test split.

---

## Outputs

Results are saved to `outputs/` after each run:

```
outputs/
├── figures/
│   ├── label_distribution.png
│   ├── confusion_matrix_<model>.png   (one per model)
│   └── model_comparison.png
└── results/
    └── model_comparison.csv
```

---

## Project Structure

```
EMAIL-Classifier/
├── run_pipeline.py         # Entry point
├── config/settings.py      # All hyperparameters and paths
├── scripts/download_data.py
├── src/
│   ├── data/               # loader, preprocessor, labeler
│   ├── features/           # tfidf, char n-grams, metadata
│   ├── models/             # nb, lr, svm, rf, nn
│   └── evaluation/         # metrics, visualizer
├── data/raw/               # enron/, phishing/
└── tests/
```

---

## Tests

```bash
python -m pytest tests/ -v
```
