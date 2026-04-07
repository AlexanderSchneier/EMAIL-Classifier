# Smart Multi-Class Email Classifier

**Pattern Recognition Final Project**
Upi Shanker & Alexander Schneier

A machine learning pipeline that classifies emails into six categories: **spam, phishing, promotions, social/notifications, personal, and work**. Goes beyond traditional binary spam detection by combining word-level TF-IDF, character n-grams, and sender metadata features across five different classifiers.

> **Active label set:** `config/settings.py` has a `LABELS` list that controls which classes are used. It is set to `["spam", "phishing", "personal", "work"]`. Phishing labels come from the Nazario phishing corpus (real ground truth); the remaining classes come from CMU-annotated Enron data.

---

## Table of Contents

- [Project Structure](#project-structure)
- [File Reference](#file-reference)
- [Setup](#setup)
- [Getting the Data](#getting-the-data)
- [Running the Project](#running-the-project)
- [Outputs](#outputs)
- [Models](#models)

---

## Project Structure

```
EMAIL-Classifier/
├── run_pipeline.py
├── requirements.txt
├── scripts/
│   └── download_data.py
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   │   ├── enron/
│   │   ├── spamassassin/
│   │   └── phishing/        ← Nazario .mbox files go here
│   └── processed/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── labeler.py
│   ├── features/
│   │   ├── tfidf_features.py
│   │   ├── ngram_features.py
│   │   └── metadata_features.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── neural_network.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualizer.py
├── outputs/
│   ├── figures/
│   └── results/
└── tests/
    ├── test_loader.py
    ├── test_preprocessor.py
    ├── test_features.py
    └── test_models.py
```

---

## File Reference

### `run_pipeline.py`

The main entry point. Runs the entire pipeline end-to-end: loads data, preprocesses it, extracts features, trains all five models, evaluates each one, and saves all results and plots. Accepts command-line arguments to control which dataset and models to use.

---

### `scripts/download_data.py`

Downloads all datasets automatically. SpamAssassin and the Nazario phishing corpus are pulled from public servers (no credentials needed). Enron is pulled via the Kaggle API (free account required). Run with `--spamassassin`, `--phishing`, `--enron`, or `--all`. See [Getting the Data](#getting-the-data) for full setup instructions.

---

### `requirements.txt`

Lists all Python dependencies. Install with `pip install -r requirements.txt`.

---

### `config/settings.py`

Central configuration file. Every constant used across the project is defined here — file paths, label names, feature hyperparameters, train/test split ratio, random seed, and keyword lists used for label assignment. Edit this file to tune the project without touching any other code.

Key settings:
- `LABELS` — the active class names (defaults to `["spam", "personal", "work"]`, the only classes with human-annotated ground truth in the included datasets)
- `MAX_TFIDF_FEATURES`, `TFIDF_NGRAM_RANGE` — controls word TF-IDF vocabulary size
- `SVD_COMPONENTS` — number of latent dimensions for Random Forest and MLP
- `CMU_LABEL_MAP` — maps CMU annotation codes to class labels (used with the brianray Enron dataset)
- `PHISHING_KEYWORDS`, `PROMOTIONS_KEYWORDS`, `SOCIAL_KEYWORDS`, `WORK_KEYWORDS`, `PERSONAL_KEYWORDS` — heuristics used to split binary labels into six classes when CMU annotations are absent

---

### `src/data/loader.py`

Reads raw `.txt` email files from disk using Python's standard library `email` module, which handles MIME encoding, multipart bodies, and charset decoding automatically.

Supports multiple Enron formats in priority order:
1. **brianray CMU-annotated dataset** (`data/raw/enron/brianray-enron-email-dataset/`) — preferred; passes CMU genre annotations to the labeler
2. **Plain CSV** (`data/raw/enron/*.csv`) — auto-detects common Kaggle column layouts
3. **spam/ham folder layout** — walks `data/raw/enron/spam/` and `data/raw/enron/ham/`

- `load_enron(path)` — loads using the highest-priority format found
- `load_spamassassin(path)` — walks `data/raw/spamassassin/` subdirectories (spam, easy_ham, hard_ham, spam_2, easy_ham_2)
- `load_all()` — calls both loaders and concatenates the results

Output columns: `subject`, `sender`, `body`, `raw_text`, `label_raw`, `source`

---

### `src/data/preprocessor.py`

Cleans and normalizes email text. Operates on the `raw_text` column and adds a `clean_text` column.

Steps applied in order:
1. Lowercase
2. Strip HTML tags
3. Remove URLs and email addresses
4. Remove non-alphabetic characters
5. Tokenize with NLTK
6. Remove stopwords
7. Lemmatize (or stem, configurable in `settings.py`)

---

### `src/data/labeler.py`

Converts binary `spam`/`ham` labels from the source datasets into the six-class schema.

**Primary path — CMU annotations (brianray Enron dataset):** genre/topic codes (`cat_i_level_1`, `cat_i_level_2`, `cat_i_weight`) are mapped to classes via `CMU_LABEL_MAP` in `settings.py`. The highest-weight annotation wins when multiple slots are populated.

**Fallback — keyword heuristics (all other data):**
- `spam` emails → checked for phishing keywords → classified as `phishing` or `spam`
- `ham` emails → checked in priority order for promotions, social, work, and personal keywords

The keyword lists are defined in `config/settings.py` and can be extended. Also logs a class distribution summary and warns if any class has fewer than 2% of samples.

---

### `src/features/tfidf_features.py`

Wraps scikit-learn's `TfidfVectorizer` to extract word-level n-gram features (unigrams and bigrams by default). Fits only on training data to prevent leakage. Returns a sparse matrix.

---

### `src/features/ngram_features.py`

Extracts character-level n-grams (3–5 character windows) using `TfidfVectorizer` with `analyzer="char_wb"`. Character n-grams are strong signals for phishing and spam because they capture suspicious substrings regardless of word boundaries. Fits only on training data.

---

### `src/features/metadata_features.py`

Extracts structural and sender-based features that do not depend on email content:

| Feature | Description |
|---|---|
| `subject_length` | Character count of the subject line |
| `body_length` | Character count of the body |
| `has_html` | 1 if the body contains HTML tags |
| `exclamation_count` | Number of `!` in the subject |
| `num_links` | Number of `http` occurrences in the body |
| `sender_domain_*` | One-hot encoding of the sender's domain (top 50) |
| `sender_frequency` | Normalised frequency of the sender in the training set |

Sender frequency is computed only from training data to avoid leakage. Returns a sparse matrix that is stacked with TF-IDF and character n-gram features.

---

### `src/models/base_model.py`

Abstract base class that all five classifiers implement. Defines the interface: `fit()`, `predict()`, `predict_proba()`, and `get_params()`. Allows `run_pipeline.py` to iterate over all models in a loop without any branching logic.

---

### `src/models/naive_bayes.py`

`MultinomialNB` classifier. Works on TF-IDF word features only — MultinomialNB requires non-negative inputs, so metadata features (which may include scaled values) are excluded. Fast baseline with strong performance on text.

---

### `src/models/logistic_regression.py`

`LogisticRegression` with `lbfgs` solver. Accepts the full combined feature matrix (TF-IDF + character n-grams + metadata). Also exposes `get_top_features_per_class()` which returns the highest-weighted TF-IDF tokens for each class — useful for interpretability and sanity checking the model.

---

### `src/models/svm.py`

`LinearSVC` wrapped in `CalibratedClassifierCV` to support probability estimates. LinearSVC is used instead of kernel SVM because it scales linearly with the number of samples and features, making it practical for the Enron dataset (~33,000 emails). The calibration wrapper uses 3-fold cross-validation.

---

### `src/models/random_forest.py`

`RandomForestClassifier` preceded by `TruncatedSVD` (Latent Semantic Analysis). Tree-based models do not benefit from high-dimensional sparse matrices, so the feature space is reduced to 300 latent dimensions before training. Uses all CPU cores (`n_jobs=-1`).

---

### `src/models/neural_network.py`

`MLPClassifier` with a 3-layer architecture `(512 → 256 → 128)`, ReLU activations, Adam optimizer, and early stopping. Also uses `TruncatedSVD` preprocessing (same as Random Forest). Labels are integer-encoded internally before fitting to ensure compatibility with newer scikit-learn versions.

---

### `src/evaluation/metrics.py`

- `compute_metrics(y_true, y_pred)` — returns accuracy plus a full per-class precision/recall/F1 report
- `save_metrics_table(metrics, model_name)` — appends the model's results to `outputs/results/model_comparison.csv`. Re-running a model overwrites its previous rows.
- `compare_models()` — loads the CSV and returns a summary DataFrame sorted by macro F1

---

### `src/evaluation/visualizer.py`

Saves all plots to `outputs/figures/` using a non-interactive matplotlib backend (no display required).

- `plot_confusion_matrix()` — row-normalised heatmap per model
- `plot_model_comparison()` — grouped bar chart of macro precision/recall/F1 across all models
- `plot_label_distribution()` — countplot of class distribution in the full dataset

---

### `tests/`

38 unit tests using pytest. No real dataset is required — tests use synthetic data and temporary directories.

| File | What it tests |
|---|---|
| `test_loader.py` | Email file parsing, Enron and SpamAssassin directory traversal |
| `test_preprocessor.py` | HTML removal, URL stripping, stopword removal, column handling |
| `test_features.py` | TF-IDF, character n-grams, and metadata feature extraction |
| `test_models.py` | Predict/predict_proba for all five classifiers |

---

## Setup

**Requirements:** Python 3.9+

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run (stopwords, wordnet, punkt).

---

## Getting the Data

A download script is included that handles both datasets automatically.

### Option A — Download everything with one command

**SpamAssassin** and **Nazario phishing** require no account. **Enron** requires a free Kaggle account (setup below).

```bash
# SpamAssassin only (no account needed)
python scripts/download_data.py --spamassassin

# Nazario phishing corpus (no account needed)
python scripts/download_data.py --phishing

# Enron only (requires Kaggle credentials)
python scripts/download_data.py --enron

# Everything at once
python scripts/download_data.py --all
```

### Kaggle setup (one-time, only needed for Enron)

1. Create a free account at https://www.kaggle.com
2. Go to **https://www.kaggle.com/settings** → **API** → click **"Create New Token"**
3. This downloads `kaggle.json` — move it to `~/.kaggle/kaggle.json`
4. Run `chmod 600 ~/.kaggle/kaggle.json`
5. Run `pip install kaggle`

Then run `python scripts/download_data.py --enron` and it will pull the dataset automatically.

### Option B — Manual download (no Kaggle API needed)

**Enron (easiest — just drop in a CSV):**

1. Go to https://www.kaggle.com/datasets/venky73/spam-mails-dataset
2. Click **Download** (you need a free Kaggle account to download, but no API setup)
3. Unzip it and place the `.csv` file here:

```
data/raw/enron/          ← put the .csv file directly in this folder
    emails.csv           ← any filename works, as long as it ends in .csv
```

The loader auto-detects the column names so it works with any of the common Enron CSV formats from Kaggle.

**SpamAssassin** — Download the `.tar.bz2` files from https://spamassassin.apache.org/old/publiccorpus/ and extract into:
```
data/raw/spamassassin/
    spam/
    easy_ham/
    hard_ham/
    spam_2/         (optional)
    easy_ham_2/     (optional)
```

You can use either dataset or both. If only one is available the pipeline will use what it finds.

---

## Running the Project

### Full pipeline (both datasets, all models)

```bash
python run_pipeline.py
```

### Choose a specific dataset

```bash
python run_pipeline.py --dataset enron
python run_pipeline.py --dataset spamassassin
python run_pipeline.py --dataset both
```

### Run specific models only

Model keys: `nb` (Naive Bayes), `lr` (Logistic Regression), `svm`, `rf` (Random Forest), `nn` (Neural Network)

```bash
python run_pipeline.py --models nb,lr,svm
python run_pipeline.py --models all
```

### Skip re-preprocessing (reuse saved CSV)

After the first run, the processed data is saved to `data/processed/emails_processed.csv`. The pipeline **automatically reuses this file** on subsequent runs — no flag needed. To force re-processing from raw data:

```bash
python run_pipeline.py --force-preprocess
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Outputs

After running the pipeline, results are saved to:

```
outputs/
├── figures/
│   ├── label_distribution.png          # Class counts across the full dataset
│   ├── confusion_matrix_Naive Bayes.png
│   ├── confusion_matrix_Logistic Regression.png
│   ├── confusion_matrix_Linear SVM.png
│   ├── confusion_matrix_Random Forest.png
│   ├── confusion_matrix_Neural Network (MLP).png
│   └── model_comparison.png            # Grouped bar chart of macro metrics
└── results/
    └── model_comparison.csv            # Per-class P/R/F1 for all models
```

---

## Models

| Key | Model | Features Used | Notes |
|---|---|---|---|
| `nb` | Naive Bayes | TF-IDF only | Requires non-negative input |
| `lr` | Logistic Regression | TF-IDF + char n-grams + metadata | Outputs top features per class |
| `svm` | Linear SVM | TF-IDF + char n-grams + metadata | Calibrated for probability output |
| `rf` | Random Forest | TruncatedSVD(300) of full features | SVD reduces sparse matrix first |
| `nn` | Neural Network MLP | TruncatedSVD(300) of full features | 512→256→128, Adam, early stopping |

All models are evaluated on a stratified 80/20 train/test split.
