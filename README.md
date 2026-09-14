# Topic Classification Model — Second Labels + Probabilities + Top-K

A notebook for training and evaluating a **35-class text topic classifier** using sentence embeddings and logistic regression. The model classifies user prompts into fine-grained topic labels (child labels) organized under 10 parent categories, and supports **Top-K evaluation** (accepting predictions as correct if the true label appears in the top-K predictions).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Label Hierarchy](#label-hierarchy)
- [Setup & Requirements](#setup--requirements)
- [How to Run](#how-to-run)
- [Notebook Pipeline](#notebook-pipeline)
- [Configuration](#configuration)
- [Outputs & Artifacts](#outputs--artifacts)
- [Troubleshooting](#troubleshooting)

---

## Overview

This notebook builds a topic classification system that:

1. **Loads** a labeled dataset from Hugging Face
2. **Filters** and cleans the data (removes duplicates, filters by flag)
3. **Encodes** 35 fine-grained topic labels into numeric IDs
4. **Merges** training data from multiple sources (generated + real-labeled)
5. **Embeds** text using `google/embeddinggemma-300m` via `sentence-transformers`
6. **Trains** a logistic regression classifier (SGDClassifier with log loss)
7. **Evaluates** with full classification report, confusion matrices, and parent-level analysis
8. **Supports Top-K** evaluation (accepting a prediction if the true label is in the top-K)

---

## Architecture

```
Input Text
    ↓
SentenceTransformer("google/embeddinggemma-300m")
    ↓
768-dim normalized embedding
    ↓
SGDClassifier(loss="log_loss")  →  35-class prediction
    ↓
[Optional] Top-K acceptance logic
```

The classifier uses **incremental learning** (`partial_fit`) with 10 epochs, shuffling data each epoch.

---

## Datasets

| Dataset | Hugging Face ID | Purpose |
|---------|----------------|---------|
| Generated prompts | `maryamdar/topic_classification_dataset_gen` | Training data (filtered to `use_for_train=True`, excludes "MiMo V2.5 Free" generator) |
| Real-labeled prompts | `maryamdar/topic-classification-dataset-real-labeled` | Test data (filters out "Not Related" and "Ambiguous" labels) |

Both datasets are **stratified-sampled** (1200 train + 800 test per source, merged into 2400 train + 1600 test total).

---

## Label Hierarchy

The 35 child labels are organized under 10 parent categories:

| Parent | Child Labels |
|--------|-------------|
| **Programming / Technology** | Desktop & Mobile & Web Development, Cybersecurity, AI / ML / Data Science, Infrastructure (DevOps, Cloud, Databases, Networking) |
| **Medical** | Clinical Diagnosis Treatment & Surgery, Medication & Pharmacology, Mental Health, Healthcare Organizations System Hospitals, Nutrition |
| **Finance** | Payments & Personal Budgeting, Investment Markets & Cryptocurrency, Corporate |
| **Science** | Physics Mathematics, Chemistry, Biology |
| **Sports** | Team Sports, Individual Sports, Fitness |
| **Engineering** | Civil Structural & Architecture, Mechanical & Electrical Engineering |
| **Personal** | Family & Relationships, Personal, Travel |
| **Business** | Marketing & Sales, Entrepreneurship & Startups, Management & Strategy & Human Resources |
| **Law** | Criminal Law, Family Law, Corporate Law, Civil Law |
| **Art** | Game, Film, Music, Literature, Painting |

---

## Setup & Requirements

### Python Packages

```bash
pip install datasets transformers scikit-learn torch sentence-transformers joblib matplotlib pandas numpy
```

### Hardware

- **GPU recommended** for faster embedding extraction (the notebook auto-detects CUDA)
- Works on CPU but will be slower

### Hugging Face Authentication

You need a Hugging Face token to access the datasets:

```python
from huggingface_hub import login
login()  # Will prompt for token
```

---

## How to Run

1. Clone or download this repository
2. Open `Topic_classification_model_secondLabels_probs_Top-k.ipynb` in Jupyter/Colab
3. Run all cells sequentially from top to bottom
4. The notebook will:
   - Install dependencies
   - Authenticate with Hugging Face
   - Load and preprocess datasets
   - Generate embeddings
   - Train the classifier
   - Run full evaluation (Normal + Top-3)
   - Save model artifacts

---

## Notebook Pipeline

### 1. Setup (Cells 1-4)
- Install packages
- Import libraries
- Hugging Face login
- Select model name (`google/embeddinggemma-300m`)

### 2. Configuration (Cells 5-6)
- Define `LABEL_ORDER` — ordered list of 35 labels
- Define `HIERARCHY` — parent-to-child mapping
- Build `CHILD_TO_PARENT` reverse mapping

### 3. Load & Filter Dataset (Cells 7-11)
- Load generated dataset from Hugging Face
- Filter: `use_for_train == True`
- Filter: exclude `generator_model == "MiMo V2.5 Free"`
- Display dataset statistics (missing values, duplicates, label counts)

### 4. Label Encoding (Cell 12)
- Map label strings → integer IDs using `LABEL_ORDER`
- Safety check for unknown labels

### 5. Train/Test Split (Cells 13-20)
- Two options: single-dataset split or multi-source merge
- Stratified sampling with configurable `TRAIN_SIZE=1200`, `TEST_SIZE=800`
- Merge train sets and test sets from both sources
- Encode labels on merged datasets

### 6. Embedding Extraction (Cells 21-25)
- Load `SentenceTransformer("google/embeddinggemma-300m")`
- Batch encode with `prompt_name="Classification"`
- Extract embeddings for train and test sets

### 7. Train Classifier (Cells 26-27)
- `SGDClassifier(loss="log_loss", max_iter=1)`
- Manual epoch loop with `partial_fit` (10 epochs)
- Track training loss and accuracy
- Plot training curves

### 8. Save Model (Cells 28-29)
- Save classifier as `classifier.joblib`
- Save label mappings as `label_mappings.json`

### 9. Evaluation Functions (Cells 30-38)
- `show_classification_report()` — accuracy, precision, recall, F1
- `plot_child_confusion_matrix_old()` — raw count confusion matrix
- `plot_child_confusion_matrix()` — percentage confusion matrix
- `plot_misclassification_confusion_matrix()` — errors only
- `show_parent_summary()` — aggregated parent-level metrics
- `plot_parent_misclassification_matrix()` — parent-level errors
- `show_random_misclassified_probabilities()` — detailed error analysis
- `save_test_results()` / `load_test_results()` — persistence

### 10. Normal Evaluation (Cells 39-40)
- Run `evaluate_model()` for standard Top-1 evaluation
- Results saved to `test_predictions.npz`

### 11. Top-K Evaluation (Cells 41-44)
- `accept_top_k_predictions()` — accept if true label is in top-K
- `evaluate_top_k_model()` — full evaluation pipeline with Top-K
- Default: Top-3 evaluation
- Reports both Top-1 and Top-K accuracy

### 12. Save Misclassified Samples (Cells 45-47)
- Export all test samples with predictions to `misclassified_samples.json`

---

## Configuration

Key parameters you can adjust:

```python
# Model
model_name = "google/embeddinggemma-300m"

# Train/Test sizes (per source)
TRAIN_SIZE = 1200
TEST_SIZE = 800
RANDOM_STATE = 42

# Classifier
EPOCHS = 10

# Top-K evaluation
K = 3
```

---

## Outputs & Artifacts

| File | Description |
|------|-------------|
| `classifier.joblib` | Trained SGD classifier |
| `label_mappings.json` | Label ID ↔ label name mappings |
| `test_predictions.npz` | Test set predictions, probabilities, and class IDs |
| `confusion_matrix.png` | Confusion matrix plots |
| `misclassification_confusion_matrix.png` | Errors-only confusion matrix |
| `misclassified_samples.json` | Full test data with predictions |

---

## Performance (Reported)

| Metric | Top-1 | Top-3 |
|--------|-------|-------|
| Accuracy | 89.31% | 98.06% |
| Macro F1 | 89.01% | 97.83% |
| Weighted F1 | 89.29% | 98.06% |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `RuntimeError: The size of tensor a (556) must match tensor b (512)` | This occurs with BERT models on long texts. The notebook uses SentenceTransformer which handles this automatically. |
| `ValueError: Unknown label type: unknown` | Ensure `label_id` column is integer-typed, not float. The notebook handles this with `.map(encode_labels)`. |
| Hugging Face login fails | Generate a token at https://huggingface.co/settings/tokens and paste when prompted. |
| GPU out of memory | Reduce `batch_size` in `extract_embeddings()` (default: 32). |
| Missing packages | Run `pip install datasets transformers scikit-learn torch sentence-transformers joblib matplotlib pandas numpy` |
