# User-Prompts-Topic-Classification

## Overview

This project classifies user prompts (short text queries) into **35 fine-grained topic labels** organized under **10 parent categories**. It compares three classification approaches—Single Classifier, Hierarchical Input Classifier, and Dynamic Oracle Classifier—using sentence embeddings from `BAAI/bge-small-en-v1.5` and Logistic Regression. The notebook also performs deep misclassification analysis using word overlap techniques.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Label Schema](#2-label-schema)
3. [Pipeline Walkthrough (Cell by Cell)](#3-pipeline-walkthrough)
   - 3.1 [Setup & Imports](#31-setup--imports)
   - 3.2 [Dataset Loading & Filtering](#32-dataset-loading--filtering)
   - 3.3 [Label Definitions & Hierarchy](#33-label-definitions--hierarchy)
   - 3.4 [Data Exploration & Cleaning](#34-data-exploration--cleaning)
   - 3.5 [Label Encoding](#35-label-encoding)
   - 3.6 [Train/Test Split](#36-traintest-split)
   - 3.7 [Embedding Extraction](#37-embedding-extraction)
   - 3.8 [Training Classifiers](#38-training-classifiers)
   - 3.9 [Evaluation Functions](#39-evaluation-functions)
   - 3.10 [Word Overlap Misclassification Analysis](#310-word-overlap-misclassification-analysis)
   - 3.11 [Single Classifier Evaluation](#311-single-classifier-evaluation)
   - 3.12 [Hierarchical Input Classifier](#312-hierarchical-input-classifier)
   - 3.13 [Dynamic Oracle Classifier](#313-dynamic-oracle-classifier)
4. [Results Summary](#4-results-summary)
5. [Key Takeaways](#5-key-takeaways)

---

## 1. Dataset

The dataset is hosted on Hugging Face Hub (`maryamdar/topic_classification_dataset_gen`) and contains **27,826 rows** of user-generated prompts. Each row includes:

| Column | Description |
|---|---|
| `text` | The raw user prompt |
| `label` | The fine-grained topic label (one of 35) |
| `parent_label` | The coarse-grained parent category (one of 10) |
| `generator_model` | The LLM used to generate/augment the sample |
| `source` | Whether the sample is `generated` or original |
| `generation_group` | The prompt group used for generation |
| `use_for_train` | Boolean flag indicating if the row should be used for training |
| `old_label` / `old_parent_label` | Previous label assignments (from earlier iterations) |

The dataset is a mix of **synthetically generated** samples (via GPT-5.6, deepseek-v4, MiMo V2.5 Free, etc.) and **real-world** user prompts.

---

## 2. Label Schema

### Fine-Grained Labels (35 total)

| # | Label | Parent |
|---|---|---|
| 0 | Desktop & Mobile & Web Development | Programming / Technology |
| 1 | Cybersecurity | Programming / Technology |
| 2 | AI / Machine Learning / Data Science | Programming / Technology |
| 3 | Infrastructure (DevOps, Cloud, Databases, Networking) | Programming / Technology |
| 4 | Clinical Diagnosis, Treatment & Surgery | Medical |
| 5 | Medication & Pharmacology | Medical |
| 6 | Mental Health | Medical |
| 7 | Healthcare Organizations, System, Hospitals | Medical |
| 8 | Nutrition | Medical |
| 9 | Payments & Personal Budgeting | Finance |
| 10 | Investment, Markets & Cryptocurrency | Finance |
| 11 | Corporate | Finance |
| 12 | Physics, Mathematics | Science |
| 13 | Chemistry | Science |
| 14 | Biology | Science |
| 15 | Team Sports | Sports |
| 16 | Individual Sports | Sports |
| 17 | Fitness | Sports |
| 18 | Civil, Structural & Architecture | Engineering |
| 19 | Mechanical & Electrical Engineering | Engineering |
| 20 | Family & Relationships | Personal |
| 21 | Personal | Personal |
| 22 | Travel | Personal |
| 23 | Marketing & Sales | Business |
| 24 | Entrepreneurship & Startups | Business |
| 25 | Management & Strategy & Human Resources | Business |
| 26 | Criminal Law | Law |
| 27 | Family Law | Law |
| 28 | Corporate Law | Law |
| 29 | Civil Law | Law |
| 30 | Game | Art |
| 31 | Film | Art |
| 32 | Music | Art |
| 33 | Literature | Art |
| 34 | Painting | Art |

---

## 3. Pipeline Walkthrough

### 3.1 Setup & Imports

**Cell: Install dependencies**

```python
!pip install -q datasets transformers scikit-learn torch
```

Installs the core libraries: `datasets` (Hugging Face), `transformers` (for tokenizer/model), `scikit-learn` (for classifiers and metrics), and `torch` (PyTorch backend).

**Cell: Import libraries**

Imports `torch`, `numpy`, Hugging Face `datasets` and `transformers`, and `sklearn` modules for Logistic Regression and metrics.

---

### 3.2 Dataset Loading & Filtering

**Cell: Load the dataset**

```python
dataset = load_dataset("maryamdar/topic_classification_dataset_gen")
```

Downloads the dataset from Hugging Face Hub. At this point the dataset has **29,618** rows in the training split.

**Cell: Filter for training data**

```python
dataset["train"] = dataset["train"].filter(lambda x: x["use_for_train"] is True)
```

Keeps only rows where `use_for_train` is `True`. This removes test-only or validation-only samples, reducing the pool to **27,826** rows.

**Cell: Remove specific generator model**

```python
dataset["train"] = dataset["train"].filter(
    lambda x: x["generator_model"] != "MiMo V2.5 Free"
)
```

Filters out samples generated by `MiMo V2.5 Free`. This is likely done to avoid data contamination since the same model might be used for inference, or because the quality of those generations was lower. After filtering, **19,108** rows remain.

> **Note:** There is also a large commented-out block that demonstrates merging additional real-world datasets (`topic-classification-dataset-real` and `topic-classification-dataset-real-2`) into the training set. This merging pipeline filters out "Not Related" labels, aligns column schemas, and concatenates the datasets.

---

### 3.3 Label Definitions & Hierarchy

**Cell: Define LABEL_ORDER**

A fixed list of all 35 fine-grained labels. The order is intentional and consistent throughout the notebook — label index 0 maps to "Desktop & Mobile & Web Development", index 1 to "Cybersecurity", and so on.

**Cell: Define HIERARCHY**

A dictionary mapping each of the 10 parent categories to their child labels:

- **Programming / Technology** → 4 children
- **Medical** → 5 children
- **Finance** → 3 children
- **Science** → 3 children
- **Sports** → 3 children
- **Engineering** → 2 children
- **Personal** → 3 children
- **Business** → 3 children
- **Law** → 4 children
- **Art** → 5 children

Also builds a reverse mapping `CHILD_TO_PARENT` for looking up the parent of any child label.

---

### 3.4 Data Exploration & Cleaning

**Cell: Dataset overview and stats**

Converts the HF dataset to a Pandas DataFrame and prints:
- Total row count and column names
- Missing values per column (notably `old_label` and `old_parent_label` have many NaNs since they're from previous iterations)
- Duplicate text check (0 duplicates found)
- Class distribution: shows count and percentage for each of the 35 labels

**Cell: Drop missing values and duplicates**

```python
df = df.dropna(subset=['text', 'label'])
df = df.drop_duplicates(subset=['text'], keep='first')
```

Removes any rows with missing text or labels, and deduplicates by text content.

**Cell: Display label examples**

Prints 3 example texts per label so you can visually inspect what each category looks like. Examples range from short queries ("phishing link") to detailed multi-sentence prompts.

---

### 3.5 Label Encoding

**Cell: Encode labels**

Creates a bidirectional mapping:
- `label_to_id`: maps each label string → integer (0–34)
- `id_to_label`: maps each integer → label string

Adds a `label_id` column to the DataFrame. Includes a safety check that raises an error if any label is not found in the mapping.

Converts the cleaned DataFrame back to a Hugging Face `Dataset` object.

---

### 3.6 Train/Test Split

**Cell: Stratified split**

```python
split_dataset = clean_dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="label_id",
    seed=42
)
```

Performs an **80/20 stratified split**, ensuring each of the 35 labels has proportional representation in both train and test sets. Results in:
- **Train:** 15,286 samples
- **Test:** 3,822 samples

**Cells: Verify distribution**

Prints label distributions for both train and test sets to confirm the stratification worked correctly. Both sets show nearly identical percentage distributions.

---

### 3.7 Embedding Extraction

**Cell: Set device and model name**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "BAAI/bge-small-en-v1.5"
```

Uses the **BGE-small** embedding model (384-dimensional, 12-layer BERT variant). It's a strong retrieval-oriented embedding model that produces high-quality sentence representations.

**Cell: Load tokenizer and model**

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
```

Downloads and loads the pre-trained model. The architecture is a standard BERT encoder with 12 transformer layers, 384 hidden dimensions, and ~33M parameters.

**Cell: Define mean pooling**

```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

**Mean pooling** converts token-level embeddings into a single sentence embedding by averaging all token vectors, weighted by the attention mask (ignoring padding tokens). This is the standard approach for BERT-style sentence embeddings.

**Cell: Define embedding extraction function**

```python
def extract_embeddings(batch):
    encoded_input = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return {"embedding": sentence_embeddings.cpu().numpy()}
```

This function:
1. Tokenizes a batch of texts (padded, truncated to 512 tokens)
2. Runs the model in `torch.no_grad()` mode (no gradient computation)
3. Applies mean pooling
4. **L2-normalizes** the embeddings (important for cosine similarity downstream)
5. Returns embeddings as NumPy arrays

**Cell: Extract embeddings for train and test**

```python
train_embedded = train_ds.map(extract_embeddings, batched=True, batch_size=32)
test_embedded = test_ds.map(extract_embeddings, batched=True, batch_size=32)
```

Processes both splits in batches of 32. Each sample gets a 384-dimensional embedding vector appended as a new column.

**Cell: Convert to NumPy arrays**

```python
X_train = np.array(train_embedded["embedding"])
y_train = np.array(train_embedded["label_id"])
X_test = np.array(test_embedded["embedding"])
y_test = np.array(test_embedded["label_id"])
```

Extracts the embedding matrices and label vectors for use with scikit-learn classifiers.

---

### 3.8 Training Classifiers

**Cell: SGD Classifier (iterative training)**

```python
classifier = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=42)
```

Trains an **SGD-based Logistic Regression** classifier using `partial_fit` over 30 epochs. Each epoch:
1. Shuffles the training data
2. Calls `partial_fit()` for one epoch
3. Computes training loss (log loss) and accuracy
4. Records metrics for plotting

The training curves show the model converges around epoch 5-10 with ~93.6% training accuracy and stable loss around 0.535.

Two plots are generated:
- **Training Loss per Epoch** — shows the loss decreasing and stabilizing
- **Training Accuracy per Epoch** — shows accuracy climbing and plateauing

**Cell: Logistic Regression (one-shot training)**

```python
classifier = LogisticRegression(max_iter=1000, C=1.0)
classifier.fit(X_train, y_train)
```

The **final classifier** used for evaluation. Logistic Regression with L2 regularization (C=1.0) and up to 1000 iterations. This is the simpler, more robust approach — one-shot fit without manual epoch control.

**Cell: Save model and label mappings**

```python
joblib.dump(classifier, "classifier.joblib")
with open("label_mappings.json", "w") as f:
    json.dump({"id_to_label": {str(k): v for k, v in id_to_label.items()}}, f)
```

Persists the trained classifier and label mappings for later use.

---

### 3.9 Evaluation Functions

A comprehensive set of evaluation helper functions is defined:

#### `predict_model(classifier, X_test)`
Simple wrapper that calls `classifier.predict()`.

#### `show_classification_report(y_test, y_pred, label_order)`
- Prints overall accuracy
- Prints the full sklearn `classification_report` with per-class precision, recall, and F1-score
- Displays a summary DataFrame with Accuracy, Macro F1, and Weighted F1

#### `plot_child_confusion_matrix_old(y_test, y_pred, label_order)`
Plots a **raw count** confusion matrix for all 35 child labels using `ConfusionMatrixDisplay`. Saves as `confusion_matrix.png`.

#### `plot_child_confusion_matrix(y_test, y_pred, label_order)`
Plots a **percentage-normalized** confusion matrix (each row sums to 100%). This makes it easier to compare across classes with different sample counts.

#### `show_parent_summary(report, hierarchy)`
Aggregates per-class metrics into **parent-level summaries**. For each parent category, computes:
- Min/Avg/Max Precision across its children
- Min/Avg/Max Recall across its children
- Min/Avg/Max F1 across its children
- Total support (number of samples)

#### `plot_parent_misclassification_matrix(y_test, y_pred, hierarchy, id_to_label)`
Creates a **parent-level misclassification matrix** that shows how many child-level errors occurred between each pair of parent categories. Only counts errors (correct predictions are ignored on the diagonal).

#### `evaluate_model(classifier, X_test, y_test, label_order, id_to_label, hierarchy)`
Orchestrates the full evaluation pipeline by calling all the above functions in sequence.

---

### 3.10 Word Overlap Misclassification Analysis

This is a deep analysis pipeline that investigates **why** misclassifications happen by checking whether the text contains top keywords from both the true and predicted labels.

#### Custom Stop Words

A large custom stop word list is defined on top of sklearn's `ENGLISH_STOP_WORDS`. It includes:
- Instruction words ("using", "make", "create", "write", "explain")
- Modal/generic request words ("should", "must", "would", "could", "need", "want")
- Generic prompt words ("question", "answer", "prompt", "text", "example")
- Generic verbs ("get", "take", "find", "know", "help")
- Filler words ("like", "also", "just", "really")
- Numbers ("one", "two", "three")

#### `extract_words(text)`
Extracts lowercase English words (3+ characters) that are NOT in the combined stop word list.

#### `get_top_words_per_label(texts, labels, label_order, top_n=10)`
Counts word frequencies per label and returns the top N most common words for each of the 35 labels.

#### `analyze_all_misclassifications(texts, y_test, y_pred, label_order, top_words)`
For each misclassified sample:
1. Checks which top-10 words from the **true label** appear in the text
2. Checks which top-10 words from the **predicted label** appear in the text
3. Records the overlap information

This helps understand whether confusion is due to **lexical ambiguity** (the text genuinely contains keywords from both categories) or **embedding/model error** (the text has no overlap with the predicted category).

#### `create_overlap_matrix(...)` and `plot_overlap_matrix(...)`
Creates and visualizes a **35×35 matrix** where each cell counts how many misclassifications between two labels had word overlap.

#### Parent-Level Overlap Analysis
- `add_parent_labels(...)`: Adds parent columns to the misclassification DataFrame
- `create_parent_overlap_matrix(...)`: Creates a 10×10 parent-level overlap matrix
- `create_within_parent_summary(...)`: Shows child-label confusions that happen **within** the same parent (e.g., "Nutrition" confused with "Mental Health" — both under "Medical")
- `create_parent_summary(...)`: Shows how much overlap confusion each parent has internally vs. with other parents

#### `run_word_overlap_analysis(...)`
Orchestrates the full overlap analysis pipeline in 7 steps:
1. Extract top words from all data
2. Find overlap misclassifications
3. Print overlap samples with details
4. Build and plot child-level overlap matrix
5. Build and plot parent-level overlap matrix
6. Show within-parent child confusions
7. Show within vs. cross-parent summary percentages

---

### 3.11 Single Classifier Evaluation

**Cell: Run evaluation**

```python
y_pred = evaluate_model(classifier, X_test, y_test, LABEL_ORDER, id_to_label, HIERARCHY)
```

Runs the full evaluation pipeline on the single Logistic Regression classifier.

**Results:**
- **Accuracy: 92.94%**
- **Macro F1: 92.73%**
- **Weighted F1: 92.92%**

Per-class highlights:
- Best: Family Law (100% precision), Healthcare Organizations (98% recall), Travel (99% recall)
- Weakest: Entrepreneurship & Startups (74% recall — likely confused with other Business labels)

**Cell: Save misclassified samples**

```python
save_misclassified_samples(df_test, y_test, y_pred, output_path="misclassified_samples.json")
```

Exports **5,565 misclassified samples** to a JSON file for manual inspection. The output includes the text, true label, predicted label, and all metadata.

**Cell: Analyze misclassification by generation group**

Breaks down misclassification rates by `generation_group` to identify which prompt generation strategies produce harder-to-classify samples. Some groups have 0% misclassification (e.g., `Desktop_single`, `one_parent_labels_3`), while others are higher (e.g., `second_label_desc_corporate_law` at 20%, `10_labels_2_shuffled` at 20%).

**Cell: Word overlap analysis for single classifier**

Runs the full word overlap analysis to understand the nature of the 5,565 misclassifications.

---

### 3.12 Hierarchical Input Classifier

This approach modifies the **input** to the classifier by prepending the parent category.

**Cell: Prepare hierarchical data**

```python
df_train["input_text"] = "Parent Category: " + df_train["parent_label"] + "\nText: " + df_train["text"]
df_test["input_text"] = "Parent Category: " + df_test["parent_label"] + "\nText: " + df_test["text"]
```

Each text is prepended with `"Parent Category: <parent>\nText: <original_text>"`. This gives the model explicit context about which broad category the text belongs to, potentially reducing cross-parent confusion.

**Cell: Re-embed and retrain**

The modified texts go through the same embedding pipeline (BGE-small), and a new Logistic Regression classifier is trained.

**Cell: Evaluate hierarchical model**

Runs the same evaluation pipeline. The hierarchical input approach may improve classification by disambiguating between similar child labels in different parents.

**Cell: Word overlap analysis for hierarchical model**

Runs the overlap analysis to see if the hierarchical input reduces lexical confusion.

---

### 3.13 Dynamic Oracle Classifier

This approach trains **separate classifiers per parent category** and uses the **ground-truth parent** at test time (oracle).

**Cell: Train child classifiers per parent**

```python
child_classifiers = {}
for parent_name in HIERARCHY.keys():
    mask = (df_train['parent_label'] == parent_name)
    X_train_sub = X_train[mask]
    y_train_sub = df_train[mask]['label'].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_sub, y_train_sub)
    child_classifiers[parent_name] = clf
```

For each of the 10 parent categories, trains a dedicated Logistic Regression on only the child labels under that parent. This means each classifier only has to distinguish between 2–5 classes instead of 35.

**Cell: Generate oracle predictions**

```python
for idx, row in df_test.iterrows():
    parent = row["parent_label"]  # ground truth parent
    clf = child_classifiers[parent]
    pred = clf.predict(X_test[idx].reshape(1, -1))[0]
    oracle_predictions.append(pred)
```

At test time, each sample is routed to its **true parent's classifier**. This is an "oracle" because in production you wouldn't know the true parent — but this establishes an **upper bound** on performance.

**Cell: Evaluate each parent separately**

For each parent category, prints:
- Accuracy, Macro F1, Weighted F1
- Full classification report for the children
- Confusion matrix (saved as PNG)

Also generates:
- Overall oracle classification report (all 35 labels)
- Oracle child confusion matrix (absolute counts and percentages)
- Oracle parent misclassification matrix
- Oracle parent summary metrics

---

## 4. Results Summary

| Approach | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Single Classifier** | 92.94% | 92.73% | 92.92% |
| **Hierarchical Input** | ~similar range | ~similar range | ~similar range |
| **Dynamic Oracle** | upper bound | upper bound | upper bound |

The single classifier already achieves strong performance (~93%) despite the challenging 35-class problem. The word overlap analysis reveals that many misclassifications occur between semantically related labels (e.g., within the same parent category), which is expected behavior for a hierarchical label schema.

---

## 5. Key Takeaways

1. **Embedding quality matters**: Using `BAAI/bge-small-en-v1.5` (a retrieval-optimized model) produces high-quality sentence embeddings that enable strong classification with a simple Logistic Regression.

2. **Simple classifiers can work well**: A Logistic Regression on top of good embeddings achieves ~93% accuracy on 35 classes without any fine-tuning of the language model.

3. **Parent-level structure helps analysis**: The hierarchical label structure (10 parents × 3-5 children) provides a useful framework for understanding confusion patterns. Most errors happen within parent categories rather than across them.

4. **Word overlap analysis is valuable**: By checking if misclassified texts contain top keywords from both true and predicted labels, you can distinguish between:
   - **Genuine ambiguity**: Texts that legitimately contain terms from multiple categories
   - **Model errors**: Texts that have no lexical reason to be confused

5. **Generation group analysis matters**: Different prompt generation strategies produce varying difficulty levels. Some groups have 0% misclassification while others reach 20%.

6. **Oracle upper bound**: The dynamic oracle approach (separate classifiers per parent) shows the theoretical ceiling when the parent category is known, which is useful for designing multi-stage classification systems.

---

## Generated Files

| File | Description |
|---|---|
| `classifier.joblib` | Trained Logistic Regression model |
| `label_mappings.json` | JSON mapping between label IDs and names |
| `misclassified_samples.json` | All misclassified test samples with metadata |
| `confusion_matrix.png` | Child-level confusion matrix (counts) |
| `confusion_matrix_2.png` | Hierarchical model confusion matrix |
| `oracle_child_confusion_matrix.png` | Oracle model confusion matrix |
| `parent_confusion_matrices/` | Directory of per-parent confusion matrices |
| `misclassified_analysis.csv` | Detailed word overlap misclassification analysis |
