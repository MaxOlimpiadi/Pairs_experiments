# Humor Classification Experiments

A research project comparing **TF-IDF + SVM**, **sentence embeddings + SVM**, and **fine-tuned BERT** on original and edited news headlines from SemEval-2020 Task 7.

The goal is to investigate how classification performance changes with the size of the training dataset and to evaluate the impact of dimensionality reduction on SVM-based classifiers.

**Models**

| Approach | Configuration |
|---|---|
| TF-IDF + SVM | Word unigrams/bigrams, up to 1,000 features |
| Embeddings + SVM | `sentence-transformers/all-MiniLM-L6-v2` |
| Fine-tuned BERT | `bert-base-uncased`, two-class classification |

Both SVM models use an RBF kernel with `C=1.0` and `gamma="scale"`.

BERT uses AdamW, a learning rate of `5e-5`, batch size 16, and up to 10 epochs. The best checkpoint is selected by validation loss, with early stopping after three epochs without improvement.

**Dimensionality Reduction (only for SVMs)**

Two dimensionality reduction techniques are evaluated:

- **Principal Component Analysis (PCA)** for dense MiniLM sentence embeddings, using full SVD.
- **Truncated Singular Value Decomposition (TruncatedSVD / LSA)** for sparse TF-IDF features, using randomized SVD with `random_state=42`.

Both techniques reduce feature representations to **8, 12, or 16 components**.

Each SVM approach is evaluated under four configurations:

| Feature representation | Reduction method | Components |
|---|---|---|
| TF-IDF | None (baseline) | Up to 1,000 |
| TF-IDF | TruncatedSVD | 8, 12, 16 |
| MiniLM embeddings | None (baseline) | 384 |
| MiniLM embeddings | PCA | 8, 12, 16 |

Dimensionality reduction is implemented within **scikit-learn pipelines** and fitted exclusively on the corresponding training subset to prevent data leakage.

The same training subsets and test set are used across all reduction configurations, enabling direct comparisons of classification performance.

Fine-tuned BERT is evaluated without dimensionality reduction as an independent end-to-end transformer baseline.

**Data**

Original headlines are labeled `0`, and edited headlines are labeled `1`.

- Training edits are filtered by `meanGrade >= 1.2`.
- The original development dataset is divided into validation and test sets, grouping all edits of the same original headline together.
- Training subsets are approximately balanced by class and nested within each seed.
- Subsets contain individual texts; original headlines and their edits are not necessarily selected together.

Validation and test edits have no supplied humor scores, so their positive labels represent **intended humorous edits**, not independently verified humor.

**Experiment Setup**

- **Training sizes:** 25, 50, 100, 200, 300, 500, 700, 900, 1100, 1300, 1500, 1750, 2000.
- **Random seeds:** 7, 10, 35.
- **Dimensionality reduction:** None, 8, 12, 16 components (SVM models only).
- **Test metrics:** Accuracy, Macro Precision, Macro Recall, Macro F1.
- **Plots:** mean metrics by training size, with ± one standard deviation.
- **Reduction comparison:** Separate plots for each metric, comparing baseline and reduced-dimensionality configurations.

**Project Structure**

```text
requirements.txt   # Main dependencies with pinned versions
config.py          # Paths, model parameters, and preparation flags
main.py            # Entry point
data_prep.py       # Data preparation and training subset generation
experiments.py     # Training, evaluation, logging, and plotting
data/
  original/        # Input CSV files
  pairs/           # Reconstructed original–edited pairs
  prepared/        # Texts with binary labels (final version of data for the experiments)
split/
  train_slices/    # Training subsets grouped by seed
  dev.csv          # Validation set
  test.csv         # Test set
```

**Requirements**



```bash
python -m pip install -r requirements.txt
```

BERT uses CUDA when available, otherwise CPU. Pretrained models may be downloaded on the first run.

**Running**

Run from the project root:

```bash
python main.py
```

To prepare everything from `data/original/train.csv` and `data/original/dev.csv`, enable these flags in `config.py`:

```python
DO_PAIRS = True
DO_TRANSFORM_PAIRS = True
CREATE_SPLITS = True
```

To reuse existing splits, set all three flags to `False`.

The pipeline runs baseline SVM experiments, transformer experiments, and SVM experiments with dimensionality reduction. Results are logged for each training subset and configuration. Plots are generated separately after the experiments have completed.

**Results**

- `experiments_log.csv` — experiment parameters and test metrics.
- `SVM_embeddings_plot.png`, `SVM_tf-idf_plot.png`, `bert_plot.png` — learning curves.
- `reduction_SVM_embeddings.png` — PCA comparison across training sizes.
- `reduction_SVM_tf-idf.png` — TruncatedSVD comparison across training sizes.
- `best_model.pth` — best BERT checkpoint for the current run.
- `test/predictions.csv` — BERT predictions and class probabilities.

**Note:** `DELETE_OLD_REPORT=True` deletes the previous experiment log. BERT checkpoints and predictions are overwritten between runs.
