# Sequential Models for Swahili News Classification

**Formative Assignment 2: NLP and Language Technologies**

**Research question:** How effectively can sequential modelling approaches classify Swahili news articles into topics, and what evidence supports the strengths and limitations of each approach?

We compare five approaches on the [Swahili News Classification Challenge](https://zindi.world/competitions/swahili-news-classification-challenge) dataset (5,151 Tanzanian news articles, five categories): two bag-of-words baselines (TF-IDF with Logistic Regression and with Naive Bayes) and three neural models that read the article as a sequence (a 1D CNN, a bidirectional LSTM, and fine-tuned AfriBERTa).

## Abstract

Swahili is spoken by more than 100 million people but has few labelled datasets. We study topic classification of Swahili news, a task with severe class imbalance (the smallest class has 17 of 5,151 articles), long articles (26.5% exceed the 512-token limit of pretrained transformers), rich morphology (53.5% of word forms occur only once) and some label noise between national and business news. Using a fixed stratified split, macro-F1 with bootstrap confidence intervals, and a logged series of experiments per model, we compare bag-of-words baselines with a CNN, a BiLSTM and AfriBERTa. The best baseline (TF-IDF + Logistic Regression) reaches 0.874 accuracy but only 0.633 macro-F1, because it never recognises the entertainment class. The CNN (0.521) and BiLSTM (0.578), trained from scratch, do not beat this baseline, while fine-tuned AfriBERTa reaches 0.782 macro-F1 and 0.921 accuracy and is the only model that recognises both small classes. On this task, the benefit of sequential modelling comes from pretraining on African-language text rather than from the architecture alone; the remaining errors are dominated by inconsistent labels between national and business news.

## Team

| Member | Contribution | Where |
|---|---|---|
| Mahlet Assefa Tilahun | Approach 1: TF-IDF + Logistic Regression | `02`, Section 2 |
| Tedla Tesfaye Godebo | Approach 2: TF-IDF + Naive Bayes | `02`, Section 3 |
| Carla Lisa Batoni | Approach 3: 1D CNN; comparison of all five approaches | `03`, Sections 2 and 5 |
| Mukunzi Ndahiro James | Approaches 4 and 5: Bidirectional LSTM, fine-tuned AfriBERTa; error analysis | `03`, Sections 3, 4 and 6 |

Contribution tracker: _[link]_ · Report (PDF): _[link]_ · Demo video: _[link]_

## Repository structure

```text
├── data/
│   ├── split_ids.csv          fixed stratified train/validation split used by every model
│   ├── Train.csv              download from Zindi (not in git)
│   └── Test.csv               download from Zindi (not in git)
├── notebooks/
│   ├── 01_eda_and_model_selection.ipynb   EDA, metrics, related work, choice of the five approaches
│   ├── 02_baselines.ipynb                 TF-IDF + Logistic Regression, TF-IDF + Naive Bayes
│   └── 03_neural_models.ipynb             CNN, BiLSTM, AfriBERTa, comparison, error analysis, limitations
├── src/
│   ├── data.py                loading, cleaning, split, paths, seed
│   ├── text.py                word vocabulary, AfriBERTa tokeniser, truncation strategies
│   ├── neural.py              CNN, BiLSTM, AfriBERTa wrapper, training loop with early stopping
│   └── evaluate.py            metrics, bootstrap CI, confusion matrices, learning curves, experiment log
├── results/                   metrics and validation predictions of each model, experiments.csv
├── figures/                   all figures used in the report
└── requirements.txt
```

## How to run

### Google Colab (recommended)

| Notebook | Runtime |
|---|---|
| `01_eda_and_model_selection.ipynb` | CPU |
| `02_baselines.ipynb` | CPU |
| `03_neural_models.ipynb` | **T4 GPU** |

1. Download `Train.csv` and `Test.csv` from the [Zindi data page](https://zindi.world/competitions/swahili-news-classification-challenge/data) (free account required).
2. In Colab, choose *File → Open notebook → GitHub*, open the notebook from this repository, and run the first cell. It clones this repository and asks you to upload the two CSV files.
3. Run the notebooks in order: `03` compares against the results saved by `02`. Because Colab sessions are separate, either run `02` and `03` in the same session, or commit the `results/` files from `02` before opening `03`.

### Locally

```bash
git clone https://github.com/mukunzindahiro-max/formativeII-NLP.git
cd formativeII-NLP
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Put `Train.csv` and `Test.csv` in `data/` before running. The neural notebook runs on CPU but is slow; AfriBERTa needs a GPU in practice.

## Methodology in brief

- **Cleaning** (`src/data.py`): 255 sports articles stored as Python list strings are parsed and re-joined; 3 near-empty articles are removed.
- **Split:** stratified 80/20 train/validation split saved in `data/split_ids.csv` (4,118 / 1,030 articles). Baselines are tuned with 5-fold stratified cross-validation on the training part; neural models use a stratified 10% development split for early stopping and model selection. The validation set is only used for reporting.
- **Metrics:** macro-F1 (primary) with a 95% bootstrap confidence interval, per-class F1, macro-F1 over the three large classes, accuracy, log loss, and McNemar's test between models.
- **Imbalance:** balanced class weights (Logistic Regression), uniform prior (Naive Bayes), square-root balanced loss weights (neural models).
- **Tokens:** AfriBERTa's SentencePiece tokeniser for all neural models (a word-level vocabulary is also tested for the CNN); 512-token inputs, with head+tail truncation tested for AfriBERTa.
- **Reproducibility:** fixed seed (42), fixed split file, and every experiment appended to `results/experiments.csv`.

## Results

Validation set (1,030 articles). Full per-class results are in `results/*_metrics.json`, the experiment log is `results/experiments.csv`, and the final comparison table is written to `results/model_comparison.csv` by notebook 03.

| Approach | Macro-F1 (95% CI) | Macro-F1, 3 large classes | Accuracy | Log loss |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.633 (0.561-0.684) | 0.879 | 0.874 | 0.457 |
| TF-IDF + Naive Bayes | 0.621 (0.538-0.680) | 0.858 | 0.852 | 0.526 |
| 1D CNN | 0.521 (0.508-0.533) | 0.868 | 0.862 | 0.406 |
| BiLSTM | 0.578 (0.494-0.666) | 0.834 | 0.828 | 0.549 |
| **AfriBERTa** | **0.782 (0.641-0.889)** | **0.925** | **0.921** | **0.392** |

Fine-tuned AfriBERTa is significantly better than every other approach (McNemar's test, p < 0.0001). The CNN and BiLSTM trained from scratch do not beat the TF-IDF + LR baseline.

## Data

The dataset belongs to its authors (David, 2020, [doi:10.5281/zenodo.4300294](https://doi.org/10.5281/zenodo.4300294)) and is distributed through Zindi, so the raw CSV files and any output containing article text (`results/errors_*.csv`) are excluded from git.
