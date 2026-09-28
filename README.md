# Sequential Models for Swahili News Classification

**Research question:** How effectively can sequential modelling approaches classify Swahili news articles into topics, and what evidence supports the strengths and limitations of each approach?

We compare five approaches on the [Swahili News Classification Challenge](https://zindi.world/competitions/swahili-news-classification-challenge) dataset (5,151 Tanzanian news articles, five categories): two bag-of-words baselines (TF-IDF with Logistic Regression and with Naive Bayes) and three neural models that read the article as a sequence (a 1D CNN, a bidirectional LSTM, and fine-tuned AfriBERTa).

## Abstract

Swahili is spoken by more than 100 million people but has few labelled datasets. We study topic classification of Swahili news, a task with severe class imbalance (the smallest class has 17 of 5,151 articles), long articles (26.5% exceed the 512-token limit of pretrained transformers), rich morphology (53.5% of word forms occur only once) and some label noise between national and business news. Using a fixed stratified split, macro-F1 with bootstrap confidence intervals, and a logged series of experiments per model, we compare bag-of-words baselines with a CNN, a BiLSTM and AfriBERTa. The best baseline (TF-IDF + Logistic Regression) reaches 0.874 accuracy but only 0.633 macro-F1, because it never recognises the entertainment class. _[Add the neural results and the main conclusion after running `03_neural_models.ipynb`.]_

## Team

| Member | Contribution | Where |
|---|---|---|
| _name_ | Data analysis, TF-IDF + Logistic Regression, TF-IDF + Naive Bayes | `01`, `02` |
| _name_ | 1D CNN | `03`, Section 2 |
| _name_ | Bidirectional LSTM | `03`, Section 3 |
| _name_ | Fine-tuned AfriBERTa | `03`, Section 4 |

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

| Notebook | Runtime | Open |
|---|---|---|
| 01 Data investigation and model selection | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mukunzindahiro-max/formativeII-NLP/blob/main/notebooks/01_eda_and_model_selection.ipynb) |
| 02 Baselines | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mukunzindahiro-max/formativeII-NLP/blob/main/notebooks/02_baselines.ipynb) |
| 03 Neural models and comparison | **T4 GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mukunzindahiro-max/formativeII-NLP/blob/main/notebooks/03_neural_models.ipynb) |

1. Download `Train.csv` and `Test.csv` from the [Zindi data page](https://zindi.world/competitions/swahili-news-classification-challenge/data) (free account required).
2. Open a notebook with the badge above and run the first cell. It clones this repository and asks you to upload the two CSV files.
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
| 1D CNN | _run notebook 03_ | | | |
| BiLSTM | _run notebook 03_ | | | |
| AfriBERTa | _run notebook 03_ | | | |

## Data

The dataset belongs to its authors (David, 2020, [doi:10.5281/zenodo.4300294](https://doi.org/10.5281/zenodo.4300294)) and is distributed through Zindi, so the raw CSV files and any output containing article text (`results/errors_*.csv`) are excluded from git.
