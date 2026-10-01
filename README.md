# Sequential Models for Swahili News Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C)
![seaborn](https://img.shields.io/badge/seaborn-0.12%2B-4C72B0)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?logo=huggingface&logoColor=black)
![SentencePiece](https://img.shields.io/badge/SentencePiece-0.1.99%2B-6A1B9A)
![Protobuf](https://img.shields.io/badge/Protobuf-3.20%2B-00897B)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

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
4. In notebook `03`, choose *Runtime → Change runtime type → T4 GPU*. The CNN and BiLSTM take a few minutes each; each AfriBERTa experiment takes roughly 5-15 minutes on a T4. Each model section can be run on its own after Section 1, but Sections 5-6 need the result files of all five models in `results/`.

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

Validation set (1,030 articles), from the Colab T4 run shown in `03_neural_models.ipynb`. Running the notebooks writes the per-class results to `results/*_metrics.json`, the validation predictions to `results/*_val_predictions.csv`, the experiment log to `results/experiments.csv` and the final comparison table to `results/model_comparison.csv`. The committed `results/` folder contains the baseline and CNN files only; the BiLSTM and AfriBERTa files are created when notebook 03 is run.

| Approach | Macro-F1 (95% CI) | Macro-F1, 3 large classes | Accuracy | Log loss |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.633 (0.561-0.684) | 0.879 | 0.874 | 0.457 |
| TF-IDF + Naive Bayes | 0.621 (0.538-0.680) | 0.858 | 0.852 | 0.526 |
| 1D CNN | 0.521 (0.508-0.533) | 0.868 | 0.862 | 0.406 |
| BiLSTM | 0.578 (0.494-0.666) | 0.834 | 0.828 | 0.549 |
| **AfriBERTa** | **0.782 (0.641-0.889)** | **0.925** | **0.921** | **0.392** |

Fine-tuned AfriBERTa is significantly better than every other approach (McNemar's test, p < 0.0001). The CNN and BiLSTM trained from scratch do not beat the TF-IDF + LR baseline.

### Experiments

15 experiments were run, three per approach; the variant with the best selection score (cross-validation for the baselines, development macro-F1 for the neural models) is the one reported above.

| Approach | Variants tested | Selection macro-F1 |
|---|---|---|
| TF-IDF + LR | unigrams with balanced weights · unigrams + bigrams · unigrams without class weights | 0.650 · 0.615 · 0.552 |
| TF-IDF + NB | multinomial, learned prior · multinomial, uniform prior · Complement NB | 0.541 · 0.642 · 0.513 |
| 1D CNN | subword tokens · subword tokens with class weights · word tokens with class weights | 0.518 · 0.524 · 0.501 |
| BiLSTM | last hidden states, 512 tokens · attention pooling, 512 tokens · last hidden states, first 256 tokens | 0.742 · 0.519 · 0.766 |
| AfriBERTa | first 510 tokens · first 510 tokens with class weights · first 128 + last 382 tokens | 0.844 · 0.836 · 0.829 |

The development split has only 1 `Burudani` and 4 `Kimataifa` articles, so selection scores for the neural models are noisy.

## Data

The dataset belongs to its authors (David, 2020, [doi:10.5281/zenodo.4300294](https://doi.org/10.5281/zenodo.4300294)) and is distributed through Zindi, so the raw CSV files and any output containing article text (`results/errors_*.csv`) are excluded from git.
