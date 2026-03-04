# Train a Word Sense Classifier for the German Language

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [How to Use](#how-to-use)
* [Code Example](#code-example)


## General Info
This package trains a word sense classifier from the data that is generated with the algorithm from [https://github.com/AndreasPadProbst/fastSense_CreateData](https://github.com/AndreasPadProbst/fastSense_CreateData).
The main structure of this code is taken from [https://github.com/texttechnologylab/fastSense](https://github.com/texttechnologylab/fastSense) (accessed: 20.02.2021) and modified to be able to process
the German version of the Wikipedia. [https://github.com/texttechnologylab/fastSense](https://github.com/texttechnologylab/fastSense) is a model based on fastSense (see [paper](https://www.aclweb.org/anthology/L18-1168/)).

## Setup
This package was used with **Python 3.9** with the following package specifications:
* **tensorflow** = 2.5

## How to Use
To train the model, the cli_train.py script needs to me executed with the following command line specifications:

* **--data**: Path to the folder containing the training, validation, and test data.
* **--models_dir**: Path to the folder where the trained model should be saved. If the folder already contains an existing model, the script will continue training the model.
* **--final_models_dir**: Path to the folder where the trained models will be saved after each epoch. Each model is placed in a timestamped subfolder, together with information on the used parameters to train that model.
* **--jobs**: Path to a JSON file containing the parameters that should be used to train the model.

## Code Example
```
python cli_train.py --data ./training_data --models_dir ./saved_model --final_ models_dir ./final_model_output --data ./train_jobs.json
```

# Thesis Findings: Influence of Word Sense Disambiguation on Legal Information Retrieval

## Summary of Chapters 3–5

**Author:** Andreas Probst | **Institution:** Technical University of Munich | **Thesis Type:** Master's Thesis (CSE)

## Table of contents

* [Preparation and Resources](#preparation-and-resources)
* [Implementation](#implementation)
* [Quantitative Evaluation](#quantitative-evaluation)
---

## Preparation and Resources

### Expert Interviews

Six lawyers from different age groups (26–55) and legal domains were interviewed in semi-structured sessions to assess whether WSD could improve legal information retrieval. The key findings were:

- Lawyers spend on average 5.25 hours/week on legal research, with significant weekly fluctuation.
- Current German online legal databases return too many irrelevant results, even with filters.
- Spelling and wording exert excessive influence on retrieved results, making it hard to find relevant documents without exact keywords.
- Ambiguous *legal* terms exist only across different legal domains, not within a single domain. Existing "legal area" filters already resolve these with 100% accuracy, making ML-based disambiguation of legal terms unnecessary.
- A sense filter for **natural language terms** (e.g., the German word "Bahn") was judged as highly beneficial by the experts. Lawyers preferred sense options to be displayed in a dropdown menu during query input.

Based on these findings, the thesis focuses exclusively on disambiguating natural language terms that appear in German court rulings.

### Legal Dataset

A corpus of **56,606 court rulings** from 1,980 German courts was provided by the legal publisher Dr. Otto Schmidt KG. The corpus covers 26 court types (dominated by OLG rulings at >33%), spans publication years from 1955–2017, and includes three legal decision types (judgements, court orders, injunctions). Within this corpus, **23,920 unique ambiguous word tokens** were identified using the sense database built from German Wikipedia.

### Manually Tagged Legal Test Set

Since no sense-labelled German legal dataset was publicly available, a test set of **3,826 manually annotated ambiguous words** was created by a German-proficient clickworker. The test set was sampled by randomly selecting text segments (≥1,000 characters) to ensure court type variety. Special labels ("NV", "RW", "EN", "AK", "?") were introduced for cases where a standard sense label could not be assigned (e.g., missing senses, idioms, proper names, abbreviations). An inter-rater reliability assessment on a 375-word subset yielded **69.6% agreement**, with disagreements stemming from overlapping sense definitions and Wikipedia redirect errors in the sense database.

---

## Implementation

### Key Idea

The implementation leverages the **fastSense algorithm**, which exploits Wikipedia disambiguation pages to create a sense inventory and sense-tagged training corpus automatically. For each ambiguous word listed on a disambiguation page, every paragraph in the linked Wikipedia article that contains that word is treated as a training example for the corresponding sense. This yields a large-scale sense-annotated dataset without manual labelling. The trained classifier can then be applied to legal texts to tag ambiguous words with their correct senses.

### Data Creation Pipeline

The pipeline is split into two components:

1. **Extractor** — Parses German Wikipedia dumps (category links, page metadata, and article content). It standardizes article titles, resolves redirections, identifies disambiguation pages, builds sense groups, and extracts paragraphs with their linked senses. The implementation uses Python multiprocessing across 6 threads. *(GitHub: [fastSense_CreateData](https://github.com/AndreasPadProbst/fastSense_CreateData))*

2. **Exporter** — Produces a SQLite sense database (mapping sense groups → individual senses → Wikipedia article URLs) and TensorFlow Records containing tokenized text segments with their sense labels. The complete dataset comprises **64,274,208 sense-tagged examples** covering **342,861 different senses**, split 70/15/15 into train/validation/test sets (stratified). *(GitHub: [sense_db](https://github.com/AndreasPadProbst/sense_db))*

### Neural Model Architectures

Two architectures were implemented and evaluated:

1. **Hashing Model** (original fastSense architecture) — Word tokens are hashed to integers, passed through an embedding layer, averaged via a square-root combiner, and fed to a softmax output layer of 342,861 nodes (one per sense). A post-processing step restricts the output to valid senses for the target word. Trained on the full 64M examples. *(GitHub: [fastSense_TrainModel](https://github.com/AndreasPadProbst/fastSense_TrainModel))*

2. **FastBERT Model** — Replaces the hashing+embedding approach with contextual embeddings from a pre-trained German BERT model (bert-base-german-cased, 109M parameters). Uses [CLS] token embeddings of sentences as input to the classification head. Trained on a reduced subset of ~314,700 examples (~32,000 senses) due to the computational cost of BERT forward passes. Train/test split: 75/25.

---

## Quantitative Evaluation

### Evaluation Metrics

Two accuracy metrics are used:

- **Micro accuracy** — Fraction of correctly classified senses over all classifications (equivalent to precision). Sensitive to sense frequency imbalance (Zipf's law), as common senses dominate.
- **Macro accuracy** — Average per-sense accuracy across all senses. A more robust metric that compensates for the skewed sense distribution by giving equal weight to each sense.

### Hashing Model Results

A systematic parameter study was conducted. Key findings:

| Parameter Variation | Best Micro Accuracy | Best Macro Accuracy |
|---|---|---|
| N-grams (1/2/3) | 1-grams: **88.30%** | 3-grams: **83.63%** |
| Lemmatization | Lemmatized: **88.30%** | Similar (~81.6%) |
| Sentences vs. Paragraphs | Similar performance | Similar performance |
| POS information | More stable micro curves | Slight drop for sentences |
| Dropout (10/20/30%) | All worse than no dropout | All worse than no dropout |
| 50 hidden nodes | **89.83%** | **84.08%** |
| 2 hidden layers (25×25) | 86.71% (overfitting) | 78.73% |

**Interpretation:** The best-performing hashing model uses 50 hidden layer nodes, achieving ~90% micro and ~84% macro accuracy. Higher n-grams improved macro accuracy (better per-sense generalization) but reduced micro accuracy. Dropout was counterproductive with only 25 neurons, as the model showed no signs of overfitting. Adding a second hidden layer caused clear overfitting due to exponential growth in parameters.

### FastBERT Results

#### On the Generated Wikipedia Test Set

| Model Configuration | Best Micro Accuracy |
|---|---|
| 12 nodes, no dropout | 84.23% |
| 25 nodes, no dropout | 84.78% |
| 50 nodes, no dropout | 85.23% |
| 100 nodes, no dropout | 85.63% |
| **25 nodes, 20% dropout** | **88.59%** |

**Interpretation:** Despite being trained on less than 1% of the hashing model's data (~315K vs. 64M examples), fastBERT achieved comparable accuracy (~89%). Dropout had the opposite effect compared to the hashing model — here, 20% dropout was beneficial, preventing overfitting that was visible as accuracy declined after 3–4 epochs. This aligns with BERT fine-tuning literature recommending 2–4 epochs.

#### On the Manually Tagged Legal Test Set (991 examples)

| Model Configuration | Best Micro Accuracy |
|---|---|
| 100 nodes, no dropout | **74.47%** |
| 25 nodes, 20% dropout | 73.76% |
| 50×25 (2 layers) | 73.16% |
| 25 nodes, no dropout | 72.75% |

**Interpretation:** Accuracy dropped significantly (~15 percentage points) when moving from Wikipedia-based test data to legal texts. A clear downward trend with increasing epochs indicated overfitting and domain mismatch — legal language differs substantially from Wikipedia prose. Additionally, the 69.6% inter-annotator agreement sets a practical upper bound on achievable accuracy for the legal test set. The best model achieved **74.47%**, which is meaningful when contextualized against this agreement ceiling. The sense labels' inherent subjectivity and occasional overlaps in sense definitions contribute to performance limits.

### Overall Takeaway

The quantitative evaluation demonstrates that automatic WSD for German texts is feasible at ~90% accuracy on general-domain text. The transfer to legal texts incurs a notable accuracy loss, but the results (~74%) remain promising given the domain gap and annotation subjectivity. The hashing model excels with abundant training data, while fastBERT is remarkably data-efficient, achieving comparable results with <1% of the training samples. For a practical legal information retrieval system, fine-tuning on legal-domain data and refining the sense database (e.g., merging duplicate senses from Wikipedia redirects) would likely improve performance further.
