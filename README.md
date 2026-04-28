# 🇮🇩 Indonesian NLP - Sentiment & Emotion Analysis
### Text Mining Portfolio | Informatika Universitas Udayana 

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-red)](https://data.mendeley.com/datasets/574v66hf2v/1)
[![IndoBERT](https://img.shields.io/badge/Model-IndoBERT-orange)](https://huggingface.co/indobenchmark/indobert-base-p1)

---

## Overview

My Biggest project so far, this project demonstrates a **complete Indonesian NLP pipeline** for sentiment and emotion classification using the [PRDECT-ID dataset](https://data.mendeley.com/datasets/574v66hf2v/1) - a publicly available collection of Tokopedia product reviews annotated with sentiment and emotion labels.

> **Research Context:** This portfolio is developed as a foundation for the thesis *"HIDDEN FOR NOW"*  - demonstrating full command of the Text Mining pipeline before applying it to the financial domain.

---

## Key Contributions

| # | Contribution |
|---|---|
| 1 | End-to-end Indonesian NLP preprocessing pipeline (slang normalization, Sastrawi stemming) |
| 2 | Comparative study: Naive Bayes vs SVM vs Logistic Regression on TF-IDF features |
| 3 | Fine-tuning **IndoBERT** (`indobenchmark/indobert-base-p1`) for sentiment classification |
| 4 | Multi-class **emotion classification** (Love, Happiness, Anger, Fear, Sadness) |
| 5 | Systematic error analysis with false positive/negative breakdown |

---

## Results

### Sentiment Classification (Binary: Positive / Negative) - WILL BE UPDATED SOON

| Model | Accuracy | Precision | Recall | F1 (weighted) |
|---|---|---|---|---|
| Naive Bayes (Unigram TF-IDF) | - | - | - | - |
| Naive Bayes (Bigram TF-IDF) | - | - | - | - |
| Linear SVM (Unigram TF-IDF) | - | - | - | - |
| **Linear SVM (Bigram TF-IDF)** | - | - | - | **-** |
| Logistic Regression | - | - | - | - |
| **IndoBERT (fine-tuned)** | - | - | - | **-** |

>  Run the notebook to populate results. Values depend on the actual PRDECT-ID dataset split.

### Emotion Classification (5-class)

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Linear SVM | - | - |
| Logistic Regression | - | - |
| Naive Bayes | - | - |

---

## Repository Structure

```
portfolio-indo-nlp/
│
├── notebooks/
│   └── sentiment_analysis_prdect_id.ipynb   # Main Colab notebook (end-to-end)
│
├── src/
│   ├── preprocessing.py                      # Indonesian NLP preprocessing module
│   ├── features.py                           # TF-IDF feature extraction
│   ├── models.py                             # ML model definitions & training
│   ├── evaluate.py                           # Evaluation & metrics utilities
│   └── predict.py                            # Inference / demo script
│
├── data/
│   └── README_data.md                        # Dataset download instructions
│
├── results/
│   ├── figures/                              # Generated plots (EDA, confusion matrix, etc.)
│   └── metrics.json                          # Saved model metrics
│
├── docs/
│   └── pipeline_diagram.md                  # NLP pipeline documentation
│
├── requirements.txt                          # Python dependencies
├── .gitignore                                
└── LICENSE
```

---

## Quick Start

### Option 1: Google Colab (Recommended)
Download Notebook and Data - put at the same folder - Run All cells - Done.

### Option 2: Local Environment
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/portfolio-indo-nlp.git
cd portfolio-indo-nlp

# Install dependencies
pip install -r requirements.txt

# Download dataset (see data/README_data.md)

# Run preprocessing
python src/preprocessing.py

# Train models
python src/models.py

# Evaluate
python src/evaluate.py
```

---

## Dataset

**PRDECT-ID** - Product Reviews Dataset for Emotions Classification Tasks (Indonesian)

| Property | Value |
|---|---|
| Source | [Mendeley Data - DOI: 10.17632/574v66hf2v.1](https://data.mendeley.com/datasets/574v66hf2v/1) |
| Size | ~5,400 labeled reviews |
| Language | Bahasa Indonesia |
| Source Platform | Tokopedia (29 product categories) |
| Labels | Sentiment (Positive/Negative) + Emotion (5 classes) |
| License | **CC BY 4.0** (free to use with attribution) |

> The dataset is **not included** in this repository. See [`data/README_data.md`](data/README_data.md) for download instructions.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP (Indonesian) | [PySastrawi](https://github.com/har07/PySastrawi), NLTK |
| ML | scikit-learn |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Pre-trained Model | [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) |
| Feature Extraction | TF-IDF (Unigram & Bigram) |
| Visualization | Matplotlib, Seaborn, WordCloud, Plotly |
| Environment | Google Colab, Python venv |

---

## References

1. Sutoyo, R., et al. (2022). *PRDECT-ID Dataset*. Mendeley Data. DOI: 10.17632/574v66hf2v.1
2. Wilie, B., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian NLU*. arXiv:2009.05387
3. Kalra, S. & Prasad, J.S. (2019). *Efficacy of News Sentiment for Stock Market Prediction*. ResearchGate.
4. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL-HLT 2019.

---

## Author

**[LEONARDO PRAMUDYO HUTOMO]**  
Mahasiswa Informatika - Universitas Udayana  

---

<p align="center">
  <i>Portfolio project untuk mendukung penelitian skripsi J1 Text Mining - Informatika Unud 2025</i>
</p>
