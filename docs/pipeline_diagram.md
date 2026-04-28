# 🔧 Indonesian NLP Pipeline — Documentation

## Overview

This document describes the full text preprocessing and modeling pipeline used in this project.

---

## Pipeline Diagram

```
Raw Text (Bahasa Indonesia)
        │
        ▼
┌─────────────────────┐
│  1. Lowercase       │  "Produknya BAGUS!!" → "produknya bagus!!"
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Noise Removal   │  Remove URLs, @mentions, #hashtags, emojis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Char Cleaning   │  Keep only [a-z] and spaces
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Slang Normalization│ "bgt" → "sangat", "gak" → "tidak"
│     (Custom Dict)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Tokenization    │  NLTK word_tokenize
│     (NLTK)          │  "produk bagus" → ["produk", "bagus"]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Stopword Removal│  Sastrawi + custom domain stopwords
│  (Sastrawi + Custom)│  Remove: "dan", "atau", "yang", etc.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  7. Stemming        │  Sastrawi ECS (Enhanced Confix Stripping)
│     (Sastrawi)      │  "pengiriman" → "kirim"
└─────────┬───────────┘
          │
          ▼
   Clean Token List
          │
     ┌────┴────┐
     │         │
     ▼         ▼
TF-IDF     IndoBERT
Unigram/   Tokenizer
Bigram     (WordPiece)
     │         │
     ▼         ▼
  Sparse    Dense
  Matrix    Embeddings
     │         │
     ▼         ▼
  ML Model  Fine-tuned
 (NB/SVM/LR) IndoBERT
     │         │
     └────┬────┘
          │
          ▼
   Sentiment / Emotion Label
```

---

## Step-by-Step Details

### Step 1: Lowercase
Convert all text to lowercase to ensure case-insensitive matching.
```
Input  : "Produknya BAGUS Banget!!"
Output : "produknya bagus banget!!"
```

### Step 2: Noise Removal
Remove non-textual elements that add noise.
- URLs: `https://...` → removed
- Mentions: `@user` → removed
- Hashtags: `#belanja` → removed
- Emojis/non-ASCII: 👍😊 → removed

### Step 3: Character Cleaning
Keep only lowercase letters and spaces.
```
Input  : "produk bagus! harga 50rb... (worth it)"
Output : "produk bagus harga rb worth it"
```

### Step 4: Slang Normalization
Map Indonesian informal/abbreviated words to formal equivalents.

| Slang | Formal |
|---|---|
| bgt, banget | sangat |
| gak, gk, nggak | tidak |
| yg | yang |
| dgn | dengan |
| udah, udh | sudah |
| kalo, kl | kalau |
| tp | tapi |
| pake | pakai |
| dapet | dapat |

> Full dictionary: [`src/preprocessing.py`](../src/preprocessing.py) → `SLANG_DICT`

### Step 5: Tokenization
Using NLTK's `word_tokenize` which handles Indonesian text reasonably well.
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize("produk sangat bagus pengiriman cepat")
# ['produk', 'sangat', 'bagus', 'pengiriman', 'cepat']
```

### Step 6: Stopword Removal
Remove high-frequency, low-information words.
- **Sastrawi stopwords**: 758 common Indonesian words
- **Custom additions**: domain-specific noise words (nih, sih, deh, dll.)

```python
# Words removed (examples)
stopwords = {'dan', 'yang', 'atau', 'ini', 'itu', 'ke', 'di', 'nih', 'sih', ...}
```

### Step 7: Stemming (Sastrawi)
Using the **Enhanced Confix Stripping (ECS)** algorithm — the standard Indonesian stemmer.

| Original | Stemmed |
|---|---|
| pengiriman | kirim |
| kecewa | kecewa |
| mengecewakan | kecewa |
| penjual | jual |
| pembelian | beli |
| berkualitas | kualitas |

> Note: For IndoBERT pipeline, stemming is **skipped** — BERT handles morphology internally.

---

## Feature Extraction

### TF-IDF Configuration
```python
TfidfVectorizer(
    ngram_range=(1, 2),     # Unigram + Bigram
    max_features=20000,     # Top 20K features
    min_df=2,               # Appear in ≥2 docs
    sublinear_tf=True,      # log(1+tf) scaling
)
```

**Why Bigram?** Captures negations and compound expressions:
- `tidak bagus` (not good) ≠ `tidak` + `bagus` separately
- `sangat puas` (very satisfied) conveys stronger sentiment than `puas` alone

### IndoBERT Tokenization
```python
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
# Uses WordPiece tokenization trained on Indonesian corpus
# max_length = 128 tokens
# padding = 'max_length'
# truncation = True
```

---

## Model Architecture

### Classical ML Stack
```
Input (TF-IDF sparse matrix)
         │
   ┌─────┼──────┐
   │     │      │
   ▼     ▼      ▼
  NB    SVM    LR
   │     │      │
   └─────┼──────┘
         │
    Best by F1
```

### IndoBERT Fine-tuning
```
Input tokens
    │
IndoBERT (12 layers, 768 hidden, 12 heads)
    │
[CLS] representation
    │
Dropout (0.1)
    │
Linear(768 → num_labels)
    │
Softmax
    │
Predicted label
```

**Training config:**
- Epochs: 3 (with early stopping, patience=2)
- Learning rate: 2e-5 (AdamW)
- Warmup steps: 100
- Batch size: 16
- Mixed precision: fp16 (if GPU)

---

## Evaluation Metrics

| Metric | Formula | When to Use |
|---|---|---|
| Accuracy | TP+TN / All | Balanced datasets |
| Precision | TP / (TP+FP) | When false positives are costly |
| Recall | TP / (TP+FN) | When false negatives are costly |
| F1 (weighted) | Harmonic mean, class-weighted | **Primary metric** (imbalanced classes) |
| F1 (macro) | Unweighted average F1 | Equal weight per class |

---

## References

1. Sutoyo, R. et al. (2022). PRDECT-ID Dataset. *Mendeley Data*. DOI: 10.17632/574v66hf2v.1
2. Wilie, B. et al. (2020). IndoNLU. *arXiv:2009.05387*
3. Asian, J. et al. (2007). Effective Techniques for Indonesian Text Retrieval. *PhD Thesis, RMIT*
4. Tala, F.Z. (2003). A Study of Stemming Effects on Information Retrieval in Bahasa Indonesia. *MSc Thesis, Amsterdam*
