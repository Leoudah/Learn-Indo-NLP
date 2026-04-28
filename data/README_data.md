# 📦 Dataset — PRDECT-ID

**This directory is intentionally empty.** The dataset is not included in this repository due to redistribution policies.

---

## Dataset Details

| Property | Value |
|---|---|
| **Name** | PRDECT-ID — Product Reviews Dataset for Emotions Classification Tasks (Indonesian) |
| **DOI** | [10.17632/574v66hf2v.1](https://doi.org/10.17632/574v66hf2v.1) |
| **Publisher** | Mendeley Data |
| **License** | **CC BY 4.0** (Attribution required) |
| **Language** | Bahasa Indonesia |
| **Platform** | Tokopedia (29 product categories) |
| **Size** | ~5,400 labeled product reviews |
| **Labels** | Sentiment (Positive/Negative) + Emotion (Love, Happiness, Anger, Fear, Sadness) |
| **Extra Features** | Location, Price, Overall Rating, Number Sold, Total Review, Customer Rating |

### Citation
```
Sutoyo, R., Chowanda, A., Achmad, S., Andangsari, E.W., Isa, S.M., et al. (2022).
Product Reviews Dataset for Emotions Classification Tasks - Indonesian (PRDECT-ID) Dataset.
Mendeley Data. DOI: 10.17632/574v66hf2v.1
```

---

## Download Instructions

### Option 1: Direct Download (Recommended)
1. Go to: https://data.mendeley.com/datasets/574v66hf2v/1
2. Click **"Download All"**
3. Extract the ZIP file to this `data/` directory

Expected file after extraction:
```
data/
└── PRDECT-ID.csv    ← main file
```

### Option 2: Automated (Python)
```python
import urllib.request
import zipfile

URL = 'https://data.mendeley.com/public-files/datasets/574v66hf2v/files/7e4b8c4c-a5dc-40d6-b5b4-e0a3ba84c555/file_downloaded'

urllib.request.urlretrieve(URL, 'data/prdect_id.zip')
with zipfile.ZipFile('data/prdect_id.zip', 'r') as z:
    z.extractall('data/')
print('Done!')
```

### Option 3: From Colab
The main notebook handles this automatically — just run the first data acquisition cells.

---

## CSV Structure

After downloading, `PRDECT-ID.csv` should have these columns:

| Column | Type | Description |
|---|---|---|
| `Review` | str | Product review text (Bahasa Indonesia) |
| `Sentiment` | str | `Positive` or `Negative` |
| `Emotion` | str | One of: `Love`, `Happiness`, `Anger`, `Fear`, `Sadness` |
| `Category` | str | Product category (29 options) |
| `Location` | str | Reviewer location |
| `Price` | float | Product price |
| `Overall Rating` | float | Product overall rating |
| `Number Sold` | int | Units sold |
| `Total Review` | int | Total review count |
| `Customer Rating` | int | Rating given by reviewer (1–5) |

---

## Directory Structure After Setup

```
data/
├── README_data.md         ← this file
└── PRDECT-ID.csv          ← download this
```

---

## License Note

This dataset is published under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
You are free to use, share, and adapt the data as long as you provide appropriate credit to the original authors.

See full license: https://creativecommons.org/licenses/by/4.0/
