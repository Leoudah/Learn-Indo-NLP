"""
preprocessing.py
================
Indonesian NLP Preprocessing Pipeline

Handles: case folding, URL/mention removal, slang normalization,
tokenization, stopword removal, and Sastrawi stemming.

Usage:
    from src.preprocessing import IndonesianPreprocessor
    prep = IndonesianPreprocessor(do_stem=True)
    clean = prep.transform(["Produknya bagus banget!"])
"""

import re
import json
from pathlib import Path
from typing import List, Optional, Union

import nltk
from nltk.tokenize import word_tokenize

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("[WARNING] PySastrawi not installed. Run: pip install PySastrawi")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────
# Slang Dictionary (Indonesian informal → formal)
# ─────────────────────────────────────────────
SLANG_DICT = {
    # Pronouns
    "gw": "saya", "gue": "saya", "gu": "saya",
    "lo": "kamu", "lu": "kamu",
    # Intensifiers
    "bgt": "sangat", "banget": "sangat", "bngt": "sangat",
    "bener": "benar", "beneran": "benar",
    # Product quality
    "ok": "baik", "oke": "baik", "okelah": "baik",
    "mantap": "bagus", "mantul": "bagus", "kece": "bagus",
    "jelek": "buruk", "jelekk": "buruk", "jlek": "buruk",
    "rusak": "rusak", "cacat": "cacat",
    # Actions
    "dapet": "dapat", "dpt": "dapat",
    "pake": "pakai", "dipake": "dipakai",
    "dikirim": "kirim", "ngirim": "kirim",
    # Common abbreviations
    "yg": "yang", "dgn": "dengan", "utk": "untuk",
    "kpd": "kepada", "sbg": "sebagai",
    "pdhl": "padahal", "bkn": "bukan",
    "tp": "tapi", "tapi": "tapi",
    "kl": "kalau", "kalo": "kalau",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah",
    "blm": "belum", "blum": "belum",
    "tmn": "teman", "temen": "teman",
    "krn": "karena", "karna": "karena",
    "jg": "juga", "jga": "juga",
    "sm": "sama",
    # Negations
    "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "enggak": "tidak", "gk": "tidak",
    # Financial domain (thesis prep)
    "ihsg": "ihsg", "idx": "idx", "bei": "bei",
    "saham": "saham", "emiten": "emiten",
    "bullish": "bullish", "bearish": "bearish",
    "naik": "naik", "turun": "turun",
}

# Custom stopwords (domain-specific additions)
CUSTOM_STOPWORDS = {
    "nih", "sih", "deh", "dong", "lah", "aja", "aj",
    "nya", "ini", "itu", "ke", "di", "yang", "dan",
    "atau", "juga", "sudah", "belum", "tidak", "dengan",
    "untuk", "dari", "pada", "oleh", "karena", "kalau",
    "tapi", "sama", "bisa", "mau", "kita", "kami", "mereka",
}


class IndonesianPreprocessor:
    """
    Full Indonesian NLP preprocessing pipeline.

    Steps:
        1. Lowercase
        2. Remove URLs, mentions (@), hashtags (#)
        3. Remove emojis and non-ASCII
        4. Remove numbers and special characters
        5. Normalize whitespace
        6. Slang normalization
        7. Tokenization (NLTK)
        8. Stopword removal (Sastrawi + custom)
        9. Stemming (Sastrawi — optional)
        10. Remove short tokens (len < min_token_len)
    """

    def __init__(
        self,
        do_stem: bool = True,
        min_token_len: int = 2,
        extra_stopwords: Optional[List[str]] = None,
        slang_dict: Optional[dict] = None,
    ):
        self.do_stem = do_stem
        self.min_token_len = min_token_len

        # Stemmer
        if SASTRAWI_AVAILABLE and do_stem:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = None

        # Stopwords
        if SASTRAWI_AVAILABLE:
            sw_factory = StopWordRemoverFactory()
            self.stopwords = set(sw_factory.get_stop_words())
        else:
            self.stopwords = set()
        self.stopwords.update(CUSTOM_STOPWORDS)
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

        # Slang dictionary
        self.slang_dict = slang_dict if slang_dict else SLANG_DICT

    # ── private helpers ──────────────────────────────

    def _remove_noise(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
        text = re.sub(r"@\w+", " ", text)                  # mentions
        text = re.sub(r"#\w+", " ", text)                  # hashtags
        text = re.sub(r"[^\x00-\x7F]+", " ", text)         # non-ASCII (emojis, etc.)
        text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_slang(self, text: str) -> str:
        tokens = text.split()
        return " ".join([self.slang_dict.get(t, t) for t in tokens])

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords and len(t) >= self.min_token_len]

    def _stem(self, tokens: List[str]) -> List[str]:
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    # ── public API ───────────────────────────────────

    def preprocess(self, text: str) -> str:
        """Preprocess a single string."""
        text = text.lower()
        text = self._remove_noise(text)
        text = self._normalize_slang(text)
        tokens = word_tokenize(text)
        tokens = self._remove_stopwords(tokens)
        if self.do_stem:
            tokens = self._stem(tokens)
        return " ".join(tokens)

    def transform(self, texts: List[str], verbose: bool = False) -> List[str]:
        """Preprocess a list of texts."""
        results = []
        for i, text in enumerate(texts):
            results.append(self.preprocess(str(text)))
            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(texts)}")
        return results

    def fit_transform(self, texts: List[str], verbose: bool = True) -> List[str]:
        """Alias for transform (sklearn-compatible naming)."""
        if verbose:
            print(f"[IndonesianPreprocessor] Processing {len(texts):,} texts...")
        result = self.transform(texts, verbose=verbose)
        if verbose:
            print(f"[IndonesianPreprocessor] Done!")
        return result


# ─────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    prep = IndonesianPreprocessor(do_stem=True)

    examples = [
        "Produknya bagus banget!! pengiriman cepet, puas bgt sama seller yg ini 👍",
        "Kecewa banget, barang tidak sesuai foto sama sekali, minta refund gak direspon",
        "IHSG menguat didorong sentimen positif dari laporan keuangan emiten besar",
        "Bursa saham Indonesia melemah akibat kekhawatiran inflasi global yg meningkat",
    ]

    print("=" * 60)
    print("Indonesian NLP Preprocessing Demo")
    print("=" * 60)
    for text in examples:
        clean = prep.preprocess(text)
        print(f"\nOriginal : {text}")
        print(f"Processed: {clean}")
    print("=" * 60)
