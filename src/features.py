"""
features.py
===========
Feature extraction for Indonesian text classification.

Supports:
- TF-IDF (Unigram, Bigram, Trigram)
- Feature importance visualization
- Saving/loading vectorizers

Usage:
    from src.features import TFIDFExtractor
    fe = TFIDFExtractor(ngram_range=(1, 2), max_features=20000)
    X_train = fe.fit_transform(train_texts)
    X_test  = fe.transform(test_texts)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


class TFIDFExtractor:
    """
    TF-IDF feature extractor with utilities for Indonesian NLP tasks.

    Args:
        ngram_range: Tuple (min_n, max_n) for n-gram range.
        max_features: Maximum number of features to keep.
        min_df: Minimum document frequency to include a term.
        sublinear_tf: Apply sublinear TF scaling (log(1+tf)).
        name: Identifier for saving/loading.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 20000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        name: str = "tfidf",
    ):
        self.name = name
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        self._is_fitted = False

    def fit_transform(self, texts: List[str]):
        """Fit vectorizer and transform training texts."""
        print(f"[TFIDFExtractor] Fitting on {len(texts):,} texts...")
        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        print(f"[TFIDFExtractor] Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"[TFIDFExtractor] Feature matrix: {X.shape}")
        return X

    def transform(self, texts: List[str]):
        """Transform texts using fitted vectorizer."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        return np.array(self.vectorizer.get_feature_names_out())

    def get_top_features(self, X, top_n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Get top N features by mean TF-IDF score."""
        if issparse(X):
            mean_scores = X.mean(axis=0).A1
        else:
            mean_scores = X.mean(axis=0)
        feature_names = self.get_feature_names()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        return feature_names[top_idx], mean_scores[top_idx]

    def plot_top_features(
        self,
        X,
        top_n: int = 20,
        title: str = "Top TF-IDF Features",
        save_path: Optional[str] = None,
    ):
        """Plot top N TF-IDF features by mean score."""
        names, scores = self.get_top_features(X, top_n)
        sorted_idx = np.argsort(scores)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names[sorted_idx], scores[sorted_idx], color="#3498db", alpha=0.85, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Mean TF-IDF Score")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[TFIDFExtractor] Figure saved to {save_path}")
        plt.show()
        return fig

    def plot_class_features(
        self,
        texts: List[str],
        labels: List[str],
        top_n: int = 15,
        save_path: Optional[str] = None,
    ):
        """Plot top features per class (side-by-side)."""
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12"]

        fig, axes = plt.subplots(1, n_classes, figsize=(8 * n_classes, 6))
        if n_classes == 1:
            axes = [axes]

        for ax, label, color in zip(axes, unique_labels, colors):
            class_texts = [t for t, l in zip(texts, labels) if l == label]
            X_class = self.transform(class_texts)
            names, scores = self.get_top_features(X_class, top_n)
            sorted_idx = np.argsort(scores)
            ax.barh(names[sorted_idx], scores[sorted_idx], color=color, alpha=0.85)
            ax.set_title(f"{label}", fontweight="bold")
            ax.set_xlabel("Mean TF-IDF Score")

        fig.suptitle("Top Features per Class", fontweight="bold", fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def save(self, directory: str = "results"):
        """Save vectorizer to disk."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[TFIDFExtractor] Saved to {path}")

    def load(self, directory: str = "results"):
        """Load vectorizer from disk."""
        path = os.path.join(directory, f"{self.name}.pkl")
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._is_fitted = True
        print(f"[TFIDFExtractor] Loaded from {path}")


# ─────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "produk bagus pengiriman cepat puas",
        "barang rusak tidak sesuai gambar kecewa",
        "kualitas oke harga sesuai lumayan",
        "seller ramah fast response recommended",
        "paket tidak sampai refund tidak diproses",
    ]
    labels = ["Positive", "Negative", "Positive", "Positive", "Negative"]

    fe = TFIDFExtractor(ngram_range=(1, 2), max_features=500, name="demo_tfidf")
    X = fe.fit_transform(sample_texts)
    print("\nTop features overall:")
    names, scores = fe.get_top_features(X, top_n=10)
    for n, s in zip(names, scores):
        print(f"  {n:<25} {s:.4f}")
