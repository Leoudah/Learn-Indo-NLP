"""
predict.py
==========
Real-time inference demo for trained models.

Usage:
    # Interactive CLI
    python src/predict.py

    # Programmatic
    from src.predict import SentimentPredictor
    pred = SentimentPredictor.load("results/models")
    result = pred.predict("Produk bagus banget, sangat puas!")
    print(result)  # {'text': ..., 'sentiment': 'Positive', 'confidence': 0.92}
"""

import os
import pickle
import sys
import numpy as np
from typing import List, Dict, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import IndonesianPreprocessor


LABEL_MAP = {0: "Negative", 1: "Positive"}
LABEL_MAP_EMOTION = {0: "Anger", 1: "Fear", 2: "Happiness", 3: "Love", 4: "Sadness"}
EMOJI_MAP = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
EMOTION_EMOJI = {"Love": "❤️", "Happiness": "😊", "Anger": "😠", "Fear": "😨", "Sadness": "😢"}


class SentimentPredictor:
    """
    Lightweight inference wrapper using saved TF-IDF + ML classifier.

    Loads:
        - vectorizer : TF-IDF vectorizer (pickle)
        - model      : Trained sklearn classifier (pickle)
        - preprocessor: IndonesianPreprocessor instance
    """

    def __init__(self, vectorizer, model, label_map: dict = None):
        self.vectorizer   = vectorizer
        self.model        = model
        self.preprocessor = IndonesianPreprocessor(do_stem=True)
        self.label_map    = label_map or LABEL_MAP

    @classmethod
    def load(cls, model_dir: str = "results/models", model_filename: str = "linear_svm.pkl"):
        """Load vectorizer + model from disk."""
        vec_path   = os.path.join(model_dir, "tfidf.pkl")
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Train the model first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print(f"[SentimentPredictor] Loaded vectorizer + {model_filename}")
        return cls(vectorizer, model)

    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for one or more texts.

        Returns dict (single) or list of dicts (batch):
            {'text': str, 'clean': str, 'label': str, 'label_id': int}
        """
        single = isinstance(text, str)
        texts  = [text] if single else text

        clean_texts = self.preprocessor.transform(texts)
        X = self.vectorizer.transform(clean_texts)
        pred_ids = self.model.predict(X)

        results = []
        for raw, clean, label_id in zip(texts, clean_texts, pred_ids):
            label = self.label_map.get(label_id, str(label_id))
            results.append({
                "text":     raw,
                "clean":    clean,
                "label":    label,
                "label_id": int(label_id),
                "emoji":    EMOJI_MAP.get(label, "⚪"),
            })

        return results[0] if single else results

    def predict_batch(self, texts: List[str], verbose: bool = True) -> List[Dict]:
        """Batch prediction with optional progress display."""
        results = self.predict(texts)
        if verbose:
            for r in results:
                print(f"  {r['emoji']} {r['label']:<12} | {r['text'][:60]}")
        return results


# ─────────────────────────────────────────────
# Demo samples
# ─────────────────────────────────────────────
DEMO_SAMPLES = [
    # Product review domain
    "Produknya sangat bagus dan pengiriman sangat cepat, saya sangat puas!",
    "Kecewa banget, barang tidak sesuai foto dan seller tidak responsif",
    "Lumayan lah, harganya sesuai dengan kualitas yang didapat",
    "Seller ramah, produk original, recommended banget buat semua!",
    "Paket rusak waktu datang, sudah lapor tapi tidak ada respon sama sekali",
    # Financial news domain (thesis preview)
    "IHSG menguat didorong sentimen positif dari laporan keuangan emiten besar",
    "Bursa saham Indonesia melemah akibat kekhawatiran inflasi global yang meningkat",
    "Saham perbankan naik setelah Bank Indonesia pertahankan suku bunga acuan",
    "IDX mencatat penurunan signifikan imbas data ekonomi AS yang mengecewakan",
    "Emiten sektor teknologi mencatatkan pertumbuhan laba bersih di atas ekspektasi",
]


def run_demo():
    """Interactive CLI demo."""
    print("=" * 65)
    print("  🇮🇩 Indonesian Sentiment Analysis — Demo")
    print("  Dataset: PRDECT-ID (Mendeley Data, CC BY 4.0)")
    print("=" * 65)

    try:
        predictor = SentimentPredictor.load()
    except FileNotFoundError:
        print("\n⚠️  No trained model found. Running in MOCK mode.")
        print("   Train the model first: python src/models.py\n")

        # Mock predictor for demo purposes
        class MockPredictor:
            preprocessor = IndonesianPreprocessor(do_stem=True)

            def predict(self, texts):
                import random
                random.seed(42)
                results = []
                for t in (texts if isinstance(texts, list) else [texts]):
                    neg_words = ["kecewa", "rusak", "buruk", "tidak", "jelek", "turun", "melemah", "penurunan"]
                    label_id = 0 if any(w in t.lower() for w in neg_words) else 1
                    label = LABEL_MAP[label_id]
                    results.append({"text": t, "clean": t, "label": label,
                                    "label_id": label_id, "emoji": EMOJI_MAP.get(label, "⚪")})
                return results if isinstance(texts, list) else results[0]

        predictor = MockPredictor()

    print("\n📋 Batch Prediction Demo:\n")
    print(f"  {'Label':<12} | Text")
    print("  " + "-" * 62)

    results = predictor.predict(DEMO_SAMPLES)
    if not isinstance(results, list):
        results = [results]

    for r in results:
        print(f"  {r['emoji']} {r['label']:<10} | {r['text'][:58]}")

    print("\n" + "=" * 65)
    print("💡 To use programmatically:")
    print("   from src.predict import SentimentPredictor")
    print("   pred = SentimentPredictor.load('results/models')")
    print("   result = pred.predict('Saham BBRI naik 3% hari ini')")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
