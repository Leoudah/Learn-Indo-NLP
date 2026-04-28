"""
models.py
=========
ML Model definitions, training, and comparison.

Models:
- Naive Bayes (MultinomialNB)
- Linear SVM (LinearSVC)
- Logistic Regression
- IndoBERT (HuggingFace fine-tuning)

Usage:
    from src.models import ClassifierSuite, IndoBERTClassifier
    suite = ClassifierSuite()
    suite.fit_all(X_train, y_train)
    suite.evaluate_all(X_test, y_test)
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42


# ─────────────────────────────────────────────
# Classical ML Suite
# ─────────────────────────────────────────────

class ClassifierSuite:
    """
    Train and compare multiple classical ML classifiers.

    Supports TF-IDF feature matrices (scipy sparse).
    """

    def __init__(self, task: str = "sentiment"):
        self.task = task
        self.classifiers = {
            "Naive Bayes": MultinomialNB(alpha=0.5),
            "Linear SVM": LinearSVC(C=1.0, max_iter=3000, random_state=SEED),
            "Logistic Regression": LogisticRegression(
                C=1.0, max_iter=1000, random_state=SEED, multi_class="ovr"
            ),
        }
        self.results: Dict[str, dict] = {}
        self._fitted = False

    def fit_all(self, X_train, y_train):
        """Fit all classifiers on training data."""
        print(f"[ClassifierSuite] Training {len(self.classifiers)} models...")
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            print(f"  ✅ {name}")
        self._fitted = True

    def evaluate_all(
        self,
        X_test,
        y_test,
        label_names: Optional[List[str]] = None,
    ) -> dict:
        """Evaluate all fitted classifiers and return results dict."""
        if not self._fitted:
            raise RuntimeError("Call fit_all() first.")

        print(f"\n[ClassifierSuite] Evaluation — Task: {self.task}")
        print(f"{'Model':<30} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
        print("-" * 65)

        for name, clf in self.classifiers.items():
            preds = clf.predict(X_test)
            acc   = accuracy_score(y_test, preds)
            prec  = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec   = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1    = f1_score(y_test, preds, average="weighted")
            self.results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_weighted": round(f1, 4),
                "predictions": preds.tolist(),
                "report": classification_report(
                    y_test, preds,
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0,
                ),
            }
            print(f"  {name:<28} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f}")

        best = max(self.results, key=lambda k: self.results[k]["f1_weighted"])
        print(f"\n  🏆 Best model: {best} (F1={self.results[best]['f1_weighted']:.4f})")
        return self.results

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """Run stratified k-fold cross-validation for all models."""
        print(f"\n[ClassifierSuite] {cv}-Fold Cross Validation")
        cv_results = {}
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

        for name, clf in self.classifiers.items():
            scores = cross_val_score(clf, X, y, cv=kfold, scoring="f1_weighted", n_jobs=-1)
            cv_results[name] = {
                "mean_f1": round(scores.mean(), 4),
                "std_f1": round(scores.std(), 4),
                "scores": scores.tolist(),
            }
            print(f"  {name:<30} F1 = {scores.mean():.4f} ± {scores.std():.4f}")

        return cv_results

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
    ):
        """Bar chart comparing all models across metrics."""
        if not self.results:
            raise RuntimeError("Call evaluate_all() first.")

        metrics = ["accuracy", "precision", "recall", "f1_weighted"]
        labels  = list(self.results.keys())
        colors  = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        x = np.arange(len(labels))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [self.results[m][metric] for m in labels]
            bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(),
                          color=color, alpha=0.85)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(f"Model Comparison — {self.task.title()} Classification", fontweight="bold")
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def plot_confusion_matrix(
        self,
        y_test,
        model_name: Optional[str] = None,
        label_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """Plot confusion matrix for a specific model (default: best)."""
        if not model_name:
            model_name = max(self.results, key=lambda k: self.results[k]["f1_weighted"])

        preds = np.array(self.results[model_name]["predictions"])
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def save_results(self, path: str = "results/metrics.json"):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {
            k: {m: v for m, v in r.items() if m != "predictions"}
            for k, r in self.results.items()
        }
        with open(path, "w") as f:
            json.dump({self.task: serializable}, f, indent=2)
        print(f"[ClassifierSuite] Results saved to {path}")

    def save_models(self, directory: str = "results/models"):
        """Pickle all fitted classifiers."""
        os.makedirs(directory, exist_ok=True)
        for name, clf in self.classifiers.items():
            fname = name.lower().replace(" ", "_") + ".pkl"
            with open(os.path.join(directory, fname), "wb") as f:
                pickle.dump(clf, f)
        print(f"[ClassifierSuite] Models saved to {directory}/")


# ─────────────────────────────────────────────
# IndoBERT Fine-tuning Wrapper
# ─────────────────────────────────────────────

class IndoBERTClassifier:
    """
    Wrapper for fine-tuning IndoBERT (indobenchmark/indobert-base-p1)
    using HuggingFace Trainer API.

    Requires: transformers, datasets, torch
    Recommended: GPU (T4 or better)
    """

    MODEL_NAME = "indobenchmark/indobert-base-p1"

    def __init__(
        self,
        num_labels: int = 2,
        max_len: int = 128,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        output_dir: str = "./indobert_output",
    ):
        self.num_labels = num_labels
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            print(f"[IndoBERT] Loading tokenizer: {self.MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        return self.tokenizer

    def _tokenize(self, texts: List[str]):
        tokenizer = self._load_tokenizer()
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def prepare_datasets(self, X_train, y_train, X_val, y_val):
        """Tokenize and prepare HuggingFace Datasets."""
        from datasets import Dataset

        tokenizer = self._load_tokenizer()

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )

        train_ds = Dataset.from_dict({"text": list(X_train), "labels": list(y_train)})
        val_ds   = Dataset.from_dict({"text": list(X_val),   "labels": list(y_val)})

        train_ds = train_ds.map(tokenize_fn, batched=True).remove_columns(["text"])
        val_ds   = val_ds.map(tokenize_fn, batched=True).remove_columns(["text"])

        train_ds.set_format("torch")
        val_ds.set_format("torch")
        return train_ds, val_ds

    def train(self, X_train, y_train, X_val, y_val):
        """Fine-tune IndoBERT."""
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            TrainingArguments, Trainer, EarlyStoppingCallback
        )
        from sklearn.metrics import accuracy_score, f1_score as _f1

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": _f1(labels, preds, average="weighted"),
            }

        print(f"[IndoBERT] GPU: {torch.cuda.is_available()}")
        train_ds, val_ds = self.prepare_datasets(X_train, y_train, X_val, y_val)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=self.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            seed=SEED,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("[IndoBERT] Starting fine-tuning...")
        trainer.train()
        self.trainer = trainer
        return trainer

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate fine-tuned model on test set."""
        from datasets import Dataset
        tokenizer = self._load_tokenizer()

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )

        test_ds = Dataset.from_dict({"text": list(X_test), "labels": list(y_test)})
        test_ds = test_ds.map(tokenize_fn, batched=True).remove_columns(["text"])
        test_ds.set_format("torch")
        results = self.trainer.evaluate(test_ds)
        print(f"[IndoBERT] Test Results: {results}")
        return results

    def save(self, path: str = "./indobert_final"):
        """Save model and tokenizer."""
        if self.model:
            self.trainer.save_model(path)
            self._load_tokenizer().save_pretrained(path)
            print(f"[IndoBERT] Saved to {path}")

    def predict(self, texts: List[str]) -> List[int]:
        """Run inference on new texts."""
        import torch
        self.model.eval()
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).tolist()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("models.py — Import as module or run from notebook.")
    print("  from src.models import ClassifierSuite, IndoBERTClassifier")
