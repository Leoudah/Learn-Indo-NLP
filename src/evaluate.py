"""
evaluate.py
===========
Evaluation utilities: metrics, plots, error analysis.

Usage:
    from src.evaluate import Evaluator
    ev = Evaluator(y_true, y_pred, label_names=['Positive', 'Negative'])
    ev.full_report()
    ev.plot_confusion_matrix()
    ev.error_analysis(texts)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud
from typing import List, Optional, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)


class Evaluator:
    """
    Comprehensive evaluation for text classification tasks.

    Args:
        y_true: Ground truth labels (array-like of ints).
        y_pred: Predicted labels (array-like of ints).
        label_names: Human-readable class names.
        task: Task name for display.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        label_names: Optional[List[str]] = None,
        task: str = "Classification",
    ):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.label_names = label_names
        self.task = task

    # ── Core Metrics ─────────────────────────

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    @property
    def f1_weighted(self) -> float:
        return f1_score(self.y_true, self.y_pred, average="weighted")

    @property
    def f1_macro(self) -> float:
        return f1_score(self.y_true, self.y_pred, average="macro")

    @property
    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

    @property
    def recall(self) -> float:
        return recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

    def summary(self) -> Dict[str, float]:
        return {
            "accuracy":     round(self.accuracy, 4),
            "precision":    round(self.precision, 4),
            "recall":       round(self.recall, 4),
            "f1_weighted":  round(self.f1_weighted, 4),
            "f1_macro":     round(self.f1_macro, 4),
        }

    def full_report(self, print_output: bool = True) -> str:
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.label_names,
            zero_division=0,
        )
        if print_output:
            print(f"\n{'='*50}")
            print(f"  Evaluation Report — {self.task}")
            print(f"{'='*50}")
            s = self.summary()
            print(f"  Accuracy  : {s['accuracy']:.4f}")
            print(f"  Precision : {s['precision']:.4f}")
            print(f"  Recall    : {s['recall']:.4f}")
            print(f"  F1 (w)    : {s['f1_weighted']:.4f}")
            print(f"  F1 (macro): {s['f1_macro']:.4f}")
            print(f"\n{report}")
        return report

    # ── Plots ─────────────────────────────────

    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        save_path: Optional[str] = None,
        figsize: tuple = (7, 6),
    ):
        cm = confusion_matrix(self.y_true, self.y_pred)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            ax=ax, linewidths=0.5,
        )
        title = f"Confusion Matrix — {self.task}"
        if normalize:
            title += " (Normalized)"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()
        return fig

    def plot_metrics_radar(
        self,
        save_path: Optional[str] = None,
    ):
        """Radar chart of per-class F1, Precision, Recall."""
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        if self.label_names is None:
            return

        cats = [l for l in self.label_names if l in report]
        metrics = ["precision", "recall", "f1-score"]
        values  = [[report[c][m] for m in metrics] + [report[c]["precision"]] for c in cats]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

        for i, (cat, vals) in enumerate(zip(cats, values)):
            ax.plot(angles, vals, color=colors[i % len(colors)], linewidth=2, label=cat)
            ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f"Per-Class Metrics — {self.task}", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    # ── Error Analysis ────────────────────────

    def error_analysis(
        self,
        texts: List[str],
        label_encoder=None,
        top_n_words: int = 50,
        save_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyse misclassified samples.
        Returns a DataFrame of errors.
        """
        true_names = self.label_names[self.y_true] if self.label_names else self.y_true
        pred_names = self.label_names[self.y_pred] if self.label_names else self.y_pred

        if self.label_names is not None:
            label_arr = np.array(self.label_names)
            true_names = label_arr[self.y_true]
            pred_names = label_arr[self.y_pred]
        else:
            true_names = self.y_true.astype(str)
            pred_names = self.y_pred.astype(str)

        error_df = pd.DataFrame({
            "text":       texts,
            "true_label": true_names,
            "pred_label": pred_names,
            "correct":    self.y_true == self.y_pred,
        })

        errors = error_df[~error_df["correct"]].copy()
        total  = len(error_df)
        n_err  = len(errors)

        print(f"\n[ErrorAnalysis] Misclassified: {n_err}/{total} ({n_err/total*100:.1f}%)")
        print("\nError breakdown:")
        print(errors.groupby(["true_label", "pred_label"]).size()
                    .reset_index(name="count").to_string(index=False))

        # Word clouds per error type
        error_types = errors.groupby(["true_label", "pred_label"])
        n_types = len(error_types)
        if n_types > 0:
            fig, axes = plt.subplots(1, n_types, figsize=(8 * n_types, 4))
            if n_types == 1:
                axes = [axes]

            for ax, ((true_l, pred_l), group) in zip(axes, error_types):
                text = " ".join(group["text"].values)
                if text.strip():
                    wc = WordCloud(
                        width=600, height=300, background_color="white",
                        colormap="Reds", max_words=top_n_words, collocations=False
                    ).generate(text)
                    ax.imshow(wc, interpolation="bilinear")
                ax.set_title(f"True: {true_l}\nPred: {pred_l}\n(n={len(group)})", fontweight="bold")
                ax.axis("off")

            fig.suptitle("Error Analysis — Word Clouds", fontweight="bold")
            plt.tight_layout()

            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, "error_wordclouds.png"), dpi=150, bbox_inches="tight")
            plt.show()

        return errors

    def save_summary(self, path: str = "results/metrics.json", model_name: str = "model"):
        """Append summary to JSON file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        existing = {}
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)

        existing[model_name] = self.summary()
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[Evaluator] Metrics saved to {path}")


if __name__ == "__main__":
    # Quick demo
    y_true = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]

    ev = Evaluator(y_true, y_pred, label_names=["Negative", "Positive"], task="Demo Sentiment")
    ev.full_report()
