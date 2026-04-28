# src/__init__.py
from .preprocessing import IndonesianPreprocessor
from .features import TFIDFExtractor
from .models import ClassifierSuite, IndoBERTClassifier
from .evaluate import Evaluator
from .predict import SentimentPredictor

__version__ = "1.0.0"
__author__ = "Informatika Universitas Udayana"
