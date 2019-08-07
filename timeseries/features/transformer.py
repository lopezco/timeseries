import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .compute import compute_tsfeatures


class TsFeaturesTransformer(BaseEstimator, TransformerMixin):
    _VARIABLE_COUNT = 0

    def __init__(self, freq=1, normalize=True, width=None, window=None):
        self._tsparams = dict(freq=freq, normalize=normalize, width=width, window=window)

    def fit(self, X, y=None):
        """Return self nothing else to do here"""
        return self

    def transform(self, X, y=None):
        """Transformer method"""
        return compute_tsfeatures(X, **self._tsparams)
