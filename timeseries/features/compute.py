import numpy as np
import pandas as pd

from timeseries.utils import NameGenerator
from .tsfeatures import *

_name_generator = NameGenerator()


def compute_tsfeatures(x, **kwargs):
    """See `ts_features_series` doc"""
    global name_generator

    if isinstance(x, pd.Series):
        features_df = compute_tsfeatures_for_series(x, **kwargs)
    elif isinstance(x, pd.DataFrame):
        _buffer = []
        for c in x.columns:
            _buffer.append(compute_tsfeatures_for_series(x[c], **kwargs))
        features_df = pd.concat(_buffer, axis=0)
    elif issubclass(x.__class__, pd.core.groupby._GroupBy):
        _buffer = []
        for i in x.groups:
            _buffer.append(compute_tsfeatures(x.get_group(i), **kwargs))
        features_df = pd.concat(_buffer, axis=0)
    else:
        raise TypeError('Unhandled input type')
    _name_generator.reset()
    return features_df


def compute_tsfeatures_for_series(x, freq=1, normalize=True, width=None, window=None):
    """
    :param x: a uni-variate time series
    :param freq: number of points to be considered as part of a single period for trend_seasonality_spike_strength
    :param normalize: TRUE: scale data to be normally distributed
    :param width: a window size for variance change and level shift, lumpiness
    :param window: a window size for KLscore
    :return:
    """
    global _name_generator
    name = x.name

    if width is None:
        width = freq if freq > 1 else 10

    if window is None:
        window = width

    if (width <= 1) | (window <= 1):
        raise ValueError("Window widths should be greater than 1.")

    # Remove columns containing all NAs
    if x.isnull().all():
        raise ValueError("All values are null")

    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

    trimx = trim(x)

    features = dict()
    features['lumpiness'] = lumpiness(x, width=width)
    if ENTROPY_PACKAGE_AVAILABLE:
        features['entropy'] = entropy(x, freq=freq, normalize=False)
    features['ACF1'] = first_order_autocorrelation(x)
    features['lshift'] = rolling_level_shift(trimx, width=width)
    features['vchange'] = rolling_variance_change(trimx, width=width)
    features['cpoints'] = n_crossing_points(x)
    features['fspots'] = flat_spots(x)
    #  features['mean'] = np.mean(x)
    #  features['var'] = np.var(x)

    varts = trend_seasonality_spike_strength(x, freq=freq)
    features['trend'] = varts['trend']
    features['linearity'] = varts['linearity']
    features['curvature'] = varts['curvature']
    features['spikiness'] = varts['spike']

    if freq > 1:
        features['season'] = varts['season']
        features['peak'] = varts['peak']
        features['trough'] = varts['trough']

    threshold = norm.pdf(38)

    try:
        kl = kullback_leibler_score(x, window=window, threshold=threshold)
        features['KLscore'] = kl['score']
        features['change_idx'] = kl['change_idx']
    except Exception:
        features['KLscore'] = np.nan
        features['change_idx'] = np.nan

    features['boxcox'] = boxcox_optimal_lambda(x)

    # Build output
    features_df = pd.Series(features).to_frame().transpose()
    features_df.index = [x.index.min()] if isinstance(x, pd.Series) else [0]
    features_df['variable'] = name if name is not None else _name_generator.get()
    return features_df
