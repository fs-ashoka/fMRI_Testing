import numpy as np

from fmri_testing.ablation.feature_ablation import ablate_feature_drops
from fmri_testing.encoding.ridge import grouped_predictions, make_feature_sets


def test_ridge_encoding_and_ablation_shapes():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(30, 12)).astype("float32")
    sae = rng.normal(size=(30, 5)).astype("float32")
    y = (sae @ rng.normal(size=(5, 4)) + 0.1 * rng.normal(size=(30, 4))).astype("float32")
    sets = make_feature_sets(raw, sae, seed=0)
    assert set(["raw", "pca", "sae", "random_projection", "shuffled_sae"]).issubset(sets)
    pred, models = grouped_predictions(sets["sae"], y, np.arange(30), [0.1, 1.0], n_splits=3)
    assert pred.shape == y.shape
    assert len(models) == 3
    drops = ablate_feature_drops(sae, y, max_features=3)
    assert drops.shape[0] == 3
    assert {"feature_id", "mean_drop", "activation_frequency", "feature_variance"}.issubset(drops.columns)
