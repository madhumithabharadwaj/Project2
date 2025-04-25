# tests/test_boosting_tree.py

import numpy as np
from model.BoostingTree import BoostingTree

def test_boosting_tree_accuracy():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    true_weights = np.array([1.2, -0.8, 0.5, 0.3])
    logits = X @ true_weights
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    model = BoostingTree(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    acc = (preds == y).mean()

    assert acc >= 0.8, f"Accuracy too low: {acc}"
