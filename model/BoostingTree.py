import numpy as np

class BoostingTree:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1, early_stopping_rounds=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.models = []         # list of decision stumps
        self.model_weights = []  # contribution of each tree

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss_grad(self, y_true, y_pred_proba):
        return y_pred_proba - y_true

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]

        F = np.zeros(n_samples)

        #  For early stopping
        best_acc = 0
        rounds_without_improve = 0

        for m in range(self.n_estimators):
            prob = self._sigmoid(F)
            gradient = self._log_loss_grad(y, prob)

            tree = self._fit_stump(X, -gradient)
            pred = self._predict_stump(tree, X)

            F += self.learning_rate * pred

            self.models.append(tree)
            self.model_weights.append(self.learning_rate)

            #  Early Stopping Logic
            y_pred = (self._sigmoid(F) > 0.5).astype(int)
            acc = (y_pred == y).mean()

            if self.early_stopping_rounds is not None:
                if acc > best_acc:
                    best_acc = acc
                    rounds_without_improve = 0
                else:
                    rounds_without_improve += 1

                if rounds_without_improve >= self.early_stopping_rounds:
                    print(f" Early stopping at round {m+1} (best accuracy: {best_acc:.4f})")
                    break

    def _fit_stump(self, X, residuals):
        best_feature, best_thresh, best_score = None, None, float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t

                if len(residuals[left_mask]) == 0 or len(residuals[right_mask]) == 0:
                    continue

                left_mean = residuals[left_mask].mean()
                right_mean = residuals[right_mask].mean()
                pred = np.where(left_mask, left_mean, right_mean)
                mse = np.mean((residuals - pred) ** 2)

                if mse < best_score:
                    best_score = mse
                    best_feature = feature
                    best_thresh = t
                    best_left_val = left_mean
                    best_right_val = right_mean

        return {
            'feature': best_feature,
            'threshold': best_thresh,
            'left_value': best_left_val,
            'right_value': best_right_val
        }

    def _predict_stump(self, stump, X):
        f_idx = stump['feature']
        thresh = stump['threshold']
        return np.where(X[:, f_idx] <= thresh, stump['left_value'], stump['right_value'])

    def predict(self, X):
        X = np.array(X)
        F = np.zeros(X.shape[0])

        for tree, alpha in zip(self.models, self.model_weights):
            F += alpha * self._predict_stump(tree, X)

        return (self._sigmoid(F) > 0.5).astype(int)
