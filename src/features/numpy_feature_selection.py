#!/usr/bin/env python3
"""Numpy-only feature selection module.

No sklearn, xgboost, torch, or compiled C extensions allowed.
Implements variance filtering, correlation filtering, and permutation importance.

Usage:
    from src.features.numpy_feature_selection import NumpyFeatureSelector

    selector = NumpyFeatureSelector(
        variance_threshold=0.01,
        correlation_threshold=0.95,
        target_n_features=40,
    )
    X_sel, names_sel = selector.fit_transform(X, y, feature_names, model=my_model)
    report = selector.get_report()
    selector.save("feature_selection.json")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VarianceFilter:
    """Remove features whose variance falls below a threshold.

    Features with near-zero variance carry almost no information and can
    introduce numerical instability during training.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.variances_: Optional[np.ndarray] = None
        self.mask_: Optional[np.ndarray] = None
        self.removed_features_: List[str] = []

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "VarianceFilter":
        self.variances_ = np.var(X, axis=0)
        self.mask_ = self.variances_ >= self.threshold
        self.removed_features_ = [
            feature_names[i]
            for i in range(len(feature_names))
            if not self.mask_[i]
        ]
        n_removed = len(self.removed_features_)
        logger.info(
            "VarianceFilter: removed %d / %d features (threshold=%.4f)",
            n_removed, len(feature_names), self.threshold,
        )
        if n_removed > 0 and n_removed <= 20:
            logger.debug("  Removed: %s", self.removed_features_)
        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        if self.mask_ is None:
            raise RuntimeError("VarianceFilter has not been fit yet.")
        X_out = X[:, self.mask_]
        names_out = [feature_names[i] for i in range(len(feature_names)) if self.mask_[i]]
        return X_out, names_out


class CorrelationFilter:
    """Remove redundant features that are highly correlated.

    When two features have absolute Pearson correlation above the threshold,
    the one with lower variance is dropped.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.drop_indices_: Optional[set] = None
        self.removed_features_: List[str] = []
        self.correlation_pairs_: List[Tuple[str, str, float]] = []

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "CorrelationFilter":
        n_features = X.shape[1]
        variances = np.var(X, axis=0)

        # Compute correlation matrix in chunks to manage memory
        # Standardize columns first for faster correlation computation
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds < 1e-12] = 1.0
        X_std = (X - means) / stds
        n_samples = X_std.shape[0]

        drop_indices = set()
        correlation_pairs = []

        # Process in blocks to avoid creating a huge n_features x n_features matrix
        # when n_features is large (e.g., 137)
        block_size = 50
        for block_start in range(0, n_features, block_size):
            block_end = min(block_start + block_size, n_features)
            # Correlate this block against all features from block_start onward
            block = X_std[:, block_start:block_end]
            rest = X_std[:, block_start:]
            corr_block = (block.T @ rest) / n_samples

            for i_local in range(block_end - block_start):
                i = block_start + i_local
                if i in drop_indices:
                    continue
                # Only check j > i (upper triangle)
                j_start = i_local + 1  # offset within corr_block columns
                for j_local in range(j_start, corr_block.shape[1]):
                    j = block_start + j_local
                    if j in drop_indices:
                        continue
                    r = abs(corr_block[i_local, j_local])
                    if r >= self.threshold:
                        # Drop the feature with lower variance
                        if variances[i] >= variances[j]:
                            drop_idx = j
                        else:
                            drop_idx = i
                        drop_indices.add(drop_idx)
                        correlation_pairs.append((
                            feature_names[i],
                            feature_names[j],
                            float(r),
                        ))
                        if drop_idx == i:
                            break  # i is dropped, no need to check more

        self.drop_indices_ = drop_indices
        self.removed_features_ = [feature_names[i] for i in sorted(drop_indices)]
        self.correlation_pairs_ = correlation_pairs

        logger.info(
            "CorrelationFilter: removed %d / %d features (threshold=%.2f)",
            len(drop_indices), n_features, self.threshold,
        )
        if len(correlation_pairs) <= 10:
            for f1, f2, r in correlation_pairs:
                logger.debug("  Correlated pair: %s <-> %s (r=%.4f)", f1, f2, r)
        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        if self.drop_indices_ is None:
            raise RuntimeError("CorrelationFilter has not been fit yet.")
        keep_mask = np.array(
            [i not in self.drop_indices_ for i in range(len(feature_names))]
        )
        X_out = X[:, keep_mask]
        names_out = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]
        return X_out, names_out


class PermutationImportance:
    """Measure feature importance by shuffling each feature and observing
    the drop in directional accuracy.

    Uses a provided model (must have a `.predict(X)` method) or falls back
    to a lightweight internal GradientBoostedStumps trained on the data.
    """

    def __init__(
        self,
        n_repeats: int = 5,
        random_state: int = 42,
        top_k: Optional[int] = None,
        min_importance: float = 0.0,
    ):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.top_k = top_k
        self.min_importance = min_importance
        self.importances_: Optional[np.ndarray] = None
        self.feature_ranking_: List[Tuple[str, float]] = []
        self.selected_indices_: Optional[np.ndarray] = None
        self.removed_features_: List[str] = []

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fraction of samples where predicted and true signs agree."""
        return float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    def _build_default_model(self, X: np.ndarray, y: np.ndarray):
        """Train a lightweight GradientBoostedStumps as a surrogate model."""
        # Import locally to avoid circular deps; reuse the class from train_numpy
        # or reimplement a minimal version here.
        try:
            import sys
            from pathlib import Path as _Path

            repo_root = _Path(__file__).resolve().parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from scripts.train_numpy import GradientBoostedStumps, NumpyScaler
        except ImportError:
            logger.warning(
                "Could not import GradientBoostedStumps from scripts.train_numpy; "
                "using inline minimal implementation."
            )
            return self._build_inline_model(X, y)

        scaler = NumpyScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostedStumps(n_estimators=100, learning_rate=0.05)
        model.fit(X_scaled, y)

        class _ScaledModel:
            """Wraps model + scaler so predict() works on raw X."""

            def __init__(self, inner_model, inner_scaler):
                self._model = inner_model
                self._scaler = inner_scaler

            def predict(self, X_raw):
                return self._model.predict(self._scaler.transform(X_raw))

        return _ScaledModel(model, scaler)

    @staticmethod
    def _build_inline_model(X: np.ndarray, y: np.ndarray):
        """Minimal inline GradientBoostedStumps (no external imports)."""

        class _MinimalGBS:
            def __init__(self):
                self.stumps = []
                self.init_pred = 0.0
                self.lr = 0.05
                self.mean_ = None
                self.std_ = None

            def fit(self, X_raw, y_raw, n_estimators=100):
                self.mean_ = np.mean(X_raw, axis=0)
                self.std_ = np.std(X_raw, axis=0)
                self.std_[self.std_ < 1e-10] = 1.0
                X = (X_raw - self.mean_) / self.std_

                n, d = X.shape
                self.init_pred = float(np.mean(y_raw))
                preds = np.full(n, self.init_pred)
                n_feat = max(1, int(d * 0.8))
                rng = np.random.RandomState(42)

                for _ in range(n_estimators):
                    residuals = y_raw - preds
                    subset = rng.choice(d, size=n_feat, replace=False)
                    best_gain = -np.inf
                    best = (0, 0.0, 0.0, 0.0)
                    for j in subset:
                        col = X[:, j]
                        for pct in [20, 40, 50, 60, 80]:
                            t = np.percentile(col, pct)
                            left = col <= t
                            right = ~left
                            nl, nr = left.sum(), right.sum()
                            if nl < 5 or nr < 5:
                                continue
                            lm = residuals[left].mean()
                            rm = residuals[right].mean()
                            gain = (nl * lm**2 + nr * rm**2) / n
                            if gain > best_gain:
                                best_gain = gain
                                best = (j, t, lm, rm)
                    feat, thresh, lv, rv = best
                    self.stumps.append(best)
                    mask = X[:, feat] <= thresh
                    preds += self.lr * np.where(mask, lv, rv)

            def predict(self, X_raw):
                X = (X_raw - self.mean_) / self.std_
                preds = np.full(X.shape[0], self.init_pred)
                for feat, thresh, lv, rv in self.stumps:
                    mask = X[:, feat] <= thresh
                    preds += self.lr * np.where(mask, lv, rv)
                return preds

        model = _MinimalGBS()
        model.fit(X, y)
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model=None,
        top_k: Optional[int] = None,
    ) -> "PermutationImportance":
        if top_k is not None:
            self.top_k = top_k

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Use a validation split to avoid overfitting the importance estimate
        split_idx = int(n_samples * 0.7)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        if X_val.shape[0] < 20:
            # Fall back to using the full data if validation set is too small
            logger.warning(
                "PermutationImportance: validation set too small (%d), "
                "using full data.", X_val.shape[0],
            )
            X_val, y_val = X, y
            X_train, y_train = X, y

        # Build or use model
        if model is None:
            logger.info("PermutationImportance: training surrogate model...")
            model = self._build_default_model(X_train, y_train)
        else:
            logger.info("PermutationImportance: using provided model.")

        # Baseline directional accuracy
        baseline_preds = model.predict(X_val)
        baseline_acc = self._directional_accuracy(y_val, baseline_preds)
        logger.info("  Baseline directional accuracy: %.4f", baseline_acc)

        # Permutation importance for each feature
        importances = np.zeros(n_features)
        for j in range(n_features):
            drops = []
            for _ in range(self.n_repeats):
                X_perm = X_val.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_preds = model.predict(X_perm)
                perm_acc = self._directional_accuracy(y_val, perm_preds)
                drops.append(baseline_acc - perm_acc)
            importances[j] = float(np.mean(drops))

        self.importances_ = importances

        # Build ranking
        order = np.argsort(-importances)
        self.feature_ranking_ = [
            (feature_names[i], float(importances[i])) for i in order
        ]

        # Determine which features to keep
        if self.top_k is not None and self.top_k > 0:
            k = min(self.top_k, n_features)
            selected = set(order[:k].tolist())
        else:
            # Keep features with importance >= min_importance
            selected = set(
                i for i in range(n_features) if importances[i] >= self.min_importance
            )
            if len(selected) == 0:
                # Keep at least the top 10
                selected = set(order[:10].tolist())

        self.selected_indices_ = np.array(sorted(selected))
        self.removed_features_ = [
            feature_names[i]
            for i in range(n_features)
            if i not in selected
        ]

        logger.info(
            "PermutationImportance: selected %d / %d features (top_k=%s)",
            len(selected), n_features, self.top_k,
        )
        # Log top 10
        for rank, (name, imp) in enumerate(self.feature_ranking_[:10]):
            logger.info("  #%d %s: %.6f", rank + 1, name, imp)

        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        if self.selected_indices_ is None:
            raise RuntimeError("PermutationImportance has not been fit yet.")
        X_out = X[:, self.selected_indices_]
        names_out = [feature_names[i] for i in self.selected_indices_]
        return X_out, names_out


class NumpyFeatureSelector:
    """Main feature selection pipeline chaining variance, correlation,
    and permutation importance filters.

    Typical usage reduces ~137 features down to 30-50.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance to keep a feature (default 0.01).
    correlation_threshold : float
        Maximum absolute correlation between feature pairs (default 0.95).
    target_n_features : int or None
        Target number of features after all filters. If None, relies on
        permutation importance min_importance threshold instead.
    n_perm_repeats : int
        Number of permutation shuffles per feature (default 5).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        target_n_features: Optional[int] = 40,
        n_perm_repeats: int = 5,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_n_features = target_n_features
        self.n_perm_repeats = n_perm_repeats
        self.random_state = random_state

        self._variance_filter = VarianceFilter(threshold=variance_threshold)
        self._correlation_filter = CorrelationFilter(threshold=correlation_threshold)
        self._perm_importance = PermutationImportance(
            n_repeats=n_perm_repeats,
            random_state=random_state,
            top_k=target_n_features,
        )

        self._is_fitted = False
        self._original_n_features = 0
        self._original_feature_names: List[str] = []
        self._selected_feature_names: List[str] = []
        self._selected_indices: List[int] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model=None,
    ) -> "NumpyFeatureSelector":
        """Fit all three filters sequentially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Target values (log returns).
        feature_names : list of str
            Names corresponding to columns in X.
        model : object with .predict(X), optional
            Model for permutation importance. If None, a surrogate
            GradientBoostedStumps is trained internally.
        """
        assert X.shape[1] == len(feature_names), (
            f"X has {X.shape[1]} columns but got {len(feature_names)} feature names"
        )
        self._original_n_features = len(feature_names)
        self._original_feature_names = list(feature_names)

        logger.info(
            "NumpyFeatureSelector.fit: starting with %d features, %d samples",
            len(feature_names), X.shape[0],
        )

        # Replace NaN/Inf with 0 for safety
        X_clean = X.copy()
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 1: Variance filter
        logger.info("Step 1/3: Variance filtering...")
        self._variance_filter.fit(X_clean, feature_names)
        X_step1, names_step1 = self._variance_filter.transform(X_clean, feature_names)
        logger.info("  After variance filter: %d features", len(names_step1))

        # Step 2: Correlation filter
        logger.info("Step 2/3: Correlation filtering...")
        self._correlation_filter.fit(X_step1, names_step1)
        X_step2, names_step2 = self._correlation_filter.transform(X_step1, names_step1)
        logger.info("  After correlation filter: %d features", len(names_step2))

        # Step 3: Permutation importance (only if we still have more than target)
        if (
            self.target_n_features is not None
            and len(names_step2) > self.target_n_features
        ):
            logger.info("Step 3/3: Permutation importance...")
            self._perm_importance.fit(
                X_step2, y, names_step2, model=model, top_k=self.target_n_features
            )
            _, names_step3 = self._perm_importance.transform(X_step2, names_step2)
        elif self.target_n_features is None:
            logger.info("Step 3/3: Permutation importance (no target_k)...")
            self._perm_importance.fit(X_step2, y, names_step2, model=model)
            _, names_step3 = self._perm_importance.transform(X_step2, names_step2)
        else:
            logger.info(
                "Step 3/3: Skipping permutation importance "
                "(%d features already <= target %d)",
                len(names_step2), self.target_n_features,
            )
            names_step3 = names_step2

        # Build final mapping back to original indices
        self._selected_feature_names = names_step3
        name_to_orig_idx = {
            name: idx for idx, name in enumerate(self._original_feature_names)
        }
        self._selected_indices = [name_to_orig_idx[n] for n in names_step3]

        self._is_fitted = True
        logger.info(
            "NumpyFeatureSelector.fit complete: %d -> %d features",
            self._original_n_features, len(self._selected_feature_names),
        )
        return self

    def transform(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply the fitted selection to new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        feature_names : list of str

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected)
        selected_names : list of str
        """
        if not self._is_fitted:
            raise RuntimeError("NumpyFeatureSelector has not been fit yet.")

        # Map selected names to indices in the provided feature_names
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        indices = []
        valid_names = []
        for name in self._selected_feature_names:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
                valid_names.append(name)
            else:
                logger.warning(
                    "Feature '%s' was selected during fit but is missing in transform input.",
                    name,
                )

        X_out = X[:, indices]
        return X_out, valid_names

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model=None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Convenience: fit then transform."""
        self.fit(X, y, feature_names, model=model)
        return self.transform(X, feature_names)

    def get_report(self) -> Dict:
        """Return a dictionary summarising what was removed and why."""
        if not self._is_fitted:
            raise RuntimeError("NumpyFeatureSelector has not been fit yet.")

        report = {
            "original_n_features": self._original_n_features,
            "selected_n_features": len(self._selected_feature_names),
            "reduction_pct": round(
                100.0
                * (1 - len(self._selected_feature_names) / max(self._original_n_features, 1)),
                2,
            ),
            "selected_features": list(self._selected_feature_names),
            "variance_filter": {
                "threshold": self.variance_threshold,
                "n_removed": len(self._variance_filter.removed_features_),
                "removed": self._variance_filter.removed_features_,
            },
            "correlation_filter": {
                "threshold": self.correlation_threshold,
                "n_removed": len(self._correlation_filter.removed_features_),
                "removed": self._correlation_filter.removed_features_,
                "correlated_pairs": [
                    {"feature_a": a, "feature_b": b, "correlation": round(r, 4)}
                    for a, b, r in self._correlation_filter.correlation_pairs_
                ],
            },
        }

        if self._perm_importance.importances_ is not None:
            report["permutation_importance"] = {
                "n_repeats": self.n_perm_repeats,
                "n_removed": len(self._perm_importance.removed_features_),
                "removed": self._perm_importance.removed_features_,
                "ranking": [
                    {"feature": name, "importance": round(imp, 6)}
                    for name, imp in self._perm_importance.feature_ranking_[:50]
                ],
            }

        return report

    def save(self, path: str) -> None:
        """Serialize selection state to JSON."""
        if not self._is_fitted:
            raise RuntimeError("NumpyFeatureSelector has not been fit yet.")

        state = {
            "version": 1,
            "params": {
                "variance_threshold": self.variance_threshold,
                "correlation_threshold": self.correlation_threshold,
                "target_n_features": self.target_n_features,
                "n_perm_repeats": self.n_perm_repeats,
                "random_state": self.random_state,
            },
            "original_feature_names": self._original_feature_names,
            "selected_feature_names": self._selected_feature_names,
            "selected_indices": self._selected_indices,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info("NumpyFeatureSelector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "NumpyFeatureSelector":
        """Load a previously saved selection state from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        params = state.get("params", {})
        selector = cls(
            variance_threshold=params.get("variance_threshold", 0.01),
            correlation_threshold=params.get("correlation_threshold", 0.95),
            target_n_features=params.get("target_n_features", 40),
            n_perm_repeats=params.get("n_perm_repeats", 5),
            random_state=params.get("random_state", 42),
        )
        selector._original_feature_names = state["original_feature_names"]
        selector._selected_feature_names = state["selected_feature_names"]
        selector._selected_indices = state["selected_indices"]
        selector._original_n_features = len(selector._original_feature_names)
        selector._is_fitted = True

        logger.info(
            "NumpyFeatureSelector loaded from %s (%d -> %d features)",
            path,
            selector._original_n_features,
            len(selector._selected_feature_names),
        )
        return selector
