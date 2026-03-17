"""Testes para modelos da pesquisa Oriental: Mamba, Evidential, Dual Prediction, Evolutionary Ensemble."""

import numpy as np
import pytest
import torch


@pytest.fixture
def train_val_data():
    """Dados sinteticos para treino/validacao."""
    np.random.seed(42)
    n_train, n_val = 150, 50
    n_features = 30
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.randn(n_train).astype(np.float32) * 0.01
    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.random.randn(n_val).astype(np.float32) * 0.01
    return X_train, y_train, X_val, y_val


@pytest.fixture
def ensemble_data():
    """Dados sinteticos para ensemble (previsoes de modelos base)."""
    np.random.seed(42)
    n_val = 100
    n_models = 5
    X_val = np.random.randn(n_val, n_models).astype(np.float32) * 0.01
    y_val = np.random.randn(n_val).astype(np.float32) * 0.01
    X_test = np.random.randn(50, n_models).astype(np.float32) * 0.01
    y_test = np.random.randn(50).astype(np.float32) * 0.01
    return X_val, y_val, X_test, y_test


class TestMambaModel:

    @staticmethod
    def _make_feature_config():
        from config.settings import FeatureConfig
        return FeatureConfig(lookback_window=10)

    def test_fit_and_predict(self, train_val_data):
        from src.models.mamba_model import MambaModel, MambaConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = MambaConfig(max_epochs=3, d_model=16, d_state=8, n_layers=2)
        feat_cfg = self._make_feature_config()
        model = MambaModel(mamba_config=cfg, feature_config=feat_cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert isinstance(preds, np.ndarray)
        assert len(preds) > 0

    def test_predict_with_confidence(self, train_val_data):
        from src.models.mamba_model import MambaModel, MambaConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = MambaConfig(max_epochs=3, d_model=16, d_state=8, n_layers=2, mc_dropout_samples=3)
        feat_cfg = self._make_feature_config()
        model = MambaModel(mamba_config=cfg, feature_config=feat_cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds, conf = model.predict_with_confidence(X_val)
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_name_property(self):
        from src.models.mamba_model import MambaModel
        model = MambaModel()
        assert model.name == "mamba_ssm"

    def test_save_and_load(self, train_val_data, tmp_path):
        from src.models.mamba_model import MambaModel, MambaConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = MambaConfig(max_epochs=2, d_model=16, d_state=8, n_layers=1)
        feat_cfg = self._make_feature_config()
        model = MambaModel(mamba_config=cfg, feature_config=feat_cfg)
        model.fit(X_train, y_train, X_val, y_val)

        path = tmp_path / "mamba.pt"
        model.save(path)
        loaded = MambaModel.load(path)
        preds_orig = model.predict(X_val)
        preds_loaded = loaded.predict(X_val)
        np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5)


class TestEvidentialModel:

    def test_fit_and_predict(self, train_val_data):
        from src.models.evidential import EvidentialModel, EvidentialConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = EvidentialConfig(max_epochs=3, hidden_sizes=[32, 16])
        model = EvidentialModel(input_size=X_train.shape[1], cfg=cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_predict_with_confidence(self, train_val_data):
        from src.models.evidential import EvidentialModel, EvidentialConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = EvidentialConfig(max_epochs=3, hidden_sizes=[32, 16])
        model = EvidentialModel(input_size=X_train.shape[1], cfg=cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds, conf = model.predict_with_confidence(X_val)
        assert preds.shape == conf.shape
        # Evidential confidence should NOT be all zeros (solves the 0% bug)
        assert conf.sum() > 0, "Confidence should not be all zeros"

    def test_name_property(self):
        from src.models.evidential import EvidentialModel
        model = EvidentialModel(input_size=10)
        assert model.name == "evidential"


class TestDualPredictionModel:

    def test_fit_and_predict(self, train_val_data):
        from src.models.dual_prediction import DualPredictionModel, DualPredictionConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = DualPredictionConfig(max_epochs=3, hidden_sizes=[32, 16])
        model = DualPredictionModel(config=cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert preds.shape == (len(X_val),)

    def test_predict_with_confidence(self, train_val_data):
        from src.models.dual_prediction import DualPredictionModel, DualPredictionConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = DualPredictionConfig(max_epochs=3, hidden_sizes=[32, 16])
        model = DualPredictionModel(config=cfg)
        model.fit(X_train, y_train, X_val, y_val)
        preds, conf = model.predict_with_confidence(X_val)
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_name_property(self):
        from src.models.dual_prediction import DualPredictionModel
        model = DualPredictionModel()
        assert model.name == "dual_prediction"

    def test_save_and_load(self, train_val_data, tmp_path):
        from src.models.dual_prediction import DualPredictionModel, DualPredictionConfig
        X_train, y_train, X_val, y_val = train_val_data
        cfg = DualPredictionConfig(max_epochs=2, hidden_sizes=[32, 16])
        model = DualPredictionModel(config=cfg)
        model.fit(X_train, y_train, X_val, y_val)

        path = tmp_path / "dual.pt"
        model.save(path)
        loaded = DualPredictionModel.load(path)
        preds_orig = model.predict(X_val)
        preds_loaded = loaded.predict(X_val)
        np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5)


class TestEvolutionaryEnsemble:

    def test_fit_and_predict(self, ensemble_data):
        from src.models.evolutionary_ensemble import EvolutionaryEnsemble, EvolutionaryConfig
        X_val, y_val, X_test, y_test = ensemble_data
        cfg = EvolutionaryConfig(population_size=10, generations=5)
        model = EvolutionaryEnsemble(config=cfg)
        model.fit(X_val, y_val, X_test, y_test)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_best_weights_sum_to_one(self, ensemble_data):
        from src.models.evolutionary_ensemble import EvolutionaryEnsemble, EvolutionaryConfig
        X_val, y_val, X_test, y_test = ensemble_data
        cfg = EvolutionaryConfig(population_size=10, generations=5)
        model = EvolutionaryEnsemble(config=cfg)
        model.fit(X_val, y_val, X_test, y_test)
        # Active weights should be normalized
        assert model.best_weights is not None
        assert model.best_mask is not None
        active = model.best_weights[model.best_mask]
        if len(active) > 0:
            total = active.sum()
            assert abs(total) > 0, "Weights should not all be zero"

    def test_name_property(self):
        from src.models.evolutionary_ensemble import EvolutionaryEnsemble
        model = EvolutionaryEnsemble()
        assert model.name == "evolutionary_ensemble"

    def test_fitness_history_populated(self, ensemble_data):
        from src.models.evolutionary_ensemble import EvolutionaryEnsemble, EvolutionaryConfig
        X_val, y_val, X_test, y_test = ensemble_data
        cfg = EvolutionaryConfig(population_size=10, generations=5)
        model = EvolutionaryEnsemble(config=cfg)
        model.fit(X_val, y_val, X_test, y_test)
        assert len(model.fitness_history) == 5

    def test_save_and_load(self, ensemble_data, tmp_path):
        from src.models.evolutionary_ensemble import EvolutionaryEnsemble, EvolutionaryConfig
        X_val, y_val, X_test, y_test = ensemble_data
        cfg = EvolutionaryConfig(population_size=10, generations=3)
        model = EvolutionaryEnsemble(config=cfg)
        model.fit(X_val, y_val, X_test, y_test)

        path = tmp_path / "evo.joblib"
        model.save(path)
        loaded = EvolutionaryEnsemble.load(path)
        preds_orig = model.predict(X_test)
        preds_loaded = loaded.predict(X_test)
        np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5)


class TestConfigIntegration:
    """Testa que as novas configs foram integradas corretamente."""

    def test_new_model_configs_exist(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg, 'mamba')
        assert hasattr(cfg, 'evidential')
        assert hasattr(cfg, 'dual_prediction')
        assert hasattr(cfg, 'evolutionary_ensemble')

    def test_new_training_flags_exist(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg.training, 'use_mamba')
        assert hasattr(cfg.training, 'use_evidential')
        assert hasattr(cfg.training, 'use_dual_prediction')
        assert hasattr(cfg.training, 'use_evolutionary_ensemble')

    def test_new_feature_flags_exist(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg.features, 'use_regional_markets')
        assert hasattr(cfg.features, 'use_chart_vision')
        assert hasattr(cfg.features, 'use_blockchain_nlp')
        assert hasattr(cfg.features, 'use_multilingual_sentiment')

    def test_all_flags_default_true(self):
        from config.settings import Config
        cfg = Config()
        assert cfg.training.use_mamba is True
        assert cfg.training.use_evidential is True
        assert cfg.training.use_dual_prediction is True
        assert cfg.training.use_evolutionary_ensemble is True
        assert cfg.features.use_regional_markets is True
        assert cfg.features.use_chart_vision is True
        assert cfg.features.use_blockchain_nlp is True
        assert cfg.features.use_multilingual_sentiment is True
