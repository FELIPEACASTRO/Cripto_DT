"""Explicabilidade de modelos com SHAP (baseado em estudos do KAIST e KIT)."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tentativa de importar shap; degrada graciosamente se nao disponivel
try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap nao disponivel — usando fallback com feature_importances_")

# Tentativa de importar matplotlib para graficos
try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib nao disponivel — graficos desabilitados")


class ModelExplainer:
    """Gera explicacoes de modelos usando SHAP values.

    Suporta modelos de arvore (XGBoost, LightGBM) via TreeExplainer
    e fallback com KernelExplainer para outros modelos.
    Se shap nao estiver instalado, utiliza feature_importances_ do modelo.
    """

    # Tipos de modelo que suportam TreeExplainer
    _TREE_MODEL_TYPES = (
        "XGBRegressor", "XGBClassifier",
        "LGBMRegressor", "LGBMClassifier",
        "GradientBoostingRegressor", "GradientBoostingClassifier",
        "RandomForestRegressor", "RandomForestClassifier",
    )

    def _is_tree_model(self, model: Any) -> bool:
        """Verifica se o modelo e baseado em arvore."""
        model_type = type(model).__name__
        return model_type in self._TREE_MODEL_TYPES

    def explain(
        self,
        model: Any,
        X: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Gera explicacoes SHAP para o modelo.

        Args:
            model: Modelo treinado (sklearn-compatible)
            X: Dados de entrada para explicar
            feature_names: Nomes das features

        Returns:
            Dicionario com shap_values, feature_importance e top_features
        """
        X_array = np.asarray(X)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

        # Caminho com SHAP disponivel
        if _SHAP_AVAILABLE:
            return self._explain_with_shap(model, X_array, feature_names)

        # Fallback sem SHAP: usar feature_importances_ se existir
        return self._explain_fallback(model, X_array, feature_names)

    def _explain_with_shap(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict:
        """Calcula SHAP values usando a biblioteca shap."""
        try:
            if self._is_tree_model(model):
                logger.info("Usando TreeExplainer para modelo de arvore")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                # KernelExplainer como fallback para modelos genericos
                logger.info("Usando KernelExplainer (modelo nao-arvore)")
                # Amostrar background para eficiencia
                bg_size = min(100, len(X))
                background = shap.sample(X, bg_size)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X, nsamples=200)

            # Calcular importancia media (|SHAP|)
            shap_values = np.asarray(shap_values)
            feature_importance = np.mean(np.abs(shap_values), axis=0)

            # Ranking de features por importancia
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_features = [
                {"feature": feature_names[i], "importance": float(feature_importance[i])}
                for i in sorted_idx
            ]

            logger.info(
                f"SHAP calculado: top feature = {top_features[0]['feature']} "
                f"(importancia = {top_features[0]['importance']:.4f})"
            )

            return {
                "shap_values": shap_values,
                "feature_importance": feature_importance,
                "top_features": top_features,
            }

        except Exception as e:
            logger.error(f"Erro ao calcular SHAP values: {e}")
            return self._explain_fallback(model, X, feature_names)

    def _explain_fallback(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict:
        """Fallback: usa feature_importances_ do modelo se disponivel."""
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
            logger.info("Usando feature_importances_ do modelo como fallback")
        else:
            # Sem informacao de importancia — retornar uniforme
            logger.warning(
                "Modelo nao possui feature_importances_ e shap nao disponivel"
            )
            importance = np.ones(len(feature_names)) / len(feature_names)

        sorted_idx = np.argsort(importance)[::-1]
        top_features = [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in sorted_idx
        ]

        return {
            "shap_values": None,
            "feature_importance": importance,
            "top_features": top_features,
        }

    def plot_importance(
        self,
        shap_values: np.ndarray | None,
        feature_names: list[str],
        save_path: str | None = None,
        top_n: int = 20,
    ) -> None:
        """Gera grafico de importancia de features.

        Args:
            shap_values: Matriz de SHAP values (amostras x features)
            feature_names: Nomes das features
            save_path: Caminho para salvar o grafico (None = exibir)
            top_n: Numero maximo de features no grafico
        """
        if not _MPL_AVAILABLE:
            logger.warning("matplotlib nao disponivel — nao e possivel gerar grafico")
            return

        if shap_values is None:
            logger.warning("SHAP values nao disponiveis — grafico nao gerado")
            return

        try:
            # Calcular importancia media
            importance = np.mean(np.abs(shap_values), axis=0)
            sorted_idx = np.argsort(importance)[::-1][:top_n]

            names = [feature_names[i] for i in sorted_idx]
            values = importance[sorted_idx]

            # Gerar grafico horizontal
            fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
            ax.barh(range(len(names)), values[::-1], color="#2196F3", edgecolor="none")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names[::-1])
            ax.set_xlabel("Media |SHAP value|")
            ax.set_title("Importancia das Features (SHAP)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Grafico salvo em {save_path}")
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar grafico de importancia: {e}")

    def explain_prediction(
        self,
        model: Any,
        X_single: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Explica uma previsao individual.

        Args:
            model: Modelo treinado
            X_single: Uma unica amostra (1D ou 2D com 1 linha)
            feature_names: Nomes das features

        Returns:
            Dicionario com prediction, contributions (por feature) e base_value
        """
        X_arr = np.asarray(X_single)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        # Previsao do modelo
        prediction = float(model.predict(X_arr)[0])

        # Calcular contribuicoes via SHAP se disponivel
        if _SHAP_AVAILABLE:
            try:
                if self._is_tree_model(model):
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(X_arr)
                    base_value = float(explainer.expected_value)
                else:
                    bg = X_arr  # amostra unica como background (simplificado)
                    explainer = shap.KernelExplainer(model.predict, bg)
                    sv = explainer.shap_values(X_arr, nsamples=100)
                    base_value = float(explainer.expected_value)

                sv = np.asarray(sv).flatten()
                contributions = {
                    feature_names[i]: float(sv[i]) for i in range(len(feature_names))
                }

                # Ordenar por valor absoluto (mais impactantes primeiro)
                contributions = dict(
                    sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                )

                return {
                    "prediction": prediction,
                    "base_value": base_value,
                    "contributions": contributions,
                }

            except Exception as e:
                logger.error(f"Erro ao explicar previsao com SHAP: {e}")

        # Fallback sem SHAP
        logger.info("Retornando explicacao sem contribuicoes individuais (sem SHAP)")
        return {
            "prediction": prediction,
            "base_value": None,
            "contributions": {name: None for name in feature_names},
        }
