"""Visualizacao de resultados: graficos de previsao, features, performance."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Visualizer:
    """Gera graficos de avaliacao dos modelos."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Real vs Previsto",
        filename: str = "predictions.png",
    ) -> Path:
        """Grafico de previsoes vs valores reais."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        # Previsoes vs Real
        axes[0].plot(y_true, label="Real", alpha=0.8, linewidth=1)
        axes[0].plot(y_pred, label="Previsto", alpha=0.8, linewidth=1)
        axes[0].set_title(title, fontsize=14)
        axes[0].set_ylabel("Retorno")
        axes[0].legend()

        # Residuos
        residuals = y_true[:len(y_pred)] - y_pred[:len(y_true)]
        axes[1].bar(range(len(residuals)), residuals, alpha=0.5, color="gray", width=1)
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
        axes[1].set_title("Residuos", fontsize=12)
        axes[1].set_ylabel("Erro")
        axes[1].set_xlabel("Amostra")

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grafico salvo: {path}")
        return path

    def plot_cumulative_returns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Retornos Acumulados da Estrategia",
        filename: str = "cumulative_returns.png",
    ) -> Path:
        """Grafico de retornos acumulados: estrategia vs buy-and-hold."""
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Estrategia: comprar quando previsao > 0
        strategy_returns = y_true * np.sign(y_pred)
        cum_strategy = np.cumsum(strategy_returns)
        cum_buyhold = np.cumsum(y_true)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cum_strategy, label="Estrategia ML", linewidth=2)
        ax.plot(cum_buyhold, label="Buy & Hold", linewidth=2, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.fill_between(
            range(len(cum_strategy)),
            cum_strategy,
            cum_buyhold,
            alpha=0.1,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Amostra")
        ax.set_ylabel("Retorno Acumulado (log)")
        ax.legend()

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grafico salvo: {path}")
        return path

    def plot_feature_importance(
        self,
        importance: dict[str, float],
        top_n: int = 20,
        title: str = "Importancia das Features",
        filename: str = "feature_importance.png",
    ) -> Path:
        """Grafico de barras com importancia das features."""
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        sorted_imp = sorted_imp[:top_n]

        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Importancia")

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grafico salvo: {path}")
        return path

    def plot_fold_performance(
        self,
        fold_metrics: list[dict],
        metric_name: str = "ensemble_dir_accuracy",
        title: str = "Performance por Fold",
        filename: str = "fold_performance.png",
    ) -> Path:
        """Grafico de metricas por fold do walk-forward."""
        folds = [m.get("fold", i) for i, m in enumerate(fold_metrics)]
        values = [m.get(metric_name, 0) for m in fold_metrics]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(folds, values, color="steelblue", edgecolor="navy")
        ax.axhline(y=np.mean(values), color="red", linestyle="--",
                   label=f"Media: {np.mean(values):.4f}")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5,
                   label="Aleatorio (50%)")
        ax.set_title(f"{title} - {metric_name}", fontsize=14)
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric_name)
        ax.legend()

        # Valor em cima de cada barra
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grafico salvo: {path}")
        return path

    def plot_confidence_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibracao de Confianca",
        filename: str = "confidence_calibration.png",
    ) -> Path:
        """Verifica se confianca alta -> melhor acuracia."""
        min_len = min(len(y_true), len(y_pred), len(confidence))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        confidence = confidence[:min_len]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_centers = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
            if mask.sum() > 0:
                acc = np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))
                bin_accs.append(acc)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_counts.append(mask.sum())

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="steelblue",
               label="Acuracia direcional")
        ax1.plot([0, 1], [0.5, 1], "r--", label="Calibracao perfeita", alpha=0.5)
        ax1.set_xlabel("Confianca do Modelo")
        ax1.set_ylabel("Acuracia Direcional")
        ax1.set_title(title, fontsize=14)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, bin_counts, "go-", alpha=0.5, label="N amostras")
        ax2.set_ylabel("Numero de Amostras")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Grafico salvo: {path}")
        return path
