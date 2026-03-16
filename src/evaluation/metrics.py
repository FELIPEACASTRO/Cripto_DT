"""Metricas de avaliacao para previsoes de criptomoedas."""

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Calcula metricas de regressao e direcional.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        prefix: Prefixo para nomes das metricas

    Returns:
        Dicionario {metrica: valor}
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Garantir mesmo tamanho
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if len(y_true) == 0:
        return {}

    metrics = {}

    # RMSE
    metrics[f"{prefix}rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # MAE
    metrics[f"{prefix}mae"] = float(np.mean(np.abs(y_pred - y_true)))

    # MAPE (evitar divisao por zero)
    mask = y_true != 0
    if mask.sum() > 0:
        metrics[f"{prefix}mape"] = float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )

    # Acuracia direcional
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    metrics[f"{prefix}dir_accuracy"] = float(np.mean(true_dir == pred_dir))

    # Correlacao
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics[f"{prefix}correlation"] = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Sharpe ratio de estrategia simples (comprar quando prev > 0)
    strategy_returns = y_true * np.sign(y_pred)
    if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
        metrics[f"{prefix}sharpe"] = float(
            np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        )

    # Hit rate (% de trades lucrativos)
    trades = y_true * np.sign(y_pred)
    if len(trades) > 0:
        metrics[f"{prefix}hit_rate"] = float(np.mean(trades > 0))

    # Profit factor
    gains = trades[trades > 0].sum()
    losses = abs(trades[trades < 0].sum())
    if losses > 0:
        metrics[f"{prefix}profit_factor"] = float(gains / losses)

    return metrics


def print_metrics(metrics: dict[str, float], title: str = "Metricas") -> None:
    """Imprime metricas formatadas."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in sorted(metrics.items()):
        print(f"  {key:30s}: {value:.6f}")
    print(f"{'='*50}")
