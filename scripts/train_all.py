#!/usr/bin/env python3
"""Script otimizado para treinar todos os modelos para todas as moedas.

Desabilita modelos excessivamente lentos em CPU (Mamba SSM) e ajusta
configuracoes para treinamento pratico.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from config.settings import config
from src.data.storage import DataStorage
from src.evaluation.visualizer import Visualizer
from src.training.trainer import Trainer

# ========================================================
# Ajustes para treinamento pratico em CPU
# ========================================================

# Desabilitar Mamba SSM (trava em CPU com selective state space)
config.training.use_mamba = False

# Reduzir epocas para modelos pesados em CPU
config.lstm.max_epochs = 60
config.bnn.max_epochs = 100
config.mdn.max_epochs = 60
config.evidential.max_epochs = 80

# Feature selection: usar mutual_info (mais rapido que boruta)
config.features.feature_selection_method = "mutual_info"


def main():
    storage = DataStorage(config)
    trainer = Trainer(config)
    visualizer = Visualizer(config.training.reports_dir)

    # Carregar dados de todas as moedas
    coins = config.data.coins
    data = {}
    for coin in coins:
        df = storage.load(coin, "1d", stage="raw")
        if not df.empty:
            data[coin] = df
            logger.info(f"  {coin}: {len(df)} candles")
        else:
            logger.warning(f"  {coin}: sem dados")

    if not data:
        logger.error("Nenhum dado disponivel. Execute collect_data.py primeiro.")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"  TREINANDO {len(data)} MOEDAS")
    logger.info(f"{'='*60}\n")

    # Treinar todos os modelos
    results = trainer.train_all(data)

    # Salvar modelos e metricas
    trainer.save_models(results)

    # Gerar graficos
    for coin, result in results.items():
        if coin.startswith("_"):
            continue

        if result.get("fold_metrics"):
            visualizer.plot_fold_performance(
                result["fold_metrics"],
                filename=f"{coin}_fold_performance.png",
                title=f"{coin} - Performance por Fold",
            )

        xgb = result["models"].get("xgb_reg")
        if xgb and hasattr(xgb, "get_feature_importance"):
            importance = xgb.get_feature_importance()
            if importance:
                visualizer.plot_feature_importance(
                    importance,
                    filename=f"{coin}_feature_importance.png",
                    title=f"{coin} - Importancia das Features",
                )

    # Resumo final
    print("\n" + "=" * 70)
    print("  RESUMO DO TREINO")
    print("=" * 70)
    for coin, result in results.items():
        if coin.startswith("_"):
            continue
        avg = result["avg_metrics"]
        dir_acc = avg.get("avg_ensemble_dir_accuracy", 0)
        rmse = avg.get("avg_ensemble_rmse", 0)
        n_models = len(result["models"])
        n_features = len(result.get("feature_columns", []))
        print(f"\n  {coin}:")
        print(f"    Acuracia Direcional: {dir_acc:.2%}")
        print(f"    RMSE:               {rmse:.6f}")
        print(f"    Modelos:            {n_models}")
        print(f"    Features:           {n_features}")
        print(f"    Folds:              {len(result['fold_metrics'])}")

    print("\n" + "=" * 70)
    print(f"  Modelos salvos em: {config.training.models_dir}")
    print(f"  Graficos em:       {config.training.reports_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
