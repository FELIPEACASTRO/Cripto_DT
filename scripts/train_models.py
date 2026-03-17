#!/usr/bin/env python3
"""Script para treinar todos os modelos de previsao."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from src.data.storage import DataStorage
from src.training.trainer import Trainer
from src.evaluation.metrics import print_metrics


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Treinar modelos de previsao")
    parser.add_argument(
        "--coins", nargs="+", default=None,
        help="Moedas para treinar (default: todas com dados)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d",
        help="Timeframe para treino (default: 1d)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    storage = DataStorage(config)
    trainer = Trainer(config)

    # Carregar dados
    available = storage.list_available("raw")
    logger.info(f"Dados disponiveis: {available}")

    coins = args.coins or config.data.coins
    data = {}
    for coin in coins:
        df = storage.load(coin, args.timeframe, stage="raw")
        if not df.empty:
            data[coin] = df
            logger.info(f"  {coin}: {len(df)} candles carregados")
        else:
            logger.warning(f"  {coin}: sem dados para {args.timeframe}")

    if not data:
        logger.error("Nenhum dado disponivel. Execute collect_data.py primeiro.")
        sys.exit(1)

    # Treinar
    logger.info(f"\nIniciando treino para {len(data)} moedas...")
    results = trainer.train_all(data)

    # Salvar modelos
    trainer.save_models(results)

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO DO TREINO")
    print("=" * 60)
    for coin, result in results.items():
        avg = result["avg_metrics"]
        dir_acc = avg.get("avg_ensemble_dir_accuracy", 0)
        rmse = avg.get("avg_ensemble_rmse", 0)
        print(f"\n  {coin}:")
        print(f"    Acuracia Direcional: {dir_acc:.2%}")
        print(f"    RMSE:               {rmse:.6f}")
        print(f"    Folds:              {len(result['fold_metrics'])}")

    print("\n" + "=" * 60)
    logger.info(f"Modelos salvos em: {config.training.models_dir}")


if __name__ == "__main__":
    main()
