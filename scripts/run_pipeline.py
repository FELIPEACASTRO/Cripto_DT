#!/usr/bin/env python3
"""Pipeline completo: coleta → treino → previsao → avaliacao."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from src.data.collector import DataCollector
from src.data.storage import DataStorage
from src.training.trainer import Trainer
from src.prediction.predictor import Predictor
from src.evaluation.visualizer import Visualizer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo de previsao de criptomoedas"
    )
    parser.add_argument(
        "--coins", nargs="+", default=None,
        help="Moedas (default: todas configuradas)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d",
        help="Timeframe (default: 1d)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Dias de historico (default: configuracao)"
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Pular coleta (usar dados existentes)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Pular treino (usar modelos existentes)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    coins = args.coins or config.data.coins
    days = args.days or config.data.history_days

    # ===== FASE 1: COLETA =====
    if not args.skip_collect:
        print("\n" + "=" * 60)
        print("  FASE 1: COLETA DE DADOS")
        print("=" * 60)

        collector = DataCollector(config)
        storage = DataStorage(config)

        for coin in coins:
            try:
                df = collector.fetch_ohlcv(coin, args.timeframe, since_days=days)
                if not df.empty:
                    storage.save(df, coin, args.timeframe, stage="raw")
            except Exception as e:
                logger.error(f"Erro ao coletar {coin}: {e}")
    else:
        logger.info("Coleta pulada (--skip-collect)")

    # ===== FASE 2: TREINO =====
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("  FASE 2: TREINO DOS MODELOS")
        print("=" * 60)

        storage = DataStorage(config)
        trainer = Trainer(config)

        data = {}
        for coin in coins:
            df = storage.load(coin, args.timeframe, stage="raw")
            if not df.empty:
                data[coin] = df

        if data:
            results = trainer.train_all(data)
            trainer.save_models(results)

            # Gerar graficos
            visualizer = Visualizer(config.training.reports_dir)
            for coin, result in results.items():
                if result["fold_metrics"]:
                    visualizer.plot_fold_performance(
                        result["fold_metrics"],
                        filename=f"{coin}_fold_performance.png",
                        title=f"{coin} - Performance por Fold",
                    )

                    # Feature importance (se XGBoost disponivel)
                    xgb = result["models"].get("xgb_reg")
                    if xgb and hasattr(xgb, "get_feature_importance"):
                        importance = xgb.get_feature_importance()
                        if importance:
                            visualizer.plot_feature_importance(
                                importance,
                                filename=f"{coin}_feature_importance.png",
                                title=f"{coin} - Importancia das Features",
                            )
        else:
            logger.error("Nenhum dado disponivel para treino")
    else:
        logger.info("Treino pulado (--skip-train)")

    # ===== FASE 3: PREVISAO =====
    print("\n" + "=" * 60)
    print("  FASE 3: PREVISOES")
    print("=" * 60)

    predictor = Predictor(config)
    config.data.coins = coins
    df_predictions = predictor.predict_all()

    if not df_predictions.empty:
        print("\n  RESULTADOS:")
        print("  " + "-" * 66)
        for _, row in df_predictions.iterrows():
            arrow = "^" if row["direction"] == "ALTA" else "v"
            print(
                f"  {row['coin']:>6s} {arrow} {row['direction']:>5s} | "
                f"${row['current_price']:>10,.2f} -> "
                f"${row['predicted_price']:>10,.2f} | "
                f"Confianca: {row['confidence']:.1%}"
            )
        print("  " + "-" * 66)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETO!")
    print("=" * 60)
    print(f"  Dados:     {config.data.raw_dir}")
    print(f"  Modelos:   {config.training.models_dir}")
    print(f"  Previsoes: {config.data.predictions_dir}")
    print(f"  Graficos:  {config.training.reports_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
