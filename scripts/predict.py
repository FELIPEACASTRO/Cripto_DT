#!/usr/bin/env python3
"""Script para gerar previsoes com modelos treinados."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from src.prediction.predictor import Predictor


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Gerar previsoes de criptomoedas")
    parser.add_argument(
        "--coins", nargs="+", default=None,
        help="Moedas para prever (default: todas com modelo treinado)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    predictor = Predictor(config)

    if args.coins:
        config.data.coins = args.coins

    logger.info("Gerando previsoes...")
    df = predictor.predict_all()

    if df.empty:
        logger.error("Nenhuma previsao gerada.")
        sys.exit(1)

    # Exibir resultado
    print("\n" + "=" * 70)
    print("  PREVISOES DE CRIPTOMOEDAS")
    print("=" * 70)

    for _, row in df.iterrows():
        emoji = "^" if row["direction"] == "ALTA" else "v"
        print(
            f"\n  {row['coin']:>6s} {emoji} {row['direction']:>5s} | "
            f"Atual: ${row['current_price']:>10,.2f} | "
            f"Previsto: ${row['predicted_price']:>10,.2f} | "
            f"Retorno: {row['predicted_return']:>+.4f} | "
            f"Confianca: {row['confidence']:.1%}"
        )

    print("\n" + "=" * 70)
    logger.info(f"Previsoes salvas em: {config.data.predictions_dir}")


if __name__ == "__main__":
    main()
