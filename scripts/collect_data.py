#!/usr/bin/env python3
"""Script para coletar dados historicos de criptomoedas."""

import argparse
import logging
import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from src.data.collector import DataCollector
from src.data.storage import DataStorage


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Coletar dados de criptomoedas")
    parser.add_argument(
        "--coins", nargs="+", default=None,
        help="Moedas para coletar (default: todas configuradas)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help="Timeframes (default: todos configurados)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Dias de historico (default: configuracao)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detalhado"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    coins = args.coins or config.data.coins
    timeframes = args.timeframes or config.data.timeframes
    days = args.days or config.data.history_days

    logger.info(f"Coletando dados para {len(coins)} moedas, {len(timeframes)} timeframes, {days} dias")

    collector = DataCollector(config)
    storage = DataStorage(config)

    total = 0
    for coin in coins:
        for tf in timeframes:
            try:
                df = collector.fetch_ohlcv(coin, tf, since_days=days)
                if not df.empty:
                    storage.save(df, coin, tf, stage="raw")
                    total += 1
                else:
                    logger.warning(f"Nenhum dado para {coin}/{tf}")
            except Exception as e:
                logger.error(f"Erro ao coletar {coin}/{tf}: {e}")

    logger.info(f"\nColeta finalizada! {total} datasets salvos.")
    logger.info(f"Dados em: {config.data.raw_dir}")


if __name__ == "__main__":
    main()
