"""Script CLI para executar o pipeline em tempo real."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.realtime_pipeline import RealtimePipeline


def main():
    parser = argparse.ArgumentParser(description="Pipeline de previsao em tempo real")
    parser.add_argument("--interval", type=int, default=60, help="Intervalo em minutos")
    parser.add_argument("--coins", nargs="+", default=None, help="Moedas (default: todas)")
    parser.add_argument("--once", action="store_true", help="Executar apenas uma vez")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = RealtimePipeline(update_interval_minutes=args.interval)

    if args.once:
        results = pipeline.run_once(args.coins)
        if results:
            print(f"Previsoes geradas para {len(results.get('predictions', {}))} moedas")
    else:
        pipeline.run_continuous(args.coins)


if __name__ == "__main__":
    main()
