"""Armazenamento e leitura de dados em formato Parquet."""

import logging
from pathlib import Path

import pandas as pd

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)


class DataStorage:
    """Gerencia leitura e escrita de dados Parquet."""

    def __init__(self, config: Config = default_config):
        self.config = config

    def _get_path(self, coin: str, timeframe: str, stage: str = "raw") -> Path:
        """Retorna o caminho do arquivo Parquet."""
        if stage == "raw":
            base = self.config.data.raw_dir
        elif stage == "processed":
            base = self.config.data.processed_dir
        else:
            base = self.config.data.predictions_dir
        return base / coin / f"{timeframe}.parquet"

    def save(
        self, df: pd.DataFrame, coin: str, timeframe: str, stage: str = "raw"
    ) -> Path:
        """Salva DataFrame como Parquet.

        Args:
            df: Dados a salvar
            coin: Simbolo da moeda
            timeframe: Intervalo temporal
            stage: "raw", "processed" ou "predictions"

        Returns:
            Caminho do arquivo salvo
        """
        path = self._get_path(coin, timeframe, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Salvo: {path} ({len(df)} linhas)")
        return path

    def load(self, coin: str, timeframe: str, stage: str = "raw") -> pd.DataFrame:
        """Carrega DataFrame de Parquet.

        Args:
            coin: Simbolo da moeda
            timeframe: Intervalo temporal
            stage: "raw", "processed" ou "predictions"

        Returns:
            DataFrame carregado ou DataFrame vazio se arquivo nao existe
        """
        path = self._get_path(coin, timeframe, stage)
        if not path.exists():
            logger.warning(f"Arquivo nao encontrado: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info(f"Carregado: {path} ({len(df)} linhas)")
        return df

    def save_predictions(self, df: pd.DataFrame, filename: str) -> Path:
        """Salva previsoes em CSV para facil leitura."""
        path = self.config.data.predictions_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Previsoes salvas: {path}")
        return path

    def list_available(self, stage: str = "raw") -> list[tuple[str, str]]:
        """Lista pares (coin, timeframe) disponiveis.

        Returns:
            Lista de tuplas (coin, timeframe)
        """
        if stage == "raw":
            base = self.config.data.raw_dir
        elif stage == "processed":
            base = self.config.data.processed_dir
        else:
            base = self.config.data.predictions_dir

        available = []
        if base.exists():
            for coin_dir in sorted(base.iterdir()):
                if coin_dir.is_dir():
                    for f in sorted(coin_dir.glob("*.parquet")):
                        available.append((coin_dir.name, f.stem))
        return available
