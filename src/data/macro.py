"""Coleta de variaveis macroeconomicas via Yahoo Finance."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroCollector:
    """Coleta dados macroeconomicos e deriva features para modelos de cripto."""

    # Mapeamento de tickers para nomes de colunas
    TICKERS = {
        "^VIX": "vix",
        "DX-Y.NYB": "dxy",
        "^GSPC": "sp500",
        "^IXIC": "nasdaq",
        "GC=F": "gold",
        "CL=F": "oil",
        "^TNX": "treasury_10y",
    }

    def fetch_macro_data(self, days: int = 365) -> pd.DataFrame:
        """Busca dados macroeconomicos via yfinance.

        Args:
            days: Numero de dias de historico para buscar

        Returns:
            DataFrame com colunas: timestamp, vix, dxy, sp500, nasdaq, gold, oil, treasury_10y
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance nao esta instalado. Execute: pip install yfinance")
            return pd.DataFrame()

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Buscando dados macro de {start_date.date()} a {end_date.date()} "
            f"({len(self.TICKERS)} ativos)"
        )

        try:
            tickers_str = " ".join(self.TICKERS.keys())
            data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error(f"Erro ao baixar dados do yfinance: {e}")
            return pd.DataFrame()

        if data.empty:
            logger.warning("yfinance retornou DataFrame vazio")
            return pd.DataFrame()

        # Extrair precos de fechamento para cada ativo
        records = {}
        for ticker, col_name in self.TICKERS.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    records[col_name] = data["Close"][ticker]
                else:
                    # Caso de um unico ticker (improvavel aqui)
                    records[col_name] = data["Close"]
            except KeyError:
                logger.warning(f"Ticker {ticker} nao encontrado nos dados")
                records[col_name] = np.nan

        result = pd.DataFrame(records)
        result.index.name = "timestamp"
        result = result.reset_index()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

        # Forward fill para preencher fins de semana e feriados
        for col in self.TICKERS.values():
            if col in result.columns:
                result[col] = result[col].ffill()

        result = result.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Dados macro coletados: {len(result)} dias")
        return result

    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge dados macro com DataFrame OHLCV e gera features derivadas.

        Args:
            df: DataFrame OHLCV com coluna 'timestamp'

        Returns:
            DataFrame com features macroeconomicas adicionadas
        """
        # Buscar dados macro cobrindo o periodo do DataFrame
        try:
            ts_min = pd.to_datetime(df["timestamp"]).min()
            ts_max = pd.to_datetime(df["timestamp"]).max()
            days_needed = (ts_max - ts_min).days + 60  # margem para rolling
            macro_df = self.fetch_macro_data(days=max(days_needed, 365))
        except Exception as e:
            logger.error(f"Erro ao buscar dados macro: {e}")
            return df

        if macro_df.empty:
            logger.warning("Dados macro vazios, retornando DataFrame original")
            return df

        df = df.copy()

        # Normalizar timestamps para data (macro e diario)
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        macro_df["_date"] = pd.to_datetime(macro_df["timestamp"]).dt.normalize()

        # Pegar apenas colunas macro para merge
        macro_cols = ["_date"] + [c for c in self.TICKERS.values() if c in macro_df.columns]
        macro_subset = macro_df[macro_cols].drop_duplicates(subset=["_date"])

        df = df.merge(macro_subset, on="_date", how="left")

        # Forward fill apos merge (para dias sem dados macro)
        for col in self.TICKERS.values():
            if col in df.columns:
                df[col] = df[col].ffill()

        # --- Features derivadas ---

        # VIX e variacao
        df["macro_vix"] = df["vix"]
        df["macro_vix_change"] = df["vix"].pct_change()

        # DXY e variacao
        df["macro_dxy"] = df["dxy"]
        df["macro_dxy_change"] = df["dxy"].pct_change()

        # Retornos diarios dos indices e commodities
        df["macro_sp500_return"] = df["sp500"].pct_change()
        df["macro_nasdaq_return"] = df["nasdaq"].pct_change()
        df["macro_gold_return"] = df["gold"].pct_change()
        df["macro_oil_return"] = df["oil"].pct_change()

        # Treasury yield
        df["macro_treasury_10y"] = df["treasury_10y"]

        # Risk-on: S&P subindo e VIX caindo
        vix_falling = df["vix"].diff() < 0
        sp500_positive = df["macro_sp500_return"] > 0
        df["macro_risk_on"] = (sp500_positive & vix_falling).astype(float)

        # Proxy de liquidez: S&P * (1/VIX), normalizado min-max
        raw_liquidity = df["sp500"] * (1.0 / df["vix"].replace(0, np.nan))
        liq_min = raw_liquidity.rolling(252, min_periods=20).min()
        liq_max = raw_liquidity.rolling(252, min_periods=20).max()
        liq_range = (liq_max - liq_min).replace(0, np.nan)
        df["macro_liquidity_proxy"] = (raw_liquidity - liq_min) / liq_range

        # Crypto beta: correlacao rolling 20 dias entre retorno cripto e S&P 500
        if "close" in df.columns:
            crypto_return = df["close"].pct_change()
            df["macro_crypto_beta"] = (
                crypto_return.rolling(20, min_periods=10)
                .corr(df["macro_sp500_return"])
            )
        else:
            df["macro_crypto_beta"] = np.nan

        # Limpar colunas intermediarias
        cols_to_drop = ["_date"] + [
            c for c in self.TICKERS.values() if c in df.columns
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        logger.info("Features macroeconomicas adicionadas com sucesso")
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features macroeconomicas geradas."""
        return [
            "macro_vix", "macro_vix_change",
            "macro_dxy", "macro_dxy_change",
            "macro_sp500_return", "macro_nasdaq_return",
            "macro_gold_return", "macro_oil_return",
            "macro_treasury_10y",
            "macro_risk_on", "macro_liquidity_proxy",
            "macro_crypto_beta",
        ]
