"""RiskAgent: monitora exposicao e aplica filtros de risco aos sinais de trading."""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.agents.base_agent import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

# Importacoes opcionais para calculo de correlacao
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class RiskLimits:
    """Limites de risco configuravies para o agente.

    Attributes:
        max_portfolio_exposure: Exposicao maxima total do portfolio (0 a 1).
        max_single_position: Tamanho maximo de uma posicao individual (0 a 1).
        max_correlated_exposure: Exposicao maxima em ativos correlacionados (0 a 1).
        correlation_threshold: Limiar de correlacao para considerar ativos relacionados.
        max_drawdown_pct: Drawdown maximo permitido antes de cortar posicoes.
        volatility_lookback: Janela de dias para calculo de volatilidade.
        volatility_scale_factor: Fator de escala para ajuste por volatilidade.
        min_confidence: Confianca minima para aceitar um sinal.
    """

    max_portfolio_exposure: float = 0.8
    max_single_position: float = 0.25
    max_correlated_exposure: float = 0.4
    correlation_threshold: float = 0.7
    max_drawdown_pct: float = 0.15
    volatility_lookback: int = 30
    volatility_scale_factor: float = 1.0
    min_confidence: float = 0.3


class RiskAgent(BaseAgent):
    """Agente responsavel por gestao de risco e protecao do portfolio.

    Aplica multiplas camadas de filtro de risco:
    1. Limites de exposicao por posicao e portfolio
    2. Filtro de correlacao (evita posicoes correlacionadas)
    3. Protecao contra drawdown
    4. Escala de posicao baseada em volatilidade
    5. Filtro de confianca minima

    Recebe sinais do TraderAgent e retorna sinais ajustados pelo risco.
    """

    name = "risk_agent"

    def __init__(self, risk_limits: RiskLimits | None = None):
        super().__init__()
        self.limits = risk_limits or RiskLimits()
        self._current_drawdown: float = 0.0
        self._portfolio_value_peak: float = 0.0

    def _filter_by_confidence(self, signals: dict) -> dict:
        """Remove sinais com confianca abaixo do minimo.

        Args:
            signals: Sinais por moeda.

        Returns:
            Sinais filtrados.
        """
        filtered = {}
        for coin, signal in signals.items():
            confidence = signal.get("confidence", 0)
            if confidence >= self.limits.min_confidence:
                filtered[coin] = signal
            else:
                logger.info(
                    f"[{self.name}] {coin}: removido por confianca baixa "
                    f"({confidence:.2%} < {self.limits.min_confidence:.2%})"
                )
        return filtered

    def _apply_position_limits(self, signals: dict) -> dict:
        """Aplica limites de tamanho por posicao individual.

        Args:
            signals: Sinais por moeda.

        Returns:
            Sinais com position_size limitado.
        """
        for coin, signal in signals.items():
            pos_size = signal.get("position_size", 0)
            if pos_size > self.limits.max_single_position:
                logger.info(
                    f"[{self.name}] {coin}: position_size reduzido de "
                    f"{pos_size:.2%} para {self.limits.max_single_position:.2%}"
                )
                signal["position_size"] = self.limits.max_single_position
                signal.setdefault("risk_adjustments", [])
                signal["risk_adjustments"].append("position_limit_applied")
        return signals

    def _apply_portfolio_exposure_limit(self, signals: dict) -> dict:
        """Garante que a exposicao total nao ultrapasse o limite do portfolio.

        Se a soma das posicoes excede o limite, escala proporcionalmente.

        Args:
            signals: Sinais por moeda.

        Returns:
            Sinais com exposicao total limitada.
        """
        total_exposure = sum(
            s.get("position_size", 0) for s in signals.values()
        )

        if total_exposure > self.limits.max_portfolio_exposure:
            scale = self.limits.max_portfolio_exposure / total_exposure
            logger.info(
                f"[{self.name}] Exposicao total {total_exposure:.2%} > "
                f"limite {self.limits.max_portfolio_exposure:.2%}, "
                f"aplicando escala {scale:.2f}"
            )
            for coin, signal in signals.items():
                signal["position_size"] = signal.get("position_size", 0) * scale
                signal.setdefault("risk_adjustments", [])
                signal["risk_adjustments"].append(
                    f"portfolio_scale_{scale:.2f}"
                )

        return signals

    def _apply_correlation_filter(
        self, signals: dict, ohlcv: dict
    ) -> dict:
        """Reduz exposicao em ativos altamente correlacionados.

        Args:
            signals: Sinais por moeda.
            ohlcv: Dados OHLCV por moeda para calculo de correlacao.

        Returns:
            Sinais com exposicao correlacionada limitada.
        """
        if not HAS_PANDAS or len(signals) < 2 or not ohlcv:
            return signals

        # Monta DataFrame de retornos para calculo de correlacao
        returns_dict = {}
        for coin in signals:
            if coin in ohlcv:
                df = ohlcv[coin]
                if hasattr(df, "columns") and "close" in df.columns:
                    returns_dict[coin] = df["close"].pct_change().dropna()

        if len(returns_dict) < 2:
            return signals

        try:
            returns_df = pd.DataFrame(returns_dict).dropna()
            if len(returns_df) < 10:
                return signals

            corr_matrix = returns_df.corr()
        except Exception as e:
            logger.debug(f"[{self.name}] Erro ao calcular correlacao: {e}")
            return signals

        # Identifica pares altamente correlacionados
        processed = set()
        for coin_a in signals:
            if coin_a in processed or coin_a not in corr_matrix.columns:
                continue
            correlated_group = [coin_a]

            for coin_b in signals:
                if coin_b == coin_a or coin_b in processed:
                    continue
                if coin_b not in corr_matrix.columns:
                    continue
                if abs(corr_matrix.loc[coin_a, coin_b]) >= self.limits.correlation_threshold:
                    correlated_group.append(coin_b)

            if len(correlated_group) > 1:
                # Calcula exposicao total do grupo correlacionado
                group_exposure = sum(
                    signals[c].get("position_size", 0) for c in correlated_group
                )
                if group_exposure > self.limits.max_correlated_exposure:
                    scale = self.limits.max_correlated_exposure / group_exposure
                    logger.info(
                        f"[{self.name}] Grupo correlacionado {correlated_group}: "
                        f"exposicao {group_exposure:.2%} > "
                        f"limite {self.limits.max_correlated_exposure:.2%}"
                    )
                    for c in correlated_group:
                        signals[c]["position_size"] = (
                            signals[c].get("position_size", 0) * scale
                        )
                        signals[c].setdefault("risk_adjustments", [])
                        signals[c]["risk_adjustments"].append(
                            f"correlation_filter_{scale:.2f}"
                        )
                processed.update(correlated_group)

        return signals

    def _apply_drawdown_protection(self, signals: dict, portfolio_value: float = 0) -> dict:
        """Protecao contra drawdown excessivo.

        Se o drawdown atual excede o limite, reduz ou zera todas as posicoes.

        Args:
            signals: Sinais por moeda.
            portfolio_value: Valor atual do portfolio.

        Returns:
            Sinais ajustados pelo drawdown.
        """
        if portfolio_value <= 0:
            return signals

        # Atualiza pico do portfolio
        if portfolio_value > self._portfolio_value_peak:
            self._portfolio_value_peak = portfolio_value

        if self._portfolio_value_peak > 0:
            self._current_drawdown = (
                1 - portfolio_value / self._portfolio_value_peak
            )

        if self._current_drawdown >= self.limits.max_drawdown_pct:
            # Drawdown critico - reduz drasticamente ou zera posicoes
            reduction = max(0, 1 - (self._current_drawdown / self.limits.max_drawdown_pct))
            logger.warning(
                f"[{self.name}] DRAWDOWN CRITICO: {self._current_drawdown:.2%} "
                f"(limite: {self.limits.max_drawdown_pct:.2%}). "
                f"Reducao: {reduction:.2%}"
            )
            for coin, signal in signals.items():
                signal["position_size"] = signal.get("position_size", 0) * reduction
                signal.setdefault("risk_adjustments", [])
                signal["risk_adjustments"].append(
                    f"drawdown_protection_{reduction:.2f}"
                )

        return signals

    def _apply_volatility_scaling(self, signals: dict, ohlcv: dict) -> dict:
        """Escala posicoes inversamente proporcional a volatilidade.

        Ativos mais volateis recebem posicoes menores.

        Args:
            signals: Sinais por moeda.
            ohlcv: Dados OHLCV por moeda.

        Returns:
            Sinais com posicoes ajustadas pela volatilidade.
        """
        if not HAS_PANDAS or not ohlcv:
            return signals

        for coin, signal in signals.items():
            if coin not in ohlcv:
                continue

            df = ohlcv[coin]
            if not hasattr(df, "columns") or "close" not in df.columns:
                continue

            try:
                returns = df["close"].pct_change().dropna()
                lookback = min(self.limits.volatility_lookback, len(returns))
                if lookback < 5:
                    continue

                vol = returns.tail(lookback).std()
                if vol <= 0:
                    continue

                # Volatilidade media historica como referencia
                hist_vol = returns.std()
                if hist_vol <= 0:
                    continue

                # Fator de escala: se vol atual > historica, reduz posicao
                vol_ratio = hist_vol / vol  # Inverso: alta vol = menor posicao
                vol_scale = np.clip(
                    vol_ratio * self.limits.volatility_scale_factor,
                    0.2,  # Minimo 20% da posicao
                    1.5,  # Maximo 150% da posicao
                )

                original_size = signal.get("position_size", 0)
                signal["position_size"] = original_size * vol_scale
                signal.setdefault("risk_adjustments", [])
                signal["risk_adjustments"].append(
                    f"vol_scale_{vol_scale:.2f}"
                )

                if vol_scale < 0.8:
                    logger.info(
                        f"[{self.name}] {coin}: volatilidade alta, "
                        f"posicao reduzida por fator {vol_scale:.2f}"
                    )
            except Exception as e:
                logger.debug(f"[{self.name}] Erro vol scaling {coin}: {e}")

        return signals

    def execute(self, context: dict) -> AgentResult:
        """Aplica filtros de risco a todos os sinais.

        Args:
            context: Deve conter (normalmente fornecido pelo TraderAgent):
                - signals (dict[str, dict]): Sinais de trading por moeda
                - ohlcv (dict[str, DataFrame]): Dados OHLCV para calculo de vol/corr
                - portfolio_value (float, opcional): Valor atual do portfolio

        Returns:
            AgentResult com data contendo:
                - risk_adjusted_signals (dict[str, dict]): Sinais ajustados
                - risk_report (dict): Relatorio de risco agregado
                - removed_signals (list[str]): Moedas com sinais removidos
        """
        signals = context.get("signals", {})
        ohlcv = context.get("ohlcv", {})
        portfolio_value = context.get("portfolio_value", 0)

        result = AgentResult()
        result.data = {
            "risk_adjusted_signals": {},
            "risk_report": {},
            "removed_signals": [],
        }

        if not signals:
            logger.info(f"[{self.name}] Nenhum sinal para avaliar")
            return result

        # Copia sinais para nao modificar originais
        adjusted = {}
        for coin, signal in signals.items():
            adjusted[coin] = dict(signal)
            adjusted[coin].setdefault("risk_adjustments", [])

        original_count = len(adjusted)

        # --- Passo 1: Filtro de confianca minima ---
        adjusted = self._filter_by_confidence(adjusted)
        removed = set(signals.keys()) - set(adjusted.keys())
        result.data["removed_signals"] = list(removed)

        # --- Passo 2: Limites de posicao individual ---
        adjusted = self._apply_position_limits(adjusted)

        # --- Passo 3: Filtro de correlacao ---
        adjusted = self._apply_correlation_filter(adjusted, ohlcv)

        # --- Passo 4: Escala de volatilidade ---
        adjusted = self._apply_volatility_scaling(adjusted, ohlcv)

        # --- Passo 5: Protecao contra drawdown ---
        adjusted = self._apply_drawdown_protection(adjusted, portfolio_value)

        # --- Passo 6: Limite de exposicao total do portfolio ---
        adjusted = self._apply_portfolio_exposure_limit(adjusted)

        result.data["risk_adjusted_signals"] = adjusted

        # --- Relatorio de risco ---
        total_exposure = sum(
            s.get("position_size", 0) for s in adjusted.values()
        )
        result.data["risk_report"] = {
            "total_signals_received": original_count,
            "total_signals_approved": len(adjusted),
            "total_signals_removed": len(result.data["removed_signals"]),
            "total_exposure": round(total_exposure, 4),
            "current_drawdown": round(self._current_drawdown, 4),
            "portfolio_value_peak": round(self._portfolio_value_peak, 2),
            "risk_limits": {
                "max_portfolio_exposure": self.limits.max_portfolio_exposure,
                "max_single_position": self.limits.max_single_position,
                "max_correlated_exposure": self.limits.max_correlated_exposure,
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "min_confidence": self.limits.min_confidence,
            },
        }

        logger.info(
            f"[{self.name}] Risco avaliado: {len(adjusted)}/{original_count} sinais aprovados, "
            f"exposicao total: {total_exposure:.2%}"
        )

        result.metadata["approved_coins"] = list(adjusted.keys())
        result.metadata["removed_coins"] = result.data["removed_signals"]

        return result
