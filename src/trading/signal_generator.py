"""Gerador de sinais de trading combinando predicoes ML com regras de gestao de risco."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Sinal de trading gerado a partir de predicoes de modelos.

    Attributes:
        coin: Simbolo da criptomoeda (ex: 'BTC').
        timestamp: Momento de geracao do sinal.
        direction: Direcao do sinal ('LONG', 'SHORT' ou 'HOLD').
        strength: Forca do sinal entre 0 e 1.
        confidence: Confianca agregada dos modelos entre 0 e 1.
        predicted_return: Retorno previsto (fracao).
        stop_loss: Preco de stop-loss.
        take_profit: Preco de take-profit.
        position_size: Fracao do capital a alocar (0 a 1).
        risk_reward_ratio: Razao risco/recompensa.
        model_agreement: Fracao dos modelos que concordam na direcao (0 a 1).
        reasons: Lista de razoes que explicam o sinal.
    """

    coin: str
    timestamp: datetime
    direction: str
    strength: float
    confidence: float
    predicted_return: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    model_agreement: float
    reasons: list[str] = field(default_factory=list)


class SignalGenerator:
    """Gera sinais de trading combinando predicoes de multiplos modelos ML.

    Combina predicoes ponderadas por confianca, calcula niveis de
    stop-loss/take-profit baseados em ATR e dimensiona posicoes
    usando uma versao simplificada do criterio de Kelly.

    Args:
        risk_per_trade: Fracao do capital arriscada por trade.
        max_position: Tamanho maximo de posicao como fracao do capital.
        min_confidence: Confianca minima para gerar sinal ativo.
        min_model_agreement: Concordancia minima entre modelos para sinal ativo.
        atr_multiplier_sl: Multiplicador do ATR para stop-loss.
        atr_multiplier_tp: Multiplicador do ATR para take-profit.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_position: float = 0.1,
        min_confidence: float = 0.6,
        min_model_agreement: float = 0.5,
        atr_multiplier_sl: float = 2.0,
        atr_multiplier_tp: float = 3.0,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.min_confidence = min_confidence
        self.min_model_agreement = min_model_agreement
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp

        logger.info(
            "SignalGenerator inicializado: risk=%.2f, max_pos=%.2f, "
            "min_conf=%.2f, min_agree=%.2f, sl_mult=%.1f, tp_mult=%.1f",
            risk_per_trade,
            max_position,
            min_confidence,
            min_model_agreement,
            atr_multiplier_sl,
            atr_multiplier_tp,
        )

    def generate_signal(
        self,
        predictions: dict,
        current_price: float,
        atr: float,
        coin: str,
    ) -> TradingSignal:
        """Gera sinal de trading a partir de predicoes de multiplos modelos.

        Args:
            predictions: Dicionario com predicoes por modelo. Formato:
                ``{"xgboost_reg": {"pred": 0.02, "conf": 0.7}, ...}``
            current_price: Preco atual do ativo.
            atr: Average True Range atual do ativo.
            coin: Simbolo da criptomoeda.

        Returns:
            TradingSignal com direcao, niveis de preco e dimensionamento.
        """
        reasons: list[str] = []

        # --- Media ponderada das predicoes (peso = confianca) ---
        total_weight = 0.0
        weighted_pred = 0.0
        confidences: list[float] = []

        for model_name, info in predictions.items():
            pred = info["pred"]
            conf = info["conf"]
            weighted_pred += pred * conf
            total_weight += conf
            confidences.append(conf)

        if total_weight > 0:
            predicted_return = weighted_pred / total_weight
        else:
            predicted_return = 0.0

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # --- Concordancia entre modelos ---
        n_long = sum(1 for info in predictions.values() if info["pred"] > 0)
        n_short = sum(1 for info in predictions.values() if info["pred"] < 0)
        n_models = len(predictions)

        if n_models > 0:
            model_agreement = max(n_long, n_short) / n_models
        else:
            model_agreement = 0.0

        majority_direction = "LONG" if n_long >= n_short else "SHORT"

        reasons.append(
            f"{n_models} modelos: {n_long} LONG, {n_short} SHORT "
            f"(concordancia={model_agreement:.0%})"
        )
        reasons.append(f"Retorno previsto ponderado: {predicted_return:+.4f}")
        reasons.append(f"Confianca media: {avg_confidence:.2f}")

        # --- Decidir direcao ---
        if model_agreement < self.min_model_agreement:
            direction = "HOLD"
            reasons.append(
                f"HOLD: concordancia {model_agreement:.0%} < "
                f"minimo {self.min_model_agreement:.0%}"
            )
        elif avg_confidence < self.min_confidence:
            direction = "HOLD"
            reasons.append(
                f"HOLD: confianca {avg_confidence:.2f} < "
                f"minimo {self.min_confidence:.2f}"
            )
        else:
            direction = majority_direction
            reasons.append(f"Sinal ativo: {direction}")

        # --- Forca do sinal ---
        strength = float(np.clip(
            model_agreement * avg_confidence * min(abs(predicted_return) * 20, 1.0),
            0.0,
            1.0,
        ))

        # --- Stop-loss e take-profit ---
        if direction == "LONG":
            stop_loss = current_price - atr * self.atr_multiplier_sl
            take_profit = current_price + atr * self.atr_multiplier_tp
        elif direction == "SHORT":
            stop_loss = current_price + atr * self.atr_multiplier_sl
            take_profit = current_price - atr * self.atr_multiplier_tp
        else:
            stop_loss = current_price
            take_profit = current_price

        # --- Razao risco/recompensa ---
        risk_distance = abs(current_price - stop_loss)
        reward_distance = abs(take_profit - current_price)
        if risk_distance > 0:
            risk_reward_ratio = reward_distance / risk_distance
        else:
            risk_reward_ratio = 0.0

        # --- Dimensionamento de posicao (Kelly simplificado) ---
        volatility = atr / (current_price + 1e-10)
        if direction == "HOLD" or volatility < 1e-10:
            position_size = 0.0
        else:
            kelly = avg_confidence * 2 * abs(predicted_return) / volatility
            position_size = float(np.clip(kelly, 0.0, self.max_position))

        reasons.append(f"Position size: {position_size:.4f}")
        reasons.append(f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | RR: {risk_reward_ratio:.2f}")

        signal = TradingSignal(
            coin=coin,
            timestamp=datetime.utcnow(),
            direction=direction,
            strength=strength,
            confidence=avg_confidence,
            predicted_return=predicted_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_reward_ratio=risk_reward_ratio,
            model_agreement=model_agreement,
            reasons=reasons,
        )

        logger.info(
            "Sinal gerado para %s: %s (forca=%.2f, confianca=%.2f, pos=%.4f)",
            coin,
            direction,
            strength,
            avg_confidence,
            position_size,
        )

        return signal

    def generate_portfolio_signals(
        self,
        all_predictions: dict[str, dict],
        prices: dict[str, float],
        atrs: dict[str, float],
    ) -> list[TradingSignal]:
        """Gera sinais para todas as moedas e aplica restricoes de portfolio.

        Gera sinais individuais, ranqueia por confianca * |retorno_previsto|
        e ajusta tamanhos de posicao para que a exposicao total nao exceda 50%.

        Args:
            all_predictions: Predicoes por moeda. Formato:
                ``{"BTC": {"xgboost": {"pred": 0.02, "conf": 0.7}, ...}, ...}``
            prices: Preco atual por moeda.
            atrs: ATR atual por moeda.

        Returns:
            Lista de TradingSignal ordenada por prioridade.
        """
        max_total_exposure = 0.5
        signals: list[TradingSignal] = []

        # Gerar sinal individual para cada moeda
        for coin, preds in all_predictions.items():
            if coin not in prices or coin not in atrs:
                logger.warning("Dados incompletos para %s, pulando.", coin)
                continue

            signal = self.generate_signal(
                predictions=preds,
                current_price=prices[coin],
                atr=atrs[coin],
                coin=coin,
            )
            signals.append(signal)

        # Ordenar por prioridade: confianca * |retorno_previsto|
        signals.sort(
            key=lambda s: s.confidence * abs(s.predicted_return),
            reverse=True,
        )

        # Ajustar posicoes para respeitar exposicao maxima
        total_exposure = 0.0
        for signal in signals:
            if signal.direction == "HOLD":
                continue

            remaining = max_total_exposure - total_exposure
            if remaining <= 0:
                signal.position_size = 0.0
                signal.direction = "HOLD"
                signal.reasons.append(
                    f"HOLD: exposicao total do portfolio atingiu {max_total_exposure:.0%}"
                )
            elif signal.position_size > remaining:
                old_size = signal.position_size
                signal.position_size = remaining
                signal.reasons.append(
                    f"Posicao reduzida de {old_size:.4f} para {remaining:.4f} "
                    f"(limite de portfolio)"
                )
                total_exposure += remaining
            else:
                total_exposure += signal.position_size

        logger.info(
            "Portfolio: %d sinais gerados, exposicao total=%.2f",
            len(signals),
            total_exposure,
        )

        return signals

    def apply_filters(
        self,
        signal: TradingSignal,
        market_regime: str | None = None,
        fear_greed: float | None = None,
        volatility_regime: str | None = None,
    ) -> TradingSignal:
        """Aplica filtros de regime de mercado ao sinal.

        Args:
            signal: Sinal de trading a ser filtrado.
            market_regime: Regime de mercado ('trending', 'ranging', etc.).
            fear_greed: Indice de medo/ganancia normalizado (0=medo extremo, 1=ganancia extrema).
            volatility_regime: Regime de volatilidade ('low', 'normal', 'high', 'extreme').

        Returns:
            Sinal ajustado com filtros aplicados.
        """
        # --- Filtro de medo/ganancia ---
        if fear_greed is not None:
            if fear_greed < 0.2:
                # Medo extremo: reduzir posicao em 50%
                signal.position_size *= 0.5
                signal.reasons.append(
                    f"Posicao reduzida 50%: medo extremo (fear_greed={fear_greed:.2f})"
                )
            elif fear_greed < 0.35:
                # Medo moderado: reduzir posicao em 25%
                signal.position_size *= 0.75
                signal.reasons.append(
                    f"Posicao reduzida 25%: medo moderado (fear_greed={fear_greed:.2f})"
                )

        # --- Filtro de regime de mercado ---
        if market_regime is not None:
            if market_regime == "trending":
                # Em tendencia, aumentar posicao em 20%
                new_size = min(signal.position_size * 1.2, self.max_position)
                if new_size != signal.position_size:
                    signal.position_size = new_size
                    signal.reasons.append(
                        f"Posicao aumentada: regime de tendencia ({market_regime})"
                    )
            elif market_regime == "ranging":
                # Em consolidacao, reduzir posicao em 30%
                signal.position_size *= 0.7
                signal.reasons.append(
                    f"Posicao reduzida 30%: mercado em consolidacao ({market_regime})"
                )

        # --- Filtro de regime de volatilidade ---
        if volatility_regime is not None:
            if volatility_regime == "extreme":
                # Volatilidade extrema: bloquear sinal
                signal.position_size = 0.0
                signal.direction = "HOLD"
                signal.reasons.append(
                    "HOLD: volatilidade extrema detectada, sinal bloqueado"
                )
            elif volatility_regime == "high":
                # Volatilidade alta: reduzir posicao em 40%
                signal.position_size *= 0.6
                signal.reasons.append(
                    "Posicao reduzida 40%: volatilidade alta"
                )

        logger.debug(
            "Filtros aplicados para %s: direcao=%s, posicao=%.4f",
            signal.coin,
            signal.direction,
            signal.position_size,
        )

        return signal

    def format_signal_report(self, signals: list[TradingSignal]) -> str:
        """Formata relatorio legivel dos sinais para o dashboard.

        Args:
            signals: Lista de sinais de trading.

        Returns:
            String formatada com resumo de todos os sinais.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("RELATORIO DE SINAIS DE TRADING")
        lines.append(f"Gerado em: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("=" * 70)

        active_signals = [s for s in signals if s.direction != "HOLD"]
        hold_signals = [s for s in signals if s.direction == "HOLD"]

        lines.append(f"\nSinais ativos: {len(active_signals)} | Em espera: {len(hold_signals)}")
        lines.append("-" * 70)

        for signal in signals:
            emoji_dir = {"LONG": "[LONG ]", "SHORT": "[SHORT]", "HOLD": "[HOLD ]"}
            lines.append(
                f"\n{emoji_dir.get(signal.direction, '[?]')} {signal.coin}"
            )
            lines.append(f"  Direcao:          {signal.direction}")
            lines.append(f"  Forca:            {signal.strength:.2%}")
            lines.append(f"  Confianca:        {signal.confidence:.2%}")
            lines.append(f"  Retorno previsto: {signal.predicted_return:+.4f}")
            lines.append(f"  Stop-loss:        {signal.stop_loss:.2f}")
            lines.append(f"  Take-profit:      {signal.take_profit:.2f}")
            lines.append(f"  Posicao:          {signal.position_size:.4f}")
            lines.append(f"  Risco/Recompensa: {signal.risk_reward_ratio:.2f}")
            lines.append(f"  Concordancia:     {signal.model_agreement:.0%}")
            if signal.reasons:
                lines.append("  Razoes:")
                for reason in signal.reasons:
                    lines.append(f"    - {reason}")

        total_exposure = sum(s.position_size for s in signals if s.direction != "HOLD")
        lines.append("\n" + "-" * 70)
        lines.append(f"Exposicao total do portfolio: {total_exposure:.2%}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dataframe(self, signals: list[TradingSignal]) -> pd.DataFrame:
        """Converte lista de sinais em DataFrame para logging e analise.

        Args:
            signals: Lista de sinais de trading.

        Returns:
            DataFrame com uma linha por sinal.
        """
        records = []
        for signal in signals:
            records.append(
                {
                    "coin": signal.coin,
                    "timestamp": signal.timestamp,
                    "direction": signal.direction,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "predicted_return": signal.predicted_return,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "position_size": signal.position_size,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "model_agreement": signal.model_agreement,
                    "reasons": "; ".join(signal.reasons),
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

        return df
