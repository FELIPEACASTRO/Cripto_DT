"""FinCoT — Structured Financial Chain-of-Thought Prompting.

Implementa prompting estruturado com blueprints de raciocinio financeiro,
baseado em arXiv:2506.16123 (FinNLP Workshop 2025).

Melhora acuracia de modelos LLM em tarefas financeiras:
- Qwen3-8B: 63.2% -> 80.5% (+17.3%)
- Reduz output em ate 8.9x vs CoT nao-estruturado

Cada template segue o padrao: Data-CoT -> Concept-CoT -> Thesis-CoT
(inspirado em FinRobot).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Snapshot do estado atual do mercado para um ativo."""

    coin: str = ""
    price: float = 0.0
    change_24h: float = 0.0
    volume_24h: float = 0.0
    rsi: float = 50.0
    macd_signal: str = "neutral"  # "bullish", "bearish", "neutral"
    bollinger_position: str = "middle"  # "upper", "lower", "middle"
    regime: str = "normal"  # "trending_up", "trending_down", "volatile", "normal"
    sentiment_score: float = 0.0  # -1 a 1
    fear_greed: int = 50  # 0-100
    volatility_level: str = "medium"  # "low", "medium", "high", "extreme"
    model_predictions: dict[str, float] = field(default_factory=dict)
    model_confidences: dict[str, float] = field(default_factory=dict)
    narrative: str = ""  # Narrativa dominante
    whale_activity: str = "normal"  # "accumulating", "distributing", "normal"


class FinCoTTemplates:
    """Templates de prompting estruturado para analise financeira cripto.

    Cada template gera um prompt que pode ser enviado a um LLM para obter
    analise estruturada, seguindo o padrao FinCoT:
    1. Data-CoT: Organizar e interpretar dados factuais
    2. Concept-CoT: Identificar padroes e conceitos relevantes
    3. Thesis-CoT: Formular tese de investimento
    """

    @staticmethod
    def technical_analysis(snapshot: MarketSnapshot) -> str:
        """Template FinCoT para analise tecnica estruturada."""
        preds = snapshot.model_predictions
        confs = snapshot.model_confidences

        # Calcular consenso dos modelos
        if preds:
            bullish = sum(1 for v in preds.values() if v > 0)
            bearish = sum(1 for v in preds.values() if v < 0)
            avg_pred = np.mean(list(preds.values()))
            avg_conf = np.mean(list(confs.values())) if confs else 0.5
        else:
            bullish = bearish = 0
            avg_pred = 0.0
            avg_conf = 0.5

        return f"""## FinCoT: Analise Tecnica Estruturada — {snapshot.coin}

### FASE 1: DATA-CoT (Dados Factuais)
- Preco atual: ${snapshot.price:,.2f}
- Variacao 24h: {snapshot.change_24h:+.2f}%
- Volume 24h: ${snapshot.volume_24h:,.0f}
- RSI(14): {snapshot.rsi:.1f} ({'SOBRECOMPRADO' if snapshot.rsi > 70 else 'SOBREVENDIDO' if snapshot.rsi < 30 else 'NEUTRO'})
- MACD: {snapshot.macd_signal.upper()}
- Bollinger: {snapshot.bollinger_position.upper()}
- Regime de mercado: {snapshot.regime.upper()}
- Volatilidade: {snapshot.volatility_level.upper()}
- Modelos bullish: {bullish}/{len(preds)} | Modelos bearish: {bearish}/{len(preds)}
- Predicao media: {avg_pred:+.6f} | Confianca media: {avg_conf:.2%}

### FASE 2: CONCEPT-CoT (Identificacao de Padroes)
Baseado nos dados acima, identifique:
1. TENDENCIA: Qual a direcao dominante? (momentum, MACD, modelos)
2. FORCA: Qual a forca do sinal? (RSI extremo? Consenso alto?)
3. RISCO: Quais sinais contraditorios existem? (divergencias, regime volatil?)
4. CATALISTA: Ha evento externo relevante? (narrativa: {snapshot.narrative})

### FASE 3: THESIS-CoT (Tese de Investimento)
Formate sua resposta EXATAMENTE assim:
- DIRECAO: [LONG/SHORT/HOLD]
- CONFIANCA: [0-100]%
- HORIZONTE: [1d/3d/1w]
- TESE: [1 frase concisa]
- STOP-LOSS: [% abaixo/acima do preco atual]
- TAKE-PROFIT: [% acima/abaixo do preco atual]
- RISCO PRINCIPAL: [1 frase]"""

    @staticmethod
    def sentiment_analysis(snapshot: MarketSnapshot) -> str:
        """Template FinCoT para analise de sentimento estruturada."""
        return f"""## FinCoT: Analise de Sentimento — {snapshot.coin}

### FASE 1: DATA-CoT (Dados de Sentimento)
- Fear & Greed Index: {snapshot.fear_greed}/100 ({'MEDO EXTREMO' if snapshot.fear_greed < 25 else 'MEDO' if snapshot.fear_greed < 45 else 'NEUTRO' if snapshot.fear_greed < 55 else 'GANANCIA' if snapshot.fear_greed < 75 else 'GANANCIA EXTREMA'})
- Sentiment Score: {snapshot.sentiment_score:+.2f} (-1=muito negativo, +1=muito positivo)
- Atividade de baleias: {snapshot.whale_activity.upper()}
- Narrativa dominante: {snapshot.narrative or 'Nenhuma identificada'}
- Regime de volatilidade: {snapshot.volatility_level.upper()}

### FASE 2: CONCEPT-CoT (Interpretacao de Sentimento)
Analise:
1. CONTRARIAN: O sentimento extremo sugere reversao? (medo extremo = compra? ganancia = venda?)
2. SMART MONEY: Baleias estao {snapshot.whale_activity}. Isso confirma ou contradiz o sentimento retail?
3. NARRATIVA: A narrativa "{snapshot.narrative}" esta em fase de hype ou declinio?
4. DIVERGENCIA: Ha divergencia entre sentimento e preco? (preco cai mas sentimento melhora?)

### FASE 3: THESIS-CoT (Conclusao de Sentimento)
- BIAS DE SENTIMENTO: [POSITIVO/NEGATIVO/NEUTRO]
- FORCA DO SINAL: [FRACO/MODERADO/FORTE]
- ACAO SUGERIDA: [1 frase]
- RISCO DE SENTIMENTO: [1 frase sobre risco de mudanca rapida de sentimento]"""

    @staticmethod
    def on_chain_analysis(snapshot: MarketSnapshot) -> str:
        """Template FinCoT para analise on-chain."""
        return f"""## FinCoT: Analise On-Chain — {snapshot.coin}

### FASE 1: DATA-CoT (Metricas On-Chain)
- Atividade de baleias: {snapshot.whale_activity.upper()}
- Regime de volume: {'ALTO' if snapshot.volume_24h > 0 else 'BAIXO'}
- Fear & Greed: {snapshot.fear_greed}/100

### FASE 2: CONCEPT-CoT (Interpretacao On-Chain)
1. ACUMULACAO/DISTRIBUICAO: Baleias estao em modo {snapshot.whale_activity}
2. VOLUME: Volume {'acima' if snapshot.volume_24h > 0 else 'abaixo'} da media sugere {'convicao' if snapshot.volume_24h > 0 else 'indecisao'}
3. REDE: Atividade de rede crescente/decrescente

### FASE 3: THESIS-CoT (Sinal On-Chain)
- SINAL ON-CHAIN: [ACUMULAR/DISTRIBUIR/NEUTRO]
- CONFIANCA: [0-100]%
- JUSTIFICATIVA: [1 frase]"""

    @staticmethod
    def macro_analysis(snapshot: MarketSnapshot) -> str:
        """Template FinCoT para analise macroeconomica."""
        return f"""## FinCoT: Analise Macro — {snapshot.coin}

### FASE 1: DATA-CoT (Contexto Macro)
- Regime de mercado: {snapshot.regime.upper()}
- Volatilidade global: {snapshot.volatility_level.upper()}
- Fear & Greed: {snapshot.fear_greed}/100
- Preco: ${snapshot.price:,.2f} ({snapshot.change_24h:+.2f}% 24h)

### FASE 2: CONCEPT-CoT (Contexto Macro)
1. CORRELACAO MACRO: Crypto esta correlacionado com risk-on/risk-off?
2. LIQUIDEZ: Ambiente de liquidez favoravel ou restritivo?
3. REGULACAO: Ha pressao regulatoria iminente?
4. CICLO: Estamos em que fase do ciclo crypto? (acumulacao/markup/distribuicao/markdown)

### FASE 3: THESIS-CoT (Tese Macro)
- BIAS MACRO: [FAVORAVEL/DESFAVORAVEL/NEUTRO]
- HORIZONTE: [CURTO/MEDIO/LONGO PRAZO]
- FATOR PRINCIPAL: [1 frase sobre o driver macro dominante]"""

    @staticmethod
    def multi_model_consensus(snapshot: MarketSnapshot) -> str:
        """Template FinCoT para consenso entre multiplos modelos."""
        preds = snapshot.model_predictions
        confs = snapshot.model_confidences

        model_lines = []
        for name, pred in sorted(preds.items(), key=lambda x: abs(x[1]), reverse=True):
            conf = confs.get(name, 0.5)
            direction = "LONG" if pred > 0 else "SHORT" if pred < 0 else "NEUTRO"
            model_lines.append(f"  - {name}: {direction} ({pred:+.6f}, conf={conf:.2%})")

        models_text = "\n".join(model_lines) if model_lines else "  Nenhum modelo disponivel"

        return f"""## FinCoT: Consenso Multi-Modelo — {snapshot.coin}

### FASE 1: DATA-CoT (Predicoes dos Modelos)
{models_text}

### FASE 2: CONCEPT-CoT (Analise de Consenso)
1. UNANIMIDADE: Todos os modelos concordam na direcao?
2. CONFIANCA: Os modelos mais confiantes concordam?
3. DIVERGENCIA: Quais modelos discordam? Por que? (modelos de regime vs tendencia?)
4. TIPO DE MODELO: Modelos tree-based vs deep learning concordam?

### FASE 3: THESIS-CoT (Decisao Final)
- DIRECAO CONSENSO: [LONG/SHORT/HOLD]
- FORCA DO CONSENSO: [FORTE/MODERADO/FRACO]
- TAMANHO DA POSICAO: [FULL/HALF/QUARTER/ZERO] baseado na forca
- MODELOS CHAVE: [Quais 2-3 modelos tiveram mais peso na decisao]"""


class FinCoTAnalyzer:
    """Analisa dados de mercado usando templates FinCoT.

    Gera analises estruturadas que podem ser:
    1. Enviadas a um LLM para processamento
    2. Parseadas programaticamente para sinais automaticos
    3. Exibidas para o trader humano tomar decisoes
    """

    def __init__(self):
        self.templates = FinCoTTemplates()

    def create_snapshot(
        self,
        coin: str,
        df: pd.DataFrame,
        model_predictions: dict[str, float] | None = None,
        model_confidences: dict[str, float] | None = None,
    ) -> MarketSnapshot:
        """Cria MarketSnapshot a partir de um DataFrame de features.

        Args:
            coin: Simbolo da moeda (ex: "BTC")
            df: DataFrame com features calculadas (ultima linha = estado atual)
            model_predictions: Dict nome_modelo -> predicao
            model_confidences: Dict nome_modelo -> confianca

        Returns:
            MarketSnapshot preenchido
        """
        if df.empty:
            return MarketSnapshot(coin=coin)

        last = df.iloc[-1]
        snapshot = MarketSnapshot(coin=coin)

        # Preco e variacao
        snapshot.price = float(last.get("close", 0))
        if "pct_return" in df.columns:
            snapshot.change_24h = float(last.get("pct_return", 0)) * 100

        # Volume
        snapshot.volume_24h = float(last.get("volume", 0))

        # RSI
        rsi_col = [c for c in df.columns if "rsi" in c.lower()]
        if rsi_col:
            snapshot.rsi = float(last.get(rsi_col[0], 50))

        # MACD
        macd_cols = [c for c in df.columns if "macd" in c.lower()]
        if len(macd_cols) >= 2:
            macd_val = float(last.get(macd_cols[0], 0))
            snapshot.macd_signal = "bullish" if macd_val > 0 else "bearish"

        # Bollinger
        bb_cols = [c for c in df.columns if "bb" in c.lower() or "bollinger" in c.lower()]
        if bb_cols:
            snapshot.bollinger_position = "middle"  # Simplificado

        # Regime
        regime_cols = [c for c in df.columns if "regime" in c.lower()]
        if regime_cols:
            regime_val = int(last.get(regime_cols[0], 0))
            regime_map = {0: "normal", 1: "trending_up", 2: "volatile"}
            snapshot.regime = regime_map.get(regime_val, "normal")

        # Volatilidade
        vol_cols = [c for c in df.columns if "volatility" in c.lower() or "garch" in c.lower()]
        if vol_cols:
            vol_val = float(last.get(vol_cols[0], 0))
            if vol_val > 0.05:
                snapshot.volatility_level = "extreme"
            elif vol_val > 0.03:
                snapshot.volatility_level = "high"
            elif vol_val > 0.01:
                snapshot.volatility_level = "medium"
            else:
                snapshot.volatility_level = "low"

        # Sentiment
        sent_cols = [c for c in df.columns if "sentiment" in c.lower()]
        if sent_cols:
            snapshot.sentiment_score = float(last.get(sent_cols[0], 0))

        # Fear & Greed
        fg_cols = [c for c in df.columns if "fear" in c.lower() or "greed" in c.lower()]
        if fg_cols:
            snapshot.fear_greed = int(last.get(fg_cols[0], 50))

        # Whale activity
        whale_cols = [c for c in df.columns if "whale" in c.lower()]
        if whale_cols:
            whale_val = float(last.get(whale_cols[0], 0))
            if whale_val > 0.5:
                snapshot.whale_activity = "accumulating"
            elif whale_val < -0.5:
                snapshot.whale_activity = "distributing"

        # Model predictions
        snapshot.model_predictions = model_predictions or {}
        snapshot.model_confidences = model_confidences or {}

        return snapshot

    def generate_full_analysis(
        self,
        coin: str,
        df: pd.DataFrame,
        model_predictions: dict[str, float] | None = None,
        model_confidences: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Gera analise completa FinCoT para uma moeda.

        Returns:
            Dict com chaves: technical, sentiment, onchain, macro, consensus
        """
        snapshot = self.create_snapshot(
            coin, df, model_predictions, model_confidences
        )

        analyses = {
            "technical": self.templates.technical_analysis(snapshot),
            "sentiment": self.templates.sentiment_analysis(snapshot),
            "onchain": self.templates.on_chain_analysis(snapshot),
            "macro": self.templates.macro_analysis(snapshot),
            "consensus": self.templates.multi_model_consensus(snapshot),
        }

        logger.info(
            f"FinCoT: analise completa gerada para {coin} "
            f"({len(model_predictions or {})} modelos)"
        )

        return analyses

    def generate_trading_signal(
        self,
        snapshot: MarketSnapshot,
    ) -> dict[str, Any]:
        """Gera sinal de trading programatico baseado em FinCoT logic.

        Aplica a logica FinCoT diretamente sem LLM, usando regras derivadas
        dos templates.

        Returns:
            Dict com direction, confidence, thesis, stop_loss, take_profit
        """
        preds = snapshot.model_predictions
        confs = snapshot.model_confidences

        if not preds:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "thesis": "Sem predicoes de modelos disponiveis",
                "stop_loss": 0.0,
                "take_profit": 0.0,
            }

        # Consenso ponderado por confianca
        weighted_sum = 0.0
        weight_total = 0.0
        for name, pred in preds.items():
            conf = confs.get(name, 0.5)
            weighted_sum += pred * conf
            weight_total += conf

        avg_weighted_pred = weighted_sum / max(weight_total, 1e-8)

        # Direcao
        bullish_count = sum(1 for v in preds.values() if v > 0)
        agreement = max(bullish_count, len(preds) - bullish_count) / len(preds)

        # Ajuste por sentimento contrarian
        sentiment_adj = 0.0
        if snapshot.fear_greed < 25:  # Medo extremo -> potencial compra
            sentiment_adj = 0.1
        elif snapshot.fear_greed > 75:  # Ganancia extrema -> potencial venda
            sentiment_adj = -0.1

        # Ajuste por RSI
        rsi_adj = 0.0
        if snapshot.rsi > 70:
            rsi_adj = -0.05  # Sobrecomprado
        elif snapshot.rsi < 30:
            rsi_adj = 0.05  # Sobrevendido

        final_signal = avg_weighted_pred + sentiment_adj + rsi_adj

        # Confianca = agreement * media de confiancas
        avg_conf = np.mean(list(confs.values())) if confs else 0.5
        final_confidence = agreement * avg_conf

        # Direcao
        if final_signal > 0.001 and final_confidence > 0.55:
            direction = "LONG"
        elif final_signal < -0.001 and final_confidence > 0.55:
            direction = "SHORT"
        else:
            direction = "HOLD"

        # Stop-loss/Take-profit baseados em volatilidade
        vol_mult = {"low": 1.0, "medium": 1.5, "high": 2.0, "extreme": 3.0}
        base_sl = 0.02 * vol_mult.get(snapshot.volatility_level, 1.5)
        base_tp = 0.03 * vol_mult.get(snapshot.volatility_level, 1.5)

        # Tese
        thesis_parts = []
        if direction == "LONG":
            thesis_parts.append(f"{bullish_count}/{len(preds)} modelos bullish")
        elif direction == "SHORT":
            thesis_parts.append(f"{len(preds) - bullish_count}/{len(preds)} modelos bearish")

        if snapshot.fear_greed < 30:
            thesis_parts.append("medo extremo (contrarian bullish)")
        elif snapshot.fear_greed > 70:
            thesis_parts.append("ganancia extrema (contrarian bearish)")

        if snapshot.regime != "normal":
            thesis_parts.append(f"regime {snapshot.regime}")

        thesis = "; ".join(thesis_parts) if thesis_parts else "Sinais mistos"

        return {
            "direction": direction,
            "confidence": float(final_confidence),
            "signal_strength": float(abs(final_signal)),
            "thesis": thesis,
            "stop_loss": float(base_sl),
            "take_profit": float(base_tp),
            "model_agreement": float(agreement),
            "n_models": len(preds),
        }
