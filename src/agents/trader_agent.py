"""TraderAgent: gera sinais e recomendacoes de trading baseados em modelos ML."""

import logging

from src.agents.base_agent import BaseAgent, AgentResult, _safe_import

logger = logging.getLogger(__name__)


class TraderAgent(BaseAgent):
    """Agente responsavel por gerar sinais de trading.

    Coordena:
    - Predictor para obter previsoes dos modelos ML
    - SignalGenerator para converter previsoes em sinais acionaveis
    - RegimeDetector para filtrar sinais por regime de mercado

    Recebe dados enriquecidos do AnalystAgent e retorna sinais
    com direcao, forca, confianca e razoes explicativas.
    """

    name = "trader_agent"

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Componentes inicializados sob demanda
        self._predictor = None
        self._signal_generator = None
        self._regime_detector = None
        self._initialized = False

    def _init_components(self):
        """Inicializa componentes de trading com importacoes seguras."""
        if self._initialized:
            return

        # Predictor - previsoes de modelos ML (componente principal)
        Cls = _safe_import("src.prediction.predictor", "Predictor")
        if Cls:
            try:
                self._predictor = Cls(self.config) if self.config else Cls()
                logger.info(f"[{self.name}] Predictor: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] Predictor: ERRO - {e}")

        # SignalGenerator - converte previsoes em sinais de trading
        Cls = _safe_import("src.trading.signal_generator", "SignalGenerator")
        if Cls:
            try:
                self._signal_generator = Cls()
                logger.info(f"[{self.name}] SignalGenerator: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] SignalGenerator: ERRO - {e}")

        # RegimeDetector - deteccao de regime de mercado para filtros
        Cls = _safe_import("src.features.regime", "RegimeDetector")
        if Cls:
            try:
                self._regime_detector = Cls()
                logger.info(f"[{self.name}] RegimeDetector: OK")
            except Exception as e:
                logger.debug(f"[{self.name}] RegimeDetector: {e}")

        self._initialized = True

    def _detect_regime(self, df, coin: str) -> dict | None:
        """Detecta regime de mercado atual para uma moeda.

        Returns:
            Dicionario com regime detectado ou None se nao disponivel.
        """
        if self._regime_detector is None:
            return None

        try:
            if hasattr(self._regime_detector, "detect"):
                return self._regime_detector.detect(df)
            elif hasattr(self._regime_detector, "fit_predict"):
                regimes = self._regime_detector.fit_predict(df)
                if regimes is not None and len(regimes) > 0:
                    current = int(regimes.iloc[-1]) if hasattr(regimes, "iloc") else int(regimes[-1])
                    return {"current_regime": current, "regimes": regimes}
        except Exception as e:
            logger.debug(f"[{self.name}] Erro deteccao regime {coin}: {e}")

        return None

    def _apply_regime_filter(self, signal: dict, regime: dict | None) -> dict:
        """Aplica filtros baseados no regime de mercado ao sinal.

        Em regimes de alta volatilidade ou bear market, reduz exposicao.
        Em regimes favoraveis, mantem ou aumenta confianca.
        """
        if regime is None:
            return signal

        current = regime.get("current_regime")
        if current is None:
            return signal

        # Regime 0 = baixa volatilidade (favoravel)
        # Regime 1 = alta volatilidade (cautela)
        # Regime 2 = crise/bear (maximo cuidado)
        regime_multipliers = {
            0: 1.0,   # Regime normal - sem alteracao
            1: 0.7,   # Volatilidade alta - reduz posicao
            2: 0.3,   # Crise - reduz drasticamente
        }

        multiplier = regime_multipliers.get(current, 0.5)

        if "position_size" in signal:
            signal["position_size"] *= multiplier
        if "strength" in signal:
            signal["strength"] *= multiplier

        signal.setdefault("regime_info", {})
        signal["regime_info"] = {
            "regime": current,
            "multiplier": multiplier,
        }

        return signal

    def execute(self, context: dict) -> AgentResult:
        """Gera previsoes e sinais de trading.

        Args:
            context: Deve conter (normalmente fornecido pelo AnalystAgent):
                - coins (list[str]): Lista de moedas
                - featured_data (dict[str, DataFrame]): DataFrames com features
                - ohlcv (dict[str, DataFrame]): Dados OHLCV originais (fallback)

        Returns:
            AgentResult com data contendo:
                - predictions (dict[str, dict]): Previsoes por moeda
                - signals (dict[str, dict]): Sinais de trading por moeda
                - regimes (dict[str, dict]): Regime de mercado por moeda
        """
        self._init_components()

        coins = context.get("coins", [])
        featured_data = context.get("featured_data", {})
        ohlcv = context.get("ohlcv", {})

        result = AgentResult()
        result.data = {
            "predictions": {},
            "signals": {},
            "regimes": {},
        }

        # --- Passo 1: Detectar regime de mercado ---
        for coin in coins:
            df = featured_data.get(coin) or ohlcv.get(coin)
            if df is not None:
                regime = self._detect_regime(df, coin)
                if regime is not None:
                    result.data["regimes"][coin] = regime

        if result.data["regimes"]:
            logger.info(
                f"[{self.name}] Regimes detectados para "
                f"{len(result.data['regimes'])} moedas"
            )

        # --- Passo 2: Gerar previsoes ---
        if self._predictor is not None:
            for coin in coins:
                try:
                    df = featured_data.get(coin) or ohlcv.get(coin)
                    if df is not None:
                        pred = self._predictor.predict_coin(coin, df=df)
                        if pred is not None:
                            result.data["predictions"][coin] = pred
                except Exception as e:
                    msg = f"Erro previsao {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)

            logger.info(
                f"[{self.name}] Previsoes geradas para "
                f"{len(result.data['predictions'])} moedas"
            )
        else:
            result.errors.append("Predictor nao disponivel")
            logger.error(f"[{self.name}] Predictor nao disponivel")

        # --- Passo 3: Gerar sinais de trading ---
        if self._signal_generator is not None and result.data["predictions"]:
            for coin, pred in result.data["predictions"].items():
                try:
                    signal = self._signal_generator.generate(pred)

                    # Converter TradingSignal para dict se necessario
                    if hasattr(signal, "__dict__") and not isinstance(signal, dict):
                        signal_dict = {
                            k: v for k, v in signal.__dict__.items()
                            if not k.startswith("_")
                        }
                    elif isinstance(signal, dict):
                        signal_dict = signal
                    else:
                        signal_dict = {"raw_signal": str(signal)}

                    # Aplicar filtro de regime
                    regime = result.data["regimes"].get(coin)
                    signal_dict = self._apply_regime_filter(signal_dict, regime)

                    result.data["signals"][coin] = signal_dict
                except Exception as e:
                    msg = f"Erro sinal {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)

            logger.info(
                f"[{self.name}] Sinais gerados para "
                f"{len(result.data['signals'])} moedas"
            )
        elif not self._signal_generator:
            logger.debug(f"[{self.name}] SignalGenerator nao disponivel")

        # Verifica se ao menos temos previsoes
        if not result.data["predictions"]:
            result.success = False
            result.errors.append("Nenhuma previsao gerada")

        # Resumo
        result.metadata["coins_predicted"] = list(result.data["predictions"].keys())
        result.metadata["coins_with_signals"] = list(result.data["signals"].keys())

        return result
