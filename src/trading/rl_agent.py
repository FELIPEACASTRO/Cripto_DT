"""Agente PPO para trading e gerador de sinais combinados."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import gymnasium as gym

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

logger = logging.getLogger(__name__)

# Configuracao padrao do PPO
DEFAULT_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 0,
}


class PPOTrader:
    """Wrapper de agente PPO para trading de criptomoedas.

    Utiliza stable_baselines3 para treinar um agente PPO em
    um ambiente Gymnasium de trading.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Inicializa o trader PPO.

        Args:
            config: Dicionario de hiperparametros. Chaves suportadas:
                learning_rate, n_steps, batch_size, n_epochs, gamma,
                clip_range, ent_coef, vf_coef, max_grad_norm, verbose.
        """
        if not HAS_SB3:
            raise ImportError(
                "stable_baselines3 e necessario para PPOTrader. "
                "Instale com: pip install stable-baselines3"
            )
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium e necessario para PPOTrader. "
                "Instale com: pip install gymnasium"
            )

        self.config = {**DEFAULT_PPO_CONFIG, **(config or {})}
        self.model: PPO | None = None
        self._trained = False

    def train(
        self,
        env: gym.Env,
        total_timesteps: int = 100_000,
        eval_env: gym.Env | None = None,
        log_dir: str | None = None,
    ) -> "PPOTrader":
        """Treina o agente PPO no ambiente fornecido.

        Args:
            env: Ambiente Gymnasium de trading para treinamento.
            total_timesteps: Numero total de timesteps de treinamento.
            eval_env: Ambiente opcional para avaliacao durante treinamento.
            log_dir: Diretorio para logs do TensorBoard.

        Returns:
            self para encadeamento.
        """
        logger.info(
            "Iniciando treinamento PPO: %d timesteps, config=%s",
            total_timesteps,
            {k: v for k, v in self.config.items() if k != "verbose"},
        )

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            verbose=self.config["verbose"],
            tensorboard_log=log_dir,
            seed=42,
        )

        callbacks = []
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_dir or "./logs/best_model",
                eval_freq=max(total_timesteps // 20, 1000),
                n_eval_episodes=3,
                deterministic=True,
                verbose=0,
            )
            callbacks.append(eval_callback)

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks if callbacks else None,
                progress_bar=False,
            )
            self._trained = True
            logger.info("Treinamento PPO concluido com sucesso.")
        except Exception:
            logger.exception("Falha no treinamento PPO.")
            raise

        return self

    def predict(self, obs: np.ndarray) -> tuple[float, float]:
        """Prediz acao e confianca para uma observacao.

        Args:
            obs: Vetor de observacao do ambiente.

        Returns:
            Tupla (action, confidence) onde:
              - action: posicao alvo em [-1, 1]
              - confidence: confianca da acao (baseada na entropia da politica)
        """
        self._check_trained()

        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action, _states = self.model.predict(obs, deterministic=True)
        action_val = float(action.flatten()[0])

        # Estimar confianca via variancia da distribuicao da politica
        try:
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_std = float(
                distribution.distribution.stddev.detach().cpu().numpy().mean()
            )
            # Confianca inversamente proporcional ao desvio padrao
            # std baixo -> alta confianca
            confidence = float(np.clip(1.0 / (1.0 + action_std), 0.0, 1.0))
        except Exception:
            # Fallback: confianca baseada na magnitude da acao
            confidence = float(np.clip(abs(action_val), 0.0, 1.0))
            logger.debug("Usando confianca fallback baseada em magnitude da acao.")

        return action_val, confidence

    def backtest(self, env: gym.Env) -> dict[str, Any]:
        """Executa backtest do agente treinado no ambiente.

        Args:
            env: Ambiente Gymnasium de trading.

        Returns:
            Dicionario com metricas:
              - total_return: retorno total (fracao)
              - sharpe: Sharpe ratio anualizado
              - max_drawdown: drawdown maximo
              - win_rate: proporcao de trades lucrativos
              - n_trades: numero total de trades
              - portfolio_values: lista de valores do portfolio
              - actions: lista de acoes tomadas
        """
        self._check_trained()

        obs, info = env.reset()
        done = False
        actions = []
        portfolio_values = [info.get("balance", 10_000)]

        while not done:
            action_val, confidence = self.predict(obs)
            action = np.array([action_val], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            actions.append(action_val)
            portfolio_values.append(info.get("balance", portfolio_values[-1]))

        # Calcular metricas
        portfolio_arr = np.array(portfolio_values)
        returns = np.diff(portfolio_arr) / (portfolio_arr[:-1] + 1e-10)

        total_return = (portfolio_arr[-1] / portfolio_arr[0]) - 1.0
        sharpe = self._compute_sharpe(returns)
        max_drawdown = self._compute_max_drawdown(portfolio_arr)
        n_trades = info.get("n_trades", 0)

        # Win rate baseada em trades individuais
        if len(returns) > 0:
            win_rate = float((returns > 0).sum() / len(returns))
        else:
            win_rate = 0.0

        metrics = {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_trades": int(n_trades),
            "portfolio_values": portfolio_values,
            "actions": actions,
        }

        logger.info(
            "Backtest concluido: return=%.2f%%, sharpe=%.3f, maxDD=%.2f%%, "
            "win_rate=%.1f%%, trades=%d",
            total_return * 100,
            sharpe,
            max_drawdown * 100,
            win_rate * 100,
            n_trades,
        )

        return metrics

    @staticmethod
    def _compute_sharpe(
        returns: np.ndarray, periods_per_year: float = 365.0
    ) -> float:
        """Calcula Sharpe ratio anualizado."""
        if len(returns) < 2 or returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    @staticmethod
    def _compute_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calcula drawdown maximo."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-10)
        return float(drawdown.max()) if len(drawdown) > 0 else 0.0

    def save(self, path: str | Path) -> None:
        """Salva o modelo treinado.

        Args:
            path: Caminho do arquivo (sem extensao, SB3 adiciona .zip).
        """
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("PPOTrader salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path, config: dict[str, Any] | None = None) -> "PPOTrader":
        """Carrega modelo PPO salvo.

        Args:
            path: Caminho do arquivo salvo.
            config: Configuracao opcional (sobrescreve defaults).

        Returns:
            Instancia de PPOTrader com modelo carregado.
        """
        if not HAS_SB3:
            raise ImportError(
                "stable_baselines3 e necessario. pip install stable-baselines3"
            )

        trader = cls(config=config)
        trader.model = PPO.load(str(path))
        trader._trained = True
        logger.info("PPOTrader carregado de %s", path)
        return trader

    def _check_trained(self) -> None:
        """Verifica se o modelo foi treinado."""
        if not self._trained or self.model is None:
            raise RuntimeError(
                "PPOTrader nao foi treinado. Chame train() ou load() primeiro."
            )


class TradingSignalGenerator:
    """Gera sinais de trading combinando predicoes ML com decisoes RL.

    Combina:
      - Predicoes do modelo ML (retorno previsto, probabilidade)
      - Decisoes do agente RL (posicao, confianca)
    Para gerar sinais unificados de buy/sell/hold.
    """

    def __init__(
        self,
        predictor_results: dict[str, Any],
        rl_agent: PPOTrader,
        ml_weight: float = 0.5,
        signal_threshold: float = 0.3,
    ):
        """Inicializa o gerador de sinais.

        Args:
            predictor_results: Resultados do preditor ML. Espera chaves:
                'predictions' (array de retornos previstos),
                'probabilities' (array de probabilidades, opcional).
            rl_agent: Agente PPO treinado.
            ml_weight: Peso das predicoes ML vs RL (0-1). Default 0.5.
            signal_threshold: Limiar para gerar sinal buy/sell. Default 0.3.
        """
        self.predictor_results = predictor_results
        self.rl_agent = rl_agent
        self.ml_weight = np.clip(ml_weight, 0.0, 1.0)
        self.rl_weight = 1.0 - self.ml_weight
        self.signal_threshold = signal_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais de trading combinando ML e RL.

        Args:
            df: DataFrame com features do mercado.

        Returns:
            DataFrame com colunas adicionais:
              - signal: 'buy', 'sell', ou 'hold'
              - position_size: tamanho da posicao recomendada [0, 1]
              - confidence: confianca do sinal [0, 1]
              - predicted_return: retorno previsto combinado
        """
        df = df.copy()

        predictions = self.predictor_results.get("predictions", np.array([]))
        probabilities = self.predictor_results.get("probabilities", None)

        n = len(df)

        # ML component
        if len(predictions) >= n:
            ml_predictions = np.array(predictions[:n], dtype=float)
        elif len(predictions) > 0:
            # Pad com zeros se predicoes insuficientes
            ml_predictions = np.zeros(n)
            ml_predictions[: len(predictions)] = predictions
            logger.warning(
                "Predicoes ML (%d) menores que DataFrame (%d). "
                "Preenchendo com zeros.",
                len(predictions),
                n,
            )
        else:
            ml_predictions = np.zeros(n)

        if probabilities is not None and len(probabilities) >= n:
            ml_confidence = np.array(probabilities[:n], dtype=float)
        else:
            # Confianca baseada na magnitude da predicao
            ml_confidence = np.clip(np.abs(ml_predictions) * 5, 0, 1)

        # RL component: usar predicoes ponto a ponto se agente estiver treinado
        rl_actions = np.zeros(n)
        rl_confidences = np.zeros(n)

        try:
            # Criar observacoes simplificadas para o agente RL
            for i in range(n):
                row = df.iloc[i]
                # Construir observacao basica (features numericas)
                numeric_vals = row.select_dtypes(include=[np.number]).values
                obs = np.array(numeric_vals, dtype=np.float32)

                # Adicionar estado do portfolio (neutro como default)
                obs = np.concatenate([obs, [0.0, 1.0, 0.0]])  # position, balance_pct, pnl

                action, conf = self.rl_agent.predict(obs)
                rl_actions[i] = action
                rl_confidences[i] = conf
        except Exception:
            logger.warning(
                "Falha ao gerar predicoes RL. Usando apenas ML.",
                exc_info=True,
            )
            rl_actions = np.zeros(n)
            rl_confidences = np.zeros(n)

        # Combinar sinais
        combined_signal = (
            self.ml_weight * ml_predictions + self.rl_weight * rl_actions
        )
        combined_confidence = (
            self.ml_weight * ml_confidence + self.rl_weight * rl_confidences
        )

        # Gerar sinais discretos
        signals = []
        for val in combined_signal:
            if val > self.signal_threshold:
                signals.append("buy")
            elif val < -self.signal_threshold:
                signals.append("sell")
            else:
                signals.append("hold")

        df["signal"] = signals
        df["position_size"] = np.clip(np.abs(combined_signal), 0.0, 1.0)
        df["confidence"] = np.clip(combined_confidence, 0.0, 1.0)
        df["predicted_return"] = combined_signal

        logger.info(
            "Sinais gerados: %d buy, %d sell, %d hold",
            signals.count("buy"),
            signals.count("sell"),
            signals.count("hold"),
        )

        return df
