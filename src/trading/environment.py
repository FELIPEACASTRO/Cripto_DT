"""Ambiente de trading compativel com Gymnasium para criptomoedas."""

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

    # Stub para permitir importacao sem gymnasium
    class _EnvStub:
        """Stub do gymnasium.Env para quando gymnasium nao esta instalado."""

        metadata: dict = {}

        def step(self, action):
            raise NotImplementedError

        def reset(self, **kwargs):
            raise NotImplementedError

        def render(self):
            raise NotImplementedError

    class _spaces_stub:
        @staticmethod
        def Box(*args, **kwargs):
            return None

    class _gym_stub:
        Env = _EnvStub

    gym = _gym_stub  # type: ignore[assignment]
    spaces = _spaces_stub  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class CryptoTradingEnv(gym.Env):
    """Ambiente Gymnasium para simulacao de trading de criptomoedas.

    Observacao: features do mercado + posicao atual + saldo percentual + PnL nao realizado.
    Acao: valor continuo em [-1, 1] representando posicao alvo.
      -1 = full short, 0 = neutro, 1 = full long.
    Recompensa: differential Sharpe ratio (atualizacao incremental).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        max_position: float = 1.0,
        render_mode: str | None = None,
    ):
        """Inicializa o ambiente de trading.

        Args:
            df: DataFrame com dados de mercado. Deve conter 'close' e as colunas em features.
            features: Lista de nomes de colunas a usar como observacao.
            initial_balance: Saldo inicial em USD.
            commission: Taxa de comissao por trade (frac, ex: 0.001 = 0.1%).
            max_position: Tamanho maximo da posicao (1.0 = 100% do saldo).
            render_mode: Modo de renderizacao (None ou 'human').
        """
        super().__init__()

        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium e necessario para CryptoTradingEnv. "
                "Instale com: pip install gymnasium"
            )

        self.df = df.reset_index(drop=True)
        self.features = features
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.render_mode = render_mode

        # Validar colunas
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing}")
        if "close" not in df.columns:
            raise ValueError("DataFrame precisa da coluna 'close'.")

        self._prices = self.df["close"].values.astype(np.float64)
        self._feature_data = self.df[features].values.astype(np.float32)

        # Preencher NaN nas features com 0
        self._feature_data = np.nan_to_num(self._feature_data, nan=0.0)

        n_features = len(features)
        n_extra = 3  # position, balance_pct, unrealized_pnl

        # Observation: features + [position, balance_pct, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + n_extra,),
            dtype=np.float32,
        )

        # Action: posicao alvo continua [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Estado interno (inicializado em reset)
        self._current_step: int = 0
        self._balance: float = initial_balance
        self._position: float = 0.0  # fracao do saldo (-1 a 1)
        self._entry_price: float = 0.0
        self._trades: list[dict] = []
        self._returns_history: list[float] = []
        self._portfolio_values: list[float] = []

        # Differential Sharpe ratio state
        self._A: float = 0.0  # media exponencial dos retornos
        self._B: float = 0.0  # media exponencial dos retornos^2
        self._eta: float = 0.01  # taxa de adaptacao

        self._max_steps = len(self.df) - 1

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reinicia o ambiente para o inicio do episodio.

        Returns:
            Tupla (observacao_inicial, info).
        """
        super().reset(seed=seed)

        self._current_step = 0
        self._balance = self.initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._trades = []
        self._returns_history = []
        self._portfolio_values = [self.initial_balance]

        self._A = 0.0
        self._B = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executa uma acao no ambiente.

        Args:
            action: Array shape (1,) com posicao alvo em [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        target_position = float(np.clip(action[0], -1.0, 1.0))
        target_position *= self.max_position

        prev_price = self._prices[self._current_step]
        self._current_step += 1
        current_price = self._prices[self._current_step]

        # Calcular retorno do passo
        price_return = (current_price - prev_price) / (prev_price + 1e-10)

        # Custo de transacao pela mudanca de posicao
        position_change = abs(target_position - self._position)
        transaction_cost = position_change * self.commission

        # Retorno do portfolio: posicao * retorno_preco - custo
        step_return = self._position * price_return - transaction_cost

        # Registrar trade se houve mudanca significativa de posicao
        if position_change > 0.01:
            self._trades.append(
                {
                    "step": self._current_step,
                    "price": current_price,
                    "old_position": self._position,
                    "new_position": target_position,
                    "cost": transaction_cost * self._balance,
                }
            )

        # Atualizar posicao e saldo
        self._position = target_position
        self._balance *= 1.0 + step_return
        self._balance = max(self._balance, 0.0)  # Nao permitir saldo negativo

        if abs(self._position) > 0.01:
            self._entry_price = current_price

        self._returns_history.append(step_return)
        self._portfolio_values.append(self._balance)

        # Calcular recompensa: differential Sharpe ratio
        reward = self._compute_differential_sharpe(step_return)

        # Verificar terminacao
        terminated = self._balance <= 0.0  # Falencia
        truncated = self._current_step >= self._max_steps

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_differential_sharpe(self, ret: float) -> float:
        """Calcula differential Sharpe ratio (Moody & Saffell, 2001).

        Atualizacao incremental do Sharpe ratio que serve como
        sinal de recompensa instantaneo.

        Args:
            ret: Retorno do passo atual.

        Returns:
            Recompensa baseada na atualizacao incremental do Sharpe.
        """
        delta_A = ret - self._A
        delta_B = ret**2 - self._B

        # Atualizar medias exponenciais
        self._A += self._eta * delta_A
        self._B += self._eta * delta_B

        # Differential Sharpe
        denominator = (self._B - self._A**2) ** 0.5
        if denominator < 1e-8:
            return 0.0

        dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (
            denominator**3 + 1e-10
        )

        return float(np.clip(dsr, -10.0, 10.0))

    def _get_observation(self) -> np.ndarray:
        """Constroi vetor de observacao."""
        features = self._feature_data[self._current_step]

        # Estado do portfolio
        balance_pct = self._balance / (self.initial_balance + 1e-10)
        current_price = self._prices[self._current_step]

        if abs(self._position) > 0.01 and self._entry_price > 0:
            unrealized_pnl = (
                self._position
                * (current_price - self._entry_price)
                / (self._entry_price + 1e-10)
            )
        else:
            unrealized_pnl = 0.0

        extra = np.array(
            [self._position, balance_pct, unrealized_pnl], dtype=np.float32
        )

        obs = np.concatenate([features, extra]).astype(np.float32)
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Retorna informacoes auxiliares do passo atual."""
        returns_arr = np.array(self._returns_history) if self._returns_history else np.array([0.0])
        portfolio_arr = np.array(self._portfolio_values)

        # Drawdown
        peak = np.maximum.accumulate(portfolio_arr)
        drawdown = (peak - portfolio_arr) / (peak + 1e-10)

        return {
            "step": self._current_step,
            "balance": self._balance,
            "position": self._position,
            "n_trades": len(self._trades),
            "total_return": (self._balance / self.initial_balance) - 1.0,
            "max_drawdown": float(drawdown.max()) if len(drawdown) > 0 else 0.0,
            "sharpe": self._compute_sharpe(returns_arr),
        }

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, periods_per_year: float = 365.0) -> float:
        """Calcula Sharpe ratio anualizado."""
        if len(returns) < 2 or returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    def render(self) -> None:
        """Imprime estado atual do ambiente."""
        info = self._get_info()
        print(
            f"Step {info['step']:>5d} | "
            f"Balance: ${info['balance']:>10,.2f} | "
            f"Position: {info['position']:>+.3f} | "
            f"Return: {info['total_return']:>+.2%} | "
            f"Trades: {info['n_trades']:>4d} | "
            f"Sharpe: {info['sharpe']:>+.3f} | "
            f"MaxDD: {info['max_drawdown']:>.2%}"
        )
