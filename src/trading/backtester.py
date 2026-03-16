"""Motor de backtesting para o sistema de predicao de criptomoedas.

Simula execucao de trades historicos aplicando comissao, slippage,
stop-loss e take-profit. Calcula metricas de desempenho completas
e compara a estrategia com buy-and-hold.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de configuracao e resultado
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Parametros de configuracao do backtesting.

    Attributes:
        initial_capital: Capital inicial em USD.
        commission_pct: Comissao por trade (fracao, ex: 0.001 = 0.1%).
        slippage_pct: Slippage estimado por trade (fracao).
        max_position_pct: Tamanho maximo de posicao como fracao do capital.
        use_stop_loss: Habilitar stop-loss automatico.
        use_take_profit: Habilitar take-profit automatico.
        risk_free_rate: Taxa livre de risco anualizada para calculo de Sharpe.
    """

    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_position_pct: float = 0.10
    use_stop_loss: bool = True
    use_take_profit: bool = True
    risk_free_rate: float = 0.02


@dataclass
class Trade:
    """Registro de um trade executado.

    Attributes:
        coin: Simbolo da criptomoeda.
        entry_price: Preco de entrada (apos slippage).
        exit_price: Preco de saida (apos slippage).
        entry_date: Data/hora de entrada.
        exit_date: Data/hora de saida.
        direction: Direcao do trade ('LONG' ou 'SHORT').
        size: Quantidade do ativo negociado.
        pnl: Lucro/prejuizo em USD.
        return_pct: Retorno percentual do trade.
    """

    coin: str
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    direction: str
    size: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Resultado completo do backtesting.

    Contem metricas de desempenho, curva de equity e lista de trades.
    """

    # --- Metricas de retorno ---
    total_return: float = 0.0
    annualized_return: float = 0.0

    # --- Metricas de risco ---
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # --- Metricas de trade ---
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0

    # --- Dados detalhados ---
    equity_curve: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converte resultado para dicionario serializavel."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_trades": self.total_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "monthly_returns": self.monthly_returns,
        }

    def summary_report(self) -> str:
        """Gera relatorio formatado em portugues com as metricas principais.

        Returns:
            String formatada para exibicao no terminal ou dashboard.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("RELATORIO DE BACKTESTING")
        lines.append(f"Gerado em: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("=" * 70)

        lines.append("")
        lines.append("--- RETORNO ---")
        lines.append(f"  Retorno total:        {self.total_return:>+10.2%}")
        lines.append(f"  Retorno anualizado:   {self.annualized_return:>+10.2%}")

        lines.append("")
        lines.append("--- RISCO ---")
        lines.append(f"  Drawdown maximo:      {self.max_drawdown:>10.2%}")
        lines.append(f"  Sharpe Ratio:         {self.sharpe_ratio:>10.3f}")
        lines.append(f"  Sortino Ratio:        {self.sortino_ratio:>10.3f}")
        lines.append(f"  Calmar Ratio:         {self.calmar_ratio:>10.3f}")
        lines.append(f"  Omega Ratio:          {self.omega_ratio:>10.3f}")

        lines.append("")
        lines.append("--- TRADES ---")
        lines.append(f"  Total de trades:      {self.total_trades:>10d}")
        lines.append(f"  Trades long:          {self.long_trades:>10d}")
        lines.append(f"  Trades short:         {self.short_trades:>10d}")
        lines.append(f"  Taxa de acerto:       {self.win_rate:>10.2%}")
        lines.append(f"  Fator de lucro:       {self.profit_factor:>10.3f}")
        lines.append(f"  Media ganho:          {self.avg_win:>+10.2f} USD")
        lines.append(f"  Media perda:          {self.avg_loss:>+10.2f} USD")
        lines.append(f"  Max ganhos seguidos:  {self.max_consecutive_wins:>10d}")
        lines.append(f"  Max perdas seguidas:  {self.max_consecutive_losses:>10d}")

        # Retornos mensais
        if self.monthly_returns:
            lines.append("")
            lines.append("--- RETORNOS MENSAIS ---")
            for month_key in sorted(self.monthly_returns.keys()):
                ret = self.monthly_returns[month_key]
                lines.append(f"  {month_key}:  {ret:>+8.2%}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motor de backtesting principal
# ---------------------------------------------------------------------------

class Backtester:
    """Motor de backtesting para estrategias baseadas em predicoes ML.

    Simula a execucao de trades com custos de transacao realistas,
    controle de risco (stop-loss / take-profit) e dimensionamento
    de posicao baseado no capital disponivel.

    Args:
        config: Configuracao de parametros do backtest.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        logger.info(
            "Backtester inicializado: capital=%.0f, comissao=%.4f, "
            "slippage=%.4f, max_pos=%.2f",
            self.config.initial_capital,
            self.config.commission_pct,
            self.config.slippage_pct,
            self.config.max_position_pct,
        )

    def run(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        signals: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Executa o backtest completo.

        Args:
            predictions_df: DataFrame com colunas ['date', 'coin', 'predicted_return',
                'confidence']. Cada linha e uma predicao para um passo a frente.
            prices_df: DataFrame com colunas ['date', 'coin', 'open', 'high', 'low',
                'close']. Precos historicos usados para simular execucao.
            signals: DataFrame opcional com sinais pre-calculados. Se fornecido,
                deve conter ['date', 'coin', 'direction', 'position_size',
                'stop_loss', 'take_profit']. Se None, sinais sao derivados
                das predicoes.

        Returns:
            BacktestResult com metricas, curva de equity e trades.
        """
        logger.info(
            "Iniciando backtest: %d predicoes, %d registros de preco",
            len(predictions_df),
            len(prices_df),
        )

        # Garantir que as datas sao datetime
        predictions_df = predictions_df.copy()
        prices_df = prices_df.copy()
        predictions_df["date"] = pd.to_datetime(predictions_df["date"])
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Se nao ha sinais pre-calculados, derivar das predicoes
        if signals is None:
            signals = self._derive_signals(predictions_df)
        else:
            signals = signals.copy()
            signals["date"] = pd.to_datetime(signals["date"])

        # Executar trades simulados
        trades = self._execute_trades(signals, prices_df)

        # Calcular curva de equity
        dates = sorted(prices_df["date"].unique())
        equity_curve = self._calculate_portfolio_value(trades, dates)

        # Calcular metricas de desempenho
        result = self._calculate_metrics(trades, equity_curve, dates)

        logger.info(
            "Backtest concluido: %d trades, retorno=%.2f%%, sharpe=%.3f, "
            "drawdown=%.2f%%",
            result.total_trades,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
        )

        return result

    # --- Metodos internos --------------------------------------------------

    def _derive_signals(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Deriva sinais de trading a partir das predicoes ML.

        Converte retornos previstos positivos em LONG, negativos em SHORT,
        e atribui tamanho de posicao proporcional a confianca.

        Args:
            predictions_df: DataFrame com predicoes.

        Returns:
            DataFrame de sinais com colunas padronizadas.
        """
        df = predictions_df.copy()

        # Determinar direcao baseada no retorno previsto
        df["direction"] = np.where(
            df["predicted_return"] > 0, "LONG",
            np.where(df["predicted_return"] < 0, "SHORT", "HOLD"),
        )

        # Tamanho da posicao proporcional a confianca, limitado pelo maximo
        confidence_col = "confidence" if "confidence" in df.columns else None
        if confidence_col is not None:
            df["position_size"] = np.clip(
                df[confidence_col] * self.config.max_position_pct,
                0.0,
                self.config.max_position_pct,
            )
        else:
            df["position_size"] = self.config.max_position_pct

        # Stop-loss e take-profit baseados na volatilidade implicita
        abs_return = df["predicted_return"].abs()
        # SL = 2x o retorno previsto; TP = 3x o retorno previsto (como referencia)
        df["stop_loss_pct"] = np.clip(abs_return * 2.0, 0.01, 0.10)
        df["take_profit_pct"] = np.clip(abs_return * 3.0, 0.015, 0.15)

        logger.info(
            "Sinais derivados: %d LONG, %d SHORT, %d HOLD",
            (df["direction"] == "LONG").sum(),
            (df["direction"] == "SHORT").sum(),
            (df["direction"] == "HOLD").sum(),
        )

        return df

    def _execute_trades(
        self,
        signals: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> list[Trade]:
        """Simula execucao de trades com comissao, slippage, SL e TP.

        Para cada sinal ativo (LONG ou SHORT), abre um trade no preco de
        abertura do periodo seguinte e avalia saida por SL/TP ou no proximo
        sinal contrario.

        Args:
            signals: DataFrame de sinais de trading.
            prices_df: DataFrame de precos OHLC.

        Returns:
            Lista de objetos Trade executados.
        """
        trades: list[Trade] = []
        capital = self.config.initial_capital

        # Filtrar apenas sinais ativos (nao HOLD)
        active_signals = signals[signals["direction"] != "HOLD"].copy()
        active_signals = active_signals.sort_values("date").reset_index(drop=True)

        if active_signals.empty:
            logger.warning("Nenhum sinal ativo encontrado para execucao.")
            return trades

        # Criar indice de precos por (date, coin) para acesso rapido
        prices_indexed = prices_df.set_index(["date", "coin"]).sort_index()

        # Datas disponiveis de precos por moeda
        coin_dates: dict[str, np.ndarray] = {}
        for coin in prices_df["coin"].unique():
            coin_mask = prices_df["coin"] == coin
            coin_dates[coin] = np.sort(prices_df.loc[coin_mask, "date"].unique())

        for _, signal_row in active_signals.iterrows():
            coin = signal_row["coin"]
            signal_date = signal_row["date"]
            direction = signal_row["direction"]
            position_pct = signal_row["position_size"]

            if coin not in coin_dates:
                continue

            dates_for_coin = coin_dates[coin]

            # Encontrar a proxima data disponivel apos o sinal (execucao T+1)
            # Converter para int64 para compatibilidade com np.searchsorted
            dates_i64 = dates_for_coin.astype("int64")
            signal_i64 = np.datetime64(signal_date).astype("int64") if not isinstance(signal_date, (int, np.integer)) else signal_date
            entry_idx_arr = np.searchsorted(dates_i64, signal_i64, side="right")
            if entry_idx_arr >= len(dates_for_coin):
                continue  # Sem dados futuros para executar

            entry_date = dates_for_coin[entry_idx_arr]

            # Obter preco de entrada (abertura do proximo periodo)
            try:
                entry_bar = prices_indexed.loc[(entry_date, coin)]
            except KeyError:
                continue

            entry_price_raw = float(entry_bar["open"])

            # Aplicar slippage na entrada
            if direction == "LONG":
                entry_price = entry_price_raw * (1.0 + self.config.slippage_pct)
            else:
                entry_price = entry_price_raw * (1.0 - self.config.slippage_pct)

            # Calcular tamanho da posicao em unidades do ativo
            position_value = capital * position_pct
            commission_entry = position_value * self.config.commission_pct
            position_value_net = position_value - commission_entry
            size = position_value_net / entry_price if entry_price > 0 else 0.0

            if size <= 0:
                continue

            # Calcular niveis de SL e TP
            sl_pct = float(signal_row.get("stop_loss_pct", 0.03))
            tp_pct = float(signal_row.get("take_profit_pct", 0.045))

            if direction == "LONG":
                sl_price = entry_price * (1.0 - sl_pct)
                tp_price = entry_price * (1.0 + tp_pct)
            else:
                sl_price = entry_price * (1.0 + sl_pct)
                tp_price = entry_price * (1.0 - tp_pct)

            # Simular a evolucao do trade barra a barra
            exit_price = None
            exit_date = None

            # Percorrer barras subsequentes
            max_holding_bars = min(30, len(dates_for_coin) - entry_idx_arr - 1)
            for offset in range(1, max_holding_bars + 1):
                bar_idx = entry_idx_arr + offset
                if bar_idx >= len(dates_for_coin):
                    break

                bar_date = dates_for_coin[bar_idx]
                try:
                    bar = prices_indexed.loc[(bar_date, coin)]
                except KeyError:
                    continue

                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])

                # Verificar stop-loss
                if self.config.use_stop_loss:
                    if direction == "LONG" and bar_low <= sl_price:
                        exit_price = sl_price
                        exit_date = bar_date
                        break
                    elif direction == "SHORT" and bar_high >= sl_price:
                        exit_price = sl_price
                        exit_date = bar_date
                        break

                # Verificar take-profit
                if self.config.use_take_profit:
                    if direction == "LONG" and bar_high >= tp_price:
                        exit_price = tp_price
                        exit_date = bar_date
                        break
                    elif direction == "SHORT" and bar_low <= tp_price:
                        exit_price = tp_price
                        exit_date = bar_date
                        break

            # Se nao atingiu SL/TP, fechar no ultimo preco disponivel do holding
            if exit_price is None:
                last_bar_idx = min(
                    entry_idx_arr + max_holding_bars,
                    len(dates_for_coin) - 1,
                )
                last_date = dates_for_coin[last_bar_idx]
                try:
                    last_bar = prices_indexed.loc[(last_date, coin)]
                    exit_price = float(last_bar["close"])
                    exit_date = last_date
                except KeyError:
                    continue

            # Aplicar slippage na saida
            if direction == "LONG":
                exit_price_adj = exit_price * (1.0 - self.config.slippage_pct)
            else:
                exit_price_adj = exit_price * (1.0 + self.config.slippage_pct)

            # Calcular PnL
            if direction == "LONG":
                pnl_raw = (exit_price_adj - entry_price) * size
            else:
                pnl_raw = (entry_price - exit_price_adj) * size

            # Comissao de saida
            commission_exit = abs(exit_price_adj * size) * self.config.commission_pct
            pnl = pnl_raw - commission_exit

            # Retorno percentual sobre o capital alocado
            return_pct = pnl / position_value if position_value > 0 else 0.0

            # Atualizar capital
            capital += pnl

            trade = Trade(
                coin=coin,
                entry_price=entry_price,
                exit_price=exit_price_adj,
                entry_date=pd.Timestamp(entry_date).to_pydatetime(),
                exit_date=pd.Timestamp(exit_date).to_pydatetime(),
                direction=direction,
                size=size,
                pnl=pnl,
                return_pct=return_pct,
            )
            trades.append(trade)

        logger.info("Execucao concluida: %d trades simulados.", len(trades))
        return trades

    def _calculate_portfolio_value(
        self,
        trades: list[Trade],
        dates: list,
    ) -> list[float]:
        """Constroi a curva de equity diaria do portfolio.

        Para cada data do periodo de backtest, soma o capital base
        mais os PnLs acumulados dos trades ja fechados.

        Args:
            trades: Lista de trades executados.
            dates: Lista de datas ordenadas do backtest.

        Returns:
            Lista de valores do portfolio para cada data.
        """
        initial = self.config.initial_capital
        equity: list[float] = []

        # PnL acumulado ate cada data
        cumulative_pnl = 0.0
        trade_idx = 0

        # Ordenar trades por data de saida
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)

        for date in dates:
            # Somar PnLs de trades fechados ate esta data
            while (
                trade_idx < len(sorted_trades)
                and sorted_trades[trade_idx].exit_date <= pd.Timestamp(date)
            ):
                cumulative_pnl += sorted_trades[trade_idx].pnl
                trade_idx += 1

            equity.append(initial + cumulative_pnl)

        return equity

    def _calculate_metrics(
        self,
        trades: list[Trade],
        equity_curve: list[float],
        dates: list,
    ) -> BacktestResult:
        """Calcula todas as metricas de desempenho do backtest.

        Args:
            trades: Lista de trades executados.
            equity_curve: Valores do portfolio ao longo do tempo.
            dates: Datas correspondentes a curva de equity.

        Returns:
            BacktestResult com todas as metricas preenchidas.
        """
        result = BacktestResult()
        result.equity_curve = equity_curve
        result.trades = trades
        result.total_trades = len(trades)

        if not trades:
            logger.warning("Nenhum trade para calcular metricas.")
            return result

        initial = self.config.initial_capital
        final = equity_curve[-1] if equity_curve else initial

        # --- Retorno total e anualizado ---
        result.total_return = (final - initial) / initial

        if len(dates) >= 2:
            total_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
            if total_days > 0:
                years = total_days / 365.25
                result.annualized_return = (
                    (1.0 + result.total_return) ** (1.0 / years) - 1.0
                )
            else:
                result.annualized_return = 0.0
        else:
            result.annualized_return = 0.0

        # --- Drawdown maximo ---
        equity_arr = np.array(equity_curve)
        if len(equity_arr) > 0:
            cummax = np.maximum.accumulate(equity_arr)
            drawdowns = (equity_arr - cummax) / np.where(cummax > 0, cummax, 1.0)
            result.max_drawdown = float(abs(np.min(drawdowns)))
        else:
            result.max_drawdown = 0.0

        # --- Retornos diarios para Sharpe/Sortino ---
        if len(equity_arr) > 1:
            daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        else:
            daily_returns = np.array([0.0])

        # Sharpe Ratio (anualizado, 365 dias para cripto)
        rf_daily = self.config.risk_free_rate / 365.0
        excess_returns = daily_returns - rf_daily
        if len(excess_returns) > 1 and np.std(excess_returns) > 1e-10:
            result.sharpe_ratio = float(
                np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(365)
            )
        else:
            result.sharpe_ratio = 0.0

        # Sortino Ratio (somente volatilidade negativa)
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 1:
            downside_std = float(np.std(negative_returns, ddof=1))
            if downside_std > 1e-10:
                result.sortino_ratio = float(
                    np.mean(excess_returns) / downside_std * np.sqrt(365)
                )
            else:
                result.sortino_ratio = 0.0
        else:
            result.sortino_ratio = 0.0

        # Calmar Ratio
        if result.max_drawdown > 1e-10:
            result.calmar_ratio = result.annualized_return / result.max_drawdown
        else:
            result.calmar_ratio = 0.0

        # Omega Ratio (limiar = 0)
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        sum_losses = float(np.abs(np.sum(losses)))
        if sum_losses > 1e-10:
            result.omega_ratio = float(np.sum(gains) / sum_losses)
        else:
            result.omega_ratio = float("inf") if len(gains) > 0 else 0.0

        # --- Metricas de trade ---
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses_arr = pnls[pnls < 0]

        result.long_trades = sum(1 for t in trades if t.direction == "LONG")
        result.short_trades = sum(1 for t in trades if t.direction == "SHORT")

        if len(pnls) > 0:
            result.win_rate = float(len(wins) / len(pnls))
        else:
            result.win_rate = 0.0

        # Fator de lucro (soma ganhos / soma perdas)
        total_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        total_losses = float(np.abs(np.sum(losses_arr))) if len(losses_arr) > 0 else 0.0
        if total_losses > 1e-10:
            result.profit_factor = total_wins / total_losses
        else:
            result.profit_factor = float("inf") if total_wins > 0 else 0.0

        result.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        result.avg_loss = float(np.mean(losses_arr)) if len(losses_arr) > 0 else 0.0

        # Sequencias consecutivas de ganhos e perdas
        result.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        result.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        # --- Retornos mensais ---
        result.monthly_returns = self._compute_monthly_returns(equity_curve, dates)

        return result

    @staticmethod
    def _max_consecutive(pnls: np.ndarray, positive: bool = True) -> int:
        """Calcula o numero maximo de trades consecutivos na mesma direcao.

        Args:
            pnls: Array de PnLs.
            positive: Se True, conta ganhos; se False, conta perdas.

        Returns:
            Maximo de trades consecutivos.
        """
        max_streak = 0
        current = 0
        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl <= 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _compute_monthly_returns(
        equity_curve: list[float],
        dates: list,
    ) -> dict[str, float]:
        """Calcula retornos mensais a partir da curva de equity.

        Args:
            equity_curve: Valores do portfolio por data.
            dates: Datas correspondentes.

        Returns:
            Dicionario com chave 'YYYY-MM' e valor retorno percentual.
        """
        if len(equity_curve) < 2 or len(dates) < 2:
            return {}

        df = pd.DataFrame({
            "date": pd.to_datetime(dates[:len(equity_curve)]),
            "equity": equity_curve,
        })
        df["month"] = df["date"].dt.to_period("M").astype(str)

        monthly: dict[str, float] = {}
        for month, group in df.groupby("month"):
            first_val = group["equity"].iloc[0]
            last_val = group["equity"].iloc[-1]
            if first_val > 0:
                monthly[str(month)] = (last_val - first_val) / first_val
            else:
                monthly[str(month)] = 0.0

        return monthly


# ---------------------------------------------------------------------------
# Comparador com benchmark
# ---------------------------------------------------------------------------

class BenchmarkComparator:
    """Compara a estrategia de backtest com buy-and-hold.

    Calcula alpha, beta e information ratio da estrategia
    em relacao ao benchmark (comprar e segurar).

    Args:
        risk_free_rate: Taxa livre de risco anualizada.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        logger.info(
            "BenchmarkComparator inicializado: risk_free_rate=%.4f",
            risk_free_rate,
        )

    def compare(
        self,
        backtest_result: BacktestResult,
        prices_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compara desempenho da estrategia com buy-and-hold.

        Calcula retorno do benchmark, alpha (Jensen), beta e
        information ratio usando retornos diarios.

        Args:
            backtest_result: Resultado do backtesting.
            prices_df: DataFrame de precos com colunas ['date', 'coin', 'close'].
                O benchmark e calculado como media ponderada de todas as moedas.

        Returns:
            Dicionario com metricas comparativas.
        """
        # Construir serie de retorno do benchmark (buy-and-hold equalizado)
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Valor agregado do mercado por data (media dos precos normalizados)
        pivot = prices_df.pivot_table(
            index="date", columns="coin", values="close", aggfunc="first",
        )
        pivot = pivot.sort_index()

        # Normalizar cada moeda pelo seu primeiro preco
        normalized = pivot / pivot.iloc[0]
        # Benchmark = media dos precos normalizados (portfolio igualmente ponderado)
        benchmark_curve = normalized.mean(axis=1).values

        if len(benchmark_curve) < 2:
            logger.warning("Dados insuficientes para comparacao com benchmark.")
            return self._empty_comparison()

        # Retornos diarios do benchmark
        bench_returns = np.diff(benchmark_curve) / benchmark_curve[:-1]

        # Retornos diarios da estrategia
        equity = np.array(backtest_result.equity_curve)
        if len(equity) < 2:
            logger.warning("Curva de equity insuficiente para comparacao.")
            return self._empty_comparison()

        strat_returns = np.diff(equity) / equity[:-1]

        # Alinhar tamanhos (usar o menor)
        n = min(len(strat_returns), len(bench_returns))
        strat_returns = strat_returns[:n]
        bench_returns = bench_returns[:n]

        if n < 2:
            return self._empty_comparison()

        # --- Beta (regressao linear simples) ---
        bench_mean = np.mean(bench_returns)
        strat_mean = np.mean(strat_returns)

        cov = np.mean((strat_returns - strat_mean) * (bench_returns - bench_mean))
        var_bench = np.var(bench_returns, ddof=1)

        if var_bench > 1e-15:
            beta = float(cov / var_bench)
        else:
            beta = 0.0

        # --- Alpha (Jensen) ---
        rf_daily = self.risk_free_rate / 365.0
        alpha_daily = strat_mean - rf_daily - beta * (bench_mean - rf_daily)
        alpha_annual = float(alpha_daily * 365.0)

        # --- Information Ratio ---
        tracking_error_arr = strat_returns - bench_returns
        tracking_error_std = float(np.std(tracking_error_arr, ddof=1))
        if tracking_error_std > 1e-10:
            information_ratio = float(
                np.mean(tracking_error_arr) / tracking_error_std * np.sqrt(365)
            )
        else:
            information_ratio = 0.0

        # Retorno do benchmark
        benchmark_total_return = float(
            (benchmark_curve[-1] - benchmark_curve[0]) / benchmark_curve[0]
        )

        # Retorno anualizado do benchmark
        total_days = n
        if total_days > 0:
            years = total_days / 365.25
            benchmark_annual = (1.0 + benchmark_total_return) ** (1.0 / years) - 1.0
        else:
            benchmark_annual = 0.0

        comparison = {
            "strategy_return": backtest_result.total_return,
            "strategy_annualized": backtest_result.annualized_return,
            "benchmark_return": benchmark_total_return,
            "benchmark_annualized": float(benchmark_annual),
            "alpha": alpha_annual,
            "beta": beta,
            "information_ratio": information_ratio,
            "excess_return": backtest_result.total_return - benchmark_total_return,
        }

        logger.info(
            "Comparacao: alpha=%.4f, beta=%.4f, IR=%.4f, excess=%.2f%%",
            alpha_annual,
            beta,
            information_ratio,
            comparison["excess_return"] * 100,
        )

        return comparison

    @staticmethod
    def _empty_comparison() -> dict[str, Any]:
        """Retorna dicionario de comparacao vazio para casos sem dados."""
        return {
            "strategy_return": 0.0,
            "strategy_annualized": 0.0,
            "benchmark_return": 0.0,
            "benchmark_annualized": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "information_ratio": 0.0,
            "excess_return": 0.0,
        }

    def format_comparison_report(self, comparison: dict[str, Any]) -> str:
        """Formata relatorio comparativo em portugues.

        Args:
            comparison: Dicionario retornado por compare().

        Returns:
            String formatada para exibicao.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("COMPARACAO: ESTRATEGIA vs BUY-AND-HOLD")
        lines.append("=" * 70)

        lines.append("")
        lines.append("--- RETORNO ---")
        lines.append(f"  Estrategia (total):    {comparison['strategy_return']:>+10.2%}")
        lines.append(f"  Benchmark (total):     {comparison['benchmark_return']:>+10.2%}")
        lines.append(f"  Excesso de retorno:    {comparison['excess_return']:>+10.2%}")

        lines.append("")
        lines.append("--- RETORNO ANUALIZADO ---")
        lines.append(f"  Estrategia:            {comparison['strategy_annualized']:>+10.2%}")
        lines.append(f"  Benchmark:             {comparison['benchmark_annualized']:>+10.2%}")

        lines.append("")
        lines.append("--- METRICAS DE RISCO RELATIVO ---")
        lines.append(f"  Alpha (Jensen):        {comparison['alpha']:>+10.4f}")
        lines.append(f"  Beta:                  {comparison['beta']:>10.4f}")
        lines.append(f"  Information Ratio:     {comparison['information_ratio']:>10.4f}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
