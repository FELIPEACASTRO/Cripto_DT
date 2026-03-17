"""Dashboard Streamlit: Cripto DT - Sistema de Previsao com IA/ML.

Execute com:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Caminhos do projeto (relativo a raiz do repositorio)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_PREDICTIONS_DIR = _DATA_DIR / "predictions"
_MODELS_DIR = _ROOT / "models"


# ===================================================================
# Funcoes auxiliares (com cache Streamlit)
# ===================================================================

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """Carrega o CSV de previsoes mais recente de ``data/predictions/``.

    Returns:
        DataFrame com as previsoes ou DataFrame vazio se nenhum arquivo
        for encontrado.
    """
    if not _PREDICTIONS_DIR.exists():
        return pd.DataFrame()

    csv_files = sorted(_PREDICTIONS_DIR.glob("predictions_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    latest = csv_files[-1]
    try:
        df = pd.read_csv(latest)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_metrics() -> dict[str, dict]:
    """Carrega metricas de treino de cada moeda em ``models/*/``.

    Procura por arquivos ``metrics.json`` em cada sub-diretorio de moeda.

    Returns:
        Dicionario ``{coin: {metrica: valor}}``.
    """
    metrics: dict[str, dict] = {}
    if not _MODELS_DIR.exists():
        return metrics

    for coin_dir in sorted(_MODELS_DIR.iterdir()):
        if not coin_dir.is_dir():
            continue
        metrics_file = coin_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics[coin_dir.name] = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return metrics


@st.cache_data(ttl=300)
def load_feature_importance(coin: str) -> pd.DataFrame:
    """Carrega importancia de features para uma moeda.

    Procura ``feature_importance.json`` em ``models/{coin}/``.
    """
    fi_path = _MODELS_DIR / coin / "feature_importance.json"
    if not fi_path.exists():
        return pd.DataFrame()
    try:
        with open(fi_path) as f:
            data = json.load(f)
        df = pd.DataFrame(
            list(data.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
        return df
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_fold_metrics(coin: str) -> pd.DataFrame:
    """Carrega metricas por fold de walk-forward para uma moeda."""
    path = _MODELS_DIR / coin / "fold_metrics.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except (json.JSONDecodeError, OSError):
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_price_history(coin: str) -> pd.DataFrame:
    """Carrega historico de precos diarios (Parquet ou CSV)."""
    # Tentar Parquet primeiro (formato padrao do storage)
    parquet_path = _DATA_DIR / "raw" / coin / "1d.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception:
            pass

    # Fallback para CSV
    for path in [
        _DATA_DIR / "processed" / f"{coin}_1d.csv",
        _DATA_DIR / "raw" / f"{coin}_1d.csv",
    ]:
        if path.exists():
            try:
                return pd.read_csv(path, parse_dates=["timestamp"])
            except Exception:
                pass

    return pd.DataFrame()


# ===================================================================
# Graficos
# ===================================================================

def plot_price_with_predictions(coin: str, df_pred: pd.DataFrame) -> go.Figure | None:
    """Grafico de preco historico com seta de previsao de direcao.

    Args:
        coin: Simbolo da moeda.
        df_pred: DataFrame de previsoes (linha da moeda selecionada).

    Returns:
        Figura Plotly ou None se dados insuficientes.
    """
    df_price = load_price_history(coin)
    if df_price.empty:
        return None

    fig = go.Figure()

    # Historico de preco
    fig.add_trace(go.Scatter(
        x=df_price["timestamp"],
        y=df_price["close"],
        mode="lines",
        name="Preco (close)",
        line={"color": "#1f77b4"},
    ))

    # Previsao como ponto no futuro
    coin_pred = df_pred[df_pred["coin"] == coin] if "coin" in df_pred.columns else df_pred
    if not coin_pred.empty:
        row = coin_pred.iloc[0]
        pred_price = row.get("predicted_price", None)
        direction = row.get("direction", "")
        if pred_price is not None:
            last_date = df_price["timestamp"].iloc[-1]
            last_price = df_price["close"].iloc[-1]
            color = "#2ca02c" if direction == "ALTA" else "#d62728"
            fig.add_trace(go.Scatter(
                x=[last_date, last_date + pd.Timedelta(days=1)],
                y=[last_price, pred_price],
                mode="lines+markers",
                name=f"Previsao ({direction})",
                line={"color": color, "dash": "dash", "width": 2},
                marker={"size": 10, "symbol": "arrow-up" if direction == "ALTA" else "arrow-down"},
            ))

    fig.update_layout(
        title=f"{coin}/USDT - Preco e Previsao",
        xaxis_title="Data",
        yaxis_title="Preco (USDT)",
        template="plotly_dark",
        height=450,
    )
    return fig


def plot_confidence_gauge(confidence: float, label: str = "Confianca") -> go.Figure:
    """Gauge semicircular de confianca (0 a 1).

    Args:
        confidence: Valor entre 0 e 1.
        label: Rotulo do gauge.

    Returns:
        Figura Plotly do gauge.
    """
    confidence = float(np.clip(confidence, 0, 1))

    color = "#2ca02c" if confidence >= 0.6 else "#ff7f0e" if confidence >= 0.4 else "#d62728"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#3c1a1a"},
                {"range": [40, 60], "color": "#3c2e1a"},
                {"range": [60, 100], "color": "#1a3c1a"},
            ],
            "shape": "angular",
        },
    ))
    fig.update_layout(height=250, template="plotly_dark")
    return fig


def plot_fear_greed_gauge(value: float) -> go.Figure:
    """Gauge do indice Fear & Greed (0-100)."""
    value = float(np.clip(value, 0, 100))

    if value <= 25:
        label = "Medo Extremo"
        color = "#d62728"
    elif value <= 45:
        label = "Medo"
        color = "#ff7f0e"
    elif value <= 55:
        label = "Neutro"
        color = "#bcbd22"
    elif value <= 75:
        label = "Ganancia"
        color = "#2ca02c"
    else:
        label = "Ganancia Extrema"
        color = "#17becf"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"Fear & Greed: {label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#3c1a1a"},
                {"range": [25, 45], "color": "#3c2e1a"},
                {"range": [45, 55], "color": "#2e2e1a"},
                {"range": [55, 75], "color": "#1a3c1a"},
                {"range": [75, 100], "color": "#1a2e3c"},
            ],
            "shape": "angular",
        },
    ))
    fig.update_layout(height=280, template="plotly_dark")
    return fig


# ===================================================================
# Layout principal
# ===================================================================

def main() -> None:
    st.set_page_config(
        page_title="Cripto DT",
        page_icon="📊",
        layout="wide",
    )

    st.title("Cripto DT - Sistema de Previsao com IA/ML")

    # ---------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------
    with st.sidebar:
        st.header("Configuracoes")

        predictions = load_predictions()
        available_coins = (
            sorted(predictions["coin"].unique().tolist())
            if not predictions.empty and "coin" in predictions.columns
            else ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"]
        )

        selected_coin = st.selectbox("Moeda", available_coins, index=0)

        timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)

        if st.button("Atualizar dados"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(
            f"Ultima atualizacao: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ---------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------
    tab_pred, tab_perf, tab_feat, tab_models, tab_sent = st.tabs([
        "Previsoes",
        "Performance",
        "Features",
        "Modelos",
        "Sentimento",
    ])

    # ===== Tab 1: Previsoes ========================================
    with tab_pred:
        st.subheader("Previsoes Atuais")

        if predictions.empty:
            st.info(
                "Nenhuma previsao encontrada. Execute o pipeline de previsao "
                "primeiro (``python scripts/predict.py``)."
            )
        else:
            # Tabela com setas coloridas
            display_df = predictions.copy()
            if "direction" in display_df.columns:
                display_df["direcao"] = display_df["direction"].apply(
                    lambda d: "⬆ ALTA" if d == "ALTA" else "⬇ BAIXA"
                )

            cols_show = [
                c for c in [
                    "coin", "direcao", "current_price",
                    "predicted_price", "confidence",
                ]
                if c in display_df.columns
            ]
            if cols_show:
                st.dataframe(
                    display_df[cols_show].style.format(
                        {
                            "current_price": "${:,.2f}",
                            "predicted_price": "${:,.2f}",
                            "confidence": "{:.2%}",
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # Grafico de preco + previsao
            fig = plot_price_with_predictions(selected_coin, predictions)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    f"Historico de precos nao disponivel para {selected_coin}."
                )

            # Gauge de confianca
            coin_row = predictions[predictions["coin"] == selected_coin] if "coin" in predictions.columns else pd.DataFrame()
            if not coin_row.empty and "confidence" in coin_row.columns:
                conf = float(coin_row["confidence"].iloc[0])
                st.plotly_chart(
                    plot_confidence_gauge(conf, f"Confianca - {selected_coin}"),
                    use_container_width=True,
                )

    # ===== Tab 2: Performance ======================================
    with tab_perf:
        st.subheader(f"Performance - {selected_coin}")

        fold_df = load_fold_metrics(selected_coin)
        if fold_df.empty:
            st.info(
                "Metricas de walk-forward nao encontradas. "
                "Treine os modelos primeiro."
            )
        else:
            # Tabela de metricas por fold
            st.markdown("**Metricas por fold (walk-forward)**")
            st.dataframe(fold_df, use_container_width=True, hide_index=True)

            # Grafico de desempenho por fold
            metric_cols = [
                c for c in fold_df.columns
                if c.startswith("ensemble_") and c != "fold"
            ]
            if metric_cols and "fold" in fold_df.columns:
                fig_fold = px.bar(
                    fold_df.melt(id_vars="fold", value_vars=metric_cols),
                    x="fold",
                    y="value",
                    color="variable",
                    barmode="group",
                    title="Metricas Ensemble por Fold",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_fold, use_container_width=True)

            # Retornos cumulativos (se disponivel)
            price_df = load_price_history(selected_coin)
            if not price_df.empty and "close" in price_df.columns:
                price_df["return"] = price_df["close"].pct_change()
                price_df["cum_return"] = (1 + price_df["return"]).cumprod() - 1
                fig_cum = px.line(
                    price_df.dropna(subset=["cum_return"]),
                    x="timestamp",
                    y="cum_return",
                    title=f"Retorno Cumulativo - {selected_coin}",
                    template="plotly_dark",
                )
                fig_cum.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig_cum, use_container_width=True)

    # ===== Tab 3: Features =========================================
    with tab_feat:
        st.subheader(f"Importancia de Features - {selected_coin}")

        fi_df = load_feature_importance(selected_coin)
        if fi_df.empty:
            st.info("Importancia de features nao disponivel para esta moeda.")
        else:
            top_n = st.slider("Top N features", 10, 50, 20)
            top_df = fi_df.head(top_n)

            fig_fi = px.bar(
                top_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top {top_n} Features - {selected_coin}",
                template="plotly_dark",
            )
            fig_fi.update_layout(yaxis={"autorange": "reversed"}, height=max(400, top_n * 22))
            st.plotly_chart(fig_fi, use_container_width=True)

        # Heatmap de correlacao
        st.subheader("Correlacao entre Features (amostra)")
        price_df = load_price_history(selected_coin)
        if not price_df.empty:
            numeric_cols = price_df.select_dtypes(include=[np.number]).columns.tolist()
            # Limitar a colunas mais relevantes para nao poluir o heatmap
            if len(numeric_cols) > 15:
                numeric_cols = numeric_cols[:15]
            if len(numeric_cols) >= 2:
                corr = price_df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    title="Heatmap de Correlacao",
                    template="plotly_dark",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Colunas numericas insuficientes para heatmap.")
        else:
            st.info("Dados de preco nao disponiveis para correlacao.")

    # ===== Tab 4: Modelos ==========================================
    with tab_models:
        st.subheader("Comparacao de Modelos")

        all_metrics = load_metrics()
        if not all_metrics:
            st.info("Nenhuma metrica de modelo encontrada.")
        else:
            rows = []
            for coin, m in all_metrics.items():
                row = {"coin": coin}
                # Extrair metricas medias de cada modelo
                for key, val in m.items():
                    if key.startswith("avg_"):
                        row[key.replace("avg_", "")] = val
                rows.append(row)

            if rows:
                model_df = pd.DataFrame(rows)
                st.dataframe(
                    model_df.style.format(
                        {
                            c: "{:.6f}"
                            for c in model_df.columns
                            if c != "coin"
                        },
                        na_rep="-",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Grafico comparativo por moeda (RMSE e dir_accuracy do ensemble)
                rmse_col = next(
                    (c for c in model_df.columns if "ensemble_rmse" in c), None
                )
                dir_col = next(
                    (c for c in model_df.columns if "ensemble_dir_accuracy" in c), None
                )

                if rmse_col:
                    fig_rmse = px.bar(
                        model_df,
                        x="coin",
                        y=rmse_col,
                        title="RMSE Ensemble por Moeda",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)

                if dir_col:
                    fig_dir = px.bar(
                        model_df,
                        x="coin",
                        y=dir_col,
                        title="Acuracia Direcional Ensemble por Moeda",
                        template="plotly_dark",
                    )
                    fig_dir.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_dir, use_container_width=True)

    # ===== Tab 5: Sentimento =======================================
    with tab_sent:
        st.subheader("Indicadores de Sentimento")

        # Fear & Greed Index
        col_fg, col_regime = st.columns(2)

        with col_fg:
            st.markdown("**Fear & Greed Index**")
            # Tentar ler do ultimo prediction ou de um arquivo dedicado
            fg_path = _DATA_DIR / "sentiment" / "fear_greed.csv"
            fg_value: float | None = None

            if fg_path.exists():
                try:
                    fg_df = pd.read_csv(fg_path)
                    if "value" in fg_df.columns and len(fg_df) > 0:
                        fg_value = float(fg_df["value"].iloc[-1])
                except Exception:
                    pass

            if fg_value is not None:
                st.plotly_chart(
                    plot_fear_greed_gauge(fg_value),
                    use_container_width=True,
                )
            else:
                st.info(
                    "Indice Fear & Greed nao disponivel. "
                    "Execute a coleta de sentimento primeiro."
                )

        with col_regime:
            st.markdown("**Regime de Mercado**")

            # Indicador de regime (se disponivel nas metricas ou previsoes)
            regime_path = _DATA_DIR / "sentiment" / "market_regime.json"
            if regime_path.exists():
                try:
                    with open(regime_path) as f:
                        regime_data = json.load(f)
                    regime_state = regime_data.get("regime", "Desconhecido")
                    regime_prob = regime_data.get("probability", 0.0)

                    regime_colors = {
                        "bull": "#2ca02c",
                        "bear": "#d62728",
                        "sideways": "#ff7f0e",
                    }
                    color = regime_colors.get(
                        regime_state.lower(), "#7f7f7f"
                    )
                    st.markdown(
                        f"<div style='text-align:center; padding:20px; "
                        f"background-color:{color}33; border-radius:10px; "
                        f"border: 2px solid {color};'>"
                        f"<h2 style='color:{color};'>{regime_state.upper()}</h2>"
                        f"<p>Probabilidade: {regime_prob:.1%}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                except (json.JSONDecodeError, OSError):
                    st.info("Dados de regime nao disponiveis.")
            else:
                st.info(
                    "Regime de mercado nao disponivel. "
                    "Execute a analise de regime primeiro."
                )

        # Historico Fear & Greed
        if fg_path.exists():
            try:
                fg_hist = pd.read_csv(fg_path, parse_dates=["timestamp"])
                if not fg_hist.empty and "value" in fg_hist.columns:
                    st.subheader("Historico Fear & Greed")
                    fig_fg = px.line(
                        fg_hist,
                        x="timestamp",
                        y="value",
                        title="Fear & Greed Index ao longo do tempo",
                        template="plotly_dark",
                    )
                    fig_fg.add_hline(y=50, line_dash="dash", line_color="gray")
                    fig_fg.add_hline(y=25, line_dash="dot", line_color="#d62728")
                    fig_fg.add_hline(y=75, line_dash="dot", line_color="#2ca02c")
                    st.plotly_chart(fig_fg, use_container_width=True)
            except Exception:
                pass


# ===================================================================
# Entrypoint
# ===================================================================

if __name__ == "__main__":
    main()
