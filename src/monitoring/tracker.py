"""Monitoramento de performance: drift de dados/modelo, alertas e relatorios."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Rastreia performance dos modelos e detecta degradacao/drift.

    Utiliza arquivos JSON para persistencia (sem banco de dados).
    Armazena previsoes, metricas historicas e alertas em data/monitoring/.
    """

    def __init__(self, log_dir: str = "data/monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Arquivos de persistencia
        self._predictions_file = self.log_dir / "predictions.json"
        self._alerts_file = self.log_dir / "alerts.json"
        self._reports_dir = self.log_dir / "reports"
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Carregar historico existente
        self._predictions = self._load_json(self._predictions_file, default=[])
        self._alerts = self._load_json(self._alerts_file, default=[])

    # ------------------------------------------------------------------
    # Persistencia JSON
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path, default=None):
        """Carrega dados de um arquivo JSON, retornando default se nao existir."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Erro ao ler %s: %s", path, exc)
        return default if default is not None else []

    @staticmethod
    def _save_json(path: Path, data) -> None:
        """Salva dados em formato JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except IOError as exc:
            logger.error("Erro ao salvar %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Logging de previsoes
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        coin: str,
        predicted: float,
        actual: float,
        model_name: str,
        timestamp: str | None = None,
    ) -> None:
        """Registra uma previsao e seu valor real para rastreamento.

        Args:
            coin: Simbolo da moeda (ex: 'BTC')
            predicted: Valor previsto pelo modelo
            actual: Valor real observado
            model_name: Nome do modelo que gerou a previsao
            timestamp: Timestamp ISO 8601 (usa hora atual se None)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        registro = {
            "coin": coin,
            "predicted": float(predicted),
            "actual": float(actual),
            "model_name": model_name,
            "timestamp": timestamp,
            "error": float(actual - predicted),
            "abs_error": float(abs(actual - predicted)),
            "direction_correct": bool(
                np.sign(predicted) == np.sign(actual)
            ),
        }

        self._predictions.append(registro)
        self._save_json(self._predictions_file, self._predictions)

        logger.debug(
            "Previsao registrada: %s/%s pred=%.6f real=%.6f (dir=%s)",
            coin,
            model_name,
            predicted,
            actual,
            "OK" if registro["direction_correct"] else "ERRADO",
        )

    # ------------------------------------------------------------------
    # Deteccao de drift de dados (PSI)
    # ------------------------------------------------------------------

    def detect_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> dict:
        """Calcula Population Stability Index (PSI) por feature.

        PSI mede a diferenca entre a distribuicao de referencia e a atual.
        Valores acima do threshold indicam drift significativo.

        Interpretacao do PSI:
            < 0.10  -> sem drift significativo
            0.10-0.25 -> drift moderado (monitorar)
            > 0.25  -> drift severo (retreinar)

        Args:
            reference_df: DataFrame de referencia (treino original)
            current_df: DataFrame atual (dados recentes)
            threshold: Limiar para gerar alerta de drift

        Returns:
            Dicionario com PSI por feature e flag de drift detectado.
        """
        # Usar apenas colunas numericas em comum
        common_cols = list(
            set(reference_df.select_dtypes(include=[np.number]).columns)
            & set(current_df.select_dtypes(include=[np.number]).columns)
        )

        resultados = {}
        features_com_drift = []

        for col in sorted(common_cols):
            ref_values = reference_df[col].dropna().values
            cur_values = current_df[col].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            psi = self._calculate_psi(ref_values, cur_values)
            tem_drift = psi > threshold

            resultados[col] = {
                "psi": float(psi),
                "drift_detected": tem_drift,
                "severity": (
                    "severo" if psi > 0.25
                    else "moderado" if psi > 0.10
                    else "normal"
                ),
            }

            if tem_drift:
                features_com_drift.append(col)

        drift_geral = len(features_com_drift) > 0

        if drift_geral:
            logger.warning(
                "Data drift detectado em %d features: %s",
                len(features_com_drift),
                features_com_drift[:10],  # Limitar log
            )
            # Gerar alerta automaticamente
            self._alerts.append(
                self.generate_alert(
                    metric="data_drift",
                    value=len(features_com_drift),
                    threshold=0,
                )
            )
            self._save_json(self._alerts_file, self._alerts)

        return {
            "features": resultados,
            "drift_detected": drift_geral,
            "n_features_with_drift": len(features_com_drift),
            "drifted_features": features_com_drift,
            "threshold": threshold,
        }

    @staticmethod
    def _calculate_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calcula o PSI entre duas distribuicoes usando histogramas.

        Divide os dados de referencia em bins e calcula a divergencia
        proporcional em cada bin.
        """
        # Definir bins baseados na distribuicao de referencia
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        # Evitar bins degenerados
        if min_val == max_val:
            return 0.0

        bins = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Proporcoes (com suavizacao para evitar log(0))
        epsilon = 1e-6
        ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
        cur_pct = (cur_counts + epsilon) / (len(current) + epsilon * n_bins)

        # PSI = sum((cur - ref) * ln(cur/ref))
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)

    # ------------------------------------------------------------------
    # Deteccao de drift de modelo
    # ------------------------------------------------------------------

    def detect_model_drift(
        self,
        coin: str,
        window: int = 30,
    ) -> dict:
        """Verifica se a acuracia recente degradou em relacao ao historico.

        Compara a acuracia direcional dos ultimos `window` registros
        contra a acuracia acumulada total. Se a queda for > 10pp,
        considera drift de modelo.

        Args:
            coin: Simbolo da moeda
            window: Numero de previsoes recentes para avaliar

        Returns:
            Dicionario com metricas historicas, recentes e flag de drift.
        """
        # Filtrar previsoes da moeda
        preds_coin = [
            p for p in self._predictions if p["coin"] == coin
        ]

        if len(preds_coin) < window:
            logger.info(
                "Dados insuficientes para drift de modelo (%s): %d/%d",
                coin, len(preds_coin), window,
            )
            return {
                "coin": coin,
                "drift_detected": False,
                "reason": "dados_insuficientes",
                "n_predictions": len(preds_coin),
            }

        # Metricas historicas (todas as previsoes)
        all_dir_acc = np.mean([p["direction_correct"] for p in preds_coin])
        all_mae = np.mean([p["abs_error"] for p in preds_coin])

        # Metricas recentes (ultimas `window` previsoes)
        recentes = preds_coin[-window:]
        recent_dir_acc = np.mean([p["direction_correct"] for p in recentes])
        recent_mae = np.mean([p["abs_error"] for p in recentes])

        # Drift se acuracia recente caiu mais de 10 pontos percentuais
        acc_drop = float(all_dir_acc - recent_dir_acc)
        drift_detectado = acc_drop > 0.10

        if drift_detectado:
            logger.warning(
                "Model drift detectado para %s: acuracia caiu %.2f%% "
                "(historico=%.4f, recente=%.4f)",
                coin, acc_drop * 100, all_dir_acc, recent_dir_acc,
            )
            self._alerts.append(
                self.generate_alert(
                    metric="model_drift",
                    value=recent_dir_acc,
                    threshold=all_dir_acc - 0.10,
                )
            )
            self._save_json(self._alerts_file, self._alerts)

        return {
            "coin": coin,
            "drift_detected": drift_detectado,
            "historical_dir_accuracy": float(all_dir_acc),
            "recent_dir_accuracy": float(recent_dir_acc),
            "accuracy_drop": acc_drop,
            "historical_mae": float(all_mae),
            "recent_mae": float(recent_mae),
            "window": window,
            "n_predictions": len(preds_coin),
        }

    # ------------------------------------------------------------------
    # Resumo de performance
    # ------------------------------------------------------------------

    def get_performance_summary(
        self,
        coin: str | None = None,
        days: int = 30,
    ) -> dict:
        """Agrega metricas de performance em um periodo.

        Args:
            coin: Filtrar por moeda (None = todas)
            days: Numero de dias para considerar

        Returns:
            Dicionario com metricas agregadas por moeda e modelo.
        """
        # Filtrar por data
        cutoff = datetime.now(timezone.utc).isoformat()
        # Calcular data de corte
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff_dt.isoformat()

        preds = [
            p for p in self._predictions
            if p["timestamp"] >= cutoff_str
        ]

        if coin is not None:
            preds = [p for p in preds if p["coin"] == coin]

        if not preds:
            return {
                "period_days": days,
                "coin": coin,
                "n_predictions": 0,
                "message": "Sem previsoes no periodo",
            }

        # Agregar por moeda
        por_moeda: dict[str, list] = {}
        for p in preds:
            por_moeda.setdefault(p["coin"], []).append(p)

        # Agregar por modelo
        por_modelo: dict[str, list] = {}
        for p in preds:
            por_modelo.setdefault(p["model_name"], []).append(p)

        def _aggregate(registros: list) -> dict:
            """Calcula metricas agregadas de uma lista de previsoes."""
            errors = [r["error"] for r in registros]
            abs_errors = [r["abs_error"] for r in registros]
            dir_correct = [r["direction_correct"] for r in registros]
            return {
                "n_predictions": len(registros),
                "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
                "mae": float(np.mean(abs_errors)),
                "directional_accuracy": float(np.mean(dir_correct)),
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
            }

        summary = {
            "period_days": days,
            "coin_filter": coin,
            "n_predictions": len(preds),
            "overall": _aggregate(preds),
            "by_coin": {
                c: _aggregate(regs) for c, regs in por_moeda.items()
            },
            "by_model": {
                m: _aggregate(regs) for m, regs in por_modelo.items()
            },
        }

        return summary

    # ------------------------------------------------------------------
    # Alertas
    # ------------------------------------------------------------------

    def generate_alert(
        self,
        metric: str,
        value: float,
        threshold: float,
    ) -> dict:
        """Cria um objeto de alerta padronizado.

        Args:
            metric: Nome da metrica que disparou o alerta
            value: Valor observado
            threshold: Limiar ultrapassado

        Returns:
            Dicionario com informacoes do alerta.
        """
        # Determinar severidade baseado na distancia do threshold
        if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
            desvio = abs(value - threshold)
            severidade = (
                "critico" if desvio > 0.20
                else "alto" if desvio > 0.10
                else "medio"
            )
        else:
            severidade = "medio"

        alerta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "value": float(value) if isinstance(value, (int, float)) else str(value),
            "threshold": float(threshold) if isinstance(threshold, (int, float)) else str(threshold),
            "severity": severidade,
            "acknowledged": False,
        }

        return alerta

    def check_all_alerts(self) -> list[dict]:
        """Executa todas as verificacoes de drift e performance.

        Verifica drift de modelo para cada moeda que possui previsoes
        registradas. Retorna lista de alertas novos gerados.

        Returns:
            Lista de dicionarios de alerta.
        """
        logger.info("Executando verificacoes de monitoramento...")
        novos_alertas = []

        # Coletar moedas unicas com previsoes
        moedas = set(p["coin"] for p in self._predictions)

        for coin in sorted(moedas):
            resultado = self.detect_model_drift(coin)
            if resultado.get("drift_detected"):
                alerta = self.generate_alert(
                    metric=f"model_drift_{coin}",
                    value=resultado["recent_dir_accuracy"],
                    threshold=resultado["historical_dir_accuracy"] - 0.10,
                )
                alerta["details"] = resultado
                novos_alertas.append(alerta)

        # Verificar performance geral (acuracia abaixo de 50%)
        summary = self.get_performance_summary(days=7)
        overall = summary.get("overall", {})
        dir_acc = overall.get("directional_accuracy", 1.0)

        if dir_acc < 0.50 and summary.get("n_predictions", 0) >= 10:
            alerta = self.generate_alert(
                metric="low_overall_accuracy",
                value=dir_acc,
                threshold=0.50,
            )
            alerta["details"] = {
                "period_days": 7,
                "n_predictions": summary["n_predictions"],
            }
            novos_alertas.append(alerta)

        if novos_alertas:
            self._alerts.extend(novos_alertas)
            self._save_json(self._alerts_file, self._alerts)
            logger.warning(
                "Monitoramento: %d novos alertas gerados", len(novos_alertas)
            )
        else:
            logger.info("Monitoramento: nenhum alerta novo")

        return novos_alertas

    # ------------------------------------------------------------------
    # Relatorio
    # ------------------------------------------------------------------

    def save_report(self, filepath: str | None = None) -> None:
        """Salva relatorio completo de monitoramento em JSON.

        Inclui resumo de performance (7 e 30 dias), alertas recentes
        e drift de modelo por moeda.

        Args:
            filepath: Caminho do arquivo (usa timestamp se None)
        """
        if filepath is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = str(self._reports_dir / f"report_{ts}.json")

        # Coletar moedas
        moedas = sorted(set(p["coin"] for p in self._predictions))

        relatorio = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_predictions": len(self._predictions),
            "total_alerts": len(self._alerts),
            "coins_tracked": moedas,
            "summary_7d": self.get_performance_summary(days=7),
            "summary_30d": self.get_performance_summary(days=30),
            "model_drift": {
                coin: self.detect_model_drift(coin) for coin in moedas
            },
            "recent_alerts": self._alerts[-20:],  # Ultimos 20 alertas
        }

        self._save_json(Path(filepath), relatorio)
        logger.info("Relatorio de monitoramento salvo em %s", filepath)
