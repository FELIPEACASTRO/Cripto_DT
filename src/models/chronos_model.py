"""Wrapper para Amazon Chronos-Bolt - modelo fundacional de series temporais.

Suporta previsao zero-shot e fine-tuning em dados de criptomoedas.
Utiliza o modelo amazon/chronos-bolt-small como base.

Se o pacote chronos-forecasting nao estiver instalado, faz fallback para
suavizacao exponencial simples como baseline.

Referencia: Ansari et al. (2024) - "Chronos: Learning the Language of Time Series"
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, class_name: str, dep_name: str = ""):
    """Importa uma classe de forma segura, retornando None se falhar."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        if dep_name:
            logger.debug(f"{class_name} nao disponivel (instale {dep_name})")
        return None


# Tentativa de importar dependencias opcionais
ChronosPipeline = _safe_import(
    "chronos", "ChronosPipeline", "chronos-forecasting"
)

try:
    import torch

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

CHRONOS_AVAILABLE = ChronosPipeline is not None and TORCH_AVAILABLE

DEFAULT_MODEL_ID = "amazon/chronos-bolt-small"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_PREDICTION_LENGTH = 1
DEFAULT_NUM_SAMPLES = 20


class _ExponentialSmoothingFallback:
    """Baseline de suavizacao exponencial simples para quando Chronos nao esta disponivel."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._last_value: float | None = None
        self._last_context: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> None:
        """Ajusta a suavizacao exponencial na serie."""
        self._last_context = series.copy()
        if len(series) == 0:
            self._last_value = 0.0
            return
        smoothed = series[0]
        for val in series[1:]:
            smoothed = self.alpha * val + (1 - self.alpha) * smoothed
        self._last_value = float(smoothed)

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Gera previsao constante (ultimo valor suavizado)."""
        if self._last_value is None:
            return np.zeros(horizon)
        return np.full(horizon, self._last_value)

    def predict_with_quantiles(
        self, horizon: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retorna mediana e quantis estimados via desvio padrao historico."""
        median = self.predict(horizon)
        if self._last_context is not None and len(self._last_context) > 1:
            std = float(np.std(self._last_context))
        else:
            std = abs(median[0]) * 0.1 if median[0] != 0 else 0.01
        low = median - 1.96 * std
        high = median + 1.96 * std
        return median, low, high


class ChronosModel(BaseModel):
    """Wrapper para Amazon Chronos-Bolt com interface BaseModel.

    Suporta dois modos de operacao:
    - Zero-shot: utiliza o modelo pre-treinado diretamente com contexto historico
    - Fine-tuned: armazena contexto de treino para gerar previsoes adaptadas

    Quando chronos-forecasting nao esta instalado, faz fallback automatico
    para suavizacao exponencial simples.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        prediction_length: int = DEFAULT_PREDICTION_LENGTH,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        feature_col: int = 0,
        fine_tune_epochs: int = 0,
        fine_tune_lr: float = 1e-4,
    ):
        self._model_id = model_id
        self._context_length = context_length
        self._prediction_length = prediction_length
        self._num_samples = num_samples
        self._feature_col = feature_col
        self._fine_tune_epochs = fine_tune_epochs
        self._fine_tune_lr = fine_tune_lr

        self._pipeline = None
        self._fallback: _ExponentialSmoothingFallback | None = None
        self._training_context: np.ndarray | None = None
        self._is_fitted = False
        self._use_fallback = not CHRONOS_AVAILABLE

        if self._use_fallback:
            logger.warning(
                "Chronos nao disponivel. Usando fallback de suavizacao exponencial. "
                "Instale com: pip install chronos-forecasting torch"
            )

    @property
    def name(self) -> str:
        if self._use_fallback:
            return "chronos_fallback_exp_smoothing"
        return "chronos_bolt"

    def _load_pipeline(self) -> None:
        """Carrega o pipeline Chronos sob demanda (lazy loading)."""
        if self._pipeline is not None:
            return

        if not CHRONOS_AVAILABLE:
            raise RuntimeError(
                "Chronos nao disponivel. Instale com: pip install chronos-forecasting torch"
            )

        logger.info(f"Carregando modelo Chronos: {self._model_id}")
        self._pipeline = ChronosPipeline.from_pretrained(
            self._model_id,
            device_map=str(DEVICE),
            torch_dtype=torch.float32,
        )
        logger.info(
            f"Modelo Chronos carregado com sucesso no dispositivo: {DEVICE}"
        )

    def _extract_series(self, X: np.ndarray) -> np.ndarray:
        """Extrai a serie temporal 1D a partir do array 2D de features.

        Args:
            X: Array (n_samples, n_features) ou (n_samples,)

        Returns:
            Serie 1D (n_samples,)
        """
        if X.ndim == 1:
            return X
        if X.ndim == 2:
            col = min(self._feature_col, X.shape[1] - 1)
            return X[:, col]
        raise ValueError(
            f"Formato de entrada nao suportado: ndim={X.ndim}. "
            "Esperado array 1D ou 2D."
        )

    def _fine_tune(self, context: np.ndarray) -> dict[str, float]:
        """Fine-tuna o modelo Chronos nos dados de contexto.

        Utiliza teacher forcing: prediz o proximo valor dado o contexto anterior.
        """
        if self._fine_tune_epochs <= 0:
            return {}

        self._load_pipeline()
        model = self._pipeline.model
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._fine_tune_lr)
        criterion = torch.nn.HuberLoss()

        losses = []
        window = min(self._context_length, len(context) - 1)

        logger.info(
            f"Iniciando fine-tuning do Chronos por {self._fine_tune_epochs} epocas "
            f"com janela de contexto={window}"
        )

        for epoch in range(self._fine_tune_epochs):
            epoch_losses = []
            # Janelas deslizantes para treino
            for i in range(window, len(context)):
                ctx = context[max(0, i - window) : i]
                target = context[i]

                ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                optimizer.zero_grad()
                # Gera previsao de 1 passo
                with torch.set_grad_enabled(True):
                    forecast = self._pipeline.predict(
                        ctx_tensor, prediction_length=1, num_samples=1
                    )
                    pred_value = forecast.median(dim=1).values.squeeze()
                    target_tensor = torch.tensor(target, dtype=torch.float32).to(DEVICE)
                    loss = criterion(pred_value, target_tensor)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(avg_loss)

            if (epoch + 1) % max(1, self._fine_tune_epochs // 5) == 0:
                logger.info(f"  Fine-tuning epoca {epoch + 1}/{self._fine_tune_epochs}: loss={avg_loss:.6f}")

        model.eval()
        metrics = {"fine_tune_final_loss": losses[-1] if losses else 0.0}
        logger.info(f"Fine-tuning concluido. Loss final: {metrics['fine_tune_final_loss']:.6f}")
        return metrics

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina/configura o modelo Chronos.

        Para zero-shot: apenas armazena o contexto de treino.
        Para fine-tuning: ajusta os pesos do modelo nos dados fornecidos.

        Args:
            X_train: Features de treino (n_samples, n_features)
            y_train: Targets de treino (n_samples,)
            X_val: Features de validacao (opcional)
            y_val: Targets de validacao (opcional)

        Returns:
            Dicionario com metricas de treino/validacao
        """
        series = self._extract_series(X_train)

        # Armazena contexto (ultimos N pontos para previsao)
        ctx_len = min(self._context_length, len(series))
        self._training_context = series[-ctx_len:].copy()

        metrics: dict[str, float] = {}

        if self._use_fallback:
            # Fallback: suavizacao exponencial
            self._fallback = _ExponentialSmoothingFallback()
            self._fallback.fit(series)
            logger.info("Modelo fallback (suavizacao exponencial) ajustado com sucesso")

            # Calcular metricas no treino
            preds = np.array([self._fallback.predict(1)[0] for _ in range(len(y_train))])
            train_rmse = float(np.sqrt(np.mean((preds - y_train) ** 2)))
            metrics["train_rmse"] = train_rmse
        else:
            # Chronos: carregar pipeline e opcionalmente fine-tunar
            self._load_pipeline()

            if self._fine_tune_epochs > 0:
                ft_metrics = self._fine_tune(series)
                metrics.update(ft_metrics)

            # Avaliar no treino com previsoes zero-shot em janelas
            try:
                eval_preds = self._predict_sliding(series, y_train)
                if len(eval_preds) > 0 and len(y_train) > 0:
                    min_len = min(len(eval_preds), len(y_train))
                    train_rmse = float(
                        np.sqrt(np.mean((eval_preds[:min_len] - y_train[:min_len]) ** 2))
                    )
                    train_dir_acc = float(
                        np.mean(np.sign(eval_preds[:min_len]) == np.sign(y_train[:min_len]))
                    )
                    metrics["train_rmse"] = train_rmse
                    metrics["train_dir_acc"] = train_dir_acc
            except Exception as e:
                logger.warning(f"Erro ao calcular metricas de treino: {e}")

        # Validacao
        if X_val is not None and y_val is not None:
            try:
                val_preds = self.predict(X_val)
                min_len = min(len(val_preds), len(y_val))
                val_rmse = float(
                    np.sqrt(np.mean((val_preds[:min_len] - y_val[:min_len]) ** 2))
                )
                val_dir_acc = float(
                    np.mean(np.sign(val_preds[:min_len]) == np.sign(y_val[:min_len]))
                )
                metrics["val_rmse"] = val_rmse
                metrics["val_dir_acc"] = val_dir_acc
            except Exception as e:
                logger.warning(f"Erro ao calcular metricas de validacao: {e}")

        self._is_fitted = True
        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def _predict_sliding(
        self, series: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Gera previsoes com janela deslizante para avaliacao."""
        preds = []
        ctx_len = min(self._context_length, len(series))

        # Avaliar apenas nos ultimos pontos para eficiencia
        n_eval = min(50, len(targets))
        start_idx = max(ctx_len, len(series) - n_eval)

        for i in range(start_idx, len(series)):
            ctx = series[max(0, i - ctx_len) : i]
            if len(ctx) < 2:
                preds.append(0.0)
                continue
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            forecast = self._pipeline.predict(
                ctx_tensor,
                prediction_length=self._prediction_length,
                num_samples=self._num_samples,
            )
            median_pred = forecast.median(dim=1).values.squeeze().cpu().numpy()
            pred_val = float(median_pred) if median_pred.ndim == 0 else float(median_pred[0])
            preds.append(pred_val)

        return np.array(preds[-len(targets):])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes pontuais (mediana das trajetorias amostradas).

        Args:
            X: Features de entrada (n_samples, n_features)

        Returns:
            Array de previsoes (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() primeiro.")

        series = self._extract_series(X)

        if self._use_fallback:
            # Fallback: atualizar contexto e prever
            return self._predict_fallback(series)

        return self._predict_chronos(series)

    def _predict_fallback(self, series: np.ndarray) -> np.ndarray:
        """Previsoes usando fallback de suavizacao exponencial."""
        if self._fallback is None:
            self._fallback = _ExponentialSmoothingFallback()
            if self._training_context is not None:
                self._fallback.fit(self._training_context)

        preds = []
        for i in range(len(series)):
            # Usar contexto acumulado para cada previsao
            if self._training_context is not None:
                combined = np.concatenate([self._training_context, series[:i]])
            else:
                combined = series[:i] if i > 0 else np.array([0.0])
            self._fallback.fit(combined[-self._context_length:])
            preds.append(self._fallback.predict(1)[0])

        return np.array(preds)

    def _predict_chronos(self, series: np.ndarray) -> np.ndarray:
        """Previsoes usando o pipeline Chronos."""
        self._load_pipeline()

        preds = []
        # Combinar contexto de treino com dados novos
        if self._training_context is not None:
            full_series = np.concatenate([self._training_context, series])
            offset = len(self._training_context)
        else:
            full_series = series
            offset = 0

        for i in range(len(series)):
            idx = offset + i
            ctx_start = max(0, idx - self._context_length)
            ctx = full_series[ctx_start:idx]

            if len(ctx) < 2:
                # Contexto insuficiente, usar ultimo valor conhecido
                preds.append(float(full_series[max(0, idx - 1)]))
                continue

            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                forecast = self._pipeline.predict(
                    ctx_tensor,
                    prediction_length=self._prediction_length,
                    num_samples=self._num_samples,
                )

            median_pred = forecast.median(dim=1).values.squeeze().cpu().numpy()
            pred_val = float(median_pred) if median_pred.ndim == 0 else float(median_pred[0])
            preds.append(pred_val)

        return np.array(preds)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera previsoes com estimativa de incerteza via quantis.

        Para Chronos: usa spread dos quantis (10%, 90%) como medida de incerteza.
        Para fallback: usa desvio padrao historico.

        Args:
            X: Features de entrada (n_samples, n_features)

        Returns:
            (predictions, confidence_scores) onde confidence_scores esta entre 0 e 1
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() primeiro.")

        series = self._extract_series(X)

        if self._use_fallback:
            return self._predict_with_confidence_fallback(series)

        return self._predict_with_confidence_chronos(series)

    def _predict_with_confidence_fallback(
        self, series: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Previsoes com confianca usando fallback."""
        preds = self._predict_fallback(series)

        # Estimar confianca pela estabilidade da serie
        if self._training_context is not None and len(self._training_context) > 1:
            hist_std = float(np.std(self._training_context))
        else:
            hist_std = float(np.std(series)) if len(series) > 1 else 1.0

        # Confianca inversamente proporcional a volatilidade local
        confidence = np.ones(len(preds))
        for i in range(len(series)):
            window_start = max(0, i - 20)
            local_data = series[window_start : i + 1]
            if len(local_data) > 1:
                local_std = float(np.std(local_data))
                ratio = local_std / hist_std if hist_std > 0 else 1.0
                confidence[i] = float(np.clip(1.0 - ratio * 0.5, 0.1, 1.0))
            else:
                confidence[i] = 0.5

        return preds, confidence

    def _predict_with_confidence_chronos(
        self, series: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Previsoes com confianca usando quantis do Chronos."""
        self._load_pipeline()

        preds = []
        confidences = []

        if self._training_context is not None:
            full_series = np.concatenate([self._training_context, series])
            offset = len(self._training_context)
        else:
            full_series = series
            offset = 0

        for i in range(len(series)):
            idx = offset + i
            ctx_start = max(0, idx - self._context_length)
            ctx = full_series[ctx_start:idx]

            if len(ctx) < 2:
                preds.append(float(full_series[max(0, idx - 1)]))
                confidences.append(0.5)
                continue

            ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                forecast = self._pipeline.predict(
                    ctx_tensor,
                    prediction_length=self._prediction_length,
                    num_samples=self._num_samples,
                )

            # forecast shape: (1, num_samples, prediction_length)
            samples = forecast.squeeze(0).cpu().numpy()  # (num_samples, prediction_length)

            if samples.ndim == 1:
                sample_values = samples
            else:
                sample_values = samples[:, 0]

            median_pred = float(np.median(sample_values))
            preds.append(median_pred)

            # Confianca baseada no spread dos quantis
            q10 = float(np.percentile(sample_values, 10))
            q90 = float(np.percentile(sample_values, 90))
            spread = abs(q90 - q10)

            # Normalizar: spread menor = confianca maior
            scale = abs(median_pred) if abs(median_pred) > 1e-8 else abs(np.mean(ctx[-20:]))
            if scale > 1e-8:
                relative_spread = spread / scale
                conf = float(np.clip(1.0 - relative_spread, 0.1, 1.0))
            else:
                conf = 0.5

            confidences.append(conf)

        return np.array(preds), np.array(confidences)

    def save(self, path: Path) -> None:
        """Salva o estado do modelo em disco.

        Salva metadados e contexto de treino. O modelo pre-treinado Chronos
        eh recarregado do HuggingFace Hub ao carregar.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() primeiro.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_id": self._model_id,
            "context_length": self._context_length,
            "prediction_length": self._prediction_length,
            "num_samples": self._num_samples,
            "feature_col": self._feature_col,
            "fine_tune_epochs": self._fine_tune_epochs,
            "fine_tune_lr": self._fine_tune_lr,
            "use_fallback": self._use_fallback,
            "is_fitted": self._is_fitted,
        }

        # Salvar metadados como JSON
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Salvar contexto de treino e estado do fallback
        data_path = path.with_suffix(".pkl")
        data = {
            "training_context": self._training_context,
            "fallback": self._fallback,
        }
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

        # Salvar pesos fine-tunados se disponivel
        if not self._use_fallback and self._pipeline is not None and self._fine_tune_epochs > 0:
            weights_path = path.with_suffix(".pt")
            try:
                torch.save(self._pipeline.model.state_dict(), weights_path)
                logger.info(f"Pesos fine-tunados salvos: {weights_path}")
            except Exception as e:
                logger.warning(f"Nao foi possivel salvar pesos fine-tunados: {e}")

        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "ChronosModel":
        """Carrega o modelo do disco.

        Args:
            path: Caminho base do modelo (sem extensao)

        Returns:
            Instancia de ChronosModel restaurada
        """
        path = Path(path)

        # Carregar metadados
        meta_path = path.with_suffix(".json")
        with open(meta_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        instance = cls(
            model_id=state["model_id"],
            context_length=state["context_length"],
            prediction_length=state["prediction_length"],
            num_samples=state["num_samples"],
            feature_col=state["feature_col"],
            fine_tune_epochs=state["fine_tune_epochs"],
            fine_tune_lr=state["fine_tune_lr"],
        )
        instance._is_fitted = state["is_fitted"]
        instance._use_fallback = state.get("use_fallback", not CHRONOS_AVAILABLE)

        # Carregar contexto e fallback
        data_path = path.with_suffix(".pkl")
        if data_path.exists():
            with open(data_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            instance._training_context = data.get("training_context")
            instance._fallback = data.get("fallback")

        # Carregar pesos fine-tunados se existirem
        if not instance._use_fallback and CHRONOS_AVAILABLE:
            weights_path = path.with_suffix(".pt")
            if weights_path.exists():
                try:
                    instance._load_pipeline()
                    weights = torch.load(weights_path, map_location=DEVICE, weights_only=True)
                    instance._pipeline.model.load_state_dict(weights)
                    logger.info(f"Pesos fine-tunados carregados: {weights_path}")
                except Exception as e:
                    logger.warning(
                        f"Nao foi possivel carregar pesos fine-tunados: {e}. "
                        "Usando modelo pre-treinado base."
                    )

        logger.info(f"Modelo carregado: {path}")
        return instance
