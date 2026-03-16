"""Tiny Time Mixers (TTM) — wrapper para o foundation model da IBM.

Arquitetura MLP-Mixer baseada em patches (1M-9.1M parametros), otimizada para CPU.
Suporta previsao zero-shot e few-shot. Quando o pacote tsfm_public (HuggingFace)
nao esta instalado, degrada graciosamente para um baseline de regressao linear.

Referencia: Ekambaram et al. (2024) - "Tiny Time Mixers (TTMs): Fast Pre-trained
Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series"
"""

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe import do tsfm_public (IBM TTM via HuggingFace)
# ---------------------------------------------------------------------------

def _safe_import(module_path: str, class_name: str):
    """Importa uma classe de forma segura, retornando None se indisponivel."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        logger.debug("Classe %s nao disponivel: %s", class_name, exc)
        return None


TinyTimeMixerForPrediction = _safe_import(
    "tsfm_public.models.tinytimemixer", "TinyTimeMixerForPrediction"
)
TinyTimeMixerConfig = _safe_import(
    "tsfm_public.models.tinytimemixer", "TinyTimeMixerConfig"
)

_HAS_TSFM = TinyTimeMixerForPrediction is not None


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

@dataclass
class TTMConfig:
    """Configuracao do wrapper TTM."""

    model_id: str = "ibm/TTM"
    context_length: int = 512
    prediction_length: int = 1
    # Fine-tuning
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    batch_size: int = 64
    # MC-Dropout para intervalos de confianca
    mc_dropout_samples: int = 30
    dropout: float = 0.1
    # Fallback
    use_fallback: bool = False


# ---------------------------------------------------------------------------
# Fallback: baseline com regressao linear (sklearn)
# ---------------------------------------------------------------------------

class _LinearFallback:
    """Baseline simples para quando tsfm_public nao esta disponivel."""

    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        self.model.fit(X, y)
        self._fitted = True
        preds = self.model.predict(X)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        return {"train_rmse": rmse}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fallback nao treinado")
        return self.model.predict(X)

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimativa de incerteza via residuos do treino."""
        preds = self.predict(X)
        # Heuristica: incerteza proporcional a magnitude da previsao
        std = np.abs(preds) * 0.1 + 1e-6
        return preds, std

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)  # noqa: S301
        self._fitted = True


# ---------------------------------------------------------------------------
# Wrapper TTM nativo (com tsfm_public)
# ---------------------------------------------------------------------------

class _TTMNative:
    """Wrapper fino sobre TinyTimeMixerForPrediction do HuggingFace."""

    def __init__(self, cfg: TTMConfig):
        self.cfg = cfg
        self.model = None
        self._fitted = False

    def _load_pretrained(self) -> None:
        """Carrega modelo pre-treinado do HuggingFace Hub."""
        logger.info("Carregando modelo TTM pre-treinado: %s", self.cfg.model_id)
        try:
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                self.cfg.model_id,
                context_length=self.cfg.context_length,
                prediction_length=self.cfg.prediction_length,
            )
            # Garantir CPU (ponto forte do TTM)
            self.model = self.model.cpu()
            self.model.eval()
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "TTM carregado com sucesso: %s parametros (%.2fM)",
                f"{n_params:,}",
                n_params / 1e6,
            )
        except Exception as exc:
            logger.error("Falha ao carregar TTM pre-treinado: %s", exc)
            raise

    def _prepare_input(self, X: np.ndarray) -> "torch.Tensor":
        """Converte numpy array para formato esperado pelo TTM.

        TTM espera: (batch, context_length, n_channels)
        Se X eh 2D (samples, features), cria janelas deslizantes.
        """
        import torch

        ctx = self.cfg.context_length

        if X.ndim == 3:
            return torch.FloatTensor(X)

        # Criar janelas deslizantes de tamanho context_length
        if len(X) <= ctx:
            # Padding com zeros se necessario
            padded = np.zeros((ctx, X.shape[1]), dtype=np.float32)
            padded[-len(X):] = X
            return torch.FloatTensor(padded).unsqueeze(0)

        windows = []
        for i in range(ctx, len(X) + 1):
            windows.append(X[i - ctx: i])
        return torch.FloatTensor(np.array(windows))

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict[str, float]:
        """Fine-tuning do TTM no dominio de criptomoedas."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self.model is None:
            self._load_pretrained()

        X_tensor = self._prepare_input(X)
        y_tensor = torch.FloatTensor(y[-len(X_tensor):]).unsqueeze(-1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Fine-tune com lr baixo para preservar pesos pre-treinados
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.finetune_lr
        )

        best_loss = float("inf")
        metrics = {}

        for epoch in range(self.cfg.finetune_epochs):
            epoch_losses = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                # TTM retorna previsoes; calcular MSE contra target
                pred = output.prediction_outputs
                if pred.ndim == 3:
                    pred = pred[:, -1, 0]
                elif pred.ndim == 2:
                    pred = pred[:, -1]
                loss = torch.nn.functional.mse_loss(pred, y_batch.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % max(1, self.cfg.finetune_epochs // 3) == 0:
                logger.info(
                    "  TTM fine-tune epoca %d/%d: loss=%.6f",
                    epoch + 1, self.cfg.finetune_epochs, avg_loss,
                )

        self.model.eval()
        self._fitted = True
        metrics["train_loss"] = best_loss
        metrics["train_rmse"] = float(np.sqrt(best_loss))

        # Validacao
        if X_val is not None and y_val is not None:
            try:
                val_preds = self.predict(X_val)
                val_y = y_val[-len(val_preds):]
                metrics["val_rmse"] = float(np.sqrt(np.mean((val_preds - val_y) ** 2)))
                metrics["val_dir_acc"] = float(
                    np.mean(np.sign(val_preds) == np.sign(val_y))
                )
            except Exception as exc:
                logger.warning("Erro na validacao TTM: %s", exc)

        logger.info("  TTM metricas de fine-tuning: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Previsao zero-shot ou fine-tuned."""
        import torch

        if self.model is None:
            self._load_pretrained()

        self.model.eval()
        X_tensor = self._prepare_input(X)

        with torch.no_grad():
            output = self.model(X_tensor)
            pred = output.prediction_outputs
            if pred.ndim == 3:
                pred = pred[:, -1, 0]
            elif pred.ndim == 2:
                pred = pred[:, -1]
            return pred.numpy()

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """MC-Dropout para estimativa de incerteza."""
        import torch

        if self.model is None:
            self._load_pretrained()

        X_tensor = self._prepare_input(X)

        # Ativar dropout para MC-Dropout
        self.model.train()
        predictions = []
        for _ in range(self.cfg.mc_dropout_samples):
            with torch.no_grad():
                output = self.model(X_tensor)
                pred = output.prediction_outputs
                if pred.ndim == 3:
                    pred = pred[:, -1, 0]
                elif pred.ndim == 2:
                    pred = pred[:, -1]
                predictions.append(pred.numpy())

        self.model.eval()
        predictions = np.array(predictions)  # (n_samples, batch)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        return mean_pred, std_pred

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dir = path.parent / f"{path.stem}_ttm"
        self.model.save_pretrained(str(save_dir))
        # Salvar flag de fitted
        meta = {"fitted": self._fitted, "config": asdict(self.cfg)}
        with open(save_dir / "wrapper_meta.json", "w") as f:
            json.dump(meta, f)
        logger.info("Modelo TTM salvo em: %s", save_dir)

    def load(self, path: Path) -> None:
        save_dir = path.parent / f"{path.stem}_ttm"
        self.model = TinyTimeMixerForPrediction.from_pretrained(str(save_dir))
        self.model = self.model.cpu()
        self.model.eval()
        meta_path = save_dir / "wrapper_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._fitted = meta.get("fitted", True)
        else:
            self._fitted = True
        logger.info("Modelo TTM carregado de: %s", save_dir)


# ---------------------------------------------------------------------------
# Modelo publico: TTMModel (BaseModel)
# ---------------------------------------------------------------------------

class TTMModel(BaseModel):
    """Wrapper para IBM Tiny Time Mixers (TTM).

    Arquitetura MLP-Mixer baseada em patches (NAO eh Transformer).
    Projetado para ser eficiente em CPU com 1M-9.1M parametros.
    Suporta previsao zero-shot e few-shot via fine-tuning.

    Se tsfm_public nao estiver instalado, utiliza fallback de regressao
    linear (Ridge) para manter o pipeline funcional.
    """

    def __init__(self, cfg: TTMConfig | None = None, force_fallback: bool = False):
        self.cfg = cfg or TTMConfig()
        self._use_fallback = force_fallback or not _HAS_TSFM

        if self._use_fallback:
            if not force_fallback:
                logger.warning(
                    "tsfm_public nao instalado. Usando fallback LinearRegression. "
                    "Instale com: pip install tsfm_public"
                )
            self._backend = _LinearFallback()
        else:
            self._backend = _TTMNative(self.cfg)

        logger.info(
            "TTMModel inicializado (backend=%s)",
            "fallback_linear" if self._use_fallback else "ttm_nativo",
        )

    @property
    def name(self) -> str:
        return "ttm"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina/fine-tuna o modelo TTM.

        Para o backend nativo, faz few-shot fine-tuning no dominio cripto.
        Para o fallback, treina regressao linear padrao.
        """
        logger.info("Iniciando treino TTM (backend=%s, amostras=%d)",
                     "fallback" if self._use_fallback else "nativo", len(X_train))

        try:
            if self._use_fallback:
                metrics = self._backend.fit(X_train, y_train)
                if X_val is not None and y_val is not None:
                    val_preds = self._backend.predict(X_val)
                    metrics["val_rmse"] = float(
                        np.sqrt(np.mean((val_preds - y_val) ** 2))
                    )
                    metrics["val_dir_acc"] = float(
                        np.mean(np.sign(val_preds) == np.sign(y_val))
                    )
            else:
                metrics = self._backend.fit(X_train, y_train, X_val, y_val)

        except Exception as exc:
            logger.error("Erro no treino TTM: %s", exc, exc_info=True)
            # Tentar fallback emergencial
            if not self._use_fallback:
                logger.warning("Tentando fallback de emergencia para LinearRegression")
                self._use_fallback = True
                self._backend = _LinearFallback()
                metrics = self._backend.fit(X_train, y_train)
                metrics["fallback_ativado"] = 1.0
            else:
                raise

        logger.info("TTM treino concluido: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes pontuais."""
        try:
            return self._backend.predict(X)
        except Exception as exc:
            logger.error("Erro na previsao TTM: %s", exc)
            raise

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera previsoes com estimativa de confianca.

        Usa MC-Dropout (backend nativo) ou heuristica de residuos (fallback).
        Retorna (previsoes, confianca) onde confianca esta entre 0 e 1.
        """
        try:
            mean_pred, std_pred = self._backend.predict_with_std(X)

            # Converter desvio padrao em score de confianca [0, 1]
            max_std = std_pred.max() if std_pred.max() > 0 else 1.0
            confidence = 1.0 - np.clip(std_pred / max_std, 0.0, 1.0)

            return mean_pred, confidence

        except Exception as exc:
            logger.error("Erro na previsao com confianca TTM: %s", exc)
            # Fallback: retornar previsao sem confianca
            preds = self.predict(X)
            confidence = np.full_like(preds, 0.5)
            return preds, confidence

    def save(self, path: Path) -> None:
        """Salva modelo em disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Salvar metadados do wrapper
        meta = {
            "use_fallback": self._use_fallback,
            "config": asdict(self.cfg),
        }
        meta_path = path.parent / f"{path.stem}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._backend.save(path)
        logger.info("TTMModel salvo em: %s", path)

    @classmethod
    def load(cls, path: Path) -> "TTMModel":
        """Carrega modelo do disco."""
        path = Path(path)
        meta_path = path.parent / f"{path.stem}_meta.json"

        cfg = TTMConfig()
        use_fallback = not _HAS_TSFM

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            use_fallback = meta.get("use_fallback", use_fallback)
            saved_cfg = meta.get("config", {})
            cfg = TTMConfig(**{
                k: v for k, v in saved_cfg.items()
                if k in TTMConfig.__dataclass_fields__
            })

        instance = cls(cfg=cfg, force_fallback=use_fallback)
        instance._backend.load(path)
        logger.info("TTMModel carregado de: %s", path)
        return instance
