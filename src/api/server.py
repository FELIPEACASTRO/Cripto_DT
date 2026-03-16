"""API REST FastAPI para o sistema de previsao de criptomoedas.

Endpoints para previsoes, sinais de trading, metricas de modelos,
sentimento de mercado, narrativas e orquestracao de agentes.

Uso:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importacoes lazy — dependencias podem nao estar instaladas
# ---------------------------------------------------------------------------
_predictor = None
_signal_generator = None
_orchestrator = None
_narrative_detector = None
_config = None


def _load_config():
    """Carrega configuracao global do sistema."""
    global _config
    if _config is not None:
        return _config
    try:
        from config.settings import Config
        _config = Config()
        logger.info("Configuracao carregada com sucesso")
    except Exception as exc:
        logger.warning("Falha ao carregar Config: %s — usando defaults internos", exc)
        _config = None
    return _config


def _get_predictor():
    """Retorna instancia singleton do Predictor (lazy)."""
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        from src.prediction.predictor import Predictor
        cfg = _load_config()
        _predictor = Predictor(config=cfg) if cfg else Predictor()
        logger.info("Predictor inicializado")
    except Exception as exc:
        logger.error("Falha ao inicializar Predictor: %s", exc)
        _predictor = None
    return _predictor


def _get_signal_generator():
    """Retorna instancia singleton do SignalGenerator (lazy)."""
    global _signal_generator
    if _signal_generator is not None:
        return _signal_generator
    try:
        from src.trading.signal_generator import SignalGenerator
        cfg = _load_config()
        if cfg:
            sg_cfg = cfg.signal_generator
            _signal_generator = SignalGenerator(
                risk_per_trade=sg_cfg.risk_per_trade,
                max_position=sg_cfg.max_position,
                min_confidence=sg_cfg.min_confidence,
                min_model_agreement=sg_cfg.min_model_agreement,
                atr_multiplier_sl=sg_cfg.atr_multiplier_sl,
                atr_multiplier_tp=sg_cfg.atr_multiplier_tp,
            )
        else:
            _signal_generator = SignalGenerator()
        logger.info("SignalGenerator inicializado")
    except Exception as exc:
        logger.error("Falha ao inicializar SignalGenerator: %s", exc)
        _signal_generator = None
    return _signal_generator


def _get_orchestrator():
    """Retorna instancia singleton do AgentOrchestrator (lazy)."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
    try:
        from src.agents.orchestrator import AgentOrchestrator
        cfg = _load_config()
        _orchestrator = AgentOrchestrator(config=cfg)
        logger.info("AgentOrchestrator inicializado")
    except Exception as exc:
        logger.error("Falha ao inicializar AgentOrchestrator: %s", exc)
        _orchestrator = None
    return _orchestrator


def _get_narrative_detector():
    """Retorna instancia singleton do NarrativeDetector (lazy)."""
    global _narrative_detector
    if _narrative_detector is not None:
        return _narrative_detector
    try:
        from src.data.narrative_detector import NarrativeDetector
        _narrative_detector = NarrativeDetector()
        logger.info("NarrativeDetector inicializado")
    except Exception as exc:
        logger.error("Falha ao inicializar NarrativeDetector: %s", exc)
        _narrative_detector = None
    return _narrative_detector


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SUPPORTED_COINS: list[str] = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "DOGE", "AVAX", "DOT", "MATIC",
]

API_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Rate limiter simples em memoria
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Rate limiter por IP usando janela deslizante em memoria.

    Nao persiste entre reinicializacoes — adequado para deploy single-instance.
    Para producao distribuida, substituir por Redis.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {ip: [timestamps]}
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Verifica se o IP pode fazer mais requests na janela atual."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Limpar timestamps antigos
        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if t > cutoff]

        if len(self._hits[client_ip]) >= self.max_requests:
            return False

        self._hits[client_ip].append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        """Retorna numero de requests restantes para o IP."""
        now = time.time()
        cutoff = now - self.window_seconds
        hits = [t for t in self._hits[client_ip] if t > cutoff]
        return max(0, self.max_requests - len(hits))


_rate_limiter = _RateLimiter(max_requests=60, window_seconds=60)

# ---------------------------------------------------------------------------
# Estado global da aplicacao
# ---------------------------------------------------------------------------
_app_state: dict[str, Any] = {
    "started_at": None,
    "predictions_cache": {},       # {coin: {resultado, timestamp}}
    "background_tasks_running": 0,
    "total_requests": 0,
}

# ---------------------------------------------------------------------------
# Pydantic v2 — Modelos de request/response
# ---------------------------------------------------------------------------

class CoinEnum(str, Enum):
    """Moedas suportadas pelo sistema."""
    BTC = "BTC"
    ETH = "ETH"
    BNB = "BNB"
    SOL = "SOL"
    XRP = "XRP"
    ADA = "ADA"
    DOGE = "DOGE"
    AVAX = "AVAX"
    DOT = "DOT"
    MATIC = "MATIC"


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str = Field(description="Status do servico: ok ou degraded")
    version: str = Field(description="Versao da API")
    uptime_seconds: float = Field(description="Tempo de atividade em segundos")
    timestamp: str = Field(description="Timestamp ISO 8601")
    components: dict[str, str] = Field(
        description="Status de cada componente do sistema"
    )


class PredictionResponse(BaseModel):
    """Previsao para uma moeda."""
    coin: str
    timestamp: str
    current_price: float
    predicted_price: float
    predicted_return: float
    direction: str
    confidence: float
    n_models: int
    model_predictions: dict[str, float] = Field(default_factory=dict)
    model_confidence: dict[str, float] = Field(default_factory=dict)
    price_lower_90: float | None = None
    price_upper_90: float | None = None
    scenarios: dict[str, float] | None = None
    cached: bool = Field(default=False, description="Se o resultado veio do cache")


class AllPredictionsResponse(BaseModel):
    """Previsoes para todas as moedas."""
    timestamp: str
    predictions: list[PredictionResponse]
    total: int


class PredictTriggerResponse(BaseModel):
    """Resposta ao disparar previsao em background."""
    message: str
    coin: str
    task_id: str


class SignalResponse(BaseModel):
    """Sinal de trading para uma moeda."""
    coin: str
    timestamp: str
    direction: str
    strength: float
    confidence: float
    predicted_return: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    model_agreement: float
    reasons: list[str] = Field(default_factory=list)


class ModelMetricsResponse(BaseModel):
    """Metricas de performance dos modelos para uma moeda."""
    coin: str
    models: dict[str, dict[str, float]] = Field(
        description="Metricas por modelo (mae, rmse, directional_accuracy, etc.)"
    )
    ensemble_metrics: dict[str, float] = Field(default_factory=dict)
    last_evaluated: str | None = None


class FeatureImportanceResponse(BaseModel):
    """Importancia de features para uma moeda."""
    coin: str
    method: str = Field(description="Metodo de calculo (shap, gain, permutation)")
    features: list[dict[str, Any]] = Field(
        description="Lista de {name, importance, rank}"
    )
    total_features: int


class SentimentResponse(BaseModel):
    """Sentimento de mercado (Fear & Greed Index)."""
    value: float = Field(description="Valor do indice (0=medo extremo, 100=ganancia extrema)")
    label: str = Field(description="Classificacao textual")
    timestamp: str
    source: str = "alternative.me"


class NarrativeScoresResponse(BaseModel):
    """Scores de narrativas cripto."""
    timestamp: str
    narratives: dict[str, float] = Field(
        description="Score por narrativa (0 a 1)"
    )
    dominant: str = Field(description="Narrativa dominante")
    dominant_score: float


class AgentRunRequest(BaseModel):
    """Request para executar pipeline de agentes."""
    coins: list[str] | None = Field(
        default=None,
        description="Moedas para analisar. Se None, usa todas configuradas."
    )
    timeframe: str = Field(default="1d", description="Timeframe OHLCV")
    since_days: int = Field(default=120, description="Dias de historico")
    portfolio_value: float = Field(default=0.0, description="Valor do portfolio")


class AgentRunResponse(BaseModel):
    """Resposta da execucao do pipeline de agentes."""
    message: str
    task_id: str
    coins: list[str]


class AgentStatusResponse(BaseModel):
    """Status dos agentes do sistema."""
    agents: dict[str, Any]
    orchestrator_available: bool
    latest_run: dict[str, Any] | None = None


class PortfolioRiskResponse(BaseModel):
    """Resumo de risco do portfolio."""
    timestamp: str
    total_exposure: float
    active_signals: int
    hold_signals: int
    risk_per_coin: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_drawdown_estimate: float | None = None


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada."""
    detail: str
    status_code: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Lifespan — startup e shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicacao: startup e shutdown."""
    # Startup
    _app_state["started_at"] = datetime.now(timezone.utc)
    logger.info("API Cripto_DT iniciando... versao %s", API_VERSION)

    # Pre-carregar config (nao bloqueia se falhar)
    _load_config()
    logger.info("Startup completo")

    yield

    # Shutdown
    logger.info("API Cripto_DT encerrando...")
    # Limpar caches e conexoes
    _app_state["predictions_cache"].clear()
    logger.info("Shutdown completo")


# ---------------------------------------------------------------------------
# Aplicacao FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cripto_DT - Sistema de Previsao de Criptomoedas",
    description=(
        "API REST para previsoes de precos de criptomoedas usando ensemble "
        "de modelos ML (LSTM, XGBoost, LightGBM, Helformer, MDN, etc.), "
        "sinais de trading com gestao de risco e pipeline multi-agente."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check e status do sistema",
        },
        {
            "name": "predictions",
            "description": "Previsoes de precos de criptomoedas",
        },
        {
            "name": "signals",
            "description": "Sinais de trading com gestao de risco",
        },
        {
            "name": "models",
            "description": "Metricas e analise dos modelos ML",
        },
        {
            "name": "market",
            "description": "Sentimento de mercado e narrativas",
        },
        {
            "name": "agents",
            "description": "Pipeline multi-agente (DataAgent, AnalystAgent, TraderAgent, RiskAgent)",
        },
        {
            "name": "portfolio",
            "description": "Analise de risco do portfolio",
        },
    ],
)

# CORS — permite acesso de frontends em qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — rate limiting e contagem de requests
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware de rate limiting e headers de controle."""
    _app_state["total_requests"] += 1

    client_ip = request.client.host if request.client else "unknown"

    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Rate limit excedido. Tente novamente em breve.",
                "status_code": 429,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    response = await call_next(request)

    # Headers informativos de rate limiting
    remaining = _rate_limiter.remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(_rate_limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-Request-ID"] = f"req-{_app_state['total_requests']}"

    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_coin(coin: str) -> str:
    """Valida e normaliza simbolo da moeda."""
    coin_upper = coin.upper()
    if coin_upper not in SUPPORTED_COINS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Moeda '{coin}' nao suportada. "
                f"Moedas disponiveis: {', '.join(SUPPORTED_COINS)}"
            ),
        )
    return coin_upper


def _get_cache(coin: str, max_age_seconds: int = 300) -> dict | None:
    """Retorna previsao do cache se ainda valida."""
    cached = _app_state["predictions_cache"].get(coin)
    if cached is None:
        return None
    age = (datetime.now(timezone.utc) - cached["cached_at"]).total_seconds()
    if age > max_age_seconds:
        return None
    return cached["data"]


def _set_cache(coin: str, data: dict):
    """Armazena previsao no cache."""
    _app_state["predictions_cache"][coin] = {
        "data": data,
        "cached_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------
def _run_prediction_background(coin: str):
    """Executa previsao em background e armazena no cache."""
    _app_state["background_tasks_running"] += 1
    try:
        predictor = _get_predictor()
        if predictor is None:
            logger.error("Predictor nao disponivel para background task de %s", coin)
            return

        result = predictor.predict_coin(coin)
        _set_cache(coin, result)
        logger.info("Previsao background para %s concluida e cacheada", coin)
    except Exception as exc:
        logger.error("Erro na previsao background de %s: %s", coin, exc)
    finally:
        _app_state["background_tasks_running"] -= 1


def _run_agents_background(
    coins: list[str] | None,
    timeframe: str,
    since_days: int,
    portfolio_value: float,
):
    """Executa pipeline de agentes em background."""
    _app_state["background_tasks_running"] += 1
    try:
        orchestrator = _get_orchestrator()
        if orchestrator is None:
            logger.error("Orchestrator nao disponivel para background task")
            return

        results = orchestrator.run(
            coins=coins,
            timeframe=timeframe,
            since_days=since_days,
            portfolio_value=portfolio_value,
        )

        # Cachear previsoes individuais dos resultados
        for coin, pred in results.get("predictions", {}).items():
            _set_cache(coin, pred)

        logger.info("Pipeline de agentes background concluido")
    except Exception as exc:
        logger.error("Erro no pipeline de agentes background: %s", exc)
    finally:
        _app_state["background_tasks_running"] -= 1


# ---------------------------------------------------------------------------
# Endpoints — Health
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check do sistema",
)
async def health_check():
    """Retorna status de saude do sistema e seus componentes."""
    started = _app_state["started_at"]
    uptime = 0.0
    if started:
        uptime = (datetime.now(timezone.utc) - started).total_seconds()

    # Verificar disponibilidade de cada componente
    components: dict[str, str] = {}

    components["config"] = "ok" if _load_config() is not None else "unavailable"

    try:
        components["predictor"] = "ok" if _get_predictor() is not None else "unavailable"
    except Exception:
        components["predictor"] = "error"

    try:
        components["signal_generator"] = (
            "ok" if _get_signal_generator() is not None else "unavailable"
        )
    except Exception:
        components["signal_generator"] = "error"

    try:
        components["orchestrator"] = (
            "ok" if _get_orchestrator() is not None else "unavailable"
        )
    except Exception:
        components["orchestrator"] = "error"

    try:
        components["narrative_detector"] = (
            "ok" if _get_narrative_detector() is not None else "unavailable"
        )
    except Exception:
        components["narrative_detector"] = "error"

    # Status geral: ok se pelo menos predictor ou orchestrator funcionam
    has_core = components.get("predictor") == "ok" or components.get("orchestrator") == "ok"
    overall_status = "ok" if has_core else "degraded"

    return HealthResponse(
        status=overall_status,
        version=API_VERSION,
        uptime_seconds=round(uptime, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


# ---------------------------------------------------------------------------
# Endpoints — Predictions
# ---------------------------------------------------------------------------
@app.get(
    "/predictions/{coin}",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Obter previsao para uma moeda",
    responses={
        404: {"model": ErrorResponse, "description": "Moeda nao suportada"},
        503: {"model": ErrorResponse, "description": "Predictor nao disponivel"},
    },
)
async def get_prediction(coin: str):
    """Retorna a previsao mais recente para uma moeda.

    Utiliza cache de 5 minutos para evitar reprocessamento.
    Se nao houver cache, executa previsao em tempo real.
    """
    coin = _validate_coin(coin)

    # Verificar cache
    cached = _get_cache(coin)
    if cached is not None:
        resp = PredictionResponse(**cached, cached=True)
        return resp

    # Executar previsao
    predictor = _get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor nao disponivel. Verifique se os modelos estao treinados.",
        )

    try:
        result = predictor.predict_coin(coin)
        _set_cache(coin, result)
        return PredictionResponse(**result, cached=False)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelos nao encontrados para {coin}: {exc}",
        )
    except Exception as exc:
        logger.error("Erro ao prever %s: %s", coin, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao gerar previsao para {coin}: {exc}",
        )


@app.get(
    "/predictions",
    response_model=AllPredictionsResponse,
    tags=["predictions"],
    summary="Obter previsoes para todas as moedas",
)
async def get_all_predictions():
    """Retorna previsoes para todas as moedas suportadas.

    Retorna dados cacheados quando disponivel. Moedas sem previsao sao omitidas.
    """
    predictions = []
    for coin in SUPPORTED_COINS:
        # Tentar cache primeiro
        cached = _get_cache(coin)
        if cached is not None:
            predictions.append(PredictionResponse(**cached, cached=True))
            continue

        # Tentar gerar previsao ao vivo
        predictor = _get_predictor()
        if predictor is None:
            continue

        try:
            result = predictor.predict_coin(coin)
            _set_cache(coin, result)
            predictions.append(PredictionResponse(**result, cached=False))
        except Exception as exc:
            logger.warning("Previsao indisponivel para %s: %s", coin, exc)

    return AllPredictionsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=predictions,
        total=len(predictions),
    )


@app.post(
    "/predict/{coin}",
    response_model=PredictTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["predictions"],
    summary="Disparar nova previsao em background",
    responses={
        404: {"model": ErrorResponse, "description": "Moeda nao suportada"},
    },
)
async def trigger_prediction(coin: str, background_tasks: BackgroundTasks):
    """Dispara uma nova previsao em background para a moeda especificada.

    Retorna imediatamente com HTTP 202. O resultado ficara disponivel
    via GET /predictions/{coin} apos o processamento.
    """
    coin = _validate_coin(coin)

    task_id = f"pred-{coin}-{int(time.time())}"
    background_tasks.add_task(_run_prediction_background, coin)

    return PredictTriggerResponse(
        message=f"Previsao para {coin} disparada em background",
        coin=coin,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Endpoints — Signals
# ---------------------------------------------------------------------------
@app.get(
    "/signals/{coin}",
    response_model=SignalResponse,
    tags=["signals"],
    summary="Obter sinal de trading para uma moeda",
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_signal(coin: str):
    """Retorna o sinal de trading para uma moeda com base na previsao mais recente.

    Combina previsoes de multiplos modelos para gerar direcao (LONG/SHORT/HOLD),
    stop-loss, take-profit e dimensionamento de posicao.
    """
    coin = _validate_coin(coin)

    # Precisamos de previsao e signal generator
    predictor = _get_predictor()
    sg = _get_signal_generator()

    if predictor is None or sg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor ou SignalGenerator nao disponivel.",
        )

    try:
        # Obter previsao (com cache)
        cached = _get_cache(coin)
        if cached is None:
            cached = predictor.predict_coin(coin)
            _set_cache(coin, cached)

        # Montar dicionario de predicoes por modelo para o SignalGenerator
        model_preds = cached.get("model_predictions", {})
        model_confs = cached.get("model_confidence", {})

        predictions_dict = {}
        for model_name in model_preds:
            predictions_dict[model_name] = {
                "pred": model_preds[model_name],
                "conf": model_confs.get(model_name, 0.5),
            }

        # Estimar ATR como 2% do preco atual (fallback simples)
        current_price = cached.get("current_price", 0.0)
        estimated_atr = current_price * 0.02

        signal = sg.generate_signal(
            predictions=predictions_dict,
            current_price=current_price,
            atr=estimated_atr,
            coin=coin,
        )

        return SignalResponse(
            coin=signal.coin,
            timestamp=signal.timestamp.isoformat(),
            direction=signal.direction,
            strength=signal.strength,
            confidence=signal.confidence,
            predicted_return=signal.predicted_return,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=signal.position_size,
            risk_reward_ratio=signal.risk_reward_ratio,
            model_agreement=signal.model_agreement,
            reasons=signal.reasons,
        )

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelos nao encontrados para {coin}: {exc}",
        )
    except Exception as exc:
        logger.error("Erro ao gerar sinal para %s: %s", coin, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar sinal para {coin}: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints — Models
# ---------------------------------------------------------------------------
@app.get(
    "/models/{coin}/metrics",
    response_model=ModelMetricsResponse,
    tags=["models"],
    summary="Obter metricas de performance dos modelos",
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_model_metrics(coin: str):
    """Retorna metricas de avaliacao (MAE, RMSE, acuracia direcional)
    para cada modelo treinado da moeda especificada.

    Le as metricas salvas em disco pelo pipeline de treinamento.
    """
    coin = _validate_coin(coin)

    cfg = _load_config()
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuracao nao disponivel.",
        )

    # Tentar carregar metricas do diretorio de reports
    import json
    from pathlib import Path

    reports_dir = cfg.training.reports_dir / coin
    metrics_path = reports_dir / "metrics.json"

    if not metrics_path.exists():
        # Tentar diretorio alternativo
        alt_path = cfg.training.models_dir / coin / "metrics.json"
        if alt_path.exists():
            metrics_path = alt_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Metricas nao encontradas para {coin}. "
                    f"Execute o pipeline de treinamento primeiro."
                ),
            )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao ler metricas de {coin}: {exc}",
        )

    # Separar metricas do ensemble das demais
    models_metrics = {}
    ensemble_metrics = {}
    for model_name, model_data in metrics_data.items():
        if model_name == "ensemble":
            ensemble_metrics = model_data if isinstance(model_data, dict) else {}
        elif isinstance(model_data, dict):
            models_metrics[model_name] = model_data

    # Tentar obter data da ultima avaliacao
    last_evaluated = None
    try:
        import os
        mtime = os.path.getmtime(metrics_path)
        last_evaluated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass

    return ModelMetricsResponse(
        coin=coin,
        models=models_metrics,
        ensemble_metrics=ensemble_metrics,
        last_evaluated=last_evaluated,
    )


# ---------------------------------------------------------------------------
# Endpoints — Features
# ---------------------------------------------------------------------------
@app.get(
    "/features/{coin}/importance",
    response_model=FeatureImportanceResponse,
    tags=["models"],
    summary="Obter importancia de features",
    responses={
        404: {"model": ErrorResponse},
    },
)
async def get_feature_importance(coin: str):
    """Retorna ranking de importancia de features para os modelos da moeda.

    Tenta carregar importancias SHAP ou gain do XGBoost/LightGBM salvas em disco.
    """
    coin = _validate_coin(coin)

    cfg = _load_config()
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuracao nao disponivel.",
        )

    import json
    from pathlib import Path

    # Procurar arquivo de importancia de features
    search_paths = [
        cfg.training.reports_dir / coin / "feature_importance.json",
        cfg.training.models_dir / coin / "feature_importance.json",
        cfg.training.reports_dir / coin / "shap_values.json",
    ]

    importance_data = None
    method = "unknown"

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    importance_data = json.load(f)
                if "shap" in path.name:
                    method = "shap"
                else:
                    method = "gain"
                break
            except Exception:
                continue

    if importance_data is None:
        # Fallback: tentar extrair do modelo XGBoost diretamente
        try:
            import joblib
            xgb_path = cfg.training.models_dir / coin / "xgb_reg.joblib"
            if xgb_path.exists():
                model = joblib.load(xgb_path)
                if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                    importances = model.model.feature_importances_

                    # Carregar nomes das features
                    cols_path = cfg.training.models_dir / coin / "feature_columns.json"
                    with open(cols_path, "r", encoding="utf-8") as f:
                        feature_names = json.load(f)

                    importance_data = {
                        name: float(imp)
                        for name, imp in zip(feature_names, importances)
                    }
                    method = "gain"
        except Exception as exc:
            logger.warning("Fallback feature importance falhou para %s: %s", coin, exc)

    if importance_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Importancia de features nao encontrada para {coin}. "
                f"Execute o pipeline de treinamento primeiro."
            ),
        )

    # Formatar como lista ordenada
    if isinstance(importance_data, dict):
        sorted_features = sorted(
            importance_data.items(), key=lambda x: abs(x[1]), reverse=True
        )
        features_list = [
            {"name": name, "importance": round(float(imp), 6), "rank": i + 1}
            for i, (name, imp) in enumerate(sorted_features)
        ]
    elif isinstance(importance_data, list):
        features_list = importance_data
    else:
        features_list = []

    return FeatureImportanceResponse(
        coin=coin,
        method=method,
        features=features_list,
        total_features=len(features_list),
    )


# ---------------------------------------------------------------------------
# Endpoints — Market
# ---------------------------------------------------------------------------
@app.get(
    "/market/sentiment",
    response_model=SentimentResponse,
    tags=["market"],
    summary="Obter sentimento de mercado (Fear & Greed Index)",
)
async def get_market_sentiment():
    """Retorna o indice Fear & Greed do mercado cripto via API alternative.me."""
    import requests as req_lib

    try:
        resp = req_lib.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        fng_data = data.get("data", [{}])[0]
        value = float(fng_data.get("value", 50))
        label = fng_data.get("value_classification", "Neutral")
        timestamp = fng_data.get("timestamp", "")

        # Converter unix timestamp para ISO
        if timestamp:
            try:
                ts = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                timestamp = ts.isoformat()
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        return SentimentResponse(
            value=value,
            label=label,
            timestamp=timestamp,
            source="alternative.me",
        )

    except Exception as exc:
        logger.warning("Erro ao buscar Fear & Greed Index: %s", exc)
        # Retornar valor neutro como fallback
        return SentimentResponse(
            value=50.0,
            label="Neutral (fallback - API indisponivel)",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="fallback",
        )


@app.get(
    "/market/narratives",
    response_model=NarrativeScoresResponse,
    tags=["market"],
    summary="Obter scores de narrativas cripto",
)
async def get_narratives():
    """Retorna scores das narrativas cripto dominantes baseado em analise de noticias.

    Narrativas incluem: DeFi, AI/Crypto, RWA, Memecoins, L2/Scaling,
    Regulacao, Institucional, Gaming/NFT.
    """
    detector = _get_narrative_detector()

    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NarrativeDetector nao disponivel.",
        )

    try:
        # Buscar artigos e detectar narrativas
        articles = detector._fetch_articles("BTC")  # Artigos gerais do mercado
        scores = detector.detect_narratives(articles)

        dominant = max(scores, key=scores.get) if scores else "none"
        dominant_score = scores.get(dominant, 0.0)

        return NarrativeScoresResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            narratives=scores,
            dominant=dominant,
            dominant_score=dominant_score,
        )
    except Exception as exc:
        logger.error("Erro ao detectar narrativas: %s", exc)
        # Retornar scores zerados como fallback
        from src.data.narrative_detector import NarrativeDetector
        empty_scores = {cat: 0.0 for cat in NarrativeDetector.NARRATIVE_KEYWORDS}
        return NarrativeScoresResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            narratives=empty_scores,
            dominant="none",
            dominant_score=0.0,
        )


# ---------------------------------------------------------------------------
# Endpoints — Agents
# ---------------------------------------------------------------------------
@app.post(
    "/agents/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["agents"],
    summary="Disparar pipeline completo de agentes",
)
async def trigger_agents(
    request: AgentRunRequest,
    background_tasks: BackgroundTasks,
):
    """Dispara execucao do pipeline multi-agente em background.

    Pipeline: DataAgent -> AnalystAgent -> TraderAgent -> RiskAgent.
    Resultados ficam disponiveis via GET /predictions e GET /signals.
    """
    orchestrator = _get_orchestrator()
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AgentOrchestrator nao disponivel.",
        )

    # Validar moedas se especificadas
    coins = request.coins
    if coins:
        coins = [_validate_coin(c) for c in coins]
    else:
        coins = SUPPORTED_COINS.copy()

    task_id = f"agents-{int(time.time())}"

    background_tasks.add_task(
        _run_agents_background,
        coins=coins,
        timeframe=request.timeframe,
        since_days=request.since_days,
        portfolio_value=request.portfolio_value,
    )

    return AgentRunResponse(
        message="Pipeline de agentes disparado em background",
        task_id=task_id,
        coins=coins,
    )


@app.get(
    "/agents/status",
    response_model=AgentStatusResponse,
    tags=["agents"],
    summary="Obter status dos agentes",
)
async def get_agents_status():
    """Retorna status de saude de cada agente do sistema e ultima execucao."""
    orchestrator = _get_orchestrator()

    agents_health: dict[str, dict[str, str]] = {}
    latest_run: dict[str, Any] | None = None

    if orchestrator is not None:
        try:
            agents_health = orchestrator.health_check()
        except Exception as exc:
            logger.warning("Erro no health check dos agentes: %s", exc)
            agents_health = {"error": {"status": "error", "message": str(exc)}}

        # Ultima execucao
        history = orchestrator.get_run_history()
        if history:
            latest_run = history[-1]
    else:
        agents_health = {
            "data_agent": {"status": "unavailable", "message": "Orchestrator nao inicializado"},
            "analyst_agent": {"status": "unavailable", "message": "Orchestrator nao inicializado"},
            "trader_agent": {"status": "unavailable", "message": "Orchestrator nao inicializado"},
            "risk_agent": {"status": "unavailable", "message": "Orchestrator nao inicializado"},
        }

    return AgentStatusResponse(
        agents=agents_health,
        orchestrator_available=orchestrator is not None,
        latest_run=latest_run,
    )


# ---------------------------------------------------------------------------
# Endpoints — Portfolio
# ---------------------------------------------------------------------------
@app.get(
    "/portfolio/risk",
    response_model=PortfolioRiskResponse,
    tags=["portfolio"],
    summary="Obter resumo de risco do portfolio",
)
async def get_portfolio_risk():
    """Retorna analise de risco do portfolio baseada nos sinais ativos.

    Calcula exposicao total, sinais ativos/inativos e risco por moeda.
    """
    sg = _get_signal_generator()
    predictor = _get_predictor()

    risk_per_coin: dict[str, dict[str, float]] = {}
    active_count = 0
    hold_count = 0
    total_exposure = 0.0

    for coin in SUPPORTED_COINS:
        cached = _get_cache(coin)
        if cached is None:
            continue

        model_preds = cached.get("model_predictions", {})
        model_confs = cached.get("model_confidence", {})
        current_price = cached.get("current_price", 0.0)

        if not model_preds or current_price <= 0:
            continue

        predictions_dict = {
            name: {"pred": model_preds[name], "conf": model_confs.get(name, 0.5)}
            for name in model_preds
        }

        if sg is not None:
            try:
                estimated_atr = current_price * 0.02
                signal = sg.generate_signal(
                    predictions=predictions_dict,
                    current_price=current_price,
                    atr=estimated_atr,
                    coin=coin,
                )

                risk_per_coin[coin] = {
                    "position_size": signal.position_size,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "confidence": signal.confidence,
                }

                if signal.direction != "HOLD":
                    active_count += 1
                    total_exposure += signal.position_size
                else:
                    hold_count += 1

            except Exception as exc:
                logger.warning("Erro ao calcular risco de %s: %s", coin, exc)
        else:
            # Sem SignalGenerator: usar dados basicos do cache
            confidence = cached.get("confidence", 0.0)
            predicted_return = cached.get("predicted_return", 0.0)
            risk_per_coin[coin] = {
                "position_size": 0.0,
                "confidence": confidence,
                "predicted_return": predicted_return,
            }
            hold_count += 1

    # Estimativa simplificada de max drawdown baseado na exposicao
    max_drawdown_estimate = None
    if total_exposure > 0:
        # Estimativa: max drawdown ~ exposicao * volatilidade media (assumindo 5% diario)
        max_drawdown_estimate = round(total_exposure * 0.05, 4)

    return PortfolioRiskResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_exposure=round(total_exposure, 4),
        active_signals=active_count,
        hold_signals=hold_count,
        risk_per_coin=risk_per_coin,
        max_drawdown_estimate=max_drawdown_estimate,
    )


# ---------------------------------------------------------------------------
# Entrypoint para execucao direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
