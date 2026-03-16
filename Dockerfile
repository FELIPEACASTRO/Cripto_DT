# =============================================================================
# Cripto DT - Sistema de Previsao de Criptomoedas com IA/ML
# Build multi-stage para imagem otimizada
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - instala dependencias e compila pacotes
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

LABEL maintainer="Cripto DT Team"
LABEL version="1.0.0"
LABEL description="Sistema de previsao de criptomoedas com IA/ML - builder stage"

WORKDIR /build

# Dependencias de sistema para compilacao (numpy, torch, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instala todas as dependencias em um virtualenv isolado
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] fastapi httpx

# ---------------------------------------------------------------------------
# Stage 2: Runtime - imagem final enxuta
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL maintainer="Cripto DT Team"
LABEL version="1.0.0"
LABEL description="Sistema de previsao de criptomoedas com IA/ML"

# Dependencias minimas de runtime (libgomp para LightGBM/XGBoost, curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copia virtualenv do builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copia codigo-fonte do projeto
COPY config/ ./config/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY requirements.txt .

# Diretorios para dados persistentes (montados via volumes)
RUN mkdir -p data/raw data/processed data/predictions models logs reports

# Portas: 8000 (API FastAPI), 8501 (Streamlit dashboard)
EXPOSE 8000 8501

# Healthcheck via endpoint /health da API
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrao: servidor FastAPI
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
