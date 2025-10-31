# Utiliser une image Python avec uv préinstallé
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Installer curl et autres dépendances système
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Variables d'environnement uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copier les fichiers de dépendances d'abord
COPY pyproject.toml uv.lock ./

# Installer les dépendances
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copier le code source
COPY . .

# Installer le projet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Installer Ollama CLI
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Créer la structure de dossiers
RUN mkdir -p chroma_db data

# Configurer l'environnement
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
EXPOSE 8501

ENTRYPOINT []