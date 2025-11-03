FROM python:3.12-slim-trixie

ENV PATH="/root/.local/bin/:$PATH" \
    STREAMLIT_PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE ${STREAMLIT_PORT}

CMD uv run python -m streamlit run app/app.py \
    --server.port=${STREAMLIT_PORT} \
    --server.address=0.0.0.0
