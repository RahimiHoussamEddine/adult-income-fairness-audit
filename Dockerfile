FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY pyproject.toml requirements.lock README.md NOTICE.md ./
COPY src ./src
COPY app.py ./
COPY docs ./docs
COPY reports ./reports
COPY data/source_manifest.json ./data/source_manifest.json
RUN python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
