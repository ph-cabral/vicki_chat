FROM python:3.11-slim

# Timezone igual al resto del stack
ARG TZ=America/Argentina/Buenos_Aires
ENV TZ=${TZ}
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        curl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Healthcheck compatible con el patrón del docker-compose existente
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

