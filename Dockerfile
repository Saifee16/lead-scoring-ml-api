FROM python:3.12-slim AS trainer

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY common ./common
COPY ml ./ml
COPY data ./data

RUN python -m ml.train


FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV APP_DEBUG=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY common ./common
COPY --from=trainer /build/artifacts ./artifacts

RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8000

HEALTHCHECK     --interval=30s     --timeout=3s     --start-period=10s     --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
