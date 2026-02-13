FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

RUN useradd --create-home --shell /bin/bash mvhe
USER mvhe

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--paper", "--strategy", "singularity"]
