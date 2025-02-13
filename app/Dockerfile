FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . ./app

# Environment variables
ENV PYTHONPATH=/app
ENV DEBUG=false

EXPOSE 8000
