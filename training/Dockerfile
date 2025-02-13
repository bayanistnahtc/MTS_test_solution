FROM bitnami/pytorch

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training.py .
COPY data ./data

CMD ["python", "train_run_example.py"]