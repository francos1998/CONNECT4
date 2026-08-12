FROM python:3.12.7

WORKDIR /home/bitnami/connect-four

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY server.py .
COPY models/cnnmodel2.h5 ./models/
COPY models/transformer2.keras ./models/

ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
