# Build image menggunakan python 3.12
FROM python:3.12-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx

WORKDIR /app/MLProject

RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    dagshub \
    python-dotenv

COPY . /app/MLProject/

RUN mkdir -p /app/dataset

ENTRYPOINT ['mlflow', 'run', '.', '--env-manager=local']