FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt .

RUN pip install --no-cache-dir -r base.txt

COPY src/ ./src/
COPY app/ ./app/

EXPOSE 7860

ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD [ "python", "app/gradio_app.py" ]