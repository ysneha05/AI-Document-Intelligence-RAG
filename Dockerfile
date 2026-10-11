FROM python:3.12-slim

WORKDIR /app
RUN mkdir -p /app/data

COPY requirements.txt .


RUN python -m pip install --no-cache-dir --retries 10 --timeout 120 --index-url https://pypi.org/simple -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]