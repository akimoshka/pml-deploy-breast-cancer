FROM python:3.11-slim

WORKDIR /app
COPY code/deployment/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api /app

COPY models /models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]