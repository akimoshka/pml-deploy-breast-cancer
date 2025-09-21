FROM python:3.11-slim

WORKDIR /app
COPY code/deployment/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app /app
COPY models /models

EXPOSE 8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
