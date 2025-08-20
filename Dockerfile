FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY shodan_to_qdrant.py .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "shodan_to_qdrant.py" ]