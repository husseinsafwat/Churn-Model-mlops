FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .
COPY preprocessor.pkl .
COPY model.pkl .

EXPOSE 8000

CMD ["python", "main.py"]