FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Config dir (gateway writes .env here after bootstrap)
RUN mkdir -p /config /projects

# start.sh — restart gateway khi .env được tạo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
