FROM python:3.10.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8008

CMD uvicorn api:app --port 8008 --host 0.0.0.0