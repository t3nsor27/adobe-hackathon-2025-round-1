FROM python:3.12.1

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# ENTRYPOINT ["python", "1b.py"]
ENTRYPOINT ["sh", "-c", "python 1a.py && python 1b.py"]