FROM python:3.9

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir -r req.txt

EXPOSE 8000

CMD ["uvicorn", "script:app", "--host", "0.0.0.0", "--port", "8000"]