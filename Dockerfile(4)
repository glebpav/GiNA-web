FROM python:3.9.6

WORKDIR /app
COPY ./scripts /app/scripts
COPY ./api_config.py /app
COPY ./app.py /app
COPY ./requirements.txt /app

RUN pip install -r /app/requirements.txt
EXPOSE 80

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "80"]
