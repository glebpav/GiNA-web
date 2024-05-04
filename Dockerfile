FROM python:3.9-alpine

WORKDIR /app
COPY ./nskit/ /app/nskit
COPY ./scripts /app/scripts
COPY ./api_config.py /app
COPY ./app.py /app
COPY ./requirements.txt /app

RUN apk add --no-cache gcc musl-dev python3-dev
RUN pip install ruamel.yaml.clib
RUN pip install /app/nskit
RUN pip install -r /app/requirements.txt
EXPOSE 80

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "80"]
