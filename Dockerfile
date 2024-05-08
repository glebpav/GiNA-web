FROM amd64/ubuntu:latest

RUN apt-get -y update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y install python3.9
RUN apt-get -y install python3.9-distutils
RUN apt-get -y install python3-pip

WORKDIR /app
COPY ./scripts /app/scripts
COPY ./api_config.py /app
COPY ./app.py /app
COPY ./requirements.txt /app

RUN python3.9 -m pip install -r /app/requirements.txt
# RUN python3.9 -m pip install  naskit -v
# RUN pip install -r /app/requirements.txt
EXPOSE 80

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "80"]
