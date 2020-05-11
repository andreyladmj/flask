FROM ubuntu:16.04
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY requirements.txt /app/requirements.txt
WORKDIR /app
#ADD install-requirements.sh /app/install-requirements.sh
#RUN /app/install-requirements.sh
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]