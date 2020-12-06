FROM python:3.7

COPY . /app
WORKDIR /app

RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN pip install -r requirements.txt
EXPOSE 8180
EXPOSE 8181

COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
