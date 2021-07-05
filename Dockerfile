FROM python:3.6-slim

WORKDIR /home

RUN apt update && apt install -y libpng-dev libfreetype6-dev pkg-config git g++ python3-tk

COPY ./ .
RUN pip install -U pip && pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home:/home/rllab"
RUN chmod u+x compile.sh && ./compile.sh
