FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
COPY ./requirements.txt /ws/requirements.txt
WORKDIR /ws
RUN pip install -r requirements.txt
RUN mkdir /ws/src
COPY src /ws/src
ENV PYTHONPATH "${PYTHONPATH}:/ws"
CMD ["python", "/ws/src/train.py"]


