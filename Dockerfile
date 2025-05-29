FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
WORKDIR /home/app
COPY  docker_requirements.txt .
EXPOSE 8095
RUN pip install -r docker_requirements.txt
