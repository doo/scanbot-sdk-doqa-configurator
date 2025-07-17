FROM python:3.13-bullseye

RUN apt update && apt install -y libgl1 # OpenCV dependency

COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ARG SCANBOT_SDK_CORE_VERSION

RUN echo Platform: `uname -m`
RUN python -m wget https://scanbotsdk-deployment.s3.eu-west-1.amazonaws.com/ScanbotSDK-Linux/${SCANBOT_SDK_CORE_VERSION}-py3-none-linux_`uname -m`.whl -o /tmp/
RUN pip install /tmp/scanbotsdk*.whl
RUN rm /tmp/scanbotsdk*.whl

COPY src/*.py /app/

ENTRYPOINT ["python", "/app/train_docker.py"]