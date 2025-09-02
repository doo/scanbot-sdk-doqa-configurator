FROM python:3.13-bullseye

RUN apt update && apt install -y libgl1 # OpenCV dependency

COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/*.py src/*ipynb src/*.svg /app/

ARG SCANBOT_SDK_CORE_VERSION

# Only download the whl file if there isn't already a scanbotsdk*.whl file we can use
COPY scanbotsdk*.whl /
RUN if ! ls scanbotsdk*.whl 2>/dev/null; then python /app/download_scanbotsdk_whl.py $SCANBOT_SDK_CORE_VERSION linux_`uname -m`; fi
RUN ls && pip install scanbotsdk*linux_`uname -m`.whl
RUN rm scanbotsdk*.whl

ENTRYPOINT ["python", "/app/train.py"]
