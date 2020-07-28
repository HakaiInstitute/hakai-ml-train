ARG VERSION=1.6.0-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install dependancies
COPY requirements.txt /opt/code/requirements.txt
RUN pip install -r /opt/code/requirements.txt

# Copy the script to the image
COPY utils /opt/code/utils
COPY models /opt/code/models
COPY seagrass/*.py /opt/code/seagrass/

# Run the cli script
ENTRYPOINT ["python", "seagrass/cli.py"]
