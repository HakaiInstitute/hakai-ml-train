ARG VERSION=1.8.1-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Install dependancies
COPY requirements.txt /opt/code/requirements.txt
RUN pip install -r /opt/code/requirements.txt

# Copy the code to the image
COPY . .

# Run the cli script
#ENTRYPOINT ["python", "models/lit_deeplabv3_resnet101_kelp.py"]
ENTRYPOINT ["python", "models/lit_lraspp_mobilenet_v3_large_kelp.py"]
