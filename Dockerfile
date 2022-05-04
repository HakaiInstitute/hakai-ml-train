ARG VERSION=1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install dependancies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install --assume-yes git gcc rsync apt-utils && \
    apt-get upgrade --assume-yes && \
    pip install -r requirements.txt

# Copy the code to the image
COPY src .

# Run python by default
CMD "python"
