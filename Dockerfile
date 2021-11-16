ARG VERSION=1.10.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install dependancies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code to the image
COPY src .

# Run python by default
CMD "python"
