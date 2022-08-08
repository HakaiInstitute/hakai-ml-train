ARG VERSION=1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install dependancies
COPY requirements.txt .
RUN apt-get update && \
    apt-get upgrade --assume-yes && \
    apt-get install --assume-yes git gcc rsync && \
    pip install -r requirements.txt

# Hack until we can use Pytorch v1.12.1 as a docker image
RUN sed 's/(not self._warned_capturable_if_run_uncaptured)/(not getattr(self, "_warned_capturable_if_run_uncaptured", False))/g' '/opt/conda/lib/python3.7/site-packages/torch/optim/optimizer.py'

# Copy the code to the image
COPY src .

# Run python by default
CMD "python"
