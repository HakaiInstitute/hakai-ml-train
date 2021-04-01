ARG VERSION=latest
FROM pytorch/pytorch:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Install dependancies
COPY requirements.txt /opt/code/requirements.txt
RUN pip install -r /opt/code/requirements.txt

# Copy the script to the image
COPY utils /opt/code/utils
COPY models /opt/code/models
COPY kelp/*.py /opt/code/kelp/
COPY ./*.py /opt/code/

# Run the cli script
ENTRYPOINT ["python", "lit_deeplabv3_kelp_presence.py"]
