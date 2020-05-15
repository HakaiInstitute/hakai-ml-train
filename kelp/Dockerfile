ARG VERSION=1.5-cuda10.1-cudnn7-devel-apex-lightning
FROM tayden/pytorch:$VERSION

ENV PATH="/opt/code:${PATH}"
ENV DOCKER=TRUE

# Set up the program in the image
COPY kelp/deeplabv3_lightning.py /opt/code/deeplabv3_lightning.py
COPY models /opt/code/models
COPY utils /opt/code/utils
WORKDIR /opt/code

# Run the train script
ENTRYPOINT ["python", "deeplabv3_lightning.py"]
