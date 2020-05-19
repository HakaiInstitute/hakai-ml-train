ARG VERSION=1.5-cuda10.1-cudnn7-devel-apex
FROM tayden/pytorch:$VERSION

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"
ENV DOCKER=TRUE

# Install dependancies
RUN pip install pytorch-lightning rasterio fire

# Set up the program in the image
COPY kelp/deeplabv3_lightning.py /opt/code/deeplabv3_lightning.py
COPY utils /opt/code/utils
WORKDIR /opt/code

# Run the train script
ENTRYPOINT ["python", "deeplabv3_lightning.py"]
