ARG VERSION=1.5-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

RUN pip install tensorboard tqdm rasterio fire

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"
ENV DOCKER=TRUE

# Set up the program in the image
COPY kelp/segment_kelp.py /opt/code/segment_kelp.py
COPY models /opt/code/models
COPY utils /opt/code/utils
WORKDIR /opt/code

# Run the train script
ENTRYPOINT ["python", "segment_kelp.py"]
