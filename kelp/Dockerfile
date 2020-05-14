ARG VERSION=20.03-py3
FROM nvcr.io/nvidia/pytorch:$VERSION

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"
ENV DOCKER=TRUE

# Install required python packages
RUN pip install rasterio fire pytorch-lightning

# Set up the program in the image
COPY kelp/segment_kelp.py /opt/code/segment_kelp.py
COPY models /opt/code/models
COPY utils /opt/code/utils
WORKDIR /opt/code

# Run the train script
ENTRYPOINT ["python", "segment_kelp.py"]
