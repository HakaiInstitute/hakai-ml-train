ARG VERSION=latest
FROM tayden/uav:$VERSION

WORKDIR /opt/code

# Copy the script to the image
COPY kelp_species/deeplabv3_lightning.py /opt/code/deeplabv3_lightning.py
COPY utils /opt/code/utils

# Run the train script
ENTRYPOINT ["python", "deeplabv3_lightning.py"]
