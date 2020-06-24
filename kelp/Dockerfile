ARG VERSION=latest
FROM tayden/uav:$VERSION

ENV PYTHONPATH /opt/code:$PYTHONPATH
WORKDIR /opt/code

# Copy the script to the image
COPY utils /opt/code/utils
COPY models /opt/code/models
COPY kelp/*.py /opt/code/kelp/

# Run the cli script
ENTRYPOINT ["python", "kelp/cli.py"]
