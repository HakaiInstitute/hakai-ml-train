FROM continuumio/miniconda3

RUN conda update -n base -c defaults conda
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "uav", "/bin/bash", "-c"]

WORKDIR /workspace
COPY . .