FROM jupyter/base-notebook:latest

USER root

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/jovyan/.cache/matplotlib && chown -R jovyan:users /home/jovyan/.cache

WORKDIR /home/jovyan/work

USER jovyan

RUN git clone https://github.com/bkieft-usa/jgi_integration.git

RUN pip install --no-cache-dir -r /home/jovyan/work/jgi_integration/docker/requirements.txt ipykernel && \
    python -m ipykernel install --user --name=jgi-integration --display-name "JGI Integration"

RUN rm -rf /home/jovyan/work/tmp/jgi_integration

ENV PYTHONPATH="/home/jovyan/work/jgi_integration"
ENV MPLCONFIGDIR="/home/jovyan/.cache/matplotlib"