FROM jupyter/base-notebook:latest

USER root

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN chmod +x /home/jovyan/.local/bin/uv

RUN mkdir -p /home/jovyan/.cache/matplotlib

RUN chown -R jovyan:users /home/jovyan/.cache

RUN chown -R jovyan:users /home/jovyan/.local

WORKDIR /home/jovyan/work

COPY docker-dev/requirements.txt /home/jovyan/work/tmp/requirements.txt

RUN /home/jovyan/.local/bin/uv pip install --system --no-cache -r /home/jovyan/work/tmp/requirements.txt

RUN rm -rf /home/jovyan/work/tmp/

USER jovyan

ENV HOME=/home/jovyan

RUN python -m ipykernel install --user --name=jgi-integration --display-name "JGI Integration"

ENV PYTHONPATH="/home/jovyan/work/jgi_integration"

ENV MPLCONFIGDIR="/home/jovyan/.cache/matplotlib"