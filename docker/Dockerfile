FROM jupyter/base-notebook:latest

USER root

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN chmod +x /home/jovyan/.local/bin/uv

RUN mkdir -p /home/jovyan/.cache/matplotlib

RUN chown -R jovyan:users /home/jovyan/.cache

RUN chown -R jovyan:users /home/jovyan/.local

WORKDIR /home/jovyan/work

USER jovyan

RUN git clone https://github.com/bkieft-usa/jgi_integration.git

RUN /home/jovyan/.local/bin/uv pip install --system --no-cache -r /home/jovyan/work/jgi_integration/docker/requirements.txt ipykernel

RUN python -m ipykernel install --user --name=jgi-integration --display-name "JGI Integration"

RUN cp /home/jovyan/work/jgi_integration/notebooks/integration_workflow.ipynb /home/jovyan/work/

ENV PYTHONPATH="/home/jovyan/work/jgi_integration"

ENV MPLCONFIGDIR="/home/jovyan/.cache/matplotlib"