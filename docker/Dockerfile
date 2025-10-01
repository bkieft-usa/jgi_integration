FROM jupyter/base-notebook:latest

######## Set up user permissions to install base software
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

######## Set up user environment
USER jovyan
ENV HOME=/home/jovyan
ENV WORK=${HOME}/work
ENV MPLCONFIGDIR="${HOME}/.cache/matplotlib"
ENV PYTHONPATH="${WORK}/jgi_integration"
RUN mkdir -p ${WORK} ${MPLCONFIGDIR}
WORKDIR ${WORK}

######## Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${HOME}/.local/bin:${PATH}"

######## Set up software environment with uv
RUN git clone https://github.com/bkieft-usa/jgi_integration.git && \
    cd jgi_integration && \
    cp notebooks/integration_workflow.ipynb ../ && \
    cp docker/jupyter_server_config.py ${HOME}/.jupyter/jupyter_server_config.py && \
    cd venv && \
    uv sync --locked
ENV VIRTUAL_ENV=${WORK}/jgi_integration/venv/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

######## Set up user workflow environment
RUN python -m ipykernel install --user \
    --name=jgi-integration \
    --display-name="JGI Integration"

######## Ensure proper permissions after setting up container
USER root
RUN chown -R jovyan:users ${HOME}
USER jovyan