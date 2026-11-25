# Release image: environment + workflow code. Built for linux/amd64 and linux/arm64 by
# .github/workflows/docker-image.yml on every push to main, and published as ghcr.io/bkieft-usa/jgi-integration:latest.
FROM quay.io/jupyter/base-notebook:latest

######## Install base software
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

######## Set up user environment
ENV HOME=/home/jovyan
ENV WORK=${HOME}/work
ENV REPO=${WORK}/jgi_integration
ENV MPLCONFIGDIR="${HOME}/.cache/matplotlib"
ENV PYTHONPATH="${REPO}"
RUN mkdir -p ${REPO}/venv ${HOME}/.cache/matplotlib ${HOME}/.cache/uv && \
    chown -R jovyan:users ${WORK} ${HOME}/.cache

######## Start as jovyan
USER jovyan
WORKDIR ${WORK}

######## Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="${HOME}/.local/bin:${PATH}"

######## Python environment (before the code, so code-only changes reuse this layer)
# dash ships a legacy JupyterLab source extension (only for inline Dash in notebooks, unused here);
# removing it stops JupyterLab from prompting for a rebuild.
COPY --chown=jovyan:users venv/pyproject.toml venv/uv.lock ${REPO}/venv/
RUN cd ${REPO}/venv && \
    uv sync --locked --python 3.11 && \
    uv cache clean && \
    rm -f .venv/share/jupyter/lab/extensions/dash-jupyterlab.tgz
ENV VIRTUAL_ENV=${REPO}/venv/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${REPO}/bin:${PATH}"
ENV MPLBACKEND=Agg

RUN python -m ipykernel install --user \
    --name=jgi-integration \
    --display-name="JGI Integration"

######## Workflow code (the build context is the repository; see .dockerignore)
COPY --chown=jovyan:users . ${REPO}
RUN cp ${REPO}/notebooks/workflow.ipynb ${WORK}/ && \
    cp ${REPO}/docker/jupyter_server_config.py ${HOME}/.jupyter/jupyter_server_config.py

# Commit the image was built from; recorded in each run's run_manifest.json (the image has no .git).
ARG GIT_SHA=""
ENV JGI_GIT_SHA=${GIT_SHA}

######## Ports: 8888 = JupyterLab, 8050 = integration explorer
EXPOSE 8888 8050
