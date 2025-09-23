FROM jupyter/base-notebook

WORKDIR /home/jgi_user/work

COPY . .

RUN pip install --no-cache-dir -r docker/requirements.txt

RUN pip install ipykernel && \
    python -m ipykernel install --user --name=jgi-integration --display-name "JGI Integration"

ENV PYTHONPATH="/home/jgi_user/work"