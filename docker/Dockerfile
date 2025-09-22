FROM jupyter/minimal-notebook:python-3.11

WORKDIR /home/jgi_user/work

COPY docker/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/home/jgi_user/work"


EXPOSE 8888

ENV JUPYTER_TOKEN=change_me_to_a_secure_token

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--no-browser", \
     "--NotebookApp.token=${JUPYTER_TOKEN}", \
     "--NotebookApp.password=${JUPYTER_PASSWORD}", \
     "--NotebookApp.notebook_dir=/home/jgi_user/work", \
     "--LabApp.allow_remote_access=True"]