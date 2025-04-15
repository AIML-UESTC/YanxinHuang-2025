## Start from this Docker image
FROM nvcr.io/nvidia/pytorch:23.07-py3

## Set workdir in Docker Container
# set default workdir in your docker container
# In other words your scripts will run from this directory
RUN mkdir /app
COPY requirements.txt /app

RUN  cd /app && python -m pip install --upgrade pip &&  python -m  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /app

## Copy all your files of the current folder into Docker Container
COPY ./ /app
RUN chmod a+x /app/inference.py

## Install requirements
#RUN pip3 install -r requirements.txt

## Make Docker container executable
ENTRYPOINT ["/opt/conda/bin/python", "/app/inference.py"]