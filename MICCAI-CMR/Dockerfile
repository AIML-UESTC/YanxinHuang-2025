## Start from this Docker image
FROM nvcr.io/nvidia/pytorch:23.07-py3

## Set workdir in Docker Container
# set default workdir in your docker container
# In other words your scripts will run from this directory
RUN mkdir /app
WORKDIR /app

## Copy all your files of the current folder into Docker Container
COPY ./ /app
RUN chmod a+x /app/inference.py

## Install requirements
RUN pip install -r requirements.txt

## Make Docker container executable
ENTRYPOINT ["/usr/bin/python", "/app/inference.py"]