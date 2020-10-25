FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install python-opencv -y

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install numpy

RUN python3 -m pip install opencv_transforms
RUN python3 -m pip install opencv-python

RUN python3 -m pip install matplotlib

RUN python3 -m pip install openpyxl
RUN python3 -m pip install efficientnet_pytorch

RUN python3 -m pip install tensorboard

RUN python3 -m pip install tiffile