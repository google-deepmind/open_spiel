# base image (3.13 GB)
FROM ubuntu:18.04 as base
RUN apt update
RUN dpkg --add-architecture i386 && apt update
RUN apt-get -y install \
    clang \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    sudo
RUN mkdir repo
WORKDIR /repo

RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install matplotlib

# install
COPY . .
RUN ./install.sh
RUN pip3 install --upgrade setuptools testresources 
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install --upgrade cmake

# build and test
RUN mkdir -p build
WORKDIR /repo/build
RUN cmake -DPython_TARGET_VERSION=${PYVERSION} -DCMAKE_CXX_COMPILER=`which clang++` ../open_spiel 
RUN make -j12
RUN ctest -j12
ENV PYTHONPATH=${PYTHONPATH}:/repo
ENV PYTHONPATH=${PYTHONPATH}:/repo/build/python
WORKDIR /repo/open_spiel

# slim image (2.26GB) for development in Python
FROM python:3.6-slim-buster as python-minimal
RUN mkdir repo
WORKDIR /repo
COPY --from=base /repo .
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install matplotlib
ENV PYTHONPATH=${PYTHONPATH}:/repo
ENV PYTHONPATH=${PYTHONPATH}:/repo/build/python
WORKDIR /repo/open_spiel
