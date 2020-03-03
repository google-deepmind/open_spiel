FROM ubuntu:18.04
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

RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install matplotlib

# install
COPY . .
RUN ./install.sh

RUN pip3 install --upgrade setuptools testresources 
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install --upgrade cmake

# build and test
RUN mkdir -p build && \
    cd build && \
    cmake -DPython_TARGET_VERSION=${PYVERSION} -DCMAKE_CXX_COMPILER=`which clang++` ../open_spiel && \
    make -j4 && \
    ctest -j4
COPY . build

# export path
ENV PYTHONPATH=${PYTHONPATH}:/open_spiel/
ENV PYTHONPATH=${PYTHONPATH}:/open_spiel/build/python

WORKDIR ./open_spiel
