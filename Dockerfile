FROM pytorch/pytorch:latest

RUN apt-get -qqy update
RUN apt-get -qqy upgrade
RUN apt-get -y install wget

RUN apt-get -qqy install python3 python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install numpy matplotlib scipy

WORKDIR /
WORKDIR app