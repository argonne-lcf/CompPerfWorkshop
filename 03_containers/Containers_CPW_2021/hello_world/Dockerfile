FROM centos:latest
MAINTAINER Romit rmaulik@anl.gov

WORKDIR /hello_world

RUN yum update -y
RUN yum groupinstall -y "Development Tools"
RUN yum install -y gcc-c++

COPY source/hello_world.cpp .
COPY submit.sh .

RUN chmod +x submit.sh

RUN g++ -o hello_world hello_world.cpp

ENTRYPOINT ["/hello_world/submit.sh"]
