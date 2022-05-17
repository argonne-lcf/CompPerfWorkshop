# Bootstrap our previous image
FROM romitmaulik1/alcftutorial1:mpi_app
MAINTAINER Romit rmaulik@anl.gov

WORKDIR /python_app
COPY source/hello_world.py .

RUN yum update -y && yum -y install epel-release
RUN yum -y install https://repo.ius.io/ius-release-el7.rpm https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

RUN yum -y makecache && yum -y install python3 python3-pip python3-devel && yum clean all

RUN pip3 install numpy tensorflow matplotlib

COPY submit.sh .

RUN chmod +x submit.sh

ENTRYPOINT ["/python_app/submit.sh"]
