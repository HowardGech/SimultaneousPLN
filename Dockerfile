FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip

RUN apt-get update && apt-get install -y \
    liblapack-dev \
    gcc \
    gfortran \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN rm /usr/lib/python*/EXTERNALLY-MANAGED

RUN pip3 install jupyter \
    && pip3 install numpy==1.26.4 \
    && pip3 install scipy==1.13.1\
    && pip3 install cython==0.29.37\
    && pip3 install networkx==3.1\
    && pip3 install pandas==2.2.3\
    && pip3 install seaborn==0.13.2\
    && pip3 install matplotlib==3.9.1

COPY . .

RUN pip3 install .

RUN useradd -m -s /bin/bash jupyteruser
# Switch to the non-root user
USER jupyteruser

# Set the working directory to the home of the non-root user

ENTRYPOINT ["jupyter", "notebook", "--ip=*"]