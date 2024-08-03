FROM debian:12-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    python3 \
    python3-pip \
    dssp \
    wget \
    && rm -rf /var/lib/apt/lists/*
RUN dssp --version
# Build dssp from source
#RUN git clone https://github.com/PDB-REDO/dssp.git /dssp
#WORKDIR /dssp
#RUN cmake -S . -B build && \
#    cmake --build build && \
#    cmake --install build

# Install Foldseek
WORKDIR /
RUN wget https://mmseqs.com/foldseek/foldseek-linux-sse2.tar.gz && \
    tar xvzf foldseek-linux-sse2.tar.gz && \
    rm foldseek-linux-sse2.tar.gz

ENV PATH="/foldseek/bin:${PATH}"

# Install prot2d + dependencies
WORKDIR /app

COPY requirements.txt ./
COPY A0A087X1C5_alphafold.pdb ./
RUN pip3 install --break-system-packages --upgrade pip

# Install pro2d dependencies: break system is needed beacause system wide package downloads
RUN pip3 install --break-system-packages -r requirements.txt 

# Install pro2d
RUN pip3 install --break-system-packages prot2d
