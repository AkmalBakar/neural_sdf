# Start from a standard Ubuntu image
FROM ubuntu:22.04

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y \
        wget \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc \
    && export PATH=/opt/conda/bin:$PATH \
    && conda create -n jax_env python=3.11 -y \
    && conda config --add channels anaconda \
    && conda config --add channels conda-forge \
    && conda install -n jax_env jaxlib=*=*cuda* jax -y \
    && conda install -n jax_env equinox optax pytorch mayavi matplotlib -y \
    && conda run -n jax_env pip install libigl \
    && conda clean -afy

# Set the default command to activate the conda environment and start a bash shell
SHELL ["conda", "run", "-n", "jax_env", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "jax_env"]
CMD ["/bin/bash"]