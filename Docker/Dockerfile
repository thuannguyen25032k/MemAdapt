ARG CUDA_VERSION
ARG NVIDIA_VERSION

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --fix-missing --no-install-recommends \
        # Build tools
        python3-pip python3-venv python3-dev \
        build-essential libffi-dev wget git kmod vim bzip2 pciutils \
        # Updated libstdc++ — fixes GLIBCXX_3.4.29 error from conda packages
        libstdc++6 \
        # OpenGL / Mesa (software fallback + hardware)
        libgl1-mesa-glx libgl1-mesa-dri libgles2-mesa libglu1-mesa \
        # GL vendor-neutral dispatch (required by CoppeliaSim Qt5)
        libglvnd0 libgl1 libglx0 libegl1 \
        # X11 / display
        libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libdbus-1-3 \
        # Qt xcb platform plugin dependencies
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
        libxcb-cursor0 libxcb-shape0 libxcb-util1 \
        # Virtual display
        xvfb x11-utils \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY . /opt/embodiedbench

WORKDIR /opt/embodiedbench

COPY Docker/scripts/install_nvidia.sh /app/
RUN NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh

############################ Setup Env for EB-Man ############################

RUN conda env create -f conda_envs/environment_eb-man.yaml && conda clean --all -y

# Make embench_man the default Python env for all subsequent RUN steps and at runtime
ENV PATH=/opt/conda/envs/embench_man/bin:${PATH}

# Fix Qt "XDG_RUNTIME_DIR not set" warning — must exist with mode 0700
RUN mkdir -p /tmp/runtime-root && chmod 0700 /tmp/runtime-root
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

RUN pip install --no-build-isolation -e .

# 1. CoppeliaSim: download to /tmp, unzip, move to /opt/CoppeliaSim
RUN cd /tmp && \
    wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz && \
    tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz && \
    rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz && \
    mv /tmp/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04 /opt/CoppeliaSim

# 2. set CoppeliaSim related env variables
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/CoppeliaSim
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim

# 3. install PyRep: clone to /opt/PyRep (outside the volume-mounted workspace)
RUN git clone https://github.com/stepjam/PyRep.git /opt/PyRep && \
    cd /opt/PyRep && \
    pip install --no-build-isolation -r requirements.txt && \
    rm -f /opt/CoppeliaSim/libcoppeliaSim.so.1 && \
    pip install --no-build-isolation -e .

# 4. install EB-Manipulation
RUN cd /opt/embodiedbench/embodiedbench/envs/eb_manipulation && \
    pip install --no-build-isolation -r requirements.txt && \
    pip install --no-build-isolation -e .

# 5. copy simAddOnScript_PyRep.lua to CoppeliaSim's installing directory
RUN cp /opt/embodiedbench/embodiedbench/envs/eb_manipulation/simAddOnScript_PyRep.lua \
       /opt/CoppeliaSim/

CMD ["/bin/bash"]