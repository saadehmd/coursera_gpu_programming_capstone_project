FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

ARG USER_UID
ARG USER_GID
ARG USERNAME

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system compilation tools and development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    gdb \
    pkg-config \
    libusb-1.0-0-dev \
    libflann-dev \
    libboost-all-dev \
    libeigen3-dev \
    libvtk9-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libqt5opengl5-dev \
    freeglut3-dev \
    libglew-dev \
    libxi-dev \
    libxmu-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone and install the latest NVIDIA CCCL.
WORKDIR /opt
RUN git clone --recursive https://github.com/nvidia/cccl.git && \
    cd cccl && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) install

# 3. Clone and build PCL with GPU/CUDA capabilities and Visualization libs.
RUN git clone https://github.com/PointCloudLibrary/pcl.git  && \
    cd pcl && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=20 \
          -DBUILD_GPU=ON \
          -DBUILD_CUDA=ON \
          -DCOMPUTE_CAPABILITY=8.9 \
          -DBUILD_visualization=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install


# 4. Environment configuration for modern CUDA toolkit paths
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV QT_X11_NO_MITSHM=1

# Add sudo support to chown the workspace folder by the user of the machine.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && apt-get update \
    && apt-get install -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p /workspace \
    && chown -R ${USERNAME}:${USERNAME} /workspace \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["/bin/bash"]

