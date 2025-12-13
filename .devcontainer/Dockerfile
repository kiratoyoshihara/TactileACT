FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    libgl1-mesa-dev \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch \
    numpy \
    matplotlib \
    h5py \
    opencv-python \
    Pillow

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /root/catkin_ws