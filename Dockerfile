FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-vcstool \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools \
    ros-jazzy-clearpath-simulator \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
    'setuptools==68.2.2' \
    wheel \
    'numpy==1.26.4' \
    torch \
    tensorboard \
    'matplotlib==3.8.4' \
    'pandas==2.2.3'

RUN rosdep init || true
RUN rosdep update
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /workspaces/clearpath_docker
CMD ["/bin/bash"]
