#!/bin/bash

container_name=$1
if [ -z "$container_name" ]; then
    container_name="pcl_cuda_cccl"
fi

docker run --gpus all -it --ipc=host\
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/workspace \
    --name $container_name \
    pcl_cuda_cccl:cu-12.2 