#!/bin/bash
xhost +localhost
docker build . -t magicstick &&\
# docker run --gpus all --rm --name openpose -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v `pwd`:/data magicstick /openpose/magic_stick/build/magicstick.bin --video /data/$1

docker run --gpus all --rm --name openpose -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v `pwd`:/data --device /dev/video0 magicstick /openpose/magic_stick/build/magicstick.bin --camera_resolution 480x320 --render_pose 0 --camera 0