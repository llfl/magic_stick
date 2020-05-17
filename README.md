# magic_stick

## 使用方法

### 确保设备支持nvidia cuda

命令行中输入nvidia-smi，查看nvidia设备信息，若无结果则，安装nvidia显卡驱动，并重启。

### 安装docker

参照[Get Docker Engine - Community for Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### 安装nvidia-docker

注：nvidia-docker需要nvidia驱动支持

参照[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### magicstick 镜像编译

1. cd到dockerOpenPose目录下执行docker build . -t magicstick 等待执行完成

2. 用nvidia-docker run即可运行magicstick镜像，附加参数--video_path 可附加video，附加参数--device 可以添加摄像头

## 脚本安装

### git

利用git 等工具或者直接下载文件放入文件夹，用sudo 运行install.sh 将会自进行上述操作

**注：应当事先安装好cuda或者nvidia驱动**

### demo运行

install之后将拍摄好的视频放入该文件夹内，运行./run_video.sh demo.mp4 其中demo.mp4为所需运行的视频文件

程序运行之后会在本目录生成output.mp4的输出文件 预测结果将在控制台显示。