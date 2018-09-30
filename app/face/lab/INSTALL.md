# 文件下载

[WFLW_final](https://wywu.github.io/projects/LAB/support/WFLW_final.tar.gz)

[WFLW_wo_mp](https://wywu.github.io/projects/LAB/support/WFLW_wo_mp.tar.gz)

解压到 ./models/WFLW

# 安装说明

## 安装CUDA

详见 https://blog.csdn.net/qq_31261509/article/details/78755968

## 安装依赖库

``sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler``

``sudo apt-get install --no-install-recommends libboost-all-dev``

``sudo apt-get install libatlas-base-dev``

``sudo apt-get install libopenblas-dev``

``sudo apt-get install libopencv-dev python-opencv python-dev``

``sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev``

## 修改部分配置

修改Makefile.config文件

``INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/``

Makefile中修改为

``LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial``

Makefile第399行之后添加

``COMMON_FLAGS += -lopencv_imgcodecs``

## 编译

到本项目根目录下，运行

``make clean``

``make all``

``make pycaffe``

## 添加系统配置

在 ~/.bashrc 文件中加入

``export PYTHONPATH=/path/to/this/python:$PYTHONPATH``

然后运行

``source ~/.bashrc``

替换 用本项目绝对路径替换掉 /path/to/this/，之后运行

``python2 face_aligner.py``

即可测试