#!/bin/bash

#1. Create a folder for your datasets. Usually, multiple users use one folder for all datasets to be able to share them. Later on, in the 
#training and inference scripts, you will need the path to the dataset.
#2. Create the EML tools folder structure, e.g. ```eml-tools```. The structure can be found here: https://github.com/embedded-machine-learning/eml-tools#interface-folder-structure
ROOTFOLDER=`pwd`

#In your root directory, create the structure. Sample code
mkdir -p eml_projects
mkdir -p venv

#3. Clone the EML tools repository into your workspace
EMLTOOLSFOLDER=./eml-tools
if [ ! -d "$EMLTOOLSFOLDER" ] ; then
  git clone https://github.com/embedded-machine-learning/eml-tools.git "$EMLTOOLSFOLDER"
else 
  echo $EMLTOOLSFOLDER already exists
fi

#4. Create the task spooler script to be able to use the correct task spooler on the device. In our case, just copy
#./init_ts.sh

# Project setup
#5. Create a virtual environment for TF2ODA in your venv folder. The venv folder is put outside of the project folder to 
#avoid copying lots of small files when you copy the project folder. Conda would also be a good alternative.
# From root
cd $ROOTFOLDER

cd ./venv

TF2ODAENV=tf23_py36
if [ ! -d "$TF2ODAENV" ] ; then
  python3 -m venv $TF2ODAENV
  source ./$TF2ODAENV/bin/activate

  # Install necessary libraries
  python -m pip install --upgrade pip --no-cache-dir
  pip install --upgrade setuptools cython wheel numpy==1.19.4
  
  # Install EML libraries
  pip install lxml xmltodict tdqm beautifulsoup4 pycocotools tdqm pandas matplotlib pillow
  
  # Install TF2ODA specifics
  pip install h5py==2.10.0
  pip install numpy==1.19.4 grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
  pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==2.3.1+nv20.12
  
  echo # Install protobuf
  PROTOC_ZIP=protoc-3.14.0-linux-aarch_64.zip
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
  unzip -o $PROTOC_ZIP -d protobuf
  rm -f $PROTOC_ZIP
 
  echo # Build TF addons
  git clone https://github.com/tensorflow/addons.git
  sudo rm /usr/lib/lib_pywrap_tensorflow_internal.so
  sudo ln -s $ROOTFOLDER/venv/$TF2ODAENV/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so /usr/lib/lib_pywrap_tensorflow_internal.so
  cd addons
  export TF_NEED_CUDA="1"
  export TF_CUDA_VERSION="10"
  export TF_CUDNN_VERSION="8"
  export CUDA_TOOLKIT_PATH="/usr/local/cuda"
  export CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu"
  sed -i -e "s/write(\"build --cxxopt=-std=c++14\")/# removed/" ./configure.py
  sed -i -e "s/write(\"build --host_cxxopt=-std=c++14\")/# removed/" ./configure.py
  python3 ./configure.py
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
  pip install artifacts/tensorflow_addons-*.whl	
  cd ..
  
  # TF models R 2.3.0
  echo # Clone tensorflow repository
  git clone https://github.com/tensorflow/models.git
  cd models
  git checkout a84f1b96053af058cb6b493c2d83e5e612a299c5
  cd research

  python -m pip install tf-slim cython contextlib2 pillow lxml jupyter 
  
  echo # Add object detection and slim to python path
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  
  echo # Prepare TF2 Proto Files
  ../../protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

  echo # Test installation
  # If all tests are OK or skipped, then the installation was successful
  python object_detection/builders/model_builder_tf2_test.py
  
  echo # Test if Tensorflow works with CUDA on the machine
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
  
  echo "Important information: If there are any library errors, you have to install the correct versions manually."

  echo # Installation complete
  
else 
  echo $TF2ODAENV already exists
fi

cd $ROOTFOLDER
source ./venv/$TF2ODAENV/bin/activate

echo Created TF2ODA environment for TF2ODA inference

