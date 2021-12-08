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

TF2ODAENV=tf24_py36
if [ ! -d "$TF2ODAENV" ] ; then
  virtualenv -p python3.8 $TF2ODAENV
  source ./$TF2ODAENV/bin/activate

  # Install necessary libraries
  python -m pip install --upgrade pip
  pip install --upgrade setuptools cython wheel
  
  # Install EML libraries
  pip install lxml xmltodict tdqm beautifulsoup4 pycocotools numpy tdqm pandas matplotlib pillow
  
  # Install TF2ODA specifics
  #pip install tensorflow==2.4.1
  
  cd $ROOTFOLDER
  
  echo # Install protobuf
  PROTOC_ZIP=protoc-3.14.0-linux-x86_64.zip
  curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/$PROTOC_ZIP
  unzip -o $PROTOC_ZIP -d protobuf
  rm -f $PROTOC_ZIP
  
  echo # Clone tensorflow repository
  git clone https://github.com/tensorflow/models.git
  cd models/research/
  cp object_detection/packages/tf2/setup.py .
  python -m pip install .
  
  echo # Add object detection and slim to python path
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  
  echo # Prepare TF2 Proto Files
  ../../protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

  echo # Test installation
  # If all tests are OK or skipped, then the installation was successful
  python object_detection/builders/model_builder_tf2_test.py
  
  echo #Test if Tensorflow works with CUDA on the machine. For TF2.4.1, you have to use CUDA 11.0
  python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
  
  echo "Important information: If there are any library errors, you have to install the correct versions manually. TFODAPI does install the latest version of "
  echo "tensorflow. However, in this script Tensorflow 2.4.1 is desired. Then, you have to uninstall the newer versions and replace with current versions."

  echo # Installation complete
  
else 
  echo $TF2ODAENV already exists
fi

cd $ROOTFOLDER
source ./venv/$TF2ODAENV/bin/activate

echo Created TF2ODA environment for TF2ODA inference

