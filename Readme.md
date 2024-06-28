## Monocular Depth estimation for interior scenarios in real time on Kria KV260
From a single RGB image, infer in real-time with a Kria KV260 a depth map using convolutional neural networks.
![Diagram](/diagram.png "Diagram MDE KV260")
### AMD Open Hardware 2024
### Team number: AOHW-305
Participants:
- Nicolás Urbano Pintos (UTN FRH /CITEDEF)
- Monal Patel Rakeshbhai (UMONS)

Supervisor:
- Carlos Valderrama (UMONS)

Steps:

<img src="steps.png" width="250" height="300">

- Train a U-NET model with NYUDEPTHV2 dataset in pytorch.
- Quantize the model with VITIS AI.
- Compile the quantized model for DPU.
- Evaluate the model in the Kria KV260 

## Prepare conda env for training
In a linux pc with conda:
```console
conda create -n mde_unet-env
conda activate mde_unet-env
git clone https://github.com/nurbano/mde-unet-kv260
cd mde-unet-kv260
pip install requirements.txt
```


## Prepare Dataset
1- Download the dataset from http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz
2- Extract the dataset. The directory structure is:
```
nyudepthv2
│
└───train
│   │
│   └───basement_0001a
│   │       │   00001.h5
│   │       │   00006.h5
│   │       │   ...
│   └───basement_0001b
│   │       │   00001.h5
│   │       │   00006.h5
│   │       │   ...
│   └───...
│   │   
└───val
    │ 
    └───official
    │       │   00001.h5
    │       │   00002.h5
    │       │   ...
```
                                                         
## Train
To train the model use this script:
```console
python train.py -d /path/to/nyudepthv2 -e 20 -n 5000
```  
The arguments are:

```console
Command line options:
 --dataset_dir    :  /path/to/
 --epochs      :  20
 --num_img      :  5000
 --evaluate      :  False
 --model_path      :  ./model.pth
```
For this project, the model was training with 50k images and 50 epochs. And the default losses (L1) and default optimizer (AdamW) with a lr=0.0001. The 50k images was subset in 90% for training and 10% for validate.

## Evaluate in PC
For evaluate the model in PC, run the jupyter notebook infer.ipynb in the conda env.

## Prepare the Vitis AI 1.4.1 for CPU
In a ubuntu PC:
- First install docker:
```console
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
- Manager Docker as non-root user:
```console
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
- Clone the Vitis AI repository:
```console
  git clone --recurse-submodules --branch 1.4.1 https://github.com/Xilinx/Vitis-AI
```
- Go to the Vitis AI directory:
```console
  cd Vitis-AI
```
- Pull the pre-built docker image:
```console
  docker pull xilinx/vitis-ai-cpu:1.4.1.978
```
- Modify the ducker_run.sh bash to add the path of the dataset. After the line 93 include this: -v /path/to/nyudepthv2:/workspace/Dataset/nyudepthv2
- Run the docker:
```console
./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978
```
- Activate the conda environment for pytorch:
```console
conda activate vitis-ai-pytorch
```
- Create a directory for the project:
```console
mkdir custom_ai_model
```
- Is necessary copy this reposotory directory to path/to/Vitis-AI/custom_ai_model

## Quantize
First calibrate the quantize model.
```console
python3 src/quantize.py -q calib -b 4
```
And then test the quantized model
```console
python3 src/quantize.py -q test
``` 
## Compile for DPU
```console
bash compile.sh kv260
```
## Prepare the SD for kria kv260
For this project is necessary use the Ubuntu 20.21, download the image from:
https://people.canonical.com/~platform/images/xilinx/kria/iot-kria-classic-desktop-2004-x03-20211110-98.img.xz?_gl=1*16usueu*_gcl_au*Mzk5MTE0ODkyLjE3MTExMDYyMjg.&_ga=2.59807085.1133415059.1711106229-1097448103.1710778457
- Extract the image:
```console
unxz iot-limerick-kria-classic-desktop-2204-x07-20230302-63.img.xz
```
- Format a SD:
```console
mkfs.exfat /dev/sdd
```
- Copy the img to the sd:
```console
dd if=iot-limerick-kria-classic-desktop-2204-x07-20230302-63.img of=/dev/sdd conv=fsync status=progress
```
- To setup the Kria KV260 following the steps of this link:
  https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit/getting-started/getting-started.html
  
## Evaluate the model in the kria kv260

- Connect to a net with a ethernet cable.
- Turn on the kria kv260.
- 
