# AMD Open Hardware 2024
## Monocular Depth estimation for interior scenarios in real time on Kria KV260
From a single RGB image, infer in real-time with a Kria KV260 a depth map using convolutional neural networks.
![Diagram](/diagram.png "Diagram MDE KV260")
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
For this project, the model was training with 50k images and 50 epochs, with L1 losses, AdamW for optimizer with and lr=0.0001. The 50k images was subset in 90% for training and 10% for validate.


