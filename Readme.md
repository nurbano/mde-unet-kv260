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
                                                  /nyudepthv2/
                                                            /train/ -> Train subset
                                                            /val/official/ -> Validation subset

## Train
```console
python train.py -d /media/nurbano/Datos7/Datasets/nyudepthv2 -e 20 -n 5000

------------------------------------
3.12.3 | packaged by conda-forge | (main, Apr 15 2024, 18:38:13) [GCC 12.3.0]
------------------------------------
Command line options:
 --dataset_dir    :  /media/nurbano/Datos7/Datasets/nyudepthv2
 --epochs      :  20
 --num_img      :  5000
 --evaluate      :  False
 --model_path      :  ./model.pth
```  
