import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from dataset import NYUDEPTHV2
from model import UNet

tran = transforms.Compose([transforms.Resize([224,224])])


folder = "train"
root = "/workspace/Dataset/nyudepthv2"

path = root + "/" + folder 

filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root, file))

filelist.sort()
data = {
    "h5": [x for x in filelist if x.endswith(".h5")]
}

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)

df2=df.sample(n=50000, replace=True).reset_index(drop="true")




def quantize(build_dir,quant_mode,batchsize):

  dset_dir = './dataset'
  float_model = './float'
  quant_model = './quant'
  print(batchsize)

  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda:0')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

  # load trained model
  model = UNet()
  model.load_state_dict(torch.load(os.path.join(float_model,'mde_v1.pth'), map_location=torch.device('cpu')), strict= False)
 
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 3, 224, 224])
  
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model


  # data loader
    
  test_dataset = NYUDEPTHV2(path, df2[int(df2.h5.size*0.99):].reset_index(drop="true"),  transform = None, transform2 = None)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batchsize, 
                                            shuffle=True)

  # evaluate 
  #test(quantized_model, device, test_loader)
  correct = 0 
  total = 0
  l1=[]  
  with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = quantized_model(images)
        l_depth = torch.mean(torch.abs(outputs - labels))
        l1.append(l_depth)
  print("l1:", torch.stack(l1).mean().item())

  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  #print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  #print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  #print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return



if __name__ == '__main__':
    run_main()
