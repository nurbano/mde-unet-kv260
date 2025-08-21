import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import argparse
import os
import shutil
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from dataset import NYUDEPTHV2


tran = transforms.Compose([transforms.Resize([224,224])])

folder = "val/official"
root = "/media/nurbano/Datos12/Datasets/nyudepthv2"

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
print(df.head())
# df = df.sample(frac=1, random_state=42)

# df2=df.sample(n=50000, replace=True).reset_index(drop="true")
df2= df
DIVIDER = '-----------------------------------------'

def generate_images(dset_dir, num_images, dest_dir):

  
  test_dataset = NYUDEPTHV2(dset_dir, df2.reset_index(drop="true"),  transform = tran, transform2 = None)
  
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, 
                                            shuffle=True)

  # iterate thru' the dataset and create images
  dataiter = iter(test_loader)
  for i in tqdm(range(num_images)):
    image, label = next(dataiter)
    #img = np.array(image)
    img = image.numpy().squeeze()
    img = ((img/2+0.5)*255).astype(np.uint8)
    img_n= np.zeros((224,224,3))
    img_n[:,:,0]= img[2,:,:]
    img_n[:,:,1]= img[1,:,:]
    img_n[:,:,2]= img[0,:,:]
    #print(img)
    #print(img.shape)
    idx = label.numpy()
    img_file=os.path.join(dest_dir, str(i)+'.png')
    de_file=os.path.join(dest_dir, str(i)+'.npy')	
    #cv2.imshow("prueba",img)
    cv2.imwrite(img_file, img_n)
    de= label*(10-0.7)+0.7

    np.save(de_file, de)



dir= "./dataset_kria"
dset_dir= root + "/" + folder

generate_images(dset_dir, 1000, dir)