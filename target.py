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

DIVIDER = '-----------------------------------------'

def generate_images(dset_dir, num_images, dest_dir):

  
  test_dataset = NYUDEPTHV2(path, df2[int(df2.h5.size*0.99):].reset_index(drop="true"),  transform = None, transform2 = None)
  
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1, 
                                            shuffle=True)

  # iterate thru' the dataset and create images
  dataiter = iter(test_loader)
  for i in tqdm(range(num_images)):
    image, label = dataiter.next()
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
    #plt.imsave(img_file, img_n)

  return


def make_target(build_dir,target,num_images,app_dir):

    dset_dir = build_dir + '/dataset'
    comp_dir = build_dir + '/compiled_model'
    target_dir = build_dir + '/target_' + target

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

    # copy compiled model
    model_path = comp_dir + '/CNN_' + target + '.xmodel'
    print('Copying compiled model from',model_path,'...')
    shutil.copy(model_path, target_dir)

    # create images
    dest_dir = target_dir + '/images'
    shutil.rmtree(dest_dir, ignore_errors=True)  
    os.makedirs(dest_dir)
    generate_images(dset_dir, num_images, dest_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',  type=str,  default='build', help='Path to build folder. Default is build')
    ap.add_argument('-t', '--target',     type=str,  default='zcu102', choices=['zcu102','zcu104','u50','vck190','kv260'], help='Target board type (zcu102,zcu104,u50,vck190). Default is zcu102')
    ap.add_argument('-n', '--num_images', type=int,  default=10000, help='Number of test images. Default is 10000')
    ap.add_argument('-a', '--app_dir',    type=str,  default='application', help='Full path of application code folder. Default is application')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --build_dir    : ', args.build_dir)
    print (' --target       : ', args.target)
    print (' --num_images   : ', args.num_images)
    print (' --app_dir      : ', args.app_dir)
    print('------------------------------------\n')


    make_target(args.build_dir, args.target, args.num_images, args.app_dir)


if __name__ ==  "__main__":
    main()
