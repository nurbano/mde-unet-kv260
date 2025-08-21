import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import h5py
import cv2

class DIODE(Dataset):
    def __init__(self, df, splits, scene_types, transform):
        self.transform= transform
        self.df= df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
 
        im_fname= self.df.image.loc[index]
        de_fname= self.df.depth.loc[index]
        de_mask_fname = self.df.mask_.loc[index]


        im = np.array(Image.open(im_fname), dtype=np.float32)
        de = np.load(de_fname).astype(np.float32).squeeze()
        
        de_mask = np.load(de_mask_fname).astype(np.float32).squeeze()

        im = np.transpose(im, (2, 0, 1))
        im = (im - np.min(im))/(np.max(im) - np.min(im))
        im = (im - 0.5)*2.0
        
        de[de_mask == 0] = np.mean(de)
        de = (de - 0.3)/(30.0 - 0.3)
        de = np.clip(de, 0.0, 1.0)
        
        de[de_mask == 0] = 1.0

        de = np.expand_dims(de, axis=0)
  
        data = np.append(im, de, axis=0)

        data = torch.from_numpy(data)
 
        if True:
            data = self.transform(data)
  
        return data[0:3], data[3].unsqueeze(dim=0)


class NYUDEPTHV2(Dataset):
    def __init__(self, dataset_path, df, transform, transform2):
        self.transform= transform
        self.transform2= transform2
        self.dataset_path= dataset_path
        
        self.df= df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
 
        h5_fname= self.df.h5.loc[index]
        h5f = h5py.File(h5_fname, "r")
        
        im = np.array(h5f['rgb'], dtype=np.float32)
        im= np.transpose(im,(1,2,0))

        im= cv2.resize(im, (224,224))
        
        de = np.array(h5f['depth'], dtype=np.float32)
        de= cv2.resize(de, (224,224))

        im = np.transpose(im, (2, 0, 1))
        im = (im - np.min(im))/(np.max(im) - np.min(im))
        im = (im - 0.5)*2.0
        
        de = (de - 0.7)/(10.0 - 0.7)
        
        de = np.clip(de, 0.0, 1.0)
        # de= de/5
        # de= np.clip(de, 0, 5)
        
        de = np.expand_dims(de, axis=0)
        
        data = np.append(im, de, axis=0)
        
        
        data= torch.from_numpy(data)  
        if self.transform != None:
            data = self.transform(data)
                  

        if self.transform2 != None:
            data[0:3]= self.transform2(data[0:3])
       
            
          
        return data[0:3], data[3].unsqueeze(dim=0)