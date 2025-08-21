from dataset import NYUDEPTHV2
from model import UNet
from loss import mde_loss
from utils import test_model
from utils import predict
from utils import preprocess_fn

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from metrics import evaluate_depth, evaluate_depth_by_ranges

#folder = "train"
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

df.head()

df2= df

root = "/media/nurbano/Datos12/Datasets/nyudepthv2"
train_folder= "train"
path = root + "/" + train_folder
print(path)
filelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        filelist.append(os.path.join(root, file))
        #print(file)
filelist.sort()
data = {
    "h5": [x for x in filelist if x.endswith(".h5")]
}


df_train = pd.DataFrame(data)

df_train.head()

tran_test = transforms.Compose([transforms.Resize([224,224])])

dset_test = NYUDEPTHV2(path, df2.reset_index(drop="true"),  transform = tran_test, transform2 = None)

dset_train= NYUDEPTHV2(path, df_train.reset_index(drop="true"),  transform = tran_test, transform2 = None)

print(f'len(dset_test): {len(dset_test)}, len(dset_train): {len(dset_train)}')

test_dataloader= DataLoader(dset_test, batch_size=32, shuffle=True, num_workers=12)
train_dataloader= DataLoader(dset_train, batch_size=32, shuffle=True, num_workers=12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet().to(device)
PATH_MODEL= "./models/mde_v1.pth"
checkpoint = torch.load(PATH_MODEL)
model.load_state_dict(checkpoint, strict= False)
model.eval()
print("Test de Evaluación")
eval = evaluate_depth(test_dataloader, model, device)
for k,v in eval.items():
    print(f"{k}: {v:.4f}")

# print("Evaluación Train")
# eval_train = evaluate_depth(train_dataloader, model, device)
# for k,v in eval_train.items():
#     print(f"{k}: {v:.4f}")

# Calcular cantidad de parámetros y memoria
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size = 4  # float32 = 4 bytes
total_memory = total_params * param_size / (1024 ** 2)  # en MB

print(f"Total parámetros: {total_params}")
print(f"Parámetros entrenables: {trainable_params}")
print(f"Memoria estimada (solo pesos): {total_memory:.2f} MB")
bins = [(0,2),(2,5),(5,10)]
results_by_range = evaluate_depth_by_ranges(test_dataloader, model, device="cuda", bins=bins)

for rng, metrics in results_by_range.items():
    print(f"Rango {rng} m:")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")