import argparse
import sys
import pandas as pd
import os
from dataset import NYUDEPTHV2
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from model import UNet
import torch
from utils import train_model, validate_model, test_model
from loss import mde_loss

def test(dataset_dir, num_img , model_path):
    print("Evaluate mode")
    #Find h5 paths
    root=dataset_dir
    folder = "val/official"
    path = root + "/" + folder 
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))
    
    filelist.sort()
    
    data = {"h5": [x for x in filelist if x.endswith(".h5")]}
    df = pd.DataFrame(data).sample(frac=1, random_state=42)
    df2=df.sample(n=num_img, replace=True).reset_index(drop="true")
    
    
    #Define the transformation
    tran_val = transforms.Compose([transforms.Resize([224,224])])
    
    #Instance the NYUDEPTHV2 dataset class
    dset_test = NYUDEPTHV2(path,
                          df2.reset_index(drop="true"),
                          transform = None, transform2 = None)
    #Instance the dataloader
    test_dataloader= DataLoader(dset_test, batch_size=32, shuffle=True, num_workers=12)

    #Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    #Define the model
    model = UNet().to(device)

    #Load the model state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint, strict= False)
    model.eval()

    rmse= test_model(model,test_dataloader, device )

    print("RMSE: ", rmse, " m")

    
    
    
    
    
def train(dataset_dir, epochs, num_imgs, model_path):
    print("Training mode")
    #Find h5 paths
    root=dataset_dir
    folder = "train"
    path = root + "/" + folder 
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))
    
    filelist.sort()
    
    data = {"h5": [x for x in filelist if x.endswith(".h5")]}
    df = pd.DataFrame(data).sample(frac=1, random_state=42)
    df2=df.sample(n=num_imgs, replace=True).reset_index(drop="true")
    
    #Split the dataset into Train and validate subset
    split_idx= int(df2.h5.size*0.8)

    #Define the transformation
    tran = transforms.Compose([transforms.Resize([224,224]),
                           transforms.RandomHorizontalFlip(p=0.5),
                           ])

    tran2 = transforms.Compose([v2.RandomChannelPermutation()])
    tran_val = transforms.Compose([transforms.Resize([224,224])])
    
    #Instance the NYUDEPTHV2 dataset class
    dset_train = NYUDEPTHV2(path,     
                            df2[0:split_idx].reset_index(drop="true"),
                            transform = tran,transform2 = tran2)
    dset_val = NYUDEPTHV2(path,
                          df2[split_idx:].reset_index(drop="true"),
                          transform = None, transform2 = None)
    print("Train subset: ",len(dset_train), "| Val subset:" , len(dset_val))

    #Instance dataloader
    train_dataloader= DataLoader(dset_train, batch_size=16, shuffle=True, num_workers=12)
    val_dataloader= DataLoader(dset_val, batch_size=16, shuffle=True, num_workers=12)

    #Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = UNet().to(device)
    model.eval()

    #Define Loss function, optimizer and Scheduler
    loss_fn = mde_loss()
    optimizer_ft = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    scheduler= None

    #Train and validate
    loss_train = []
    loss_val= []
    rmse_val= []
    rmse_train= []
    lr = []

    #Define the weights for the losses
    
    w1=0 #SSIM Loss
    w2=0 #Smothness Loss
    w3=1 #L1 Loss
    w4=0 #L2 Loss
    
    for epoch in tqdm(range(epochs)):
        #Train
        loss_t, rmse, l = train_model(model, train_dataloader, optimizer_ft, loss_fn, device, scheduler,w1, w2, w3, w4)

        loss_train.append(loss_t)
        lr.append(l)
        rmse_train.append(rmse)
        #Validation
        loss_v, rmse_v = validate_model(model, val_dataloader,loss_fn, device, w1, w2, w3, w4)
        print(f"epoch: {epoch+1} / {epochs} |TRAIN-> loss: {loss_t:.4f} rmse: {rmse:.4f} |VAL-> loss: {loss_v:.4f} rmse: {rmse_v:.4f}")

    loss_val.append(loss_v)
    rmse_val.append(rmse_v)
    print("--------------------------")
    print("Training Finished")
    print("Save model")
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset_dir',  type=str,  default='data', help='Path to dataset folder. Default is data')
    ap.add_argument('-e', '--epochs', type=int,  default=50, help='Number of test images. Default is 50')
    ap.add_argument('-n', '--num_img', type=int,  default=500, help='Number of train images. Default is 500')
    ap.add_argument('-E', '--evaluate', type=bool,  default=False, help='Evaluate a model. Default is False')
    ap.add_argument('-m', '--model_path', type=str,  default="./model.pth", help='Path to the model, only for evaluate. Default is model.pth')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --dataset_dir    : ', args.dataset_dir)
    print (' --epochs      : ', args.epochs)
    print (' --num_img      : ', args.num_img)
    print (' --evaluate      : ', args.evaluate)
    print (' --model_path      : ', args.model_path)
    
    print('------------------------------------\n')

    if args.evaluate== True:
        test(args.dataset_dir, args.num_img , args.model_path)
    else:
        train(args.dataset_dir, args.epochs, args.num_img, args.model_path )

    


if __name__ ==  "__main__":
    main()
