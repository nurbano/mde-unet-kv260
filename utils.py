import torch
from tqdm.auto import tqdm

def train_model(model, train_dataloader, optimizer, loss_fn, device, scheduler, w1, w2, w3, w4):

    losses= []
    RMSE= []

    with torch.autograd.set_detect_anomaly(True):
        for inputs, labels in train_dataloader:
            

            model.train(True)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(labels, outputs,w1, w2, w3, w4, device)
            losses.append(loss)
            
            loss.backward()
            optimizer.step()

            #Compute the RMSE in meters
            de= labels*(10.0 - 0.7)+0.7
            de_p= outputs*(10.0 - 0.7)+0.7
            
            rmse= torch.sqrt(torch.mean(torch.pow((de_p-de),2)))
            RMSE.append(rmse)
            
            if scheduler!= None:
                scheduler.step(loss)

    if scheduler!= None:
        lr = scheduler.get_last_lr()
    else:
        lr = [0]

    return torch.stack(losses).mean().item(),    torch.stack(RMSE).mean().item(), lr


def validate_model(model,val_dataloader, loss_fn, device, w1, w2, w3, w4):

    losses_val= []
    RMSE= []
    for inputs, labels in val_dataloader:
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        dif = labels-outputs
        dif_isnan = torch.isnan(dif).any()
        if dif_isnan==False:
            loss_val_ = loss_fn(labels, outputs,w1, w2, w3, w4, device)
            losses_val.append(loss_val_)
            de= labels*(10.0 - 0.7)+0.3
            
            de_p= outputs*(10.0 - 0.7)+0.3
            
            rmse= torch.sqrt(torch.mean(torch.pow((de_p-de),2)))
            RMSE.append(rmse)
            

    return torch.stack(losses_val).mean().item(), torch.stack(RMSE).mean().item()

def test_model(model,test_dataloader, device ):

    
    RMSE= []
    for inputs, labels in test_dataloader:
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        dif = labels-outputs
        dif_isnan = torch.isnan(dif).any()
        if dif_isnan==False:
            
            de= labels*(10.0 - 0.7)+0.7
            de_p= outputs*(10.0 - 0.7)+0.7
            
            rmse= torch.sqrt(torch.mean(torch.pow((de_p-de),2)))
            RMSE.append(rmse)
            

    return torch.stack(RMSE).mean().item()

def predict(img, model, device):
    xb =img.unsqueeze(0).to(device)
    yb = model(xb)
    out=yb.cpu().detach().numpy()
    return out

def preprocess_fn(im_fname):
    
    im = np.array(Image.open(im_fname).resize((224,224)), dtype=np.float32)
    im = np.transpose(im, (2, 0, 1))
    im = (im - np.min(im))/(np.max(im) - np.min(im))
    im = (im - 0.5)*2.0
    
    return im