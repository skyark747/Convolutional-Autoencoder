from ConvAutoEncoder import Convautoencoder
from utils import add_gaussian_noise
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split



transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Extract all images and labels
train_images_tensor, train_labels_tensor = next(iter(train_loader))

# Convert tensors to NumPy arrays
train_images = train_images_tensor.numpy()  
train_label = train_labels_tensor.numpy()  


train_images,val_images,train_label,val_label=train_test_split(train_images,train_label,test_size=0.2,random_state=42)

# adding gaussian noise
val_set=[]
for i in range(len(val_images)):
    val_image=val_images[i]
    val_image=add_gaussian_noise(val_image)
    val_set.append(val_image)


#to save parameters of model 
params_c,params_d=[],[]

# n_c is for how many convolutions ad deconvolutions layers are there , for now conv == deconv layer but
# can be changed according to needs
n_c=3


for i in range(n_c):
    params_c.append({'kernels':None,'b':None})
    params_d.append({'kernels':None,'b':None})


# initializing the model
# load param is for when you want to continue training from previously saved state
model=Convautoencoder(load=False,params_c=params_c,params_d=params_d)

epochs=10

running_loss=0
n=len(train_images)
wait=0
patience=5
best_loss=float('inf')

for epoch in range(epochs):
    running_loss=0  
    val_loss=0

    for i in range(n):
        image=train_images[i]   
        noisy_img=add_gaussian_noise(image)
        loss=model.train(noisy_img,image,0.001,n)       
        running_loss+=loss

    for j in range(len(val_set)):
        val_loss+=model.val_test(val_set[j])
    val_loss/=len(val_set)

    if val_loss<best_loss:
        best_loss=val_loss
        wait=0
    else:
        wait+=1
        if wait>=patience:
            print("early stopping triggerd")
            break
        
    print(f"epoch : {epoch+1} , avg training loss : {running_loss/n:.4f} , avg validation loss : {val_loss:.4f}")


#get model
params_c,params_d=model.get_params()

# Flatten and save the params
np.savez("params1.npz",
**{f"c_{i}_kernels": layer["kernels"] for i, layer in enumerate(params_c)},
**{f"c_{i}_b": layer["b"] for i, layer in enumerate(params_c)},
**{f"d_{i}_kernels": layer["kernels"] for i, layer in enumerate(params_d)},
**{f"d_{i}_b": layer["b"] for i, layer in enumerate(params_d)})
print("model saved")  
