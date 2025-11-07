from ConvAutoEncoder import Convautoencoder
from utils import add_gaussian_noise,display_results
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


transform = transforms.Compose([
    transforms.ToTensor(),
])


test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Extract all images and labels
test_images_tensor, test_labels_tensor = next(iter(test_loader))

# Convert tensors to NumPy arrays
test_images = test_images_tensor.numpy()  
test_label = test_labels_tensor.numpy()  


# load the params 
data = np.load("params1.npz")

# Reconstruct params_c
params_c = [
    {"kernels": data[f"c_{i}_kernels"], "b": data[f"c_{i}_b"]}
    for i in range(len([key for key in data if key.startswith("c_")]) // 2)
]

# Reconstruct params_d
params_d = [
    {"kernels": data[f"d_{i}_kernels"], "b": data[f"d_{i}_b"]}
    for i in range(len([key for key in data if key.startswith("d_")]) // 2)
]



# make test set for testing the model
test_set=[]
for i in range(20,27):
    test_image=test_images[i]
    test_image=add_gaussian_noise(test_image)
    test_set.append(test_image)


# testing the model
test_model=Convautoencoder(load=True,params_c=params_c,params_d=params_d)

# concatenating the results
results=[]
for i in range(len(test_set)):
    results.append(test_model.test(test_set[i]))


# display the results
display_results(test_images=test_images,test_set=test_set,results=results)