import matplotlib.pyplot as plt
import numpy as np



def add_gaussian_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy


def display_results(test_images,test_set,results):
    fig,axs=plt.subplots(3,7,figsize=(6,4))
    ax=axs.flatten()

    j=0
    ax[j+3].set_title('original Images')
        
    for i in range(20,27):
        ax[j].imshow(np.transpose(test_images[i],(1,2,0)),cmap='gray')
        ax[j].axis('off')
        j+=1

    ax[j+3].set_title('noisy Input')

    for i in range(7):
        ax[j].imshow(np.transpose(test_set[i],(1,2,0)),cmap='gray')
        ax[j].axis('off')
        j+=1
        
    ax[j+3].set_title('reconstructed Images')
        
    for i in range(7):
        ax[j].imshow(np.transpose(results[i],(1,2,0)),cmap='gray')
        ax[j].axis('off')
        
        j+=1

    plt.show()



def visualize_dataset(test_loader):

    # for visualizing the dataset

    test_images = []
    test_labels = []

    for images, labels in test_loader:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())

    # Concatenate into full NumPy arrays
    test_images = np.concatenate(test_images, axis=0)   
    test_labels = np.concatenate(test_labels, axis=0)   


    f,ax=plt.subplots(1,2,figsize=(2,2))
    ax[0].imshow(np.transpose(train_images[0],(1,2,0)),cmap='gray')
    ax[1].imshow(np.transpose(test_images[0],(1,2,0)),cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')

    plt.show()