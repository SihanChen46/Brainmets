import numpy as np
import matplotlib.pyplot as plt

def show_single_pair(img, mask, index):
    figs,axes = plt.subplots(1,2)
    axes[0].imshow(img[index])
    axes[1].imshow(mask[index])
    
    
def show_single_image(img, mask):
    for i in range(len(img)):
        show_single_pair(np.clip(img,0,None),mask,i)
    
    
def show_all_images(dataset):
    for idx in range(len(dataset)):
        img = dataset[idx][0][0]
        mask = dataset[idx][1][0]
        for i in range(len(img)):
            if (mask[i].max()==1):
                figs,axes = plt.subplots(1,2,figsize=(15,15))
                axes[0].title.set_text(' slice: '+str(i))
                axes[1].title.set_text('mask')
                axes[0].imshow(img[i],cmap='gray',vmin=-1,vmax=1)
                axes[1].imshow(mask[i],cmap='gray')