import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import time
import matplotlib as mpl
#import pyrealsense2.pyrealsense2 as rs

image_path = "bagPNGs/"
image = cv2.imread(image_path+'tableFruits.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.45]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=24, points_per_batch=8)

sam_predictor = SamPredictor(sam)
sam_predictor.set_image(image)

print("RGB Segmentation...")
start = time.time()
masks = mask_generator.generate(image)


np.save("randomItems_rgb_masks.npy", masks)
print("Segmentation time", time.time()-start)


blank = np.zeros((image.shape[0], image.shape[1], 4))
plt.figure(figsize=(10,10))
plt.imshow(blank)
show_anns(masks)
plt.axis('off')
plt.show() 
