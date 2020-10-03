from os import listdir
from PIL import Image
from random import seed, choice
from re import search

import numpy as np


def read_image_and_label(data_path, idx, augment):
    img = Image.open(f"{data_path}/sat/{idx}_15.tiff")
    lab = Image.open(f"{data_path}/map/{idx}_15.tif")

    img.thumbnail((1024, 1024), Image.ANTIALIAS)
    lab.thumbnail((1024, 1024), Image.ANTIALIAS)
    
    if augment:
        rot_method = choice((None, 
                             Image.ROTATE_90, 
                             Image.ROTATE_180, 
                             Image.ROTATE_270))
        if rot_method:
            img = img.transpose(rot_method)
            lab = lab.transpose(rot_method)
        flip_method = choice((None,
                              Image.FLIP_LEFT_RIGHT))
        if flip_method:
            img = img.transpose(flip_method)
            lab = lab.transpose(flip_method)
    
    # convert to numpy arrays
    img = np.array(img, dtype=np.float32) / 255
    lab = (np.array(lab, dtype=np.uint8)[:,:,0,None] / 255).astype(np.bool_)

    return img, lab


def generator(data_path, batch_size, augment=True, a=None):
    seed(a)
    paths = listdir(f'{data_path}/map')
    idxs = list(map(lambda x: search(r'^(\d+)', x).group(1), paths))
    while True:
        batch_images = np.empty((batch_size, 1024, 1024, 3), dtype=np.float32)
        batch_labels = np.empty((batch_size, 1024, 1024, 1), dtype=np.bool_)
        for i in range(batch_size):
            idx = choice(idxs)
            img, lab = read_image_and_label(data_path, idx, augment)
            batch_images[i] = (img)
            batch_labels[i] = (lab)
        yield batch_images, batch_labels
