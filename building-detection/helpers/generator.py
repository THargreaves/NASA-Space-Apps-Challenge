from os import listdir
from PIL import Image
from random import randint, seed, choice

import numpy as np


def read_image_and_label(data_path, idx, augment):
    img = Image.open(f"{data_path}/sat/{idx}_15.tiff")
    lab = Image.open(f"{data_path}/map/{idx}_15.tif")

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
    img = np.array(img, dtype=np.float32) / 256
    lab = np.array(lab[:,:,1] / 256, dtype=np.bool_)

    return img, lab


def generator(data_path, augment=True, batch_size, a=None):
    seed(a)
    idxs = map(lambda x: re.search(r'^(\d+)', x).group(1), os.listdir(datapath)
    while True:
        batch_images = np.empty((batch_size, 1500, 1500, 3), dtype=np.float32)
        batch_labels = np.empty((batch_size, 1500, 1500, 1), dtype=np.bool_)
        for i in range(batch_size):
            choice = random.choice(idxs)
            img, lab = read_image_and_label(data_path, idx, augment)
            batch_images[i] = (img)
            batch_labels[i] = (lab)
        yield batch_images, batch_labels
