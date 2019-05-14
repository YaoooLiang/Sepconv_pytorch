import os
import torch
import torch.utils.data as data
from scipy.misc import imread, imsave, imresize
import numpy as np
import random


def random_flip(img_group):
    if random.random() < 0.5:
        return [np.fliplr(img) for img in img_group]
    else:
        return img_group


def norm(img_group):
    return img_group / 255.0


class vimeo_dataloader(data.Dataset):
    """docstring for vimeo_dataloader"""
    def __init__(self, root):
        super(vimeo_dataloader, self).__init__()
        self.img_list = []

        dataset_txt = root + '/tri_trainlist.txt'
        image_folder = root + '/sequences'

        with open(dataset_txt) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                self.img_list.append(os.path.join(image_folder, line))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        folder = self.img_list[idx]
        img_paths = os.listdir(folder)
        img_paths.sort()

        images = []

        for i in range(3):
            img = imread(os.path.join(folder, img_paths[i])).astype(np.float32)

            images.append(img)

        # flip
        images = random_flip(images)

        # norm
        for i in range(3):
            images[i] = norm(images[i])
            images[i] = torch.from_numpy(np.transpose(images[i], (2, 0, 1))).contiguous().float()

        return images[0], images[1], images[2]









