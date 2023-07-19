from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class CelebADataset(Dataset):
    """CelebA Dataset with attributes"""

    def __init__(self, file, root_dir, labels, transform=None):
        """
        Args:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #read and transform attribution dataset
        a_f = pd.read_csv(file,sep='\s+')
        a_f = a_f.replace(-1,0)
        a_f = a_f.loc[:,a_f.columns.intersection([labels[0],labels[1],labels[2],labels[3]])]
        a_f['Label'] = a_f[labels[0]].map(str) + a_f[labels[1]].map(str) + a_f[labels[2]].map(str) + a_f[labels[3]].map(str)
        

        self.attributions_frame = a_f[[os.path.isfile(root_dir+i) for i in a_f.index.values]]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attributions_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.attributions_frame.index.values[idx]
        img_path = os.path.join(self.root_dir,
                                self.attributions_frame.index.values[idx])

            
        image = Image.open(img_path)
        label = self.attributions_frame.loc[img_name][4]
        
        label = int(label,2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])


        return sample
