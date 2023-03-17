import torchvision
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from numpy import asarray
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

class MMwaveDataset(Dataset):
    def __init__(self, mat_root, RI_concat=True):
        """
        :param mat_root: the path of .mat file
        :param RI_concat: whether we split the real and imaginary parts of measurments to two channel
        """
        super(MMwaveDataset, self).__init__()

        self.mat_root = mat_root
        self.data = loadmat(self.mat_root)
        self.RI_concat = RI_concat

    def __getitem__(self, index):
        x = np.array(self.data['inputsTrain'])[index]
        y = np.array(self.data['labelsTrain'])[index]
        if self.RI_concat:
            input = torch.reshape(torch.from_numpy(x), (1, 1, -1))
        else:
            input = torch.reshape(torch.from_numpy(x), (1, 2, -1))

        label = torch.reshape(torch.from_numpy(y), (1, 256, 256)) / 255.

        return input, label

    def __len__(self):
        return len(self.data['inputsTrain'])


if __name__ == "__main__":
    #data = MMwaveDataset(mat_root='../data/activations_USAF1951202301111510.mat')
    data = loadmat('../data/activations_USAF1951202301111510.mat')
    x = np.array(data['inputsTrain'])[1]
    print(x.shape)
    y = np.array(data['labelsTrain'])[1].reshape(256, 256)
    print(y.shape)

    dataset = MMwaveDataset(mat_root='../data/activations_USAF1951202301111510.mat')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4)
    print(len(dataloader))
    for model_input, ground_truth in dataloader:
        print(model_input.shape)
        print(ground_truth.shape)
