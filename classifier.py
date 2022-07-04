import torch
import time
import os
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import models
from skimage import io, img_as_float
import albumentations as A
from albumentations.pytorch import ToTensorV2

!wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
!unzip hymenoptera_data.zip -d ./data/

def get_data(path):

    train_ants_path = path + '/train/ants/'
    val_ants_path = path + '/val/ants/'
    train_bees_path = path + '/train/bees/'
    val_bees_path = path + '/val/bees/'

    train_ants = os.listdir(train_ants_path)
    val_ants = os.listdir(val_ants_path)
    train_bees = os.listdir(train_bees_path)
    val_bees = os.listdir(val_bees_path)

    train_ants = [train_ants_path + i for i in train_ants]
    val_ants = [val_ants_path + i for i in val_ants]
    train_bees = [train_bees_path + i for i in train_bees]
    val_bees = [val_bees_path + i for i in val_bees]

    return train_ants + train_bees, val_ants + val_bees

class DataGenerator(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = img_as_float(io.imread(image_filepath)).astype(np.float32)[..., :3]
        io.imshow(image)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "ants":
            label = 1
        else:
            label = 0
        if self.transform is not None:
             return self.transform(image=image)["image"].float(), label
        return image, label

model = models.resnet18(pretrained = True)
input_size = 224
features = 512
classes = 2
model.fc = nn.Linear(features, classes)
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

train_transform = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.SmallestMaxSize(max_size=input_size),
        A.RandomCrop(height=input_size, width=input_size),
        A.RGBShift(r_shift_limit=0.5, g_shift_limit=0.5, b_shift_limit=0.5, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=input_size),
        A.CenterCrop(height=input_size, width=input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)
train, val = get_data('data/hymenoptera_data')
train = DataGenerator(train, train_transform)
val = DataGenerator(val, val_transform)
train, val = get_data('data/hymenoptera_data')
