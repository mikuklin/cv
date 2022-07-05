import torch
import time
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torchvision import models
from skimage import io, img_as_float
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "ants":
            label = 1
        else:
            label = 0
        if self.transform is not None:
             return self.transform(image=image)["image"].float(), label
        return torch.tensor(image).float(), label

model = models.resnet18(pretrained = True)
input_size = 224
features = 512
classes = 2
model.fc = nn.Linear(features, classes)
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
for param in model.layer4[1].parameters():
    param.requires_grad = True

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
image_datasets = {'train':train, 'val': val}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                   shuffle=True,) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))

def train_model(model, data, device, loss_func, optimizer, epoches):
    start = time.time()
    train_accuracy_hist = []
    val_accuracy_hist = []
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        for X, y in data['train']:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(dim = 0)
            _, preds = torch.max(output, dim = 1)
            epoch_correct += torch.sum(preds == y.data)
        train_accuracy_hist.append((epoch_correct/len(data['train'].dataset)).item())
        train_loss_hist.append(epoch_loss/len(data['train'].dataset))
        print('train Loss: {:.4f}, Acc: {:.4f}'.format(train_loss_hist[-1], train_accuracy_hist[-1]))

        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_correct = 0
            for X, y in data['val']:
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                loss = loss_func(output, y)
                epoch_loss += loss.item() * X.size(dim = 0)
                _, preds = torch.max(output, dim = 1)
                epoch_correct += torch.sum(preds == y.data)
        val_accuracy_hist.append((epoch_correct/len(data['val'].dataset)).item())
        val_loss_hist.append(epoch_loss/len(data['val'].dataset))
        epoch_time = time.time() - epoch_start
        print('val Loss: {:4f}, Acc: {:.4f}'.format(val_loss_hist[-1], val_accuracy_hist[-1]))
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))


    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(max(val_accuracy_hist)))

    return model, train_accuracy_hist, val_accuracy_hist, train_loss_hist, val_loss_hist

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=0.0001)
model = model.to(device).float()
model, ta, va, tl, vl = train_model(model, dataloaders_dict, device, nn.CrossEntropyLoss(), optimizer, epoches = 40)

sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(data = {"train":ta, "val":va})
plt.xlabel("epoch", fontsize=16)
plt.ylabel("accuracy", fontsize=16)
plt.show()
sns.lineplot(data = {"train":tl, "val":vl})
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.show()
