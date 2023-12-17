import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

dir = './Cars_dataset'

transform = transforms.Compose([
    transforms.ToTensor()
])
def df_set(dir = dir):

    """ Создание Датафрейма по трем маркам авто: Kia, Audi, BMW"""

    labels = []
    path_img  = []
    for i in os.listdir(dir):
        labels.append(i.split('_')[0])

        path_img.append(os.path.join(dir, i))

    data = pd.DataFrame({'Label': labels, 'Path': path_img})
    df1 = data[data['Label'] == 'Kia']
    df2 = data[data['Label'] == 'Audi']
    df3 = data[data['Label'] == 'BMW']
    new_df = pd.concat([df1,df2,df3], ignore_index=True)
    new_df['Label'].replace(['Kia', 'Audi', 'BMW'], [0,1,2], inplace = True)

    return new_df

df = df_set()


def train_test(data = df):

    """Разделение на обучающую и валидационную выборку"""

    X = df['Path']
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test()

class CarsDataset(Dataset):

    """Создание кастомного датасета"""

    def __init__(self, X, y, transforms = transform):

        self.path = list(X)
        self.labels = list(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128,128), cv2.INTER_AREA)
        if self.transform:
            image = transform(image)

        label = self.labels[idx]

        return image, label

train_dataset = CarsDataset(X_train, y_train)
test_dataset = CarsDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    
class ConvNet2(nn.Module):

    """Сверточная нейросеть с одним слоем свертки"""

    def __init__(self):
        super().__init__()

        self.act = nn.ReLU()
        self.flat = nn.Flatten()
        self.pooling = nn.MaxPool2d((2, 2))
        self.conv0 = nn.Conv2d(3, 32, 3)

        self.linear1 = nn.Linear(63*63*32, 256)
        self.linear2 = nn.Linear(256, 3)

    def forward(self, x):

        out = self.conv0(x)
        out = self.act(out)
        out = self.pooling(out)

        out = self.flat(out)

        out = self.linear1(out)
        out = self.act(out)

        out = self.linear2(out)

        return out

model = ConvNet2()

def count_parameters(model):

    """Сумма параметров сети"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(pred, label):
    
    answer = F.softmax(pred.detach(), dim=-1).numpy().argmax(1) == label.numpy().argmax(1) 
    return answer.mean()


device = torch.device('cuda' if torch.cuda.is_available else 'cpu') # Расчет на ГПУ

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_model(dataloader = train_dataloader):

    """Обучение на 10 эпох"""

    epochs = 10
    for epoch in range(epochs):

        loss_val = 0
        acc_val = 0

        for img, lab in (pbar := tqdm(dataloader)):
            optimizer.zero_grad()

            img = img.to(device)
            lab = lab.to(device)

            pred = model(img)
            labels = F.one_hot(lab, num_classes=3).float()
            
            loss = loss_fn(pred, labels)
            loss_item = loss.item()
            loss_val +=loss_item

            loss.backward()
            optimizer.step()

            acc_current = accuracy(pred.cpu().float(), labels.cpu().float())
            acc_val +=acc_current
            

            pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        print(loss_val/len(train_dataloader))
        print(acc_val/len(train_dataloader))

def test_model(dataloader = test_dataloader):
    
    """Тестирование"""

    test_loss = 0
    test_acc = 0

    for img, lab in (pbar := tqdm(test_dataloader)):

        img = img.to(device)
        lab = lab.to(device)

        pred = model(img)
        labels = F.one_hot(lab, num_classes=3).float()
        
        loss = loss_fn(pred, labels)
        loss_item = loss.item()
        test_loss +=loss_item


        acc_current = accuracy(pred.cpu().float(), labels.cpu().float())
        test_acc +=acc_current
        

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
    print(test_loss/len(test_dataloader))
    print(test_acc/len(test_dataloader))