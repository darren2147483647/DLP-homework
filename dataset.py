import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import json

def to_one_hot(label, label_mapping):
    num_classes = len(label_mapping)
    one_hot_labels = np.zeros((num_classes), dtype=np.float32)
    for i in (label):
        one_hot_labels[label_mapping[i]]=1
    return torch.tensor(one_hot_labels)

def save(model,filename,root="pt"):
   filename=os.path.join(root,filename)
   torch.save(model.state_dict(),filename)

def trans():
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize
    ])
    return transformer


class dataset_(Dataset):
    def __init__(self,root="iclevr",mode="train",transform=trans()):
        self.root=root
        self.mode=mode
        self.trans=transform
        with open('objects.json', 'r') as file:
            self.objects = json.load(file)
        if mode=="train":
            with open('train.json', 'r') as file:
                self.train_list = json.load(file)
        if mode=="test":
            with open('test.json', 'r') as file:
                self.list = json.load(file)
        if mode=="new_test":
            with open('new_test.json', 'r') as file:
                self.list = json.load(file)
        if mode=="label_set":
            with open('label_set.json', 'r') as file:
                self.list = json.load(file)
    def __len__(self):
        if self.mode=="train":
            return len(self.train_list)
        return len(self.list)
        
    def __getitem__(self, index):
        if self.mode=="train":
            filename=list(self.train_list.keys())[index]
            image=Image.open(os.path.join(self.root, filename)).convert('RGB')
            label=self.train_list[filename]
            image=self.trans(image)
            label=to_one_hot(label,self.objects)
            return image, label
        label=self.list[index]
        label=to_one_hot(label,self.objects)
        return label

class dataloader_():
    def __init__(self,batchsize):
        self.batchsize=batchsize
        self.dataset=dataset_(mode='train', root='iclevr')
        self.dataloader=DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        

if __name__ == '__main__':
    data = dataloader_(batchsize=8)
    print(len(data.dataloader))
    for x,y in data.dataloader:
        plt.imshow(torchvision.utils.make_grid(x).permute(1,2,0))
        plt.show()
        print(x,y)
        break