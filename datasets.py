import torch 
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class IncrementalFashionMNIST:
    '''Automatically create data loaders for incremental learning on FashionMNIST dataset'''
    def __init__(self):
        self.train_set = datasets.FashionMNIST(
                                                root="data",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([transforms.ToTensor()])

                                        )
        self.test_set = datasets.FashionMNIST(
                                                root="data",
                                                train=False,
                                                download=True,
                                                transform=transforms.Compose([transforms.ToTensor()])
                                        )
    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        indices = [i for i, (_, lbl) in enumerate(data) if lbl in label]
        subset = torch.utils.data.Subset(data, indices)
        

        return subset
    def get_loader(self, mode, label, batch_size=64, shuffle=True):
        subset = self.get_set(mode, label)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        return loader

class IncrementalTinyImageNet:
    '''Automatically create data loaders for incremental learning on TinyImageNet dataset'''
    def __init__(self):
        self.train_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/train",
                                                transform=transforms.Compose([
                                                    transforms.Resize((64, 64)),
                                                    transforms.ToTensor()
                                                ])
                                        )
        self.test_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/test",
                                                transform=transforms.Compose([
                                                    transforms.Resize((64, 64)),
                                                    transforms.ToTensor()
                                                ])
                                        )
        self.val_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/val",
                                                transform=transforms.Compose([
                                                    transforms.Resize((64, 64)),
                                                    transforms.ToTensor()
                                                ])
                                        )
       
    

    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        indices = [i for i, (_, lbl) in enumerate(data) if lbl in label]
        subset = torch.utils.data.Subset(data, indices)
        
        return subset

    def get_loader(self, mode, label, batch_size=64, shuffle=None):
        subset = self.get_set(mode, label)
        if mode == 'val' or mode == 'test':
            shuffle = False
        else:
            shuffle = True
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        return loader
    