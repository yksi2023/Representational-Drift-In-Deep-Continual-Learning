import torch 
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class IncrementalFashionMNIST:
    '''Automatically create data loaders for incremental learning on FashionMNIST dataset'''
    def __init__(self, val_ratio: float = 0.1, seed: int = 42):
        full_train = datasets.FashionMNIST(
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
        # Create a fixed train/val split from the training data
        num_samples = len(full_train)
        if val_ratio > 0.0:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(num_samples, generator=generator).tolist()
            split = int(val_ratio * num_samples)
            val_idx = indices[:split]
            train_idx = indices[split:]
            self.train_set = torch.utils.data.Subset(full_train, train_idx)
            self.val_set = torch.utils.data.Subset(full_train, val_idx)
        else:
            self.train_set = full_train
            # Use an empty subset for validation if disabled
            self.val_set = torch.utils.data.Subset(full_train, [])
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
    def get_loader(self, mode, label, batch_size=64, shuffle=True):
        subset = self.get_set(mode, label)
        if mode == 'val' or mode == 'test':
            shuffle = False
        else:
            shuffle = True
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
    