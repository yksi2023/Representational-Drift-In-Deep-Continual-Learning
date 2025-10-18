import torch 
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class IncrementalFashionMNIST:
    '''Automatically create data loaders for incremental learning on FashionMNIST dataset'''
    def __init__(self,increment, batch_size=64, shuffle=True):
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
        self.increment = increment
        self.current_class = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_set(self, mode ):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        indices = [i for i, (_, label) in enumerate(data) if label in range(self.current_class - self.increment, self.current_class)]
        subset = torch.utils.data.Subset(data, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=self.shuffle)
        
        return loader
    
    def next_task(self):
        self.current_class += self.increment

    def get_whole_set(self, mode):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle)
        return loader

