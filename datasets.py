import torch 
import os
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
        # cache for subsets by (mode, labels)
        self._cache_indices = {"train": {}, "test": {}, "val": {}}
    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        # normalize labels into a sorted tuple key for caching
        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        indices = [i for i, (_, lbl) in enumerate(data) if lbl in label]
        subset = torch.utils.data.Subset(data, indices)
        cache[label_key] = subset

        return subset
    def get_loader(self, mode, label, batch_size=64, shuffle=True):
        subset = self.get_set(mode, label)
        if mode == 'val' or mode == 'test':
            shuffle = False
        else:
            shuffle = True

        # dataloader performance defaults
        use_cuda = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_count // 2))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": use_cuda,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })

        loader = torch.utils.data.DataLoader(subset, **loader_kwargs)
        return loader

class IncrementalTinyImageNet:
    '''Automatically create data loaders for incremental learning on TinyImageNet dataset'''
    def __init__(self, resize=64):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.train_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/train",
                                                transform=transforms.Compose([
                                                                                transforms.RandomResizedCrop((resize,resize), scale=(0.8, 1.0)),
                                                                                transforms.RandomHorizontalFlip(),
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                                                                    ])
                                            )

        self.val_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/val",
                                                transform=transforms.Compose([
                                                                                transforms.Resize((resize,resize)),
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                                                                    ])
                                            )

        self.test_set = datasets.ImageFolder(
                                                root="data/tiny-imagenet-200-processed/test",
                                                transform=transforms.Compose([
                                                                                transforms.Resize((resize,resize)),
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                                                                        ])
                                            )
        # cache for subsets by (mode, labels)
        self._cache_indices = {"train": {}, "test": {}, "val": {}}
    

    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        # normalize labels into a sorted tuple key for caching
        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        indices = [i for i, (_, lbl) in enumerate(data) if lbl in label]
        subset = torch.utils.data.Subset(data, indices)
        cache[label_key] = subset
        
        return subset

    def get_loader(self, mode, label, batch_size=64, shuffle=None):
        subset = self.get_set(mode, label)
        if mode == 'val' or mode == 'test':
            shuffle = False
        else:
            shuffle = True
        # dataloader performance defaults
        use_cuda = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_count // 2))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "pin_memory": use_cuda,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            })
        loader = torch.utils.data.DataLoader(subset, **loader_kwargs)
        return loader
    