import torch 
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class IncrementalFashionMNIST:
    '''Automatically create data loaders for incremental learning on FashionMNIST dataset'''
    def __init__(self, val_ratio: float = 0.1, seed: int = 42):
        full_train = datasets.FashionMNIST(
                                                root="../data",
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([transforms.ToTensor()])

                                        )
        self.test_set = datasets.FashionMNIST(
                                                root="../data",
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
        
        # Precompute class indices for fast lookup
        self._class_indices = {"train": {}, "test": {}, "val": {}}
        self._precompute_class_indices()
        # cache for subsets by (mode, labels)
        self._cache_indices = {"train": {}, "test": {}, "val": {}}
    
    def _precompute_class_indices(self):
        """Precompute indices for each class to avoid repeated iteration."""
        for mode, dataset in [("train", self.train_set), ("test", self.test_set), ("val", self.val_set)]:
            class_to_indices = {}
            for i, (_, lbl) in enumerate(dataset):
                lbl = int(lbl)
                if lbl not in class_to_indices:
                    class_to_indices[lbl] = []
                class_to_indices[lbl].append(i)
            # Convert to tensor for faster concatenation
            for lbl in class_to_indices:
                class_to_indices[lbl] = torch.tensor(class_to_indices[lbl], dtype=torch.long)
            self._class_indices[mode] = class_to_indices
    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'")

        # normalize labels into a sorted tuple key for caching
        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        # Use precomputed class indices for fast lookup
        class_indices = self._class_indices[mode]
        indices_list = [class_indices[lbl] for lbl in label_key if lbl in class_indices]
        if indices_list:
            indices = torch.cat(indices_list).tolist()
        else:
            indices = []
        
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
        
        # Precompute class indices for fast lookup using ImageFolder.targets
        self._class_indices = {"train": {}, "test": {}, "val": {}}
        self._precompute_class_indices()
        # cache for subsets by (mode, labels)
        self._cache_indices = {"train": {}, "test": {}, "val": {}}
    
    def _precompute_class_indices(self):
        """Use ImageFolder.targets for fast vectorized indexing."""
        for mode, dataset in [("train", self.train_set), ("test", self.test_set), ("val", self.val_set)]:
            targets = torch.tensor(dataset.targets, dtype=torch.long)
            class_to_indices = {}
            for lbl in range(200):  # TinyImageNet has 200 classes
                mask = (targets == lbl)
                if mask.any():
                    class_to_indices[lbl] = mask.nonzero(as_tuple=True)[0]
            self._class_indices[mode] = class_to_indices
    

    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'")

        # normalize labels into a sorted tuple key for caching
        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        # Use precomputed class indices for fast lookup
        class_indices = self._class_indices[mode]
        indices_list = [class_indices[lbl] for lbl in label_key if lbl in class_indices]
        if indices_list:
            indices = torch.cat(indices_list).tolist()
        else:
            indices = []
        
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


class IncrementalCIFAR100:
    '''Automatically create data loaders for incremental learning on CIFAR-100.

    Args:
        num_classes: keep only the first ``num_classes`` of CIFAR-100 (100 = full dataset).
        resize: image size. CIFAR-100 is 32x32 natively; set larger (e.g. 224) for BiT-style backbones.
        val_ratio: fraction of training data reserved for validation (per-class stratified).
        seed: RNG seed for the train/val split.
    '''
    def __init__(self, num_classes: int = 100, resize: int = 32,
                 val_ratio: float = 0.1, seed: int = 42):
        if not (1 <= num_classes <= 100):
            raise ValueError(f"num_classes must be in [1, 100], got {num_classes}")
        self.num_classes = num_classes

        # CIFAR-100 per-channel stats (on full dataset)
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

        if resize == 32:
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            eval_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            train_tf = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.RandomCrop(resize, padding=resize // 8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            eval_tf = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        full_train = datasets.CIFAR100(root="data", train=True, download=True, transform=train_tf)
        full_test = datasets.CIFAR100(root="data", train=False, download=True, transform=eval_tf)

        # Optional: subset to first ``num_classes`` classes
        if num_classes < 100:
            train_keep = [i for i, t in enumerate(full_train.targets) if t < num_classes]
            test_keep = [i for i, t in enumerate(full_test.targets) if t < num_classes]
            full_train = torch.utils.data.Subset(full_train, train_keep)
            self.test_set = torch.utils.data.Subset(full_test, test_keep)
        else:
            self.test_set = full_test

        # Train / val split (shuffled, deterministic)
        n = len(full_train)
        if val_ratio > 0.0:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            split = int(val_ratio * n)
            self.train_set = torch.utils.data.Subset(full_train, perm[split:])
            self.val_set = torch.utils.data.Subset(full_train, perm[:split])
        else:
            self.train_set = full_train
            self.val_set = torch.utils.data.Subset(full_train, [])

        self._class_indices = {"train": {}, "test": {}, "val": {}}
        self._precompute_class_indices()
        self._cache_indices = {"train": {}, "test": {}, "val": {}}

    def _precompute_class_indices(self):
        """Precompute class -> indices dict by iterating labels once."""
        for mode, dataset in [("train", self.train_set), ("test", self.test_set), ("val", self.val_set)]:
            class_to_indices = {}
            for i, (_, lbl) in enumerate(dataset):
                lbl = int(lbl)
                class_to_indices.setdefault(lbl, []).append(i)
            for lbl in class_to_indices:
                class_to_indices[lbl] = torch.tensor(class_to_indices[lbl], dtype=torch.long)
            self._class_indices[mode] = class_to_indices

    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'")

        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        class_indices = self._class_indices[mode]
        indices_list = [class_indices[lbl] for lbl in label_key if lbl in class_indices]
        if indices_list:
            indices = torch.cat(indices_list).tolist()
        else:
            indices = []

        subset = torch.utils.data.Subset(data, indices)
        cache[label_key] = subset
        return subset

    def get_loader(self, mode, label, batch_size=128, shuffle=None):
        subset = self.get_set(mode, label)
        shuffle = (mode == 'train')

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
        return torch.utils.data.DataLoader(subset, **loader_kwargs)


class IncrementalImageNet21kP200:
    '''ImageNet-21k-P 200-class subset (disjoint from ImageNet-1k) for CL.

    Expects the pre-split on-disk layout produced by
    ``cnn/tools/prepare_imagenet21k_p200.py``::

        <root>/train/<wnid>/*.JPEG
        <root>/val/<wnid>/*.JPEG
        <root>/test/<wnid>/*.JPEG

    Args:
        root: directory containing the ``train/``, ``val/``, ``test/`` folders.
        num_classes: keep only the first ``num_classes`` wnids (alphabetical
            by folder name; ImageFolder's default ordering). 200 = full subset.
        resize: input resolution. 224 is the BiT / IN1k default.
    '''
    def __init__(self, root: str = "data/imagenet21k_p200",
                 num_classes: int = 200, resize: int = 224):
        if not (1 <= num_classes <= 200):
            raise ValueError(f"num_classes must be in [1, 200], got {num_classes}")
        self.num_classes = num_classes

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(int(round(resize * 256 / 224))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

        full_train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        full_val = datasets.ImageFolder(os.path.join(root, "val"), transform=eval_tf)
        full_test = datasets.ImageFolder(os.path.join(root, "test"), transform=eval_tf)

        n_on_disk = len(full_train.classes)
        if n_on_disk < num_classes:
            raise ValueError(
                f"num_classes={num_classes} exceeds the {n_on_disk} classes on disk at {root}")

        if num_classes < n_on_disk:
            self.train_set = torch.utils.data.Subset(
                full_train, [i for i, t in enumerate(full_train.targets) if t < num_classes])
            self.val_set = torch.utils.data.Subset(
                full_val, [i for i, t in enumerate(full_val.targets) if t < num_classes])
            self.test_set = torch.utils.data.Subset(
                full_test, [i for i, t in enumerate(full_test.targets) if t < num_classes])
        else:
            self.train_set = full_train
            self.val_set = full_val
            self.test_set = full_test

        self._class_indices = {"train": {}, "test": {}, "val": {}}
        self._precompute_class_indices()
        self._cache_indices = {"train": {}, "test": {}, "val": {}}

    def _precompute_class_indices(self):
        for mode, dataset in [("train", self.train_set),
                              ("test", self.test_set),
                              ("val", self.val_set)]:
            # Handle both ImageFolder (has .targets) and Subset(ImageFolder).
            if hasattr(dataset, "targets"):
                targets = torch.tensor(dataset.targets, dtype=torch.long)
            else:
                base_targets = dataset.dataset.targets
                targets = torch.tensor([base_targets[i] for i in dataset.indices],
                                       dtype=torch.long)
            class_to_indices = {}
            for lbl in range(self.num_classes):
                mask = (targets == lbl)
                if mask.any():
                    class_to_indices[lbl] = mask.nonzero(as_tuple=True)[0]
            self._class_indices[mode] = class_to_indices

    def get_set(self, mode, label):
        if mode == 'train':
            data = self.train_set
        elif mode == 'test':
            data = self.test_set
        elif mode == 'val':
            data = self.val_set
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'")

        try:
            label_key = tuple(sorted(int(x) for x in label))
        except TypeError:
            label_key = (int(label),)

        cache = self._cache_indices[mode]
        if label_key in cache:
            return cache[label_key]

        class_indices = self._class_indices[mode]
        indices_list = [class_indices[lbl] for lbl in label_key if lbl in class_indices]
        if indices_list:
            indices = torch.cat(indices_list).tolist()
        else:
            indices = []

        subset = torch.utils.data.Subset(data, indices)
        cache[label_key] = subset
        return subset

    def get_loader(self, mode, label, batch_size=64, shuffle=None):
        subset = self.get_set(mode, label)
        shuffle = (mode == 'train')

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
        return torch.utils.data.DataLoader(subset, **loader_kwargs)


DATASET_CHOICES = ("fashion_mnist", "tiny_imagenet", "cifar100", "imagenet21k_p200")


def build_dataset(name: str, **kwargs):
    """Instantiate an incremental dataset manager by name."""
    name = name.lower()
    if name == "fashion_mnist":
        return IncrementalFashionMNIST(val_ratio=kwargs.get("val_ratio", 0.1))
    if name == "tiny_imagenet":
        return IncrementalTinyImageNet(resize=kwargs.get("img_size", 64))
    if name == "cifar100":
        return IncrementalCIFAR100(
            num_classes=kwargs.get("num_classes", 100),
            resize=kwargs.get("img_size", 32),
            val_ratio=kwargs.get("val_ratio", 0.1),
        )
    if name == "imagenet21k_p200":
        return IncrementalImageNet21kP200(
            root=kwargs.get("root", "data/imagenet21k_p200"),
            num_classes=kwargs.get("num_classes", 200),
            resize=kwargs.get("img_size", 224),
        )
    raise ValueError(f"Unknown dataset '{name}'. Valid: {DATASET_CHOICES}")
    