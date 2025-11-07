from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(TransformDataset, self).__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transforms(x), y


class MedsDataPipeline:
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
        self.train_dataset_raw = train_dataset
        self.val_dataset_raw = val_dataset
        self.test_dataset_raw = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        self.eval_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        # placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # ---- setup stage ----
    def setup(self):
        self.train_dataset = TransformDataset(self.train_dataset_raw, self.train_transforms)
        self.val_dataset = TransformDataset(self.val_dataset_raw, self.eval_transforms)
        self.test_dataset = TransformDataset(self.test_dataset_raw, self.eval_transforms)

    # ---- loaders ----
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
