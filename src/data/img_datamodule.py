from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
from torchvision.datasets import MNIST, ImageNet, CIFAR100


class ImgDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # if image.shape[0] == 1 or len(image.shape) == 2:
        #     image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)
        else:
            image = image.unsqueeze(0).float() / 255.0

        return image, label

class MyDataModule(LightningDataModule):
    def __init__(self, dataset_to_down: str, img_size: tuple, data_dir: str = './data', batch_size: int = 64):
        super().__init__()
        self.dataset_to_down = MNIST if dataset_to_down == 'mnist' \
            else ImageNet if dataset_to_down == 'imagenet' \
            else CIFAR100 if dataset_to_down == 'cifar100' \
            else None
        if self.dataset_to_down is None:
            raise ValueError("dataset_to_down must be 'mnist', 'imagenet' or 'cifar100'")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255]),
        ]) if dataset_to_down != 'mnist' else None

    def prepare_data(self):
        print("Downloading dataset...")
        self.train_dataset = self.dataset_to_down(root="data/", train=True, transform=None, download=True)
        self.test_dataset = self.dataset_to_down(root="data/", train=False, transform=None, download=True)

    def n_classes(self):
        return np.unique(self.train_dataset.targets).reshape(-1).shape[0]

    def setup(self, stage=None):
        train_images, val_images, train_labels, val_labels = train_test_split(self.train_dataset.data, self.train_dataset.targets, test_size=0.2, random_state=42)
        test_images, test_labels = self.test_dataset.data, self.test_dataset.targets
        if stage == 'fit' or stage is None:
            self.train_ds = ImgDataset(train_images, train_labels, transform=self.transform)
            self.val_ds = ImgDataset(val_images, val_labels, transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_ds = ImgDataset(test_images, test_labels, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
