import kagglehub
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def download_dataset():
    path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
    print("Dataset downloaded at:", path)
    return path


def get_data_loaders(batch_size=32):
    data_path = download_dataset()

    # image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # dataset folders (train/valid/test assumed)
    train_dir = os.path.join(data_path, "train")
    valid_dir = os.path.join(data_path, "valid")
    test_dir  = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes

    return train_loader, valid_loader, test_loader, class_names