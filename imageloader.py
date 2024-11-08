import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def getimages(train_path, val_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize tất cả ảnh về 128x128
        transforms.ToTensor(),          # Chuyển ảnh thành Tensor
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa ảnh
    ])
    # Load dữ liệu từ thư mục theo cấu trúc
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

    # Tạo DataLoader để load batch dữ liệu
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader
