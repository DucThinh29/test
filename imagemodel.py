import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # Lớp Conv1: từ 3 kênh đầu vào (RGB), với 16 bộ lọc, kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Giảm kích thước ảnh đi một nửa

        # Lớp Conv2: từ 16 kênh, với 32 bộ lọc
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Lớp Conv3: từ 32 kênh, với 64 bộ lọc
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Lớp fully connected đầu tiên (Linear): flatten đầu ra sau Conv3 và pooling
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Giả sử ảnh đầu vào là 128x128

        # Lớp fully connected cuối cùng (số lớp = số class)
        self.fc2 = nn.Linear(512, 10)  # Ví dụ có 10 lớp phân loại

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + Pool + ReLU
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + Pool + ReLU
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + Pool + ReLU

        x = x.view(-1, 64 * 16 * 16)  # Flatten: biến 64x16x16 thành vector 1 chiều
        x = F.relu(self.fc1(x))  # Fully connected + ReLU
        x = self.fc2(x)  # Output

        return x
