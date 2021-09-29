import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 16, (4, 4))
        self.conv4 = nn.Conv2d(16, 120, (5, 5))
        self.fc = nn.Linear(120, 15)

    def forward(self, x):
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))
        x = self.pool(nn.ReLU(self.conv3(x)))
        x = nn.ReLU(self.conv4(x))
        x = nn.Softmax(self.fc(x))
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)



