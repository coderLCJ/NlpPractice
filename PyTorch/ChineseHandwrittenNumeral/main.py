import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import getData

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 16, (4, 4))
        self.fc1 = nn.Linear(16*5*5, 84)
        self.fc2 = nn.Linear(84, 32)
        self.fc3 = nn.Linear(32, 15)

    def get_features(self, x):
        size = x.size()[1:]
        features = 1
        for i in size:
            features *= i
        return features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(-1, self.get_features(x)) # 此步必须有
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    epochs = 100
    batch_size = 4000
    train_data, train_label = getData.getTrainData()
    t = 0
    while t < 6:
        acc = 0.0
        los = 0.0
        inputs, target = train_data, train_label
        for i in range(batch_size):
            net.zero_grad()
            outputs = net(inputs[i])
            loss = criterion(outputs, target[i].long())
            loss.backward()
            optimizer.step()
            los += loss
            predict = 0
            for j in range(15):
                if outputs[0][j] == max(outputs[0]):
                    predict = j
                    break
            # print(predict, ' === ', train_label[i])
            if predict == target[i]:
                acc += 1

        print('epoch: %d acc: %.2f%% loss: %.3f' % (t, acc / batch_size * 100, los / batch_size))
        t += 1

    torch.save(net.state_dict(), 'net.pt')  # 保存模型