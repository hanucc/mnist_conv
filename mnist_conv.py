import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

batch_size = 256
epochs = 10
learning_rate = 0.01
train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for i, (x, y) in enumerate(train_dataloader):
#     print(x.shape)
#     break
# # torch.Size([256, 1, 28, 28])


def plot_curve(data, name):
    plt.plot(range(len(data)), data, color='blue')
    plt.legend([name], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')


class Net(nn.Module):
    '''公式 out = (i-k+2p+1)/s, conv满5往上取整, pooling往下取整'''
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),  # [256, 1, 28, 28] => [256, 32, , ]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        # x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 64*5*5)
        x = self.fc(x)

        return x


# device = torch.device('cuda')

net = Net()#.to(device)
criteon = nn.CrossEntropyLoss()#.to(device)
opt = optim.Adam(net.parameters(), lr=learning_rate)
print(net)
train_loss = []
all_train_loss = []
num = 0.0
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(train_dataloader):
        # x, y = x.to(device), y.to(device)
        out = net(x)
        # loss = criteon(out, y)
        loss = criteon(out.to(float), F.one_hot(y).to(float))

        opt.zero_grad()
        loss.backward()
        opt.step()

        all_train_loss.append(loss.item())
        num += 1
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

    train_loss.append(loss.item() / num)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_curve(train_loss, 'train loss')
plt.subplot(1, 2, 2)
plot_curve(all_train_loss, 'all_train_loss')
plt.show()

# 模型评估
pred_all = []
y_all = []
for x, y in test_dataloader:
    # x = x.to(device)
    x = net(x)
    pred = x.argmax(dim=1).cpu()
    pred_all.extend(pred.numpy())
    y_all.extend(y.numpy())

test_accurary = accuracy_score(y_all, pred_all)
test_precision = precision_score(y_all, pred_all, average='weighted')
test_recall = recall_score(y_all, pred_all, average='weighted')
test_f1 = f1_score(y_all, pred_all, average='weighted')

print("[test] accurary:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
    test_accurary, test_precision, test_recall, test_f1
))
print(classification_report(y_all, pred_all))

