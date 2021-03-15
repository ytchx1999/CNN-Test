import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from conv import Conv2d

import time

print(torch.__version__)

# ## 1.加载数据

start_time = time.time()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_data = torchvision.datasets.mnist.MNIST('./data/',
                                              train=True,
                                              download=False,
                                              transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,
                                           shuffle=True)

test_data = torchvision.datasets.mnist.MNIST('./data/',
                                             train=False,
                                             download=False,
                                             transform=transform)
test_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=128,
                                          shuffle=True)


# plt.imshow(train_data.data[0], cmap='gray')
# plt.show()

# ## 2.定义网络


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1 = Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        slice_times = dot_times = sum_times = 0

        x, slice_time, dot_time, sum_time = self.conv1(x)

        slice_times += slice_time
        dot_times += dot_time
        sum_times += sum_time

        x = self.pool(F.relu(x))

        x, slice_time, dot_time, sum_time = self.conv2(x)

        slice_times += slice_time
        dot_times += dot_time
        sum_times += sum_time

        x = self.pool(F.relu(x))

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, slice_times, dot_times, sum_times


net = Net()
print(net)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net.to(device)
print(device)

# ## 3.定义损失函数和优化器


loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(params=net.parameters(),
                      lr=0.01,
                      momentum=0.5)

# ## 4.训练网络

train_losses = []
train_acces = []

total_slice_time = []
total_dot_time = []
total_sum_time = []

epoch = 5

for i in range(epoch):
    train_loss = 0
    train_acc = 0
    net.train()

    slice_times = dot_times = sum_times = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        out, slice_time, dot_time, sum_time = net(inputs)

        slice_times += slice_time
        dot_times += dot_time
        sum_times += sum_time

        loss = loss_func(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, pred = torch.max(out, dim=1)
        correct = (pred == labels).sum().item()
        train_acc += correct / labels.shape[0]

    train_losses.append(train_loss / len(train_loader))
    train_acces.append(train_acc / len(train_loader))

    total_slice_time.append(slice_times)
    total_dot_time.append(dot_times)
    total_sum_time.append(sum_times)

    print('epoch:{},train_loss:{:.4f},train_acc:{:.4f}'.format(i,
                                                               train_loss /
                                                               len(train_loader),
                                                               train_acc / len(train_loader)))

print("Average slice time:", np.mean(total_slice_time), 's')
print("Average dot time:", np.mean(total_dot_time), 's')
print("Average sum time:", np.mean(total_sum_time), 's')

# ## 5.测试网络


test_loss = 0
test_acc = 0
net.eval()

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    out, slice_time, dot_time, sum_time = net(inputs)

    loss = loss_func(out, labels)

    test_loss += loss.item()

    _, pred = torch.max(out, dim=1)
    correct = (pred == labels).sum().item()
    test_acc += correct / labels.shape[0]

test_loss /= len(test_loader)
test_acc /= len(test_loader)

print('test_loss:{:.4f},test_acc:{:.4f}'.format(test_loss, test_acc))

end_time = time.time()
print('total time:', (end_time - start_time), 's')

# ## 6.可视化训练过程


# In[19]:


# epoches = np.arange(0, epoch, 1)
# # 画出训练结果
# plt.plot(epoches, train_losses, 'b', label='train_loss')
# plt.plot(epoches, train_acces, 'r', label='train_acc')
# plt.legend()
# plt.show()

# # ## 7.预测
#
# # In[20]:
#
#
# import random
#
# # In[21]:
#
#
# pos = random.randint(0, len(train_data.data))
# plt.imshow(train_data.data[pos], cmap='gray')
# plt.show()
#
# # In[22]:
#
#
# img = train_data.data[pos]
# img = img.unsqueeze(0)
# img = img.unsqueeze(0)
# img = img.to(device)
# img = img.float()  # 一定要转换成float才可以进行预测
# img.shape
#
# # In[23]:
#
#
# net.eval()
# out = net(img)
# _, pred = torch.max(out, dim=1)
#
# # In[24]:
#
#
# print('prediction is: {}'.format(pred.item()))
# print('real is: {}'.format(train_data.targets[pos].item()))
#
# # In[ ]:
