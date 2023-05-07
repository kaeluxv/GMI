#!/usr/bin/env python
# coding: utf-8

# In[91]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# In[92]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=3),
            # nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 5, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return [feature, out]

class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        self.feat_dim = 512
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return [out]


# In[93]:


# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device =", device)
# Hyperparameters

learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0001
batch_size = 64
num_epochs = 10

# 데이터를 다운로드하고 로드합니다
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
# test_dataset = test_dataset[:len(train_dataset) * 0.9]

# DataLoader를 만듭니다
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("Train data =", len(train_dataset))
print("Test  data =", len(test_dataset))


# In[94]:


# 모델을 정의합니다
model = MCNN(num_classes=10).to(device)

# 손실 함수와 옵티마이저를 정의합니다
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# 모델을 학습합니다
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 입력 데이터를 device로 보냅니다
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward 계산
        _, outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward 계산 및 Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (i+1) % 100 == 0:
            print('[MCNN] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# 모델의 dict_state를 저장합니다
torch.save(model.state_dict(), 'mcnn_dict_state.pt')

# Initialize MCNN model
model.load_state_dict(torch.load('mcnn_dict_state.pt'))

# Set model to evaluation mode
model.eval()

# Test the model on the test dataset
correct = 0
total = 0
for images, labels in test_loader:
    # Forward pass
    features, outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('[MCNN] Accuracy on test set: %d %%' % (100 * correct / total))


# In[ ]:


# 모델을 정의합니다
model = SCNN(num_classes=10).to(device)

# 손실 함수와 옵티마이저를 정의합니다
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델을 학습합니다
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 입력 데이터를 device로 보냅니다
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward 계산
        _, outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward 계산 및 Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if (i+1) % 100 == 0:
            print('[SCNN] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# 모델의 dict_state를 저장합니다
torch.save(model.state_dict(), 'scnn_dict_state.pt')

# Initialize SCNN model
model.load_state_dict(torch.load('scnn_dict_state.pt'))

# Set model to evaluation mode
model.eval()

# Test the model on the test dataset
correct = 0
total = 0
for images, labels in test_loader:
    # Forward pass
    features, outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('[SCNN] Accuracy on test set: %d %%' % (100 * correct / total))

