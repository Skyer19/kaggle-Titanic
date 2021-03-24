#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 挂载到谷歌云盘
from google.colab import drive
drive.mount('/content/drive')


# In[2]:


## 进入到文件所在位置
import os
os.chdir("/content/drive/My Drive/Colab Notebooks")


# In[3]:


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import copy
from torch.autograd.gradcheck import zero_gradients
import pandas  as pd 
import seaborn as sns
import re
import torch.utils.data as data


# In[4]:


## 读取文件
test = pd.read_csv("./titanic/test.csv")
train = pd.read_csv("./titanic/train.csv")


# In[5]:


train['Embarked'].value_counts()


# In[6]:


## 用Embarked中的众数弥补测试文件中的空白
test['Embarked'].fillna(
    test.Embarked.mode().values[0], inplace=True)


# In[7]:


## 用Embarked中的众数弥补训练文件中的空白
train['Embarked'].fillna(
    test.Embarked.mode().values[0], inplace=True)


# In[8]:


train['Age'].describe()


# In[9]:


## 弥补测试和训练文件中的Age空白
test['Age'].fillna(29.699118, inplace=True)
train['Age'].fillna(29.699118, inplace=True)


# In[10]:


## 弥补Fare的空白
test['Fare'].fillna(32.204208, inplace=True)


# In[11]:


test.info()


# In[12]:


## 用独热编码处理训练数据
dummy_fields=['Pclass','Sex','Embarked']
for each in dummy_fields:
    dummies= pd.get_dummies(train[each], prefix= each, drop_first=False)
    train = pd.concat([train, dummies], axis=1)
train.head()    

fields_to_drop=['PassengerId', 'Cabin', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked']
df=train.drop(fields_to_drop,axis=1)
df.head()


# In[13]:


## 用独热编码处理测试数据
dummy_fields=['Pclass', 'Sex', 'Embarked']
for each in dummy_fields:
    dummies= pd.get_dummies(test[each], prefix= each, drop_first=False)
    test = pd.concat([test, dummies], axis=1)
# test.head()  

fields_to_drop=['PassengerId','Cabin', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked']
df_test=test.drop(fields_to_drop,axis=1)
df_test.head()


# In[14]:


## Age和Fare中的数据相差过大，容易造成不便
to_normalize=['Age','Fare']
for each in to_normalize:
    mean, std= df[each].mean(), df[each].std()
    df.loc[:, each]=(df[each]-mean)/std

df.head()


# In[15]:


## Age和Fare中的数据相差过大，容易造成不便
to_normalize=['Age','Fare']
for each in to_normalize:
    mean, std= df_test[each].mean(), df_test[each].std()
    df_test.loc[:, each]=(df_test[each]-mean)/std

df_test.head()


# In[16]:


titanic_train_data_X = df.drop(['Survived'], axis=1)
titanic_train_data_Y = df['Survived']
titanic_test_data = df_test


# In[17]:


## 将数据转换格式
train_data = torch.from_numpy(titanic_train_data_X.values).float()
train_label = torch.from_numpy(titanic_train_data_Y.values).float()
test_data = torch.from_numpy(titanic_test_data.values).float()


# In[45]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# 数据封装
train_dataset = TensorDataset(train_data, train_label)
trainLoader = DataLoader(train_dataset, batch_size=4,
                         shuffle=True, num_workers=2)


import torch.nn as nn
import torch.nn.functional as F

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(titanic_train_data_X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


net = Net()


# In[46]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(net.parameters(), lr=0.001)  

import time
start = time.time()
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data  
        optimizer.zero_grad()  

        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()
        # if i % 20 == 19:
        #     # 每 20 次迭代打印一次信息
        #     print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
        #     running_loss = 0.0
print('Finish Traning! Total cost time: ', time.time()-start)


# In[47]:


correct = 0
total = 0
with torch.no_grad():
    for data in trainLoader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %
      (correct / total * 100))


# In[48]:


output = torch.max(net(test_data),1)[1]
## 存入到文件
submission = pd.read_csv('./titanic/gender_submission.csv')
submission['Survived'] = output
submission.to_csv('./titanic/gender_submission.csv', index=False)


# ## 不同超参数情况下的准确率
# Epoach 200 lr =0.001<br>
# *   64  50 2 88%   0.74641
# *   40 100 2 87%.  0.76076
# *   64 100 2 87%.  0.77033
# *  80 100 2 88%.  0.74880
# *   64 110 2 87%.  0.76555
# *   64 125 2 87%.  0.75837
# *   64 150 2 88%.  0.73684
# <br>
# 
# Epoach 200 lr =0.015<br>
# *   64 100 2 61%.  0.62200
# 
# 
