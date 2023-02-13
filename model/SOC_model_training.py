# coding = utf-8
"""
作者   : Hilbert
时间   :2022/3/6 15:27
"""
import sys
import os
from warnings import simplefilter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]  # 上一级目录
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
simplefilter(action='ignore', category=Warning)
simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lib import *
import pickle
from torch.utils import data

sensor_num = 4
sequence_len = 100
stride = 10
batch_size = 16
epochs = 200
learning_rate = 1e-4

# data path
output_dir = fr'../results/December/model/sensors_{sensor_num}_sequence_{sequence_len}_stride_{stride}_batch_' \
             fr'{batch_size}_epochs_{epochs}_lr_{learning_rate}/'
mkdir(output_dir)

# load data
input_path = fr'../results/December'
training_dataset = ModelDataset(input_path=input_path, state='training', sequence_len=sequence_len, stride=stride)
training_dataloader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

# testing_dataset = ModelDataset(input_path=input_path, state='testing', sequence_len=sequence_len, stride=stride)
# testing_dataloader = data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset = ModelDataset(input_path=input_path, state='validation', sequence_len=sequence_len, stride=stride)
# val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# model
net = CNN_LSTM(sensor_num=sensor_num,
               sequence_len=sequence_len,
               lstm_hidden_size=16,
               lstm_layers=2,
               reg_hidden_size=8,
               dropout=0.2,
               encoder_bidirectional=False)
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #8e-5)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

# train
net.train()
best_crite = None
train_loss = []
val_loss = []
test_loss = []

trainer = Trainer(model=net, criterion=criterion, train_dataloader=training_dataloader, verbose=True,
                  saving_path='../results/December', val_dataloader=training_dataloader,
                  test_dataloader=training_dataloader)
trainer.train(epochs=epochs, optimizer=optimizer)
# trainer.test(mode='best')

# for epoch in range(epochs):
#     batch_loss = []
#     for input, tar in training_dataloader:
#         # input = input.to(torch.float)
#         # tar = tar.to(torch.float).squeeze()
#         pre = net(input)
#         loss = criterion(pre, tar)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         batch_loss.append(loss.item())
#     train_loss.append(sum(batch_loss) / len(batch_loss))
#     print(train_loss[-1])

