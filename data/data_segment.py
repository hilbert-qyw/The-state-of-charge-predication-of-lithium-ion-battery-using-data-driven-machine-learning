# coding = utf-8
"""
作者   : Hilbert
时间   :2022/3/3 21:07
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
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pickle
from lib import *

file_path = r'F:\hilbert-研究生\项目\华电电池储能\程序\system_day_data\results\December\aligned_interpolation_data'

listdir = os.listdir(file_path)[0:3] + os.listdir(file_path)[5:7]
days_data = {}
for file_name in listdir:
    if '.pk' in file_name:
        data_path = os.path.join(file_path, file_name)
        date = file_name.split('_S')[0]

        with open(data_path, 'rb') as f:
            days_data[date] = pickle.load(f)

data_segment = DataGenerate(days_data=days_data, saving_path='./results/December')
normal_data = data_segment.data_detection()
continuous_data = data_segment.data_segment(normal_data)
data_segment.slide_cycle(continuous_data)

data_segment.slide_cycle(continuous_data, operation_mode='day')

dataset_path = './results/December/day_cycle_data.pk'
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
    split_data = Dataset(raw_data=dataset, saving_path='./results/December/', state='day', sequence_len=100, stride=10,
                         padding=True, padding_zero=False, saving_data=True)
    split_data.dataset_split()
