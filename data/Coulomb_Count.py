# coding = utf-8
"""
作者   : Hilbert
时间   :2022/2/25 21:15
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
from lib import *
from tqdm import tqdm
import pandas as pd
import pickle


file_path = r'F:\hilbert-研究生\项目\华电电池储能\程序\system_day_data\results\December\aligned_interpolation_data'
for file_name in tqdm(os.listdir(file_path)):
    if file_name.split('.')[1] == 'pk':
        data_path = os.path.join(file_path, file_name)

        # load data
        with open(data_path, 'rb') as f:
            sensors_data = pickle.load(f)

        # coulomb count
        saving_path = './results/December/aligned_interpolation_data/coulomb'
        mkdir(saving_path)
        date = file_name.split('.')[0].split('_')[0]
        SOC_coulomb = Coulomb(saving_path=saving_path, sensors_data=sensors_data, capacity=280, date=date)
        SOC_coulomb.coulomb_count()
    else:
        continue