# coding = utf-8
"""
作者   : Hilbert
时间   :2021/11/8 14:31
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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib import *
from tqdm import tqdm


file_path = r'F:\hilbert-研究生\项目\华电电池储能\数据集-合同版\七到八月数据'
for file_name in tqdm(os.listdir(file_path)):
    data_path = os.path.join(file_path, file_name)

    columns = ['index', 'number', 'sensor', 'previous_data', 'current_data', '0', 'time', '1']
    data = pd.read_table(data_path, header=None, names=columns, parse_dates=True)

    sensors = DataAnalysis(data, saving_path=r'.\results')
    # sensors.sensor_csv()
    # sensors.sensor_info()
    _ = sensors.sensor_data(number=[18, 19, 20, 21, 24, 27, 34, 60, 61, 62, 63, 66, 69, 1346, 1408, 1409, 1433,
                                1434, 1505])