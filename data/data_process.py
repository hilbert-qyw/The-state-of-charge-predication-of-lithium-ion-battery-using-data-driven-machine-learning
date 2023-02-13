# coding = utf-8
"""
作者   : Hilbert
时间   :2022/2/24 20:15
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


file_path = r'F:\hilbert-研究生\项目\华电电池储能\数据集-合同版\December'
everyday_sensors = []
month_sensors = {}
month_SOC_corr = []
# sensor_index = [18, 19, 20, 21, 24, 27, 34, 60, 61, 62, 63, 66, 69, 1346, 1408, 1409, 1433, 1434, 1505]
# SBMS data
sensor_index = [60, 61, 62, 66, 69, 1408, 1409, 1433, 1434]
"""
    18 --> 'MBMS总电压'
    19 --> 'MBMS电流'
    20 --> 'MBMSSOC'
    21 --> 'MBMSSOH'
    24 --> 'MBMS单体平均电压值'
    27 --> 'MBMS单体平均温度值'
    34 --> 'MBMS环境湿度'
    60 --> 'SBMS电池簇总电压'
    61 --> 'SBMS电流'
    62 --> 'SBMSSOC'
    63 --> 'SBMSSOH'
    66 --> 'SBMS单体平均电压值'
    69 --> 'SBMS单体平均温度值'
    1346 --> '输出功率'
    1408 --> '总有功功率'
    1409 --> '总无功功率'
    1433 --> '有功功率'
    1434 --> '无功功率'
    1505 --> '负载功率'
    """

for file_name in tqdm(os.listdir(file_path)):
    data_path = os.path.join(file_path, file_name)

    columns = ['index', 'number', 'sensor', 'previous_data', 'current_data', '0', 'time', '1']
    data = pd.read_table(data_path, header=None, names=columns, parse_dates=True)

    # read data
    sensors = DataAnalysis(data, saving_path=r'.\results')
    raw_sensors_data = sensors.sensor_data(number=sensor_index, plot=False)
    everyday_sensors.append(raw_sensors_data)
    if len(month_sensors) == 0:
        month_sensors = raw_sensors_data
    else:
        for key, value in month_sensors.items():
            month_sensors[key] = pd.concat([month_sensors[key], raw_sensors_data[key]], axis=0)

    # data interpolation and resampling
    date = file_name.split('_')[-1].split('.')[0]
    saving_path = f'./results/December/aligned_interpolation_data'
    mkdir(saving_path)
    sensors_interpolation = DataReconstrction(saving_path=saving_path, raw_data=everyday_sensors, date=date)
    interpolated_data = sensors_interpolation.resample(interval=60, saving_data=True)
    # month_SOC_corr.append(sensors_interpolation.correlation_coefficient(saving_corr=True))

# # write SOC corr to excel
# SOC_excel(saving_path=r'./results/December/aligned_interpolation_data', month_SOC_corr=month_SOC_corr)
#
# # sensor data varies with month
# month_data_curve(saving_path='./results', month_sensors=month_sensors)

