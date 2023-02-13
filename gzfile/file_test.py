# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/18 16:19
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

path = './data/Bank02_20220701.csv.gz'
data = pd.read_csv(path, compression='gzip', engine='python')

print(data.shape)
