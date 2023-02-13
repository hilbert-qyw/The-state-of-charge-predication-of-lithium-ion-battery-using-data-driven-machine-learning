# coding = utf-8
"""
作者   : Hilbert
时间   :2022/8/12 18:26
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


w = int(input('请输入货物重量（g）'))    # 输入货物重量
hurry = input('请选择是否加急，y代表加急，n代表不加急')

# 判断是否加急
if hurry == 'n':
    # 不加急
    fee = 0
else:
    # 加急
    fee = 5

if w <= 1000:
    # 如果小于1000g，输出费用9元
    fee = fee + 9
else:
    if (w - 1000) % 500 == 0 :
        # 剩余货物重量刚好为500g的倍数
        fee = fee + int((w - 1000) / 500) * 4 + 9
    else:
        # 不足500g按照500g计算
        fee = fee + (int((w - 1000) / 500) + 1) * 4 + 9

print(fee)

