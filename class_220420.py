# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #3
# (1)
# Reading Data
input_data = pd.read_csv('Data_Base/chap2_data.csv')\
    .drop(['data'], axis=1).to_numpy(dtype='float')

# Setting Variable1
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length






