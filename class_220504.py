# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chap.3 실습
# Data Refinement and Schematic
# Reading Data
input_data = pd.read_csv('Data_Base/logistic_regression_data.csv', \
                         index_col=0).to_numpy(dtype='float')

# Setting Variable
in_x0 = input_data[:, 0] # 입력; x0
in_x1 = input_data[:, 1] # 입력; x1
out_y = input_data[:, 2] # 출력; y

list_color = []
input_data_0 = np.zeros((int(len(input_data) / 2), 3))
input_data_1 = np.zeros((int(len(input_data) / 2), 3))
for n in range(len(input_data)):
    if n < len(input_data) / 2:
        list_color.append('r')
        input_data_0[n, :] = input_data[n, :]
    elif n >= len(input_data) / 2:
        list_color.append('b')
        input_data_1[n - 250, :] = input_data[n, :]
        
in_x0_0 = input_data_0[:, 0] # 입력; x0
in_x1_0 = input_data_0[:, 1] # 입력; x1
# out_y_0 = input_data_0[:, 2] # 출력; y

in_x0_1 = input_data_1[:, 0] # 입력; x0
in_x1_1 = input_data_1[:, 1] # 입력; x1
# out_y_1 = input_data_1[:, 2] # 출력; y

# Drawing Data Base
plt.figure()
plt.scatter(in_x0_0, in_x1_0, marker='o', c='r', s=10)
plt.scatter(in_x0_1, in_x1_1, marker='x', c='b', s=10)
plt.legend(['0', '1'])
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base')
plt.grid(True, alpha=0.5)
plt.show()

# (1)













































