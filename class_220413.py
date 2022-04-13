# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #3
# (1)
# Reading Data
input_data = pd.read_csv('Data_Base/lin_regression_data_01.csv', \
                         header=None).to_numpy(dtype='float')

# Setting Variable1
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length

# plot에 그려진 Linear Combination을 통해 실제값의 y가 단조 증가가 아님을 확인
sorted_input_data = np.zeros((25, 2)) # 정렬에 사용하기 위한 input_data 행렬 복사본 정의
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(25):
    sorted_input_data[n, :] = input_data[n, :]
sorted_input_data = sorted_input_data[sorted_input_data[:, 0].argsort()] 

sorted_in_x = sorted_input_data[:, 0] # x를 오름차순으로 정렬
y_real = sorted_input_data[:, 1] # 실제값 배열 정의

time_noise = 10

def Gen_AugmentedSet(data, time):
    gen_set = np.zeros((len(data) * time + len(data), 2))
    
    for n in range(time):
        data[:, 0] = np.concatenate(data[:, 0], \
                                    np.random.rand() * time - (time / 2) + data[:, 0])
        data[:, 1] = np.concatenate(data[:, 1], \
                                    np.random.rand() * time - (time / 2) + data[:, 1])
    gen_set[:, 0] = data[:, 0]
    gen_set[:, 1] = data[:, 1]
    return gen_set

# agumented_original_set = np.zeros((len(sorted_input_data) * time_noise \
#                                    + len(sorted_input_data), 2))

agumented_original_set = Gen_AugmentedSet(sorted_input_data, time_noise)


                 
def Dist_Set(data):
    
    return True
























