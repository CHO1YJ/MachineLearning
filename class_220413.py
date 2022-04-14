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

time_noise = 10 # 데이터 수 증폭 값
temp_data = [] # original data와 random data의 합 기록

def Gen_AugmentedSet(data, time):
    gen_set = np.zeros((len(data) * time + len(data), 2)) # 발생시킬 증강 DB

    # 아무래도 numpy보다 pandas에서 데이터의 정제가 간편했음
    df_x = pd.DataFrame(data[:, 0]) # 데이터 정제를 위해 numpy인 x에 관한 데이터셋 생성
    df_y = pd.DataFrame(data[:, 1]) # 데이터 정제를 위해 numpy인 y에 관한 데이터셋 생성
    for n in range(time):
        # 1. origianl data + random data
        temp_data_x = data[:, 0] + round(np.random.rand(), 1) * 3 - 1.5
        # 2. 데이터 정제를 위해 '1.'에서 생성한 데이터를 데이터셋으로 변환
        df_rand_x = pd.DataFrame(temp_data_x)
        # 3. 열을 기준으로 데이터셋을 추가
        df_x = pd.concat([df_x, df_rand_x])      
        
        # x에서와 작업 동일
        temp_data_y = data[:, 1] + round(np.random.rand(), 1) * 3 - 1.5
        df_rand_y = pd.DataFrame(temp_data_y)
        df_y = pd.concat([df_y, df_rand_y])   

    # 정제된 데이터를 numpy 데이터로 변환
    gen_set = pd.concat([df_x, df_y], axis=1).to_numpy()
    
    return gen_set # 증강된 DB를 반환

# DB 증강 함수를 호출 및 반환된 값을 Original data로 초기화
agumented_original_set = Gen_AugmentedSet(sorted_input_data, time_noise)

# 생성된 Original data를 그래프를 생성하기 위해 필요로 하는 x와 y로 구분
data_aug_x = agumented_original_set[:, 0]
data_aug_y = agumented_original_set[:, 1]

# Drawing Linear Regression
plt.figure()
plt.scatter(data_aug_x, data_aug_y, color='0')
plt.scatter(in_x, out_y, color="red")
plt.legend(['Augmented Combination', 'Linear Combination'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Data Base')
plt.grid(True, alpha=0.5)
plt.show()
                 

# (2)
def Dist_Set(data):
    
    return True
























