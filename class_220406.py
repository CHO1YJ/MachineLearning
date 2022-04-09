# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #3
# (1)
# Reading Data
input_data = pd.read_csv('lin_regression_data_01.csv', \
                         header=None).to_numpy(dtype='float')

# Setting Variable1
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length

# plot에 그려진 Linear Combination을 통해 실제값의 y가 단조 증가가 아님을 확인
temp_input_data = np.zeros((25, 2)) # 정렬에 사용하기 위한 input_data 행렬 복사본 정의
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(25):
    temp_input_data[n, :] = input_data[n, :]
temp_input_data = temp_input_data[temp_input_data[:, 0].argsort()] 

sort_in_x = temp_input_data[:, 0] # x를 오름차순으로 정렬
y_real = temp_input_data[:, 1] # 실제값 배열 정의

K_GBF = 3 # 기저함수의 개수
y_gbf = np.zeros((len(sort_in_x), K_GBF)) # bias를 제외한 입력에 대한 행렬 
u_gbf = [] # 기저함수의 개수에 따른 가우스함수의 평균값 기록함
variance_gbf = 0 # 기저함수의 개수에 따른 가우스함수의 분산값
func_bias = np.ones((25, 1)) # bias 행렬

# Gaussian Base Function 생성 함수
def Gen_GBF(x, K): # 입력; 1. 정렬된 데이터 입력 / 2. 기저함수의 개수
    # 가우스 함수 평균 및 분산 계산    
    for k in range(K):
        u_gbf.append(np.min(x) + (np.max(x) - np.min(x)) / (K - 1) * k)
    variance_gbf = (np.max(x) - np.min(x)) / (K - 1)
    # bias를 제외한 입력에 대한 행렬 초기화
    for n in range(len(x)):
        for k in range(K):
            y_gbf[n][k] = np.exp(-1 / 2 * pow((x[n] - u_gbf[k]) / variance_gbf, 2))
    
    return y_gbf, variance_gbf # 입력에 대한 행렬 및 분산 반환

# Weight 생성 함수
def Gen_Weight(y, phi): # 입력; 1. 정렬된 데이터 출력 / 2. bias를 포함한 입력에 대한 행렬
    # w = np.linalg.inv(np.transpose(phi) * phi) * np.transpose(phi) * y
    # 행렬의 곱이기 때문에 A.dot(B) 이용
    w = np.linalg.inv(np.transpose(phi).dot(phi)).dot(np.transpose(phi)).dot(y)
    
    return w # 가중치 반환

Gen_GBF(sort_in_x, K_GBF) # 가우스 기저함수 생성 함수 호출 
variance_gbf = Gen_GBF(sort_in_x, K_GBF)[1] # 가우스 기저함수 분산 초기화
phi_GBF = np.append(y_gbf, func_bias, axis=1) # bias 첨부; 입력에 대한 행렬에!

print(Gen_Weight(y_real, phi_GBF)) # 가중치 계산 결과 출력

# (2)
weight = Gen_Weight(y_real, phi_GBF) # 가중치 값 초기화
y_hat = 0 # 예측값 정의

# 가중치에 따른 예측값 y_hat 초기화
for n in range(K_GBF):
    y_hat = y_hat + weight[n] * \
        np.exp(-1 / 2 * pow((sort_in_x - u_gbf[n]) / variance_gbf, 2))
y_hat = y_hat + weight[K_GBF]

# Drawing Linear Regression
plt.figure()
plt.scatter(in_x, out_y, color='0')
plt.plot(sort_in_x, y_hat, 'r')
plt.legend(['Analytic solution', 'Linear Combination'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Linear Regression')
plt.grid(True, alpha=0.5)
plt.show()

# 기저함수 개수, 가중치 값 등에 따른 MSE 값 초기화
value_CF_MSE = np.sum(pow(y_hat - y_real, 2)) / len(sort_in_x)

print(value_CF_MSE) # MSE 계산 결과 출력

# (3)
history_mse = [] # 기저함수 K에 따른 MSE 값 기록함
history_part_weight =[] # (2)를 위한 기저함수 K에 따른 가중치 값 기록함

# 이하 계산과정 동일
for K in range(3, 11):
    y_gbf = np.zeros((len(sort_in_x), K))
    u_gbf = []
    variance_gbf = 0
    func_bias = np.ones((25, 1))
    
    Gen_GBF(sort_in_x, K)
    variance_gbf = Gen_GBF(sort_in_x, K)[1]
    phi_GBF = np.append(y_gbf, func_bias, axis=1)
    
    weight = Gen_Weight(y_real, phi_GBF) 
    # (2)의 weight 표 제작을 위한 데이터 수집
    if K == 3 or K == 6 or K == 8:
        history_part_weight.append(weight)
    y_hat = 0

    for n in range(K):
        y_hat = y_hat + weight[n] * \
            np.exp(-1 / 2 * pow((sort_in_x - u_gbf[n]) / variance_gbf, 2))
    y_hat = y_hat + weight[K]
    
    value_CF_MSE = np.sum(pow(y_hat - y_real, 2)) / len(sort_in_x)
    history_mse.append(value_CF_MSE)

# 정의역으로 사용될 기저함수 K 구간 정의
list_K_GBF = np.arange(3, 11, 1)

# Drawing MSE
plt.figure()
plt.plot(list_K_GBF, history_mse, 'g--')
plt.legend(['MSE'], loc='center right')
plt.xlabel('K; Count of Gauss Base Function')
plt.ylabel('MSE')
plt.title('Mean Square Error')
plt.grid(True, alpha=0.5)
plt.show()



















