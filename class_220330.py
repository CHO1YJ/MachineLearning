# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #1
# (1) & (2)
# Reading Data
input_data = pd.read_csv('Data_Base/lin_regression_data_01.csv', header=None).to_numpy(dtype='float')

# Setting Variable1
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length
weight_0 = 0. # weight0; 가중치 0
weight_1 = 0. # weight1; 가중치 1
value_CF_MSE = 0. # Cost function; 비용함수

# Calculating Weight
weight_0 = np.sum(out_y * (in_x - np.sum(in_x) / len(in_x))) / len(in_x) \
    / (np.sum(pow(in_x, 2)) / len(in_x) - pow(np.sum(in_x) / len(in_x), 2))
weight_1 = sum(out_y - weight_0 * in_x) / len(in_x)
    
# Drawing Line1
func_y = weight_0 * in_x + weight_1 # 선형회귀모델에 사용될 in_x에 대응되는 y값

# Drawing Linear Regression1
plt.figure()
plt.scatter(in_x, out_y, color='0')
plt.plot(in_x, func_y, 'r')
plt.legend(['Analytic solution', 'Linear Combination'])
# plt.legend(['Linear Combination'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Linear Regression1')
plt.grid(True, alpha=0.5)
plt.show()

# (3)
# Calculating MSE
y_hat = np.zeros((25, 1)) # 예측값 배열 정의
for n in range(25):
    y_hat[n] = sorted(func_y)[n] # 예측값 배열 초기화
    
# plot에 그려진 Linear Combination을 통해 실제값의 y가 단조 증가가 아님을 확인
temp_input_data = np.zeros((25, 2)) # 정렬에 사용하기 위한 input_data 행렬 복사본 정의
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(25):
    temp_input_data[n, :] = input_data[n, :]
temp_input_data = temp_input_data[temp_input_data[:, 0].argsort()] 

sort_in_x = temp_input_data[:, 0] # x를 오름차순으로 정렬
y_real = temp_input_data[:, 1] # 실제값 배열 정의
    
# MSE 계산

# 기존 방식
# sum_result = 0
# for n in range(25):
#     sum_result = sum_result + pow(y_hat[n] - y_real[n], 2)
# value_CF_MSE = value_CF_MSE[0]

# 개선 방식
value_CF_MSE = np.sum(pow(y_hat[n] - y_real[n], 2)) / len(sort_in_x)
print(value_CF_MSE) # MSE값 출력

# 실습과제 #2
# Setting Variable2
history_w0 = [] # weight0 기록함
history_w1 = [] # weight1 기록함
history_mse = [] # MSE 기록함
history_mse_dw0w1 = [] # MSE의 미분계수 기록함

history_w0.append(np.random.rand() * 2 + 2) # weight0 초기값 초기화
history_w1.append(np.random.rand() * 2 + 4) # weight1 초기값 초기화

# 함수의 입력 초기
weight_w0 = history_w0[0]
weight_w1 = history_w1[0]
learning_rate = 0.015# a; alpha / 학습률

def Func_Gradient_Decent_Method(w0, w1, a): # 경사하강법 함수 정의
    for t in range(6000):
        # 경사하강법 기반 y_hat, weight0, weight1 초기화
        y_hat_2 = w0 * sort_in_x + w1
        w0 = w0 - a * np.sum((y_hat_2 - y_real) * sort_in_x) / len(sort_in_x)
        w1 = w1 - a * np.sum(y_hat_2 - y_real) / len(sort_in_x)
        
        # weight0, weight1, MSE, MSE의 미분계수 기록
        history_w0.append(w0)
        history_w1.append(w1)
        history_mse.append(np.sum(pow(y_hat_2 - y_real, 2)) / len(sort_in_x))
        history_mse_dw0w1.append(np.sum((y_hat_2 - y_real) * sort_in_x) / len(sort_in_x))
           
    # 출력할 값들을 return
    return history_w0, history_w1, history_mse, history_mse_dw0w1, y_hat_2

# 경사하강법 함수 호출
Func_Gradient_Decent_Method(weight_w0, weight_w1, learning_rate)

# Setting step
step1 = np.arange(0, 6001, 1)
step2 = np.arange(0, 6000, 1)

# Drawing Weight
plt.figure()
plt.plot(step1, history_w0, 'r--')
plt.plot(step1, history_w1, 'b--')
plt.legend(['w0', 'w1'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('w0, w1')
# plt.xlim([0, 6000])
plt.ylim([-1, 8])
plt.title('Gradient Decent Method1')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing MSE
plt.figure()
plt.plot(step2, history_mse_dw0w1, 'g--')
plt.plot(step2, history_mse, 'k--')
plt.legend(['dw0w1_MSE', 'MSE'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('MSE, dw0w1_MSE')
# plt.xlim([0, 6000])
plt.ylim([-1, 3])
plt.title('Gradient Decent Method2')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing Line2
func_y_2 = history_w0[5999] * in_x + history_w1[5999] # 모델에 사용될 in_x에 대응되는 y값

# Drawing Linear Regression2
plt.figure()
plt.scatter(in_x, out_y, color='0')
plt.plot(in_x, func_y_2, 'y--')
plt.plot(in_x, func_y_2, 'rx')
plt.legend(['Analytic solution', 'y_hat', 'Linear Combination'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Linear Regression2')
plt.grid(True, alpha=0.5)
plt.show()


print('실습과제1의 가중치 w0는 {:.6f}'.format(weight_0))
print('실습과제1의 가중치 w1은 {:.6f}'.format(weight_1))
print('실습과제2의 가중치 w0는 {:.6f}'.format(history_w0[5999]))
print('실습과제2의 가중치 w1은 {:.6f}'.format(history_w1[5999]))
print('실습과제1의 MSE는 {:.6f}'.format(value_CF_MSE))
print('실습과제2의 MSE는 {:.6f}'.format(history_mse[5999]))
print('실습과제2의 MSE의 미분계수는 {:.6f}'.format(history_mse_dw0w1[5999]))























