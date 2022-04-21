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

flag_div_703 = False
flag_overfitting = True

# 데이터를 학습 데이터, 검증 데이터, 평가 데이터로 분할하는 함수
def Dist_Set(data): # 원본 데이터를 입력값으로 함
    # 입력 전 방법 안내
    print("Data set에서 train, validation, test set으로의 분할 비율을 입력하세요.")
    print("(주의) 비율의 총합은 10입니다.")
    
    # 예외처리
    while(True):
        # 입력 직전 방법 안내 및 비율값 입력
        print("(Example) 입력 비율 : 2  (입력값은 0부터 9까지의 자연수))")
        ratio_train = int(input("Train set의 입력 비율 : "))
        ratio_val = int(input("Validaion set의 입력 비율 : "))
        ratio_test = int(input("Test set의 입력 비율 : "))
        ratio_sum = ratio_train + ratio_val + ratio_test
        if ratio_sum == 10: # 비율값들의 합이 10이어야만 탈출
            break
        else: # 주의사항 상기 후 재입력 유도
            print("")
            print("비율의 합이 10이 아닙니다.")
            print("(주의) 비율의 총합은 10이어야 합니다.")
            print("다시 입력해주세요")
    
    print(ratio_train, ratio_val, ratio_test)
    if ratio_train == 7 and ratio_val == 0 and ratio_test == 3:
        flag = True
    
    # 학습, 검증, 평가 데이터의 개수 초기화
    num_train = int(round(len(data) * ratio_train / 10, 0))
    num_val = int(round(len(data) * ratio_val / 10, 0))
    num_test = len(data) - num_train - num_val
    
    # 앞서와 마찬가지로 간편한 데이터 정제를 위해 pandas 활용
    # 데이터 랜덤 비복원 추출 방식
    df_xy = pd.DataFrame(data)
    # 학습 데이터 초기화
    train_set = df_xy.sample(n=num_train, replace=False)
    df_xy = df_xy.drop(train_set.index)
    # 검증 데이터 초기화
    val_set = df_xy.sample(n=num_val, replace=False)
    df_xy = df_xy.drop(val_set.index)
    # 평가 데이터 초기
    test_set = df_xy.sample(n=num_test, replace=False)
    
    # 정제된 데이터를 numpy 데이터로 변환
    train_set = train_set.to_numpy()
    val_set = val_set.to_numpy()
    test_set = test_set.to_numpy()
    
    return train_set, val_set, test_set, flag # 분할된 DB 반환

# 함수 호출 및 반환값에 대한 학습, 검증, 평가 DB 초기화
training_set, validation_set, test_set, flag_div_703 = Dist_Set(input_data)

# 분할된 DB의 그래프를 생성하기 위해 필요로 하는 정의역과 치역 초기화
# 학습 데이터부 초기화
data_train_x = training_set[:, 0] # 정의역
data_train_y = training_set[:, 1] # 치역

# 검증 데이터부 초기화
data_val_x = validation_set[:, 0] # 정의역
data_val_y = validation_set[:, 1] # 치역

# 평가 데이터부 초기화
data_test_x = test_set[:, 0] # 정의역
data_test_y = test_set[:, 1] # 치역

training_set = training_set[training_set[:, 0].argsort()] 
test_set = test_set[test_set[:, 0].argsort()] 

sorted_train_x = training_set[:, 0]
sorted_train_y = training_set[:, 1]
sorted_test_x = test_set[:, 0]
sorted_test_y = test_set[:, 1]


# # (3)
# history_test_MSE = []
# history_training_MSE = []
# if flag_div_703 == True:
#     flag_div_703 = False
#     print("DB 준비 완료!")
    
# # 지난주차의 기저함수 및 가중치 발진 함수 활용
# K_GBF = 3 # 기저함수의 개수
# y_gbf = np.zeros((len(in_x), K_GBF)) # bias를 제외한 입력에 대한 행렬 
# u_gbf = [] # 기저함수의 개수에 따른 가우스함수의 평균값 기록함
# variance_gbf = 0 # 기저함수의 개수에 따른 가우스함수의 분산값
# # Gaussian Base Function 생성 함수
# def Gen_GBF(x, K): # 입력; 1. 정렬된 데이터 입력 / 2. 기저함수의 개수
#     # 가우스 함수 평균 및 분산 계산    
#     for k in range(K):
#         u_gbf.append(np.min(x) + (np.max(x) - np.min(x)) / (K - 1) * k)
#     variance_gbf = (np.max(x) - np.min(x)) / (K - 1)
#     # bias를 제외한 입력에 대한 행렬 초기화
#     for n in range(len(x)):
#         for k in range(K):
#             y_gbf[n][k] = np.exp(-1 / 2 * pow((x[n] - u_gbf[k]) / variance_gbf, 2))
    
#     return y_gbf, variance_gbf # 입력에 대한 행렬 및 분산 반환

# Weight 생성 함수
def Gen_Weight(y, phi): # 입력; 1. 정렬된 데이터 출력 / 2. bias를 포함한 입력에 대한 행렬
    # w = np.linalg.inv(np.transpose(phi) * phi) * np.transpose(phi) * y
    # 행렬의 곱이기 때문에 A.dot(B) 이용
    w = np.linalg.inv(np.transpose(phi).dot(phi)).dot(np.transpose(phi)).dot(y)
    
    return w # 가중치 반환

# for K in range(3, 20):
#     y_gbf = np.zeros((len(sorted_train_x), K))
#     u_gbf = []
#     variance_gbf = 0
#     func_bias = np.ones((len(sorted_train_x), 1))
#     Gen_GBF(sorted_train_x, K)
#     variance_gbf = Gen_GBF(sorted_train_x, K)[1]
#     phi_GBF = np.append(y_gbf, func_bias, axis=1)
    
#     weight = Gen_Weight(sorted_train_y, phi_GBF) 
#     y_hat_train = 0
#     for n in range(K):
#         y_hat_train = y_hat_train + weight[n] * \
#             np.exp(-1 / 2 * pow((sorted_train_x - u_gbf[n]) / variance_gbf, 2))
#     y_hat_train = y_hat_train + weight[K]
#     value_CF_MSE_train = np.sum(pow(y_hat_train - sorted_train_y, 2)) / len(sorted_train_x)
#     history_training_MSE.append(value_CF_MSE_train)
    
    
#     y_gbf = np.zeros((len(sorted_test_x), K))
#     u_gbf = []
#     variance_gbf = 0
#     func_bias = np.ones((len(sorted_test_x), 1))
#     Gen_GBF(sorted_test_x, K)
#     variance_gbf = Gen_GBF(sorted_test_x, K)[1]
#     phi_GBF = np.append(y_gbf, func_bias, axis=1)
#     weight = Gen_Weight(sorted_test_y, phi_GBF) 
#     y_hat_test = 0
#     for n in range(K):
#         y_hat_test = y_hat_test + weight[n] * \
#             np.exp(-1 / 2 * pow((sorted_test_x - u_gbf[n]) / variance_gbf, 2))
#     y_hat_test = y_hat_test + weight[K]
#     value_CF_MSE_test = np.sum(pow(y_hat_test - sorted_test_y, 2)) / len(sorted_test_x)
#     if value_CF_MSE_test > 3 and flag_overfitting == True:
#         print("최적의 가우시안 기저함수 개수 K는 ", K)
#         flag_overfitting = False
#     history_test_MSE.append(value_CF_MSE_test)

# # 정의역으로 사용될 기저함수 K 구간 정의
# list_K_GBF = np.arange(3, 20, 1)

# # Drawing Training ans Validation and Test Set
# plt.figure()
# plt.scatter(data_train_x, data_train_y, color='red')
# plt.scatter(data_val_x, data_val_y, color='blue')
# plt.scatter(data_test_x, data_test_y, color='green')
# plt.plot(sorted_train_x, y_hat_train)
# plt.legend(['train_as', 'Training set', 'Validation set', 'Test set'])
# plt.xlabel('x; Weight')
# plt.ylabel('y; Length')
# plt.title('Distrubute Original Set')
# plt.grid(True, alpha=0.5)
# plt.show()

# # Drawing MSE
# plt.figure()
# plt.plot(list_K_GBF, history_training_MSE, 'g--')
# plt.plot(list_K_GBF, history_test_MSE, 'b--')
# plt.legend(['MSE of training', 'MSE of test'], loc='upper left')
# plt.xlabel('K; Count of Gauss Base Function')
# plt.ylabel('MSE')
# plt.title('Mean Square Error')
# plt.grid(True, alpha=0.5)
# plt.show()

# (4)
def Gen_PBF(x, K): # 입력; 1. 정렬된 데이터 입력 / 2. 기저함수의 개수
    for n in range(len(x)):
        for k in range(K):
            y_pbf[n][k] = pow(x, k)
    return y_pbf

history_training_MSE_PBF = []
for K in range(3, 20):
    y_pbf = np.zeros((len(sorted_train_x), K))
    func_bias = np.ones((len(sorted_train_x), 1))
    Gen_PBF(sorted_train_x, K)
    phi_PBF = np.append(y_pbf, func_bias, axis=1)
    
    weight = Gen_Weight(sorted_train_y, phi_PBF) 
    y_hat_train_PBF = 0
    for n in range(K):
        y_hat_train_PBF = y_hat_train_PBF + weight[n] * pow(sorted_train_x, n)
    y_hat_train_PBF = y_hat_train_PBF + weight[K]
    value_CF_MSE_train_PBF = np.sum(pow(y_hat_train_PBF - sorted_train_y, 2)) / len(sorted_train_x)
    history_training_MSE_PBF.append(value_CF_MSE_train_PBF)

# Drawing Training ans Validation and Test Set
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
# plt.scatter(data_val_x, data_val_y, color='blue')
# plt.scatter(data_test_x, data_test_y, color='green')
plt.plot(sorted_train_x, y_hat_train_PBF)
# plt.legend(['train_as', 'Training set', 'Validation set', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Distrubute Original Set')
plt.grid(True, alpha=0.5)
plt.show()










