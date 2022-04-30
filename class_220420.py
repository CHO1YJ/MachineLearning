# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #3
# (1)
# Reading Data
# data 열은 index 열로 우리의 실습에서는 필요 없으므로 drop한다.
input_data = pd.read_csv('Data_Base/chap2_data.csv')\
    .drop(['data'], axis=1).to_numpy(dtype='float')

# Setting Variable1
# 이미 정렬이 되어있으므로 데이터 그대로 사용한다.
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length

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
    # if ratio_train == 7 and ratio_val == 0 and ratio_test == 3:
    #     flag = True
    
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
    
    return train_set, val_set, test_set # 분할된 DB 반환

# 함수 호출 및 반환값에 대한 학습, 검증, 평가 DB 초기화
training_set, validation_set, test_set = Dist_Set(input_data)

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

# (3)   
# 지지난주차의 기저함수 및 가중치 발진 함수 활용
K_GBF = 3 # 기저함수의 개수
y_gbf = np.zeros((len(in_x), K_GBF)) # bias를 제외한 입력에 대한 행렬 
u_gbf = [] # 기저함수의 개수에 따른 가우스함수의 평균값 기록함
variance_gbf = 0 # 기저함수의 개수에 따른 가우스함수의 분산값
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

history_test_MSE = [] # 가우시안 기저함수의 평가 DB MSE 기록함
history_training_MSE = [] # 가우시안 기저함수의 훈련 DB MSE 기록함
flag_overfitting_GBF = False # GBF에서의 overfitting 발생 여부 확인
K_iter_gbf = 150 # 가우시안 기저함수의 최대 개수
K_optimal_GBF = 0 # GBF의 이상적 기저함수 개수
# 가우시안 기저함수 개수 K에 따른 기저함수 발생 함수와 가중치 발생 함수를 
# 활용하여 y_hat과 MSE를 계산하는 반복문
# 세부 계산 내용은 지지난주차의 내용과 동일하므로 추가된 부분에 대하여 설명
for K in range(3, K_iter_gbf):
    # Training set에 대한 y_hat, MSE 계산 과정
    y_gbf = np.zeros((len(sorted_train_x), K))
    u_gbf = []
    variance_gbf = 0
    func_bias = np.ones((len(sorted_train_x), 1))
    Gen_GBF(sorted_train_x, K)
    variance_gbf = Gen_GBF(sorted_train_x, K)[1]
    phi_GBF = np.append(y_gbf, func_bias, axis=1)
    
    weight_GBF = Gen_Weight(sorted_train_y, phi_GBF) 
    y_hat_train = 0
    for n in range(K):
        y_hat_train = y_hat_train + weight_GBF[n] * \
            np.exp(-1 / 2 * pow((sorted_train_x - u_gbf[n]) / variance_gbf, 2))
    y_hat_train = y_hat_train + weight_GBF[K]
    value_CF_MSE_train = np.sum(pow(y_hat_train - sorted_train_y, 2)) / len(sorted_train_x)
    history_training_MSE.append(value_CF_MSE_train)
    
    # Test set에 대한 y_hat, MSE 계산 과정
    y_gbf = np.zeros((len(sorted_test_x), K))
    u_gbf = []
    variance_gbf = 0
    func_bias = np.ones((len(sorted_test_x), 1))
    Gen_GBF(sorted_test_x, K)
    variance_gbf = Gen_GBF(sorted_test_x, K)[1]
    phi_GBF = np.append(y_gbf, func_bias, axis=1)
    weight_GBF = Gen_Weight(sorted_test_y, phi_GBF) 
    y_hat_test = 0
    for n in range(K):
        y_hat_test = y_hat_test + weight_GBF[n] * \
            np.exp(-1 / 2 * pow((sorted_test_x - u_gbf[n]) / variance_gbf, 2))
    y_hat_test = y_hat_test + weight_GBF[K]
    value_CF_MSE_test = np.sum(pow(y_hat_test - sorted_test_y, 2)) / len(sorted_test_x)
    # 평가 DB를 통한 MSE 값을 활용하여 Training model 검토
    # MSE가 1보다 커지는 순간 MSE 곡선의 기울기 부호가 변화함을 인식
    if value_CF_MSE_test > 1 and flag_overfitting_GBF == False:
        # 현재의 K는 MSE가 1을 넘어선 K이므로 1을 감소
        # 따라서, Optimal K = K - 1
        print("최적의 가우시안 기저함수 개수 K는 ", K - 1)
        print("현재 K에 대한 MSE 값 : ", value_CF_MSE_test)
        # 가우시안 기저함수의 이상적 K 개수를 초기화
        K_optimal_GBF = K - 1
        # Overfitting이 발생하였으므로 flag를 False에서 True로 전환
        flag_overfitting_GBF = True
    history_test_MSE.append(value_CF_MSE_test)

# 정의역으로 사용될 기저함수 K 구간 정의
list_K_GBF = np.arange(3, K_iter_gbf, 1)

# Drawing Training and Validation and Test Set
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
plt.scatter(data_val_x, data_val_y, color='blue')
plt.scatter(data_test_x, data_test_y, color='green')
plt.plot(sorted_train_x, y_hat_train, 'b')
plt.legend(['train_as', 'Training set', 'Validation set', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('1. Distrubute Original Set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing MSE
plt.figure()
plt.plot(list_K_GBF, history_training_MSE, 'g--')
plt.plot(list_K_GBF, history_test_MSE, 'b--')
plt.legend(['MSE of training', 'MSE of test'], loc='upper left')
plt.xlabel('K; Count of Gauss Basis Function')
plt.ylabel('MSE')
plt.title('2. Mean Square Error')
plt.grid(True, alpha=0.5)
plt.show()

# 가우시안 기저함수의 이상적 K 개수에 대한 Analytic Solution을 도식화
for K in range(3, K_optimal_GBF): # 이상적인 기저함수 개수의 범위 제시
    y_gbf = np.zeros((len(sorted_train_x), K))
    u_gbf = []
    variance_gbf = 0
    func_bias = np.ones((len(sorted_train_x), 1))
    Gen_GBF(sorted_train_x, K)
    variance_gbf = Gen_GBF(sorted_train_x, K)[1]
    phi_GBF = np.append(y_gbf, func_bias, axis=1)
    
    weight_GBF = Gen_Weight(sorted_train_y, phi_GBF) 
    y_hat_train = 0
    for n in range(K):
        y_hat_train = y_hat_train + weight_GBF[n] * \
            np.exp(-1 / 2 * pow((sorted_train_x - u_gbf[n]) / variance_gbf, 2))
    y_hat_train = y_hat_train + weight_GBF[K]

# Drawing Linear Regression1 - Training DB
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
plt.plot(sorted_train_x, y_hat_train, 'b')
plt.legend(['Analytic solution', 'Training set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('3-1. Linear Regression')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing Linear Regression2 - Test DB
plt.figure()
plt.scatter(data_test_x, data_test_y, color='green')
plt.plot(sorted_train_x, y_hat_train, 'b')
plt.legend(['Analytic solution', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('3-2. Linear Regression')
plt.grid(True, alpha=0.5)
plt.show()

# (4)
def Gen_PBF(x, K): # 입력; 1. 정렬된 데이터 입력 / 2. 기저함수의 개수
    for n in range(len(x)):
        for k in range(K):
            y_pbf[n][k] = pow(x[n], k + 1)
    return y_pbf

history_training_MSE_PBF = [] # 가우시안 기저함수의 평가 DB MSE 기록함
history_test_MSE_PBF = [] # 가우시안 기저함수의 훈련 DB MSE 기록함
flag_overfitting_PBF = False # PBF에서의 overfitting 발생 여부 확인
K_iter_pbf = 20 # 다항식 기저함수의 최대 개수
K_optimal_PBF = 0 # PBF의 이상적 기저함수 개수
# GBF에서와 반복문 생성 근거가 동일
for K in range(3, K_iter_pbf):
    # Training set에 대한 y_hat, MSE 계산 과정
    y_pbf = np.zeros((len(sorted_train_x), K))
    func_bias = np.ones((len(sorted_train_x), 1))
    Gen_PBF(sorted_train_x, K)
    phi_PBF = np.append(y_pbf, func_bias, axis=1)
    weight_PBF = Gen_Weight(sorted_train_y, phi_PBF) 
    y_hat_train_PBF = 0
    for n in range(K):
        y_hat_train_PBF = y_hat_train_PBF + weight_PBF[n] * pow(sorted_train_x, n + 1)
    y_hat_train_PBF = y_hat_train_PBF + weight_PBF[K]
    value_CF_MSE_train_PBF = np.sum(pow(y_hat_train_PBF - sorted_train_y, 2)) / len(sorted_train_x)
    history_training_MSE_PBF.append(value_CF_MSE_train_PBF)
    
    # Test set에 대한 y_hat, MSE 계산 과정
    y_pbf = np.zeros((len(sorted_test_x), K))
    func_bias = np.ones((len(sorted_test_x), 1))
    Gen_PBF(sorted_test_x, K)
    phi_PBF = np.append(y_pbf, func_bias, axis=1)
    weight_PBF = Gen_Weight(sorted_test_y, phi_PBF) 
    y_hat_test_PBF = 0
    for n in range(K):
        y_hat_test_PBF = y_hat_test_PBF + weight_PBF[n] * pow(sorted_test_x, n + 1)
    y_hat_test_PBF = y_hat_test_PBF + weight_PBF[K]
    value_CF_MSE_test_PBF = np.sum(pow(y_hat_test_PBF - sorted_test_y, 2)) / len(sorted_test_x)
    # GBF에서와 모델 평가 검토 근거가 동일
    if value_CF_MSE_test_PBF > 1 and flag_overfitting_PBF == False:
        print("최적의 다항식 기저함수 개수 K는 ", K - 1)
        print("현재 K에 대한 MSE 값 : ", value_CF_MSE_test_PBF)
        # 다항식 기저함수 개수의 이상적 K 개수를 초기화
        K_optimal_PBF = K - 1
        flag_overfitting_PBF = True
    history_test_MSE_PBF.append(value_CF_MSE_test_PBF)

# 정의역으로 사용될 기저함수 K 구간 정의
list_K_GBF = np.arange(3, K_iter_pbf, 1)

# Drawing Training ans Validation and Test Set
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
plt.scatter(data_test_x, data_test_y, color='green')
plt.plot(sorted_train_x, y_hat_train_PBF)
plt.legend(['train_as', 'Training set', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('4. Distrubute Original Set')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing MSE
plt.figure()
plt.plot(list_K_GBF, history_training_MSE_PBF, 'g--')
plt.plot(list_K_GBF, history_test_MSE_PBF, 'b--')
plt.legend(['MSE of training', 'MSE of test'], loc='upper left')
plt.xlabel('K; Count of Polynomial Basis Function')
plt.ylabel('MSE')
plt.title('5. Mean Square Error')
plt.grid(True, alpha=0.5)
plt.show()

# 다항식 기저함수의 이상적 K 개수에 대한 Analytic Solution을 도식화
for K in range(3, K_optimal_PBF): # 이상적인 기저함수 개수의 범위 제시
    y_pbf = np.zeros((len(sorted_train_x), K))
    func_bias = np.ones((len(sorted_train_x), 1))
    Gen_PBF(sorted_train_x, K)
    phi_PBF = np.append(y_pbf, func_bias, axis=1)
    weight_PBF = Gen_Weight(sorted_train_y, phi_PBF) 
    y_hat_train_PBF = 0
    for n in range(K):
        y_hat_train_PBF = y_hat_train_PBF + weight_PBF[n] * pow(sorted_train_x, n + 1)
    y_hat_train_PBF = y_hat_train_PBF + weight_PBF[K]

# Drawing Linear Regression1 - Training DB
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
plt.plot(sorted_train_x, y_hat_train_PBF, 'b')
plt.legend(['Analytic solution', 'Training set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('6-1. Linear Regression')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing Linear Regression2 - Test DB
plt.figure()
plt.scatter(data_test_x, data_test_y, color='green')
plt.plot(sorted_train_x, y_hat_train_PBF, 'b')
plt.legend(['Analytic solution', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('6-2. Linear Regression')
plt.grid(True, alpha=0.5)
plt.show()






