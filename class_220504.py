# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chap.3 실습
# Data Refinement and Schematic
# Reading Data
input_data = pd.read_csv('Data_Base/logistic_regression_data.csv', \
                         index_col=0).to_numpy(dtype='float')

# plot에 그려진 Linear Combination을 통해 실제값의 y가 단조 증가가 아님을 확인
sorted_input_data = np.zeros((len(input_data), 3)) # 정렬에 사용하기 위한 input_data 행렬 복사본 정의
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(len(input_data)):
    sorted_input_data[n, :] = input_data[n, :]
sorted_input_data = sorted_input_data[sorted_input_data[:, 0].argsort()] 

sorted_in_x0 = sorted_input_data[:, 0] # x0를 오름차순으로 정렬
sorted_in_x1 = sorted_input_data[:, 1] # x1을 오름차순으로 정렬
y_real = sorted_input_data[:, 2] # 실제값 배열 정의
    
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
plt.scatter(in_x0_0, in_x1_0, marker='o', c='r', s=25)
plt.scatter(in_x0_1, in_x1_1, marker='x', c='b', s=25)
plt.legend(['0', '1'])
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base')
plt.grid(True, alpha=0.5)
plt.show()

# (1) Logistic Regression
# Setting Variable2
history_w0 = [] # weight0 기록함
history_w1 = [] # weight1 기록함
history_w2 = [] # weight1 기록함

history_p = []
history_y_hat = []

history_w0.append(np.random.rand() * 2 + 3) # weight0 초기값 초기화
history_w1.append(np.random.rand() * 2 + 4) # weight1 초기값 초기화
history_w2.append(np.random.rand() * 2 - 6) # weight1 초기값 초기화
# 함수의 입력 초기값
weight_w0 = history_w0[0]
weight_w1 = history_w1[0]
weight_w2 = history_w2[0]
learning_rate = 0.015# a; alpha / 학습률
step = 1000

def Func_Logistic_Regression(datax0, datax1, y_r, w0, w1, w2, a): # 로지스틱 회귀 함수 정의
    p = p_init
    for t in range(step):       
        w0 = w0 - a * np.sum((p - y_r) * datax0) / len(datax0)
        w1 = w1 - a * np.sum((p - y_r) * datax1) / len(datax0)
        w2 = w2 - a * np.sum(p - y_r) / len(datax0)
        
        p = 1 / (1 + np.exp(-(w0 * datax0 + w1 * datax1 + w2)))
            
        history_y_hat.append(y_hat)
        history_p.append(p)
        
        history_w0.append(w0)
        history_w1.append(w1)
        history_w2.append(w2)
           
    # 출력할 값들을 return
    return True

# (2)
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

training_set, validation_set, test_set = Dist_Set(sorted_input_data)

training_set = training_set[training_set[:, 0].argsort()] 
test_set = test_set[test_set[:, 0].argsort()] 

# 학습 데이터부 정렬 및 초기화
sorted_train_x0 = training_set[:, 0]
sorted_train_x1 = training_set[:, 1]
sorted_train_y = training_set[:, 2]

container_train_x0_0 = []
container_train_x1_0 = []
container_train_x0_1 = []
container_train_x1_1 = []
for n in range(len(sorted_train_x0)):
    if sorted_train_y[n] == 0:
        container_train_x0_0.append(sorted_train_x0[n])
        container_train_x1_0.append(sorted_train_x1[n])
    elif sorted_train_y[n] == 1:
        container_train_x0_1.append(sorted_train_x0[n])
        container_train_x1_1.append(sorted_train_x1[n])

# 평가 데이터부 정렬 및 초기화
sorted_test_x0 = test_set[:, 0]
sorted_test_x1 = test_set[:, 1]
sorted_test_y = test_set[:, 2]

container_test_x0_0 = []
container_test_x1_0 = []
container_test_x0_1 = []
container_test_x1_1 = []
for n in range(len(sorted_test_x0)):
    if sorted_test_y[n] == 0:
        container_test_x0_0.append(sorted_test_x0[n])
        container_test_x1_0.append(sorted_test_x1[n])
    elif sorted_test_y[n] == 1:
        container_test_x0_1.append(sorted_test_x0[n])
        container_test_x1_1.append(sorted_test_x1[n])

# Drawing Training and Test Set
plt.figure()
plt.scatter(container_train_x0_0, container_train_x1_0, marker='o', c='r', s=25)
plt.scatter(container_train_x0_1, container_train_x1_1, marker='x', c='r', s=25)
plt.scatter(container_test_x0_0, container_test_x1_0, marker='o', c='g', s=25)
plt.scatter(container_test_x0_1, container_test_x1_1, marker='x', c='g', s=25)
plt.legend(['Training set - 0', 'Training set - 1', \
            'Test set - 0', 'Test set - 1'])
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base')
plt.grid(True, alpha=0.5)
plt.show()

# (3) 정확도 및 CEE
y_hat = weight_w0 * sorted_train_x0 + weight_w1 * sorted_train_x1 + weight_w2
p_init = 1 / (1 + np.exp(-y_hat))
Func_Logistic_Regression(sorted_train_x0, sorted_train_x1, sorted_train_y, \
                         weight_w0, weight_w1, weight_w2, learning_rate)

decided_y_hat = np.zeros((step, len(sorted_train_x0)))
def Decide_y_hat(p):
    for n in range(step):
        for m in range(len(sorted_train_x0)):
            if p[n][m] >= 0.5:
                decided_y_hat[n][m] = 1
            else:
                decided_y_hat[n][m] = 0
    return True

Decide_y_hat(history_p)

matrix_accuracy = np.zeros((step, len(sorted_train_x0)))
accuracy = 0.
history_accuracy = []
def Measure_Accuracy(dcd_y_hat, train_y):
    for n in range(step):
        for m in range(len(sorted_train_x0)):
            if dcd_y_hat[n][m] == train_y[m]:
                matrix_accuracy[n][m] = True
            else:
                matrix_accuracy[n][m] = False
    count_accuracy = 0
    for n in range(step):
        for m in range(len(sorted_train_x0)):
            if matrix_accuracy[n][m] == 1:
                count_accuracy = count_accuracy + 1
        accuracy = (count_accuracy / 350) * 100
        history_accuracy.append(accuracy)
        count_accuracy = 0
    return True

Measure_Accuracy(decided_y_hat, sorted_train_y)
history_CEE = []
def Measure_CEE(p, train_y):
    for n in range(step):
        CEE = -np.sum(train_y * np.log(p[n]) + (1 - train_y) * np.log(1 - p[n])) / len(train_y)
        history_CEE.append(CEE)
    return True

Measure_CEE(history_p, sorted_train_y)

# Setting step
step1 = np.arange(0, step + 1, 1)
step2 = np.arange(0, step, 1)

# Drawing Weight
plt.figure()
plt.plot(step1, history_w0, 'rv-', markevery = 50)
plt.plot(step1, history_w1, 'gx-', markevery = 50)
plt.plot(step1, history_w2, 'bo-', markevery = 50)
plt.legend(['w0', 'w1', 'w2'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('w0, w1, w2')
plt.title('Logistic Regression - Weight')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing Accuracy
plt.figure()
plt.plot(step2, history_accuracy, 'ko-', markevery = 50)
plt.legend(['Accuracy'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('Accuracy')
plt.title('Logistic Regression - Accuracy')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing CEE
plt.figure()
plt.plot(step2, history_CEE, 'ko-', markevery = 50)
plt.legend(['CEE'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('CEE')
plt.title('Logistic Regression - CEE')
plt.grid(True, alpha=0.5)
plt.show()





























