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
# 정렬에 사용하기 위한 input_data 행렬 복사본 정의
sorted_input_data = np.zeros((len(input_data), 3))
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(len(input_data)):
    sorted_input_data[n, :] = input_data[n, :]
sorted_input_data = sorted_input_data[sorted_input_data[:, 0].argsort()] 

# 데이터를 y에 대한 값에 따라 분류
list_color = []
input_data_0 = np.zeros((int(len(input_data) / 2), 3)) # 출력이 0인 데이터 기록함
input_data_1 = np.zeros((int(len(input_data) / 2), 3)) # 출력이 1인 데이터 기록함
# 기존 데이터에서 250을 기준으로 0과 1로 나뉘므로 이를 통해 index를 조정하여 초기화
for n in range(len(input_data)):
    if n < len(input_data) / 2:
        list_color.append('r')
        input_data_0[n, :] = input_data[n, :] # 250 이하의 y가 0인 데이터
    elif n >= len(input_data) / 2:
        list_color.append('b')
        input_data_1[n - 250, :] = input_data[n, :] # 250 이상의 y가 1인 데이터
        
in_x0_0 = input_data_0[:, 0] # 입력; x0
in_x1_0 = input_data_0[:, 1] # 입력; x1

in_x0_1 = input_data_1[:, 0] # 입력; x0
in_x1_1 = input_data_1[:, 1] # 입력; x1

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

history_p = [] # 확률 probability 기록함
history_y_hat = [] # y_hat 기록함

history_w0.append(np.random.rand() * 2 + 3) # weight0 초기값 초기화
history_w1.append(np.random.rand() * 2 + 4) # weight1 초기값 초기화
history_w2.append(np.random.rand() * 2 - 6) # weight1 초기값 초기화
# 함수의 입력 초기값
weight_w0 = history_w0[0] # w0 초기값
weight_w1 = history_w1[0] # w1 초기값
weight_w2 = history_w2[0] # w2 초기값
learning_rate = 0.015# a; alpha / 학습률
step = 1000 # 학습 횟수

# Logistic Regression 함수 구현
# x1, x0 y_real 데이터와 가중치 w0, w1, w2 그리고 학습률 a가 함수의 입력
def Func_Logistic_Regression(datax0, datax1, y_r, w0, w1, w2, a):
    # 학습 횟수에 따라 w0, w1, w2 값 갱신
    p = p_init # 확률의 초기값 설정
    for t in range(step):       
        w0 = w0 - a * np.sum((p - y_r) * datax0) / len(datax0)
        w1 = w1 - a * np.sum((p - y_r) * datax1) / len(datax0)
        w2 = w2 - a * np.sum(p - y_r) / len(datax0)
        
        # 갱신된 w0, w1, w2에 따른 확률값 p 갱신
        p = 1 / (1 + np.exp(-(w0 * datax0 + w1 * datax1 + w2)))
            
        # 데이터 확인 및 추후 도식화와 함수에 사용하기 위한 데이터 수집
        history_y_hat.append(y_hat)
        history_p.append(p)
        history_w0.append(w0)
        history_w1.append(w1)
        history_w2.append(w2)
           
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

# 훈련, 검증, 평가 DB 초기화; 입력은 정렬된 초기 데이터
training_set, validation_set, test_set = Dist_Set(sorted_input_data)

# 랜덤으로 추출된 훈련, 평가 DB를 다시 정렬
training_set = training_set[training_set[:, 0].argsort()] 
test_set = test_set[test_set[:, 0].argsort()] 

# 학습 데이터부 정렬 및 초기화
sorted_train_x0 = training_set[:, 0]
sorted_train_x1 = training_set[:, 1]
sorted_train_y = training_set[:, 2]

# 정렬된 데이터의 출력 y를 0과 1로 구분하여 기록
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

# 정렬된 데이터의 출력 y를 0과 1로 구분하여 기록
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

# (3) 정확도 및 CEE
# y_hat의 초기값 초기화
y_hat = weight_w0 * sorted_train_x0 + weight_w1 * sorted_train_x1 + weight_w2
# 확률의 초기값 초기화
p_init = 1 / (1 + np.exp(-y_hat))

# Logistic Regression 함수 호출
Func_Logistic_Regression(sorted_train_x0, sorted_train_x1, sorted_train_y,\
                         weight_w0, weight_w1, weight_w2, learning_rate)

# y_hat 값 기록함 정의
decided_y_hat = np.zeros((step, len(sorted_train_x0)))
# y_hat을 확률 값에 따라 1과 0으로 구분하는 함수 정의; 입력값은 확률값 p
def Decide_y_hat(p, data):
    for n in range(step):
        for m in range(len(data)):
            if p[n][m] >= 0.5:
                decided_y_hat[n][m] = 1
            else:
                decided_y_hat[n][m] = 0
    return True

# y_hat 결정 함수 호출
Decide_y_hat(history_p, sorted_train_x0)

# 정확도 기록함 행렬 정의; 각각의 성분에 대하여 True와 False로 표현
matrix_accuracy = np.zeros((step, len(sorted_train_x0)))
# 정확도 값 정의
accuracy = 0.
# 정확도 기록함 정의; 학습 횟수에 따라 확률로 표현
history_accuracy = []
# 정확도 측정 함수 정의; 입력은 결정된 y_hat 값과 훈련 DB의 y(0과 1로 구성)값
def Measure_Accuracy(dcd_y_hat, data):
    for n in range(step):
        for m in range(len(data)):
            # 결정된 y_hat과 훈련 DB의 y값이 같으면 True 아니면 False로 초기화
            if dcd_y_hat[n][m] == data[m]:
                matrix_accuracy[n][m] = True
            else:
                matrix_accuracy[n][m] = False
    # 정확도의 정도에 대한 카운트 정의
    count_accuracy = 0
    for n in range(step):
        for m in range(len(data)):
            # True는 곧 1이므로 1이 많으면 정확도가 높은 것!
            if matrix_accuracy[n][m] == 1:
                count_accuracy = count_accuracy + 1
        # 정확도 초기화 및 기록
        accuracy = (count_accuracy / 350) * 100
        history_accuracy.append(accuracy)
        # 학습횟수마다 count를 해야하므로 0으로 초기화 후 다시 count
        count_accuracy = 0
    return True

# 정확도 측정 함수 호출
Measure_Accuracy(decided_y_hat, sorted_train_y)
# CEE 기록함 정의
history_CEE = []
# CEE 측정 함수 정의; 입력은 확률값 p와 훈련 DB의 y값
def Measure_CEE(p, train_y):
    for n in range(step):
        CEE = -np.sum(train_y * np.log(p[n]) + (1 - train_y) * np.log(1 - p[n])) / len(train_y)
        history_CEE.append(CEE)
    return True

# CEE 측정 함수 호출
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

# (4) Test DB 분류 정확도 측정
modeling_weight0 = history_w0[step] # Modeling Weight0
modeling_weight1 = history_w1[step] # Modeling Weight1
modeling_weight2 = history_w2[step] # Modeling Weight2

history_test_p = [] # Modeling Probability 기록함
# Modeling Probability 발생 함수; 입력은 Modeling 가중치들과 Test DB
def Gen_probability(datax0, datax1, w0, w1, w2):
    p = 1 / (1 + np.exp(-(w0 * datax0 + w1 * datax1 + w2))) # 확률 p 초기화
    history_test_p.append(p) # 확률 p의 기록
    return True

# Modeling Probability 발생 함수 호출
Gen_probability(sorted_test_x0, sorted_test_x1, modeling_weight0, \
                modeling_weight1, modeling_weight2)

# Test DB에 의해 결정될 y_hat
decided_y_hat_test = np.zeros((len(sorted_test_y), 1))
# 검증 및 평가를 위해 생성된 y_hat 결정 함수
def Decide_y_hat_test(p, data): # 입력은 Modeling된 확률 값과 데이터
    for n in range(len(data)):
        # 0.5를 기준으로 이상이면 '1' 미만이면 '0'으로 y_hat을 초기화
        if p[0][n] >= 0.5:
            decided_y_hat_test[n] = 1
        else:
            decided_y_hat_test[n] = 0
    return True

# y_hat 결정 함수 호출
Decide_y_hat_test(history_test_p, sorted_test_y)

# 정확도 측정 함수; 입력은 Modeling Test DB에 의해 결정된 y_hat과 Test DB
def Measure_Accuracy(dcd_y_hat, data):
    count_accuracy = 0
    for n in range(len(data)):
        # 실제 데이터(test_y)와 예측 데이터(y_hat)가 같은 지가 기준!
        if dcd_y_hat[n] == data[n]:
            count_accuracy = count_accuracy + 1 # 정확도 카운트 상승
    print("총 정확도 카운트 값은 ", count_accuracy)
    # 정확도 초기화 및 기록
    test_acc = (count_accuracy / 150) * 100
    # 학습횟수마다 count를 해야하므로 0으로 초기화 후 다시 count
    count_accuracy = 0
    return test_acc # 평가 분류 정확도 반환

# 정확도 측정 함수 호출 및 평가 분류 정확도 초기화
test_accuracy = Measure_Accuracy(decided_y_hat_test, sorted_test_y)
print("Logistic Regression Model의 평가 결과 : ", round(test_accuracy, 3))

# (5)
# Decision Boundary를 위한 정의역과 치역을 초기화
x_x0 = np.linspace(0, 6, 1000)
y_x1 = -(modeling_weight0 / modeling_weight1) * x_x0 \
    -(modeling_weight2 / modeling_weight1)

# Drawing Training Set and Decision Boundary & Test Set and Decision Boundary
plt.subplot(2, 1, 1)
plt.scatter(container_train_x0_0, container_train_x1_0, marker='o', c='r', s=25, label='Training set - 0')
plt.scatter(container_train_x0_1, container_train_x1_1, marker='x', c='r', s=25, label='Training set - 1')
plt.plot(x_x0, y_x1, color='violet', label='Decision Boundary')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base - Training Set & Decision Boundary')
plt.grid(True, alpha=0.5)

plt.subplot(2, 1, 2)
plt.scatter(container_test_x0_0, container_test_x1_0, marker='o', c='g', s=25, label='Test set - 0')
plt.scatter(container_test_x0_1, container_test_x1_1, marker='x', c='g', s=25, label='Test set - 1')
plt.plot(x_x0, y_x1, color='violet', label='Decision Boundary')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base - Test Set & Decision Boundary')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# Drawing Training and Test Set and Decision Boundary
plt.figure()
plt.scatter(container_train_x0_0, container_train_x1_0, marker='o', c='r', s=25, label='Training set - 0')
plt.scatter(container_train_x0_1, container_train_x1_1, marker='x', c='r', s=25, label='Training set - 1')
plt.scatter(container_test_x0_0, container_test_x1_0, marker='o', c='g', s=25, label='Test set - 0')
plt.scatter(container_test_x0_1, container_test_x1_1, marker='x', c='g', s=25, label='Test set - 1')
plt.plot(x_x0, y_x1, color='violet', label='Decision Boundary')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.xlabel('x0; Input0')
plt.ylabel('x1; Input1')
plt.title('Data Base - Total')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing Accuracy
plt.figure()
plt.plot(step2, history_accuracy, 'ko-', markevery = 50)
plt.scatter(step, test_accuracy, c='r')
plt.legend(['Accuracy', 'Test Result'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('Accuracy')
plt.title('Logistic Regression - Training Accuracy')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing CEE
plt.figure()
plt.plot(step2, history_CEE, 'ko-', markevery = 50)
plt.legend(['CEE'], loc='center right')
plt.xlabel('t; step')
plt.ylabel('CEE')
plt.title('Logistic Regression - Training CEE')
plt.grid(True, alpha=0.5)
plt.show()











