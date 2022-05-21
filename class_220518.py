# # Setting Module
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Chap.4 실습 - First week
# Reading Data
input_data = pd.read_csv('Data_Base/NN_data.csv', \
                         index_col=0).to_numpy(dtype='float')
    
temp_input_data = input_data[:, 0:3]

# (1) One-Hot Encoding 구현
def One_Hot_Encoding(data):
    Encoding_y = []
    data_y = data[:, 3]
    y_class = [] # class 종류 및 개수 데이터
    
    # Check Class
    y_class.append(data_y[0])
    for q in range(len(input_data) - 1):
        if data_y[q] != data_y[q + 1]:
            y_class.append(data_y[q + 1])
    y_class.append(len(y_class))
    print(y_class)
    
    # One-Hot Encoding
    for q in range(len(data)):
        if y_class[0] == data_y[q]:
            Encoding_y.append([1, 0, 0])
        elif y_class[1] == data_y[q]:
            Encoding_y.append([0, 1, 0])
        elif y_class[2] == data_y[q]:
            Encoding_y.append([0, 0, 1])
            
    Encoding_y = np.array(Encoding_y)
    
    return Encoding_y
    
y_One_Hot_Encoding = One_Hot_Encoding(input_data)
    
coverted_data = np.concatenate([np.delete(input_data, 3, axis=1) , \
                                y_One_Hot_Encoding], 1)
    
# (2) Two-Layer Neural Network 구현
# Setting Variable
num_hidden_layer = 2 # Hidden layer의 Node 수
num_output_layer = 3 # Output layer의 Node 수
# history_a = [] # Hidden layer의 y_hat
history_b =[] # Hidden layer의 출력

def Two_Layer_Neural_Network(num_l, num_q):
    # Hidden Layer
    a = np.zeros((num_l, 1))
    b = np.ones((num_l + 1, 1))
    M = len(temp_input_data)
    print("Input 속성 수 : ", M + 1)
    for l in range(num_l):
        list_v = []
        for n in range(len(input_data) + 1):
            v = np.random.rand() * 2 + 3
            list_v.append(v)
        for m in range(len(input_data)):
            a[l] = np.sum(np.array(list_v[m]) * temp_input_data[m]) + np.array(list_v[900]) * 1
        b[l] = 1 / (1 + np.exp(-a[l]))
          
    # Output Layer
    beta = np.zeros((num_q, 1))
    y_hat = np.zeros((num_q, 1))
    for q in range(num_q):
        list_w = []
        for n in range(len(input_data) + 1):
            w = np.random.rand() * 2 + 4
            list_w.append(w)
        for l in range(len(input_data)):
            beta[q] = np.sum(np.array(list_w[m]) * temp_input_data[m])
        y_hat[q] = 1 / (1 + np.exp(-beta[q]))
    
    Q = len(y_hat)
    print("Output 속성 수 : ", Q)
    # return a, b, list_v, beta, y_hat, list_w
    return y_hat
    
y_hat = Two_Layer_Neural_Network(num_hidden_layer, num_output_layer)
# y_hat.resize(1, 3)

# (3) Accuracy 함수 구현
# y_hat 값 기록함 정의
decided_y_hat = np.zeros((len(y_hat), 1))
# y_hat을 확률 값에 따라 1과 0으로 구분하는 함수 정의; 입력값은 확률값 p
def Decide_y_hat(p, data):
    for m in range(len(data)):
        if p[m] >= 0.5:
            decided_y_hat[m] = 1
        else:
            decided_y_hat[m] = 0
    return True

# y_hat 결정 함수 호출
Decide_y_hat(y_hat, y_hat)
decided_y_hat = np.reshape(decided_y_hat, (1, 3))

# 정확도 기록함 행렬 정의; 각각의 성분에 대하여 True와 False로 표현
matrix_accuracy = np.zeros((len(y_One_Hot_Encoding), 1))
# 정확도 값 정의
accuracy = 0.
# 정확도 측정 함수 정의; 입력은 결정된 y_hat 값과 훈련 DB의 y(0과 1로 구성)값
def Measure_Accuracy(dcd_y_hat, data):
    for m in range(len(data)):
        # 결정된 y_hat과 훈련 DB의 y값이 같으면 True 아니면 False로 초기화
        if dcd_y_hat.tolist() == data[m].tolist():
            matrix_accuracy[m] = True
        else:
            matrix_accuracy[m] = False
    # 정확도의 정도에 대한 카운트 정의
    count_accuracy = 0
    for m in range(len(data)):
        # True는 곧 1이므로 1이 많으면 정확도가 높은 것!
        if matrix_accuracy[m] == 1:
            count_accuracy = count_accuracy + 1
    # 정확도 초기화
    acc = (count_accuracy / 900) * 100
    # 학습횟수마다 count를 해야하므로 0으로 초기화 후 다시 count
    count_accuracy = 0
    
    return acc

# 정확도 측정 함수 호출
accuracy = Measure_Accuracy(decided_y_hat[0], y_One_Hot_Encoding)

print("정확도 : ", accuracy)

    
    
    
    
    
    
    
    
    
    