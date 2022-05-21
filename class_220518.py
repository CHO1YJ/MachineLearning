# # Setting Module
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Chap.4 실습 - First week
# Reading Data
input_data = pd.read_csv('Data_Base/NN_data.csv', \
                         index_col=0).to_numpy(dtype='float')
    
temp_input_data = input_data[:, 0:3]
input_data_added_bias = np.concatenate((temp_input_data , np.ones((900, 1))), axis=1)

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

def Two_Layer_Neural_Network(num_l, num_q):
    # Hidden Layer
    M = temp_input_data.shape[1]
    print("Input 속성 수 : ", M)
    
    list_v = np.zeros((input_data_added_bias.shape[1], num_l))
    for n in range(input_data_added_bias.shape[1]):
        for l in range(num_l):
            list_v[n][l] = np.random.randn()
    alpha = input_data_added_bias.dot(list_v)  
    b = 1 / (1 + np.exp(-alpha))
    b = np.concatenate((b , np.ones((900, 1))), axis=1)
          
    # Output Layer
    Q = num_q
    print("Output 속성 수 : ", Q)
    
    list_w = np.zeros((num_l + 1, Q))
    for l in range(num_l + 1):
        for q in range(Q):
            list_w[l][q] = np.random.randn()
    
    beta = b.dot(list_w)
    y_hat = 1 / (1 + np.exp(-beta))
    
    return list_v, b, list_w, y_hat
    
list_v, b_output_hidden_layer, list_w, y_hat = \
    Two_Layer_Neural_Network(num_hidden_layer, num_output_layer)

# (3) Accuracy 함수 구현
# y_hat 값 기록함 정의
decided_y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1]))
# y_hat을 확률 값에 따라 1과 0으로 구분하는 함수 정의; 입력값은 확률값 p
def Decide_y_hat(p, data):
    for n in range(data.shape[0]):
        for m in range(data.shape[1]):
            if p[n][m] >= 0.5:
                decided_y_hat[n][m] = 1
            else:
                decided_y_hat[n][m] = 0
    return True

# y_hat 결정 함수 호출
Decide_y_hat(y_hat, y_hat)

# 정확도 기록함 행렬 정의; 각각의 성분에 대하여 True와 False로 표현
matrix_accuracy = np.zeros((len(y_One_Hot_Encoding), 1))
# 정확도 값 정의
accuracy = 0.
# 정확도 측정 함수 정의; 입력은 결정된 y_hat 값과 훈련 DB의 y(0과 1로 구성)값
def Measure_Accuracy(dcd_y_hat, data):
    for m in range(len(data)):
        # 결정된 y_hat과 훈련 DB의 y값이 같으면 True 아니면 False로 초기화
        if dcd_y_hat[m].tolist() == data[m].tolist():
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
    acc = (count_accuracy / len(data)) * 100
    # 학습횟수마다 count를 해야하므로 0으로 초기화 후 다시 count
    count_accuracy = 0
    
    return acc

# 정확도 측정 함수 호출
accuracy = Measure_Accuracy(decided_y_hat, y_One_Hot_Encoding)

print("정확도 : ", accuracy)

    
    
    
    
    
    
    
    
    
    