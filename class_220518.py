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
num_hidden_layer = 3 # Hidden layer의 Node 수
# history_a = [] # Hidden layer의 y_hat
history_b =[] # Hidden layer의 출력

def Hidden_Layer(num_l):
    a = np.zeros((num_l, 1))
    b = np.ones((num_l + 1, 1))
    list_v = []
    for n in range(len(input_data) + 1):
        v = np.random.rand() * 2 + 3
        list_v.append(v)
    for l in range(num_l):
        for m in range(len(input_data)):
            a[l] = np.sum(list_v[m] * temp_input_data[m]) + list_v[900] * 1
            b[l] = 1 / (1 + np.exp(-a[l]))
    # for l in range(num_l):
    #     for m in range(len(input_data)):
    #         a[l] = np.sum(list_v[m] * input_data[m, :])
    #         b[l] = 1 / (1 + np.exp(-a[l]))
    return a, b, list_v
    
a_y_hat, b_output, v_list = Hidden_Layer(num_hidden_layer)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    