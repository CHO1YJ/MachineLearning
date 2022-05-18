# # Setting Module
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Chap.4 실습 - First week
# Reading Data
input_data = pd.read_csv('Data_Base/NN_data.csv', \
                         index_col=0).to_numpy(dtype='float')
    
def One_Hot_Encoding(data):
    Encoding_y = np.zeros((len(data), 1))
    data_y = data[:, 3]
    y_class = [] # class 종류 및 개수 데이터
    count_class = 1
    
    # Check Class
    y_class.append(data_y[0])
    for q in range(len(input_data) - 1):
        if data_y[q] != data_y[q + 1]:
            y_class.append(data_y[q + 1])
    y_class.append(len(y_class))
    print(y_class)
    
    # One-Hot Encoding
    for n in range(y_class[-1] - 1):
        for q in range(len(data)):
            if y_class[n] == data_y[q]:
                # if count_class 어떻게 처리할 지!
                Encoding_y[q] = "100"
        count_class = count_class + 1
    
    return Encoding_y
    
y = One_Hot_Encoding(input_data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    