# # Setting Module
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Chap.4 실습 - First week
# Reading Data
input_data = pd.read_csv('Data_Base/NN_data.csv', \
                         index_col=0).to_numpy(dtype='float')
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    