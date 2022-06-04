# # Setting Module
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Chap.4 실습 - First week
# Reading Data
input_data = pd.read_csv('Data_Base/NN_data.csv', \
                         index_col=0).to_numpy(dtype='float')
    
# 출력을 필터링하고 bias를 붙임
temp_input_data = input_data[:, 0:3]
input_data_added_bias = np.concatenate((temp_input_data , np.ones((900, 1))), axis=1)

# (1) One-Hot Encoding 구현
def One_Hot_Encoding(data):
    Encoding_y = [] # One-Hot Encoding 결과값 기록함
    data_y = data[:, 3] # 원본 데이터의 출력 초기화
    y_class = [] # class 종류 및 개수 데이터
    
    # Check Class
    y_class.append(data_y[0])
    for q in range(len(input_data) - 1):
        if data_y[q] != data_y[q + 1]:
            y_class.append(data_y[q + 1])
    y_class.append(len(y_class)) # 마지막에 class 개수를 첨부
    # 출력에 따라 정렬이 되어있으므로 따로 정렬할 필요는 없음
    # 임의의 데이터라면 정렬이 필요할 것으로 생각됨
    # 성분의 앞에서부터 오름차순으로 class의 성분들이 나타남
    # 마지막 성분은 class의 개수로 초기화
    print(y_class)
    
    # One-Hot Encoding
    for q in range(len(data)):
        if y_class[0] == data_y[q]:
            Encoding_y.append([1, 0, 0])
        elif y_class[1] == data_y[q]:
            Encoding_y.append([0, 1, 0])
        elif y_class[2] == data_y[q]:
            Encoding_y.append([0, 0, 1])
            
    Encoding_y = np.array(Encoding_y) # numpy array로 초기화
    return Encoding_y, y_class
    
# One-Hot Encoding 함수 호출
y_One_Hot_Encoding, y_class = One_Hot_Encoding(input_data)
    
# (2) Two-Layer Neural Network 구현
# Setting Variable
num_hidden_layer = 2 # Hidden Layer의 속성 수
num_output_layer = y_class[-1] # Output layer의 Node 수

def Two_Layer_Neural_Network(data, num_l, num_q):
    # Hidden Layer
    # bias가 붙기 전 입력의 열의 개수가 속성 수
    M = temp_input_data.shape[1]
    print("Input 속성 수 : ", M)
    
    # 가중치 v는 입력과 Hidden Layer의 node 수에 따라 size가 결정
    list_v = np.zeros((data.shape[1], num_l))
    for n in range(data.shape[1]):
        for l in range(num_l):
            # 가우시안 함수에 따라 랜덤하게 가중치 값 초기화
            list_v[n][l] = np.random.randn() * 10
    alpha = data.dot(list_v) # Hidden Layer의 입력 초기화
    b = 1 / (1 + np.exp(-alpha)) # Hidden Layer의 출력 초기화
    b = np.concatenate((b , np.ones((len(data), 1))), axis=1) # bias 첨부
          
    # Output Layer
    # Output Layer의 속성 수는 출력 class의 수
    Q = num_q
    print("Output 속성 수 : ", Q)
    
    # 가중치 w는 Hidden Layer와 출력의 node 수에 따라 size가 결정
    list_w = np.zeros((num_l + 1, Q))
    for l in range(num_l + 1):
        for q in range(Q):
            list_w[l][q] = np.random.randn() * 10
    
    beta = b.dot(list_w) # Output Layer의 입력 초기화
    y_hat = 1 / (1 + np.exp(-beta)) # Output Layer의 출력 초기화
    
    # 가중치 v와 w 그리고 Hidden, Output Layer의 출력을 반환
    return list_v, b, list_w, y_hat

# (3) Accuracy 함수 구현
# y_hat을 확률 값에 따라 1과 0으로 구분하는 함수 정의; 입력값은 확률값 p
def Decide_y_hat(p, data):
    for n in range(data.shape[0]):
        for m in range(data.shape[1]):
            if p[n][m] >= 0.5:
                decided_y_hat[n][m] = 1
            else:
                decided_y_hat[n][m] = 0
    return True


# 정확도 값 정의
accuracy = 0.
# 정확도 측정 함수 정의; 입력은 결정된 y_hat 값과 훈련 DB의 y(0과 1로 구성)값
def Measure_Accuracy(dcd_y_hat, data):
    # 정확도 기록함 행렬 정의; 각각의 성분에 대하여 True와 False로 표현
    matrix_accuracy = np.zeros((len(data), 1))
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

# (4) 데이터 분할 함수
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
training_set, validation_set, test_set = Dist_Set(input_data_added_bias)

# (5)
def shuffle_data(data):
    np.random.shuffle(data)
    OHE_y = np.zeros((len(data), 3))
    for m in range(data.shape[0]):
        for n in range(len(input_data_added_bias)):
            if data[m].tolist() == input_data_added_bias[n].tolist():
                OHE_y[m] = y_One_Hot_Encoding[n]
    return OHE_y

# (6) 신경망 학습 함수 및 가중치 갱신
# 신경망 학습 함수
learning_rate = 0.001
def learning_ANN(data, v, w, in_y_hat, y_real, in_b, num_l, num_q):
    u = learning_rate
    
    renewal_v = np.zeros((data.shape[1], num_l))
    dmse_vml = np.zeros((data.shape[1], num_l))
    renewal_w = np.zeros((num_l + 1, num_q))
    dmse_wlq = np.zeros((num_l + 1, num_q))
    
    list_mem2 = []
    list_mem1 = []
    for n in range(data.shape[0]):
        for m in range(data.shape[1]):
            for l in range(num_l):
                for q in range(num_q):
                    dmse_vml[m][l] = 2 * np.sum((in_y_hat[n][q] - y_real[n][q]) \
                            * in_y_hat[n][q] * (1 - in_y_hat[n][q]) * w[l][q]) \
                            * in_b[n][l] * (1 - in_b[n][l]) * data[n][m]
        list_mem1.append(dmse_vml)
        renewal_v = v - u * dmse_vml
    
        for l in range(num_l + 1):
            for q in range(num_q):
                dmse_wlq[l][q] = 2 * (in_y_hat[n][q] - y_real[n][q]) \
                                  * in_y_hat[n][q] * (1 - in_y_hat[n][q]) * in_b[n][l]
        list_mem2.append(dmse_wlq)
        renewal_w = w - u * dmse_wlq
        
        # for l in range(num_l):
        #     for m in range(data.shape[1]):
        #         dmse_vml[m][l] = 2 * np.sum(((in_y_hat[n] - y_real[n]) \
        #                 * in_y_hat[n] * (1 - in_y_hat[n])).dot(np.transpose(w))) \
        #                 * in_b[n][l] * (1 - in_b[n][l]) * data[n][m]
        # # dmse_vml = np.reshape(dmse_vml, (data.shape[1], 1))
        # list_mem1.append(dmse_vml)
        # renewal_v = v - u * dmse_vml
    
    # # 가중치 v 갱신; for문 중복
    # # m하고 l하고 q
    # # for n in range(~~)
    # dmse_vml = np.zeros(data.shape[1])
    # for n in range(data.shape[0]):
    #     for l in range(num_l):
    #         list_mem1 = []
    #         for m in range(data.shape[1]):
    #             dmse_vml = 2 * np.sum((y_hat - y_real) * y_hat * (1 - y_hat) * w[l])  \
    #                             * b[n][l] * (1 - b[n][l]) * data[m]
    #             list_mem1.append(dmse_vml)
    #             dmse_vml = np.reshape(dmse_vml, (data.shape[1], 1))
    #             renewal_v = v - u * dmse_vml
    #             # renewal_v[:, l] = v[:, l] - u * dmse_vml
    # # 가중치 w 갱신
    # dmse_wlq = np.zeros(num_q)
    # for l in range(num_l + 1):
    #     list_mem2 = []
    #     for q in range(num_q):
    #         dmse_wlq = 2 * (y_hat[q] - y_real[q]) * y_hat[q] * (1 - y_hat[q]) * b[l]
    #         list_mem2.append(dmse_wlq)
    #         renewal_w[:, q] = w[:, q] - u * dmse_wlq
    
    # 갱신된 v, w를 통한 y_hat 계산
    alpha = data.dot(renewal_v) # Hidden Layer의 입력 초기화
    renewal_b = 1 / (1 + np.exp(-alpha)) # Hidden Layer의 출력 초기화
    renewal_b = np.concatenate((renewal_b , np.ones((len(data), 1))), axis=1) # bias 첨부
    
    beta = renewal_b.dot(renewal_w) # Output Layer의 입력 초기화
    renewal_y_hat = 1 / (1 + np.exp(-beta)) # Output Layer의 출력 초기화
    
    # 가중치 v와 w 그리고 b, y_hat, y_real의 출력을 반환
    return renewal_v, renewal_w, renewal_b, renewal_y_hat, list_mem1, list_mem2

# 가중치 갱신
train_data = training_set
# Shuffle data
epoch = 10
initial_v, b_output_hidden_layer, initial_w, y_hat = \
    Two_Layer_Neural_Network(train_data, num_hidden_layer, num_output_layer)
weight_v = initial_v
weight_w = initial_w
b = b_output_hidden_layer
sorted_OHE_y = shuffle_data(train_data)
for n in range(epoch):
    print(n)
    # y_hat 값 기록함 정의
    decided_y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1]))
    # y_hat 결정 함수 호출
    Decide_y_hat(y_hat, y_hat)
        
    # 정확도 측정 함수 호출
    accuracy = Measure_Accuracy(decided_y_hat, sorted_OHE_y)
    print("정확도 : ", accuracy)
           
    weight_v, weight_w, b, y_hat, mem1, mem2 = learning_ANN(train_data, weight_v, weight_w, y_hat, sorted_OHE_y, \
                    b, num_hidden_layer, num_output_layer)
    # weight_v, weight_w, learned_y_hat = learning_ANN(train_data, \
    #             initial_v, initial_w, y_hat, y_One_Hot_Encoding, \
    #                 num_hidden_layer, num_output_layer)

































