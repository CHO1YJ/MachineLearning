# # Setting Module
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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
            list_v[n][l] = np.random.randn()
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
            list_w[l][q] = np.random.randn()
    
    beta = b.dot(list_w) # Output Layer의 입력 초기화
    y_hat = 1 / (1 + np.exp(-beta)) # Output Layer의 출력 초기화
    
    # 가중치 v와 w 그리고 Hidden, Output Layer의 출력을 반환
    return list_v, list_w, y_hat

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
    for m in range(data.shape[0]): # 630
        for n in range(len(input_data_added_bias)): # 900
            if data[m].tolist() == input_data_added_bias[n].tolist():
                OHE_y[m] = y_One_Hot_Encoding[n]
    return OHE_y

# (6) 신경망 학습 함수 및 가중치 갱신
# 신경망 학습 함수
learning_rate = 0.02 # 학습률
# ANN Model 구현
def learning_ANN(data, v, w, y_real, num_l, num_q):
    u = learning_rate # 학습률
    list_mem2 = [] # 가중치 v의 dms 기록함 - 코드가 정상적인지 확인하기 위함
    list_mem1 = [] # 가중치 w의 dms 기록함 - 코드가 정상적인지 확인하기 위함
    MSE = 0 # 모델의 MSE 값 정의
    
    # 가중치 v, w 초기값 및 갱신값 적용
    renewal_v = v
    renewal_w = w
    
    # Error Back Propagation Algorithm of ANN 구현
    for n in range(data.shape[0]):
        # 가중치 v, w 초기화
        dmse_vml = np.zeros((train_data.shape[1], num_hidden_layer))
        dmse_wlq = np.zeros((num_hidden_layer + 1, num_output_layer))
        
        # 갱신된 v, w를 통한 y_hat 계산
        renewal_alpha = data.dot(renewal_v) # Hidden Layer의 입력 초기화
        renewal_b = 1 / (1 + np.exp(-renewal_alpha)) # Hidden Layer의 출력 초기화
        renewal_b = np.concatenate((renewal_b , np.ones((len(data), 1))), axis=1) # bias 첨부
        
        renewal_beta = renewal_b.dot(renewal_w) # Output Layer의 입력 초기화
        renewal_y_hat = 1 / (1 + np.exp(-renewal_beta)) # Output Layer의 출력 초기화
        
        # 가중치 v 갱신
        for m in range(data.shape[1]):
            for l in range(num_l):
                dmse_vml[m][l] = 0
                for q in range(num_q):
                    dmse_vml[m][l] = dmse_vml[m][l] + 2 * (renewal_y_hat[n][q] - y_real[n][q]) \
                        * renewal_y_hat[n][q] * (1 - renewal_y_hat[n][q]) * renewal_w[l][q]
                dmse_vml[m][l] = dmse_vml[m][l] * renewal_b[n][l] * (1 - renewal_b[n][l]) * data[n][m]
        list_mem1.append(dmse_vml)
        renewal_v = renewal_v - u * dmse_vml
    
        # 가중치 w 갱신
        for l in range(num_l + 1):
            for q in range(num_q):
                dmse_wlq[l][q] = 2 * (renewal_y_hat[n][q] - y_real[n][q]) \
                    * renewal_y_hat[n][q] * (1 - renewal_y_hat[n][q]) * renewal_b[n][l]
        list_mem2.append(dmse_wlq)
        renewal_w = renewal_w - u * dmse_wlq
    
    # 모델의 MSE 계산
    for q in range(num_q):
        MSE = MSE + np.sum((renewal_y_hat[:, q] - y_real[:, q]) ** 2) / len(renewal_y_hat)
    mem_MSE.append(MSE)
    
    # y_hat 결정 함수 호출
    Decide_y_hat(renewal_y_hat, renewal_y_hat)
        
    # 정확도 측정 함수 호출 $ 모델의 정확도 측정
    training_accuracy = Measure_Accuracy(decided_y_hat, sorted_OHE_y)
    mem_accuracy.append(training_accuracy)
    print("모델 정확도 :", training_accuracy)
    
    # 가중치 v와 w 그리고 b, y_hat, y_real의 출력을 반환
    return renewal_v, renewal_w, list_mem1, list_mem2

# 가중치 갱신
train_data = training_set
# Shuffle data
epoch = 50 # 학습 횟수
# 초기값 설정
initial_v, initial_w, y_hat = \
                Two_Layer_Neural_Network(train_data, num_hidden_layer, num_output_layer)
# 초기값으로 초기화
weight_v = initial_v
weight_w = initial_w

# y_hat 값 기록함 정의
decided_y_hat = np.zeros((y_hat.shape[0], y_hat.shape[1]))

# 모델 정확도 및 MSE 기록함
mem_accuracy = []
mem_MSE = []

for n in range(epoch):
    print("훈련 횟수 :", n) 
    
    sorted_OHE_y = shuffle_data(train_data)    
    
    weight_v, weight_w, mem1, mem2 = \
        learning_ANN(train_data, weight_v, weight_w, sorted_OHE_y, num_hidden_layer, num_output_layer)

# 모델 평가 결과 확인
test_set = test_set
# 평가 결과에 따른 y_hat 발생 함수
def test_model(data, test_v, test_w):
    # 갱신된 v, w를 통한 y_hat 계산
    test_alpha = data.dot(test_v) # Hidden Layer의 입력 초기화
    test_b = 1 / (1 + np.exp(-test_alpha)) # Hidden Layer의 출력 초기화
    test_b = np.concatenate((test_b , np.ones((len(data), 1))), axis=1) # bias 첨부
    
    test_beta = test_b.dot(test_w) # Output Layer의 입력 초기화
    test_y_hat = 1 / (1 + np.exp(-test_beta)) # Output Layer의 출력 초기화
    return test_y_hat

# 평가 결과 y_hat 반환
result_y_hat = test_model(test_set, weight_v, weight_w)

# 평가 DB에 대응되는 index의 One-Hot Encoding y 발생 함수
def sorted_OHE_y(data):
    OHE_y = np.zeros((len(data), 3))
    for m in range(data.shape[0]):
        for n in range(len(input_data_added_bias)):
            if data[m].tolist() == input_data_added_bias[n].tolist():
                OHE_y[m] = y_One_Hot_Encoding[n]
    return OHE_y

# 결과 y_hat 값 기록함 정의
decided_y_hat = np.zeros((result_y_hat.shape[0], result_y_hat.shape[1]))
# y_hat 결정 함수 호출
Decide_y_hat(result_y_hat, result_y_hat)
# test set에 대응하는 One-Hot-Encoding 구현
test_OHE_y = sorted_OHE_y(test_set)
# 정확도 측정 함수 호출
test_accuracy = Measure_Accuracy(decided_y_hat, test_OHE_y)
print("평가 정확도 :", test_accuracy)

# 평가 MSE
result_MSE = 0
for q in range(num_output_layer):
    result_MSE = result_MSE + np.sum((result_y_hat[:, q] - test_OHE_y[:, q]) ** 2) / len(result_y_hat)

# Setting step
epoch1 = np.arange(0, epoch, 1)

# Drawing Accuracy
plt.figure()
plt.plot(epoch1, mem_accuracy, 'ko-', markevery = 50)
plt.scatter(epoch, test_accuracy, c='r')
plt.legend(['Accuracy', 'Test Result'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('ANN (Aritificial Neural Network)')
plt.grid(True, alpha=0.5)
plt.show()

# Drawing MSE
plt.figure()
plt.plot(epoch1, mem_MSE, 'ko-', markevery = 50)
plt.scatter(epoch, result_MSE, c='r')
plt.legend(['MSE', 'Test MSE'], loc='center right')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('ANN - MSE')
plt.grid(True, alpha=0.5)
plt.show()

# Generate Confusion Matrix
# Confusion Matrix 정의
Confusion_Matrix_by_Result = np.zeros((len(result_y_hat), num_output_layer))
# Confusion Matrix의 참과 거짓에 따른 성분 정의 및 초기화
# 1행 성분들
t_element_100 = 0
f_element_100_010 = 0
f_element_100_001 = 0
f_element_100_else = 0
#2행 성분들
t_element_010 = 0
f_element_010_100 = 0
f_element_010_001 = 0
f_element_010_else = 0
# 3행 성분들
t_element_001 = 0
f_element_001_100 = 0
f_element_001_010 = 0
f_element_001_else = 0
# 4행 성분 - "000"을 걸러내기 위한 행으로 나머지 3개 성분은 필요 없는 성분!
f_element_else = 0
# Algorithm Counting Elements
for n in range(len(decided_y_hat)):
    # Filtering diagonal elements
    if decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][0] == 1: # 100-100
        t_element_100 = t_element_100 + 1
    elif decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][1] == 1: # 010-010
        t_element_010 = t_element_010 + 1
    elif decided_y_hat[n].tolist() == test_OHE_y[n].tolist() and decided_y_hat[n][2] == 1: # 001-001
        t_element_001 = t_element_001 + 1 
    
    # Filtering elements of first row
    elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][0] == 1:
        if test_OHE_y[n].tolist() == [0, 1, 0]: # 100-010
            f_element_100_010 = f_element_100_010 + 1
        elif test_OHE_y[n].tolist() == [0, 0, 1]: # 100-001
            f_element_100_001 = f_element_100_001 + 1
        else: # 100-else
            f_element_100_else = f_element_100_else + 1
    
    # Filtering elements of second row
    elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][1] == 1:
        if test_OHE_y[n].tolist() == [1, 0, 0]: # 010-100
            f_element_010_100 = f_element_010_100 + 1
        elif test_OHE_y[n].tolist() == [0, 0, 1]: # 010-001
            f_element_010_001 = f_element_010_001 + 1
        else: # 010-else
            f_element_010_else = f_element_010_else + 1
    # Filtering elements of third row
    elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n][2] == 1:
        if test_OHE_y[n].tolist() == [1, 0, 0]: # 001-100
            f_element_001_100 = f_element_001_100 + 1
        elif test_OHE_y[n].tolist() == [0, 1, 0]: # 001-010
            f_element_001_010 = f_element_001_010 + 1
        else: # 001-else
            f_element_001_else = f_element_001_else + 1
    # Filtering elements of fourth row
    elif decided_y_hat[n].tolist() != test_OHE_y[n].tolist() and decided_y_hat[n].tolist() == [0, 0, 0]:
        f_element_else = f_element_else + 1

# 성분들을 확률로 나타내기 위한 행 별로 전체 개수 계산
total_100 = t_element_100 + f_element_100_010 + f_element_100_001 + f_element_100_else
total_010 = f_element_010_100 + t_element_010 + f_element_010_001 + f_element_010_else
total_001 = f_element_001_100 + f_element_001_010 + t_element_001 + f_element_001_else
list_total = np.array([total_100, total_010, total_001, f_element_else])

# Confusion_Matrix_Result 초기화
Confusion_Matrix_by_Result = \
    np.array([[t_element_100, f_element_100_010, f_element_100_001, f_element_100_else], \
              [f_element_010_100, t_element_010, f_element_010_001, f_element_010_else], \
                  [f_element_001_100, f_element_001_010, t_element_001, f_element_001_else], \
                      [0, 0, 0, f_element_else]], dtype='float')

# Confusion Matrix_Result 성분들의 확률화
for n in range(Confusion_Matrix_by_Result.shape[0]):
    Confusion_Matrix_by_Result[n] = Confusion_Matrix_by_Result[n] / list_total[n]

# 데이터 프레임으로 변환
df_cm = pd.DataFrame(Confusion_Matrix_by_Result, index = \
                     [i for i in ["100", "010", "001", "Else"]], \
                     columns=[i for i in ["100", "010", "001", "Else"]])
# Confusion Matrix 시각화
plt.figure(figsize=(8, 8))
plt.title("Confusion Matrix", fontsize=35)
sns.heatmap(df_cm, annot=True, linewidths=.5, annot_kws={"size": 20})
plt.xlabel("Target - One Hot Encoding of y", fontsize=20)
plt.ylabel("Output Class - y_hat", fontsize=20)
plt.tight_layout()
plt.show()


















