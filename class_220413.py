# # Setting Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실습과제 #3
# (1)
# Reading Data
input_data = pd.read_csv('Data_Base/lin_regression_data_01.csv', \
                         header=None).to_numpy(dtype='float')

# Setting Variable1
in_x = input_data[:, 0] # 입력; 추의 무게 / x; weight
out_y = input_data[:, 1] # 출력; 늘어난 용수철의 길이 / y; Length

# plot에 그려진 Linear Combination을 통해 실제값의 y가 단조 증가가 아님을 확인
sorted_input_data = np.zeros((25, 2)) # 정렬에 사용하기 위한 input_data 행렬 복사본 정의
# 0열을 기준으로 작은 값부터 순서대로 행렬 정렬
for n in range(25):
    sorted_input_data[n, :] = input_data[n, :]
sorted_input_data = sorted_input_data[sorted_input_data[:, 0].argsort()] 

sorted_in_x = sorted_input_data[:, 0] # x를 오름차순으로 정렬
y_real = sorted_input_data[:, 1] # 실제값 배열 정의

time_noise = 10 # 데이터 수 증폭 값
temp_data = [] # original data와 random data의 합 기록

# 증강 데이터 생성 함수
def Gen_AugmentedSet(data, time): # 원본 데이터와 배수를 입력으로 함
    # 아무래도 numpy보다 pandas에서 데이터의 정제가 간편했음
    df_x = pd.DataFrame(data[:, 0]) # 데이터 정제를 위해 numpy인 x에 관한 데이터셋 생성
    df_y = pd.DataFrame(data[:, 1]) # 데이터 정제를 위해 numpy인 y에 관한 데이터셋 생성
    for n in range(time):
        # 1. origianl data + random data
        temp_data_x = data[:, 0] + round(np.random.rand(), 1) * 3 - 1.5
        # 2. 데이터 정제를 위해 '1.'에서 생성한 데이터를 데이터셋으로 변환
        df_rand_x = pd.DataFrame(temp_data_x)
        # 3. 열을 기준으로 데이터셋을 추가
        df_x = pd.concat([df_x, df_rand_x])      
        
        # x에서와 작업 동일
        temp_data_y = data[:, 1] + round(np.random.rand(), 1) * 3 - 1.5
        df_rand_y = pd.DataFrame(temp_data_y)
        df_y = pd.concat([df_y, df_rand_y])   

    # 정제된 데이터를 numpy 데이터로 변환
    gen_set = pd.concat([df_x, df_y], axis=1).to_numpy()
    
    return gen_set # 증강된 DB를 반환

# DB 증강 함수를 호출 및 반환된 값을 Original data로 초기화
agumented_original_set = Gen_AugmentedSet(sorted_input_data, time_noise)

# 생성된 Original data의 그래프를 생성하기 위해 필요로 하는 정의역과 치역을 초기화
data_aug_x = agumented_original_set[:, 0] # 정의역
data_aug_y = agumented_original_set[:, 1] # 치역

# Drawing Origianl DB
plt.figure()
plt.scatter(data_aug_x, data_aug_y, color='0')
plt.scatter(in_x, out_y, color="red")
plt.legend(['Augmented Combination', 'Linear Combination'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Data Base')
plt.grid(True, alpha=0.5)
plt.show()
                 

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
training_set, validation_set, test_set = Dist_Set(agumented_original_set)

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

# Drawing Training ans Validation and Test Set
plt.figure()
plt.scatter(data_train_x, data_train_y, color='red')
plt.scatter(data_val_x, data_val_y, color='blue')
plt.scatter(data_test_x, data_test_y, color='green')
plt.legend(['Training set', 'Validation set', 'Test set'])
plt.xlabel('x; Weight')
plt.ylabel('y; Length')
plt.title('Distrubute Original Set')
plt.grid(True, alpha=0.5)
plt.show()

# (3)


















