import sys
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

import numpy as np
import matplotlib.pyplot as plt

 

print('Python version :', sys.version)
print('Tensorflow version :', tf.__version__)
print('Keras version :', keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# temp_train = x_train[1, :, :]
# plt.imshow(temp_train)

# 채널 개수를 size에 반영
size_of_channel = 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], size_of_channel)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], size_of_channel)
 
# 정규화 + 데이터 타입을 실수로 형 변환
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_classes = 10 # 출력 클래스 수; 0~9
batch_size = 1000 # 몇개씩 보느냐!
epochs = 10 # 학습 횟수

# One-Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
 

# CNN model
model = Sequential() # 비어있는 모델을 생성; 모델을 구축해 나가기 위함!
# 커널(필터) 32층, 5by5, 1칸씩, zero padding, 활성화함수 ReLU, 입력 사이즈
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), \
                 padding='same', activation='relu', input_shape=(28, 28, 1)))
# 보통 pool size와  stride  사이즈는 동일
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Conv랑 MaxPooling은 세트로 움직이기!
model.add(Conv2D(3, kernel_size=(3, 3), strides=(1, 1), \
                 padding='same', activation='relu')) # input_shape은 빠짐!
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

hist = model.fit(x_train, y_train, \
                 batch_size=batch_size, epochs=epochs,
                 verbose=1, validation_data=(x_test, y_test))

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test)
Y_pred = np.argmax(y_pred, 1)
Y_test = np.argmax(y_test, 1)

mat = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
sns.heatmap(mat, annot=True, fnt='d')

# n번째의 모델 평가!
n = 0
predicted_class = np.argmax(model.predict(x_test[n]).reshape(1, 28, 28, 1))
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys')
plt.title('Predict :' + str(predicted_class))