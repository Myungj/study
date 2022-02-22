from pyparsing import col
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# [실습] cifar10을 넣어서 완성할 것

# vgg trainable : True, False
# Flatten / GAP
# 위 4가지 결과와, 지금까지 본인이 cifar10을 돌려서 최고치가 나온거와 비교!!!
# time도 명시해라!!!

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
n = x_train.shape[0]                           
x_train_reshape = x_train.reshape(n,-1)                
x_train_transe = scaler.fit_transform(x_train_reshape) 
x_train = x_train_transe.reshape(x_train.shape)    

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3))

# vgg16.summary()
# vgg16.trainable = False     # 가중치를 동결시킨다!

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.trainable = False

########################### 2번에서 아래만 추가 ###############################

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
import time

lr = 0.0001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics='accuracy')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k67_4_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es, reducelr])#, mcp])

end = time.time() - start

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('accurcy : ', round(acc,4))
print("시간 : ", round(end, 4), '초')



# 출력결과
# 기존
# loss :  0.9427627921104431
# accuracy :  0.6711000204086304

# VGG : False, Flatten
# learning rate :  0.0002
# loss :  1.1418
# accurcy :  0.6034
# 시간 :  588.8179 초

# VGG : False, GAP
# learning rate :  0.0002
# loss :  1.1373
# accurcy :  0.6053
# 시간 :  502.3811 초

# VGG : True, Flatten
# learning rate :  0.0001
# loss :  0.5568
# accurcy :  0.8345
# 시간 :  551.099 초

# VGG : True, GAP
# learning rate :  0.0001
# loss :  0.5484
# accurcy :  0.8427
# 시간 :  582.124 초




