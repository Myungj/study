# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=200,
    # color_mode='grayscale',
    class_mode='binary',
    shuffle=True
)       # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(50, 50),
    batch_size=200,
    # color_mode='grayscale',
    class_mode='binary'    
)       # Found 120 images belonging to 2 classes.

# 증폭
augment_size = 5000
randidx =  np.random.randint(xy_train[0][0].shape[0], size = augment_size)   # 랜덤한 정수값을 생성   / x_train.shape[0] or 5000이라고 써도 된다.
x_agumented = xy_train[0][0][randidx].copy()
y_agumented = xy_train[0][1][randidx].copy()

print(randidx.shape)       # (340,)
print(type(randidx))       # <class 'numpy.ndarray'>

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],50,50,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],50,50,3)

# 증폭한 데이터 합침
augment_data = train_datagen.flow(x_agumented, 
                                  y_agumented,
                                  batch_size=augment_size,
                                  shuffle=False,
                                #   save_to_dir="../_temp"
                                  )

x_train = np.concatenate((x_train, augment_data[0][0]))
y_train = np.concatenate((xy_train[0][1], augment_data[0][1]))

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(50,50,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(16, (2,2)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k50_4_brain_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100,
                 batch_size = 256, 
                 validation_split = 0.3, 
                 callbacks = [es,mcp])
end = time.time() - start

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
scores = model.evaluate_generator(xy_test)
print("%s: %.2f" %(model.metrics_names[0], scores[0]))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

predict = model.predict(x_test)
print(predict[:3])

# loss: 19.44
# acc: 92.50%