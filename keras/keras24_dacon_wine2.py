from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


#1 데이터
path = "../_data/dacon/wine/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['id', 'quality'], axis =1)
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

print(x)                          # type column의 white, red를 0,1로 변환
print(x.shape)                    # (3231, 12)

# from tensorflow.keras.utils import to_categorical
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))

test_file = test_file.drop(['id'], axis=1)
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']
# print(y.unique())                # [6 7 5 8 4]
y = get_dummies(y)
# print(y)                         #        4  5  6  7  8
                                   #  0     0  0  1  0  0
                                   #  1     0  0  0  1  0
                                   #  2     0  0  1  0  0
                                   #  3     0  1  0  0  0
                                   #  4     0  0  0  1  0

# # y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)




# lr = LogisticRegression()
# lr.fit(x_train, y_train)


#2 모델구성

# LogisticRegression
# lr = LogisticRegression(n_jobs=-1, random_state=66)
# lr.fit(x_train, y_train)
# lr_score = lr.score(x_test, y_test)

#KNeighborsClassifier
# knn = KNeighborsClassifier(n_jobs=-1)
# knn.fit(x_train, y_train)
# knn_score = knn.score(x_test, y_test)

# # #DecisionTreeClassifier
# tree = DecisionTreeClassifier(random_state=66)
# tree.fit(x_train, y_train)
# tree_score = tree.score(x_test, y_test)

# #RandomForestClassifier
# rf = RandomForestClassifier(random_state=66, n_jobs=-1)
# rf.fit(x_train, y_train)
# rf_score = rf.score(x_test, y_test)

# #GradientBoostingClassifier
# gb = GradientBoostingClassifier(random_state=66)
# gb.fit(x_train, y_train)
# gb_score = gb.score(x_test, y_test)

# print("lr : ", lr_score)
# print("knn : ", knn_score)
# print("tree : ", tree_score)
# print("rf : ", rf_score)
# print("gb : ", gb_score)

# #LogisticRegression
# lr = LogisticRegression(n_jobs=-1, random_state=66)
# lr.fit(train, y_train)
# lr_score = lr.score(x_test, y_test)

# #KNeighborsClassifier
# knn = KNeighborsClassifier(n_jobs=-1)
# knn.fit(train, y_train)
# knn_score = knn.score(x_test, y_test)

# #DecisionTreeClassifier
# tree = DecisionTreeClassifier(random_state=66)
# tree.fit(train, y_train)
# tree_score = tree.score(x_test, y_test)

# #RandomForestClassifier
# rf = RandomForestClassifier(random_state=66, n_jobs=-1)
# rf.fit(train, y_train)
# rf_score = rf.score(x_test, y_test)

# #GradientBoostingClassifier
# gb = GradientBoostingClassifier(random_state=66)
# gb.fit(train, y_train)
# gb_score = gb.score(x_test, y_test)

model = Sequential()
model.add(Dense(29, input_dim=x.shape[1], activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(5, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다
from tensorflow.keras.callbacks import EarlyStopping
patience_num = 50
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs = 1000, batch_size =10, validation_split=0.2, callbacks=[es])


#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0])                      # List 형태로 제공된다
print("accuracy : ",loss[1])



############################### 제출용 ########################################
result = model.predict(test_file)
print(result[:5])
result_recover = np.argmax(result, axis=1).reshape(-1,1) + 4
print(result_recover[:5])
print(np.unique(result_recover))                           # value_counts = pandas에서만 먹힌다. 
submission['quality'] = result_recover

# print(submission[:10])
submission.to_csv(path + "bbbbbb.csv", index = False)

print(result_recover)

'''
MinMax
loss :  1.0104622840881348
accuracy :  0.5703245997428894

robust
loss :  0.9793218970298767
accuracy :  0.6027820706367493

MaxAbs
loss :  0.9894062280654907
accuracy :  0.5672333836555481

Standard
loss :  1.0083285570144653
accuracy :  0.5656877756118774

'''
