# -*- coding: utf-8 -*-
# train/学習処理。結果ファイル保存。
# 
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils

if __name__ == '__main__':
    #cifar10をダウンロード
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    #
#    print(x_train.shape, y_train.shape )
#    print(x_test.shape  , y_test.shape )
    print(x_train.shape, y_train.shape )
    print(x_test.shape  , y_test.shape )
    #quit()
    #画像を0-1の範囲で正規化
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0

    #正解ラベルをOne-Hot表現に変換
    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)

#
#モデルを構築
model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

batch_size=32
epoch_num=10
#fit
history=model.fit(x_train,y_train
    ,batch_size=batch_size
    ,nb_epoch=epoch_num
    ,verbose=1,validation_split=0.1)


#モデルと重みを保存
json_string=model.to_json()
open('cifar10_cnn.json',"w").write(json_string)
model.save_weights('cifar10_cnn.h5')


#モデルの表示
#model.summary()
# モデルの評価
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)
