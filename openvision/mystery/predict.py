from scipy import misc
import numpy as np
import sys
import os

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


data_dir = "./data/BmpMoban"

images = os.listdir(data_dir)

data_size =  len(images) #len(images) #(len(images),
dataset = np.zeros((data_size,68,68,1))
labels = np.zeros(data_size)

for i in range(data_size):
    name = images[i]
    file_name = os.path.join(data_dir, name)
    img = misc.imread(file_name)
    label = int(name.split("_")[0])

    dataset[i,:,:,:] = np.reshape(img,(68, 68,1))
    labels[i] = label 

import sklearn.cross_validation
import sklearn.datasets

import sklearn.metrics

x_train, x_test,y_train, y_test= sklearn.cross_validation.train_test_split(dataset, labels, random_state=1)

print x_train.shape, y_train.shape, y_test.shape


batch_size = 32
num_classes = 1000
epochs = 8


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          nb_epoch=epochs,
#          validation_data=(x_test, y_test),
#          shuffle=True)


model.load_weights('my_model_weights.h5')

total_same = 0

for i in range(dataset.shape[0]):
    result = model.predict( dataset[i:i+1,:, :, :])
    index = np.argmax(result)
    if int(labels[i]) == int(index):
        total_same += 1

    if i%100 == 0:
        print float(total_same)/(i+1)

print "final", total_same/data_size

#print x_train.shape, x_test.shape,y_train.shape, y_test.shape
#automl = autosklearn.classification.AutoSklearnClassifier()
#automl.fit(X_train, y_train)
#y_hat = automl.predict(X_test)
#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


