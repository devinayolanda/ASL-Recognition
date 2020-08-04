import cv2, os
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, TimeDistributed, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from random import shuffle

train_path = os.getcwd() + '/Trainframe'
train_dir_list = os.listdir(train_path)
test_path = os.getcwd() + '/Testframe'
test_dir_list = os.listdir(test_path)

num_classes = 26
num_epoch = 50
time_steps = 10
heightwidth = 128
num_batch = time_steps

data_list = []
img_data_list = []
img_label = []

# TRAIN
# Shuffle the dataset
for dataset in train_dir_list:
    img_list = os.listdir(train_path + '/' + dataset)
    for img in img_list:
        path = train_path + '/' + dataset + "/" + img #D:\IT\Skripsi\PyCharm/Train/Z/Z6-1.jpg
        path = path.split("-")[0] #D:\IT\Skripsi\PyCharm/Train/Z/Z6
        data_list.append(path)
data_list = list(dict.fromkeys(data_list)) #biar ga kembar
shuffle(data_list)

# Label
for i in data_list:
    for j in range(10):
        path = i + "-" + str(j) + ".jpg" #D:\IT\Skripsi\PyCharm/Train/Z/Z6-1.jpg
        input_img = cv2.imread(path)
        input_img = cv2.resize(input_img, (heightwidth, heightwidth))
        img_data_list.append(input_img)
        split = i.split("/")[-2] #Z
        img_label.append(split)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

for i in range(num_of_samples):
    labels[i] = ord(img_label[i]) - 65

# convert class labels to on-hot encoding
y = np_utils.to_categorical(labels, num_classes)

# Make X_train and y_train
X_train = img_data
y_train = y

data_list.clear()
img_data_list.clear()
img_label.clear()

# Test
# Shuffle the dataset
for img in test_dir_list:
    path = test_path + '/' + img #D:\IT\Skripsi\PyCharm/Test/Z6-1.jpg
    path = path.split("-")[0] #D:\IT\Skripsi\PyCharm/Test/Z6
    data_list.append(path)
data_list = list(dict.fromkeys(data_list))
shuffle(data_list)

# Label
for i in data_list:
    for j in range(10):
        path = i + "-" + str(j) + ".jpg" #D:\IT\Skripsi\PyCharm/Test/Z6-1.jpg
        input_img = cv2.imread(path)
        input_img = cv2.resize(input_img, (heightwidth, heightwidth))
        img_data_list.append(input_img)
        split = i.split("/")[-1] #Z
        split = split[0]
        img_label.append(split)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

for i in range(num_of_samples):
    labels[i] = ord(img_label[i]) - 65

# convert class labels to on-hot encoding
y = np_utils.to_categorical(labels, num_classes)

# Make X_test and y_test
X_test = img_data
y_test = y

# reshape
X_train = X_train.reshape(int(X_train.shape[0] / time_steps), time_steps, X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_test = X_test.reshape(int(X_test.shape[0] / time_steps), time_steps, X_test.shape[1], X_test.shape[2], X_test.shape[3])
y_train = y_train.reshape(int(y_train.shape[0] / time_steps), time_steps, y_train.shape[1])
y_test = y_test.reshape(int(y_test.shape[0] / time_steps), time_steps, y_test.shape[1])

y_train = np.squeeze(y_train[:,-1,:])
y_test = np.squeeze(y_test[:,-1,:])

model = Sequential()
model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1)), input_shape=X_test.shape[1:]))
model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(Conv2D(16, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.25))
model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation("relu")))
model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
model.add(Dropout(0.25))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(1024)))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=False, input_shape=(X_test.shape[2], X_test.shape[3])))
model.add(Dense(128))
model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.Adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
fname = "weights.hdf5"

model.summary()

checkpoint = ModelCheckpoint(fname, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
hist = model.fit(X_train, y_train, batch_size=num_batch, nb_epoch=num_epoch, verbose=2, validation_data=(X_test, y_test), callbacks=[checkpoint], shuffle=False)

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(num_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])
plt.show()