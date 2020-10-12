##################################################################
# Mnist dataset with lenet-5 network
# Shanshan Xie, 1012/2020
# Accuracy: (top-1)


from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras.models import Model
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.utils import normalize
from sec_ops import relu_layer as relu_layer_op
from sec_ops import softmax_layer as softmax_layer_op
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import Tensor
from keras.layers import GlobalMaxPooling2D
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


###### Parameters #######.
batch_size = 100
epochs = 30
channels = 1
img_rows = 32
img_cols = 32

classes = 10
epsilon = 1e-6
momentum = 0.9
weight_decay = 0.0004
use_bias = False


###### Argu ######

weight_name = "weight_1011_lenet.hdf5"

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
ap.add_argument("-l", "--load-model", type=int, default=-1,
        help="(optional) whether or not pre-trained model should be loaded")

args = vars(ap.parse_args())

###### Functions ######
def relu_layer(x):
    return relu_layer_op(x)

def softmax_layer(x):
    return softmax_layer_op(x)




(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# convert class vectors to binary class matrices
train_labels = to_categorical(train_labels, classes)
test_labels = to_categorical(test_labels, classes)

test_data = test_data/255
train_data = train_data/ 255


# X_train_padded = np.zeros(shape=(train_data.shape[0], train_data.shape[1]+4, train_data.shape[2]+4))
# X_train_padded[:train_data.shape[0], 2:train_data.shape[1]+2, 2:train_data.shape[2]+2] = train_data
#
# X_test_padded = np.zeros(shape=(test_data.shape[0], test_data.shape[1]+4, test_data.shape[2]+4))
# X_test_padded[:test_data.shape[0], 2:test_data.shape[1]+2, 2:test_data.shape[2]+2] = test_data

train_data = train_data.reshape(train_data.shape[0], 28,28,1)
test_data = test_data.reshape(test_data.shape[0], 28,28,1)

# X_test_padded = tf.expand_dims(test_data, 3)

print(test_data.shape, 'train data')
print(train_data.shape, 'train_labels')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                patience=3,
                                verbose=1,
                                factor=0.2,
                                min_lr=1e-6)

# Data Augmentation
datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1)
datagen.fit(train_data)
model.fit_generator(datagen.flow(train_data, train_labels, batch_size=100), steps_per_epoch=len(train_data)/100,
                    epochs=30, validation_data=(test_data, test_labels), callbacks=[reduce_lr])

print("[INFO] dumping weights to file...")
model.save_weights(weight_name, overwrite=True)

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])















