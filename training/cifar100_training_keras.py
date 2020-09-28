from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import np_utils
from keras.models import Model
from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.utils import normalize
from sec_ops import relu_layer as relu_layer_op
from sec_ops import softmax_layer as softmax_layer_op
import argparse

###### Parameters #######.
batch_size = 32
epochs = 100
channels = 3
img_rows = 32
img_cols = 32

classes = 10
epsilon = 1e-6
momentum = 0.9
weight_decay = 0.0004
use_bias = False

###### Argu ######

weight_name = "weight_0927_cifar10.hdf5"

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


(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# convert class vectors to binary class matrices
train_labels = to_categorical(train_labels, classes)
test_labels = to_categorical(test_labels, classes)

print(train_data.shape, 'train data')
print(train_labels.shape, 'train_labels')

model = Sequential()

#Conv1 and ReLU1
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(img_rows, img_cols, channels), data_format='channels_last',
                 kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv1'))
model.add(Activation(relu_layer, name='act_conv1'))

#Conv2 and ReLU2
model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same',
                 use_bias=use_bias, name='conv2'))
model.add(Activation(relu_layer, name='act_conv2'))

#Pool1
model.add(MaxPooling2D(pool_size=(2,2), name='pool1', data_format='channels_last'))

# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(img_rows, img_cols, channels), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv3'))
# model.add(Activation(relu_layer, name='act_conv3'))
#
# #Conv2 and ReLU2
# model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv4'))
# model.add(Activation(relu_layer, name='act_conv4'))
#
# #Pool1
# model.add(MaxPooling2D(pool_size=(2,2), name='pool2', data_format='channels_first'))

# reduce overfitting
model.add(Dropout(0.25))

#Conv3 and ReLU3
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal',
                 padding='same', use_bias=use_bias, name='conv5'))
model.add(Activation(relu_layer, name='act_conv5'))

#Conv4 and ReLU4
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv6'))
model.add(Activation(relu_layer, name='act_conv6'))

#Pool2
model.add(MaxPooling2D(pool_size=(2,2), name='pool3', data_format='channels_last'))

model.add(Dropout(0.25))

model.add(Flatten())

#FC1, Batch Normalization and ReLU5
model.add(Dense(512, use_bias=True, name='FC1', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn1'))
model.add(Activation(relu_layer, name='act_fc1'))

model.add(Dropout(0.5))

#FC2, Batch Normalization and ReLU6
model.add(Dense(classes, use_bias=True, name='FC2', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn2'))
model.add(Activation(softmax_layer, name='act_fc2'))

#Optimizers
opt = RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

# WEIGHTS_FNAME = args["weights"]
WEIGHTS_FNAME = weight_name
#model.load_weights(WEIGHTS_FNAME, by_name=True)


if args["load_model"] > 0:
    print('Loading existing weights')
    model.load_weights(WEIGHTS_FNAME)
else:
    checkpoint_scheduler = ModelCheckpoint('output/weights.{epoch:02d}.hdf5',
                                           monitor='val_acc',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='max', period=1)

    history = model.fit(train_data, train_labels,
                        batch_size=batch_size, epochs=(epochs),
                        verbose=1, validation_data=(test_data, test_labels),
                        callbacks=[checkpoint_scheduler], shuffle=True)
    print("[INFO] dumping weights to file...")
    model.save_weights(WEIGHTS_FNAME, overwrite=True)


score = model.evaluate(test_data, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])















