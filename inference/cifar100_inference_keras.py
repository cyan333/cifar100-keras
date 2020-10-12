from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import np_utils
from keras.models import Model
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.utils import normalize
from sec_ops import relu_layer as relu_layer_op
from sec_ops import softmax_layer as softmax_layer_op
from binary_ops import relu_layer as relu_layer_op
from binary_ops import softmax_layer as softmax_layer_op
from binary_ops import floor_func as floor_func_op
from binary_layers import BinaryDense, BinaryConv2D
import argparse
import csv
import math


###### Parameters #######.
batch_size = 32
epochs = 100
channels = 3
img_rows = 32
img_cols = 32

classes = 100
epsilon = 1e-6
momentum = 0.9
weight_decay = 0.0004
use_bias = False

###### Argu ######

weight_name = "weight_0927_cifar100.hdf5"
layers_array = ["scaling1", "scaling2", 'scaling3','scaling4']

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str,
        help="(optional) path to weights file")
ap.add_argument("-p", "--print_layers", type=int, default=-1,
        help="(optional) To print intermediate layer pkl files")

args = vars(ap.parse_args())


def relu_layer(x):
    return relu_layer_op(x)


def softmax_layer(x):
    return softmax_layer_op(x)


def floor_func(x,divisor):
    return floor_func_op(x,divisor)


def clip_func(x):
    low_values_flags = x < -127
    x[low_values_flags] = 0

    high_values_flags = x > 127
    x[high_values_flags] = 128
    return x


def floor_func(x,divisor):
    return floor_func_op(x,divisor)


######################
#load scale
with open('max_dict.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    data_read = [row for row in reader]

conv_scale = []
for i in range(0,5):
    conv_scale.append(round(127/float(data_read[i*2][1]), 2))
    print(str(i) + '--->' + str(conv_scale[i]))

######################


#### Load data ####
(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

# convert class vectors to binary class matrices
train_labels = to_categorical(train_labels, classes)
test_labels = to_categorical(test_labels, classes)

test_data = test_data/255
train_data = train_data/ 255

print(train_data.shape, 'train data')
print(train_labels.shape, 'train_labels')


model = Sequential()

#Conv1 and ReLU1
model.add(Conv2D(96, kernel_size=(3,3), strides=(2,2), input_shape=(32,32,3), data_format='channels_last',
                 kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv1'))
# model.add(Lambda(lambda x: floor_func(x, conv_scale[0]),name='scaling1'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
model.add(Activation(relu_layer, name='act_conv1'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), data_format='channels_last'))

#Conv2 and ReLU2
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), data_format='channels_last', kernel_initializer='he_normal', padding='same',
                 use_bias=use_bias, name='conv2'))
# model.add(Lambda(lambda x: floor_func(x, conv_scale[1]),name='scaling2'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
model.add(Activation(relu_layer, name='act_conv2'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1), data_format='channels_last'))

#Pool1
# model.add(Dropout(0.1))

##################
#Conv3 and ReLU3
model.add(Conv2D(384, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv3'))
# model.add(Lambda(lambda x: floor_func(x, conv_scale[2]),name='scaling3'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
model.add(Activation(relu_layer, name='act_conv3'))

#Conv4 and ReLU4
model.add(Conv2D(384, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv4'))
# model.add(Lambda(lambda x: floor_func(x, conv_scale[3]),name='scaling4'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
model.add(Activation(relu_layer, name='act_conv4'))

model.add(Conv2D(256, kernel_size=(1,1), strides=(1,1), kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv5'))
# model.add(Lambda(lambda x: floor_func(x, conv_scale[4]),name='scaling5'))  ## Dividing by 27 (MAV) and 18.296 (Instead of 128), so need to multiply by factor of 7 in gain stage
model.add(Activation(relu_layer, name='act_conv5'))
#Pool2
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# reduce overfitting
# model.add(Dropout(0.25))

#################################
model.add(Flatten())
#FC1, Batch Normalization and ReLU5
model.add(Dense(4096, use_bias=True, name='FC1', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn1'))
model.add(Activation(relu_layer, name='act_fc1'))

# model.add(Dropout(0.25))

model.add(Dense(4096, use_bias=True, name='FC2', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn2'))
model.add(Activation(relu_layer, name='act_fc2'))

# model.add(Dropout(0.5))

#FC2, Batch Normalization and ReLU6
model.add(Dense(classes, use_bias=True, name='FC3', kernel_initializer='he_normal'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn3'))
model.add(Activation(softmax_layer, name='act_fc3'))

# Optimizers
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['accuracy', 'top_k_categorical_accuracy'])
model.summary()

# WEIGHTS_FNAME = args["weights"]

WEIGHTS_FNAME = weight_name
model.load_weights(WEIGHTS_FNAME, by_name=True)

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print ('top-k accuracy:', score[2])
    # accr_list.append(score[1])
    # top_5_acc.append(score[2])

if args["print_layers"] > 0:
    for i in layers_array:
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(i).output)
        intermediate_output = intermediate_layer_model.predict([test_data])

        file_name = "output/" + i + ".pkl"

        print("Dumping layer {} outputs to file {}".format(i, file_name))
        intermediate_output.dump(file_name)
















