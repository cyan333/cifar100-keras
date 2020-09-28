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
import argparse

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

weight_name = "weight_0916.hdf5"
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

# def floor_func(x,divisor):
#     return floor_func_op(x,divisor)

def clip_func(x):
    low_values_flags = x < -127
    x[low_values_flags] = 0

    high_values_flags = x > 127
    x[high_values_flags] = 128
    return x


#### Load data ####
(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

train_labels = to_categorical(train_labels, classes)
test_labels = to_categorical(test_labels, classes)

print(test_data.shape, 'test data')
print(test_labels.shape, 'test_labels')

model = Sequential()

#Conv1 and ReLU1
model.add(Conv2D(32, kernel_size=(2,1), input_shape=(img_rows, img_cols, channels), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv1'))
model.add(Activation(relu_layer, name='act_conv1'))

#Conv2 and ReLU2
model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv2'))
model.add(Activation(relu_layer, name='act_conv2'))

#Pool1
model.add(MaxPooling2D(pool_size=(2,2), name='pool1', data_format='channels_last'))

# reduce overfitting
# model.add(Dropout(0.25))

#Conv3 and ReLU3
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv3'))
model.add(Activation(relu_layer, name='act_conv3'))

#Conv4 and ReLU4
model.add(Conv2D(64, kernel_size=(3,3), data_format='channels_last', kernel_initializer='he_normal', padding='same', use_bias=use_bias, name='conv4'))
model.add(Activation(relu_layer, name='act_conv4'))

#Pool2
model.add(MaxPooling2D(pool_size=(2,2), name='pool2', data_format='channels_last'))

# model.add(Dropout(0.25))

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
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc', 'top_k_categorical_accuracy'])
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
















