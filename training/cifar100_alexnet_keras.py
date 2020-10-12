import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D
from keras.layers import Flatten
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import np_utils
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets import cifar10, cifar100
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD


weight_name = "weight_0927_cifar100.hdf5"
batch = 32
epoch = 100
classes = 100
use_bias = False
epsilon = 1e-6
momentum = 0.9

### CIFAR-10 ###

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# for visualization
# CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Training Dataset: This is the group of our dataset used to train the neural network directly.
#                   Training data refers to the dataset partition exposed to the neural network during training.

# Validation Dataset: This group of the dataset is utilized during training
#                       to assess the performance of the network at various iterations.

# Test Dataset: This partition of the dataset evaluates the performance
#               of our network after the completion of the training phase.

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

# Visualize image
# plt.figure(figsize=(20,20))
# for i, (image, label) in enumerate(train_ds.take(5)):
#     ax = plt.subplot(5,5,i+1)
#     plt.imshow(image)
#     plt.title(CLASS_NAMES[label.numpy()[0]])
#     plt.axis('off')


def process_images(image, label):
    """
    resize image to 227x227, since alexnet only take 227x227 input
    normalize and standardize input so it will learn faster
    """
    # Normalize images to have a mean of 0 and standard deviation of 1
    # per_image_standardization is preferred, which normalize the entire image to mean zero and std 1.
    # It also make learning fast.
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label


train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds
            .map(process_images)
            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=batch, drop_remainder=True))
test_ds = (test_ds
           .map(process_images)
           .shuffle(buffer_size=train_ds_size)
           .batch(batch_size=batch, drop_remainder=True))
validation_ds = (validation_ds
                 .map(process_images)
                 .shuffle(buffer_size=train_ds_size)
                 .batch(batch_size=batch, drop_remainder=True))

# model = Sequential()
# model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227,227,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#
# model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
#
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same"))
# model.add(Activation('relu'))
#
# model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same"))
# model.add(Activation('relu'))
#
# model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
#
# model.add(Flatten())
# model.add(Dense(4096, use_bias=True, name='FC1', kernel_initializer='he_normal'))
# model.add(Activation('relu', name='act_fc1'))
# model.add(Dropout(0.5))
#
# model.add(Dense(4096, use_bias=True, name='FC2', kernel_initializer='he_normal'))
# model.add(Activation('relu', name='act_fc2'))
# model.add(Dropout(0.5))
#
# model.add(Dense(classes, use_bias=True, name='FC3', kernel_initializer='he_normal'))
# model.add(Activation('softmax'))


# model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), input_shape=(227,227,3)),
#     keras.layers.Activation('elu'),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same"),
#     keras.layers.Activation('elu'),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same"),
#     keras.layers.Activation('elu'),
#     keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same"),
#     keras.layers.Activation('elu'),
#     keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same"),
#     keras.layers.Activation('elu'),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='elu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(4096, activation='elu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(classes, activation='softmax')
# ])

opt = tf.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
model.summary()
WEIGHTS_FNAME = weight_name

# checkpoint_scheduler = ModelCheckpoint('output/weights.{epoch:02d}.hdf5',
#                                        monitor='val_acc',
#                                        verbose=0,
#                                        save_best_only=True,
#                                        save_weights_only=True,
#                                        mode='max', period=1)

history = model.fit(train_ds,
                    batch_size=batch,
                    epochs=epoch,
                    verbose=1,
                    validation_data=(test_ds))

print("[INFO] dumping weights to file...")
model.save_weights(WEIGHTS_FNAME, overwrite=True)


score = model.evaluate(test_ds, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
