"""
目的： 实现M2NIST的图像分割。
方法： 自己搭建的类似Unet的网络，acc=94% .
评价： 非常简单的任务，做这件事的目的是熟悉下keras的使用，虽然看了很多视频、教程，自己动手的时候还是很捉急的。
      没有仔细的调参，acc上升的很快，应该是几个epoch就到了local mininum。
      重要的是学到了关于loss和metirc的东西，比如如何选择loss和metric，以及custom my loss
      etc. 这部分在本程序中没有体现，有机会写到blog里。
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


def load_data(data_path):
    combined = np.load(os.path.join(data_path, "combined.npy"))
    segmented = np.load(os.path.join(data_path, "segmented.npy"))

    print("data_shape:", combined.shape)
    print("label_shape:", segmented.shape)
    return combined, segmented


def show_data(data, img_index=np.arange(4)):
    data_shape = data.shape
    fig, axes = plt.subplots(2, 2)

    print(data[img_index[0], :, :].shape)
    axes[0, 0].imshow(data[img_index[0], :, :])
    axes[0, 1].imshow(data[img_index[1], :, :])
    axes[1, 0].imshow(data[img_index[2], :, :])
    axes[1, 1].imshow(data[img_index[3], :, :])
    plt.show()


def watch_aug_img(img_data, datagen):
    i = 0
    for batch in datagen.flow(img_data, batch_size=1):
        plt.figure(i)
        plt.imshow(batch[0].reshape(img_data.shape[1:-1]))
        i = i + 1
        if i > 4:
            break
    plt.show()


class MyUnet():
    def __init__(self, input_shape, batchNorm=True):
        self.input_shape = input_shape
        self.batchNorm = batchNorm

    def self_conv(self, input_layer, kernel_size, chan_num):
        x = layers.Conv2D(chan_num, kernel_size=kernel_size, strides=(1, 1),
                          padding="same", activation='relu')(input_layer)

        x = layers.Conv2D(chan_num, kernel_size=kernel_size, strides=(1, 1),
                          padding="same", activation='relu')(x)
        x = layers.MaxPool2D()(x)

        if self.batchNorm:
            x = layers.BatchNormalization()(x)

        return x

    def self_deconv(self, input_layer, chan_num, kernel_size):
        x = layers.Conv2D(chan_num, kernel_size=kernel_size, strides=(1, 1),
                          padding="same", activation='relu')(input_layer)
        x = layers.Conv2D(chan_num, kernel_size=kernel_size, strides=(1, 1),
                          padding="same", activation='relu')(x)
        x = layers.UpSampling2D()(x)

        if self.batchNorm:
            x = layers.BatchNormalization()(x)
        return x

    def get_model(self):
        input = keras.Input(shape=self.input_shape)

        conv_1 = self.self_conv(input, kernel_size=(3, 3), chan_num=32)
        conv_2 = self.self_conv(conv_1, kernel_size=(3, 3), chan_num=64)

        deconv_2 = self.self_deconv(conv_2, 64, (1, 1))

        deconv_1 = layers.Concatenate()([deconv_2, conv_1])

        deconv_0 = self.self_deconv(deconv_1, 32, (1, 1))

        deconv_0 = layers.Concatenate()([deconv_0, input])

        deconv_0 = layers.Conv2D(24, kernel_size=(3, 3), strides=(1,1),
                                 padding="same", activation='relu')(deconv_0)

        output = layers.Conv2D(11, kernel_size=(3, 3), strides=(1, 1),
                                 padding="same", activation='softmax')(deconv_0)

        model = keras.Model(inputs=input, outputs=output, name="my_net")

        keras.utils.plot_model(model, "my_model.png", show_shapes=True)

        return model


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

epochs = 50
batch_size = 100
data_path = r"G:\DeepLearningPractise\kaggle\img_seg\M2NIST"
model_save_path = r'G:\DeepLearningPractise\kaggle\img_seg\M2NIST\Model_Save'
log_file = r'G:\DeepLearningPractise\kaggle\img_seg\M2NIST\Model_Save\Log_file'
csv_file = r'G:\DeepLearningPractise\kaggle\img_seg\M2NIST\Model_Save\CSV_file'

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
if not os.path.exists(log_file):
    os.mkdir(log_file)
if not os.path.exists(csv_file):
    os.mkdir(csv_file)

combined, segmented = load_data(data_path)
# show_data(combined)
shuffle_index = np.random.permutation(combined.shape[0])

combined_afer_shuffle = combined[shuffle_index, :, :]
segmented_after_shuffle = segmented[shuffle_index, :, :]

combined_afer_shuffle = combined_afer_shuffle[:, :, :, np.newaxis]
print("new shape: ", combined_afer_shuffle.shape)

datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=False,
                             vertical_flip=False, validation_split=0.2,
                             shear_range=0.1, zoom_range=0.2, fill_mode='nearest')
# watch_aug_img(combined_afer_shuffle, datagen)
my_unet = MyUnet(combined_afer_shuffle.shape[1:], )
model = my_unet.get_model()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

model_check_point = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='accuracy',
                    save_best_only=True, save_weights_only=False, save_freq='epoch')
model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=1e-4,
                                                        patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1,
                                                  patience=5)
tensor_board = tf.keras.callbacks.TensorBoard(log_file)
csv_log = tf.keras.callbacks.CSVLogger(os.path.join(csv_file, 'training.csv'))
call_backs = [model_check_point, model_early_stopping, lr_scheduler, lr_plateau, tensor_board,
              csv_log]


model.fit(zip(datagen.flow(combined_afer_shuffle, batch_size=batch_size, subset='training',
                    seed=1), datagen.flow(segmented_after_shuffle, batch_size=batch_size, subset='training',
                                     seed=1)), epochs=epochs, steps_per_epoch=len(combined_afer_shuffle)//batch_size,
                    validation_data=zip(datagen.flow(combined_afer_shuffle, batch_size=batch_size,
                                                     subset='validation', seed=1), datagen.flow(segmented_after_shuffle,
                     batch_size=batch_size, subset='validation', seed=1)),
                    validation_steps=len(combined_afer_shuffle)//batch_size, callbacks=call_backs)



