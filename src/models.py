from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense

from src.generators import RandomFiguresDataGenerator


def classify_overlap(shape=(1, 100, 200)):
    '''
    Return a model architecture to classify the disparity of an auto-stereogram
    :param shape: shape of the training images
    :return: a triplet with the model, a training data generator and a validation data generator
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(5, 5), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(7, 7), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(5, 5), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(5, 5), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(5, 5), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(shape[2]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generators
    training_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=1000,
                                                    mode='multiclass_overlapping', normalize=True, add_channel=True)
    validation_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=500,
                                                      mode='multiclass_overlapping', normalize=True, add_channel=True)

    return model, training_generator, validation_generator



