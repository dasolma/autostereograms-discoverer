from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, BatchNormalization, MaxPooling2D
from generators import RandomFiguresDataGenerator
from layers import OverlapingLayer, StereoConv
from keras.layers import Lambda, multiply
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam

def classify_overlap(shape=(1, 100, 200)):
    '''
    Return a model architecture to classify the disparity of an auto-stereogram
    :param shape: shape of the training images
    :return: a triplet with the model, a training data generator and a validation data generator
    '''
    model = Sequential()


    def add_block(model):
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=shape,
                         data_format='channels_first', activation='relu'))
        model.add(Conv2D(32, kernel_size=(5, 5), input_shape=shape,
                         data_format='channels_first', activation='relu'))
        model.add(Conv2D(32, kernel_size=(7, 7), input_shape=shape,
                         data_format='channels_first', activation='relu'))
        model.add(MaxPooling2D())
    add_block(model)
    add_block(model)
    add_block(model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(shape[2]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Generators
    training_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=1000,
                                                    mode='multiclass_overlapping', normalize=True, add_channel=True)
    validation_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=500,
                                                      mode='multiclass_overlapping', normalize=True, add_channel=True)

    return model, training_generator, validation_generator




def classifly_with_custom_layer(custom_layer, shape=(100, 200), train_custom_layer=True):

    model = Sequential()
    model.add(custom_layer(input_shape=shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((5, 5)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(shape[-1]))
    model.add(Activation('softmax'))
    
    model.layers[0].trainable = train_custom_layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generators
    training_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=1000,
                                       mode='multiclass_overlapping', normalize=True, add_channel=True)
    validation_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=500,
                                         mode='multiclass_overlapping', normalize=True, add_channel=True)


    return model, training_generator, validation_generator


def classifly_with_overlaping_layer(shape=(100, 200), train_custom_layer=True):
    return classifly_with_custom_layer(OverlapingLayer, shape, train_custom_layer=train_custom_layer)

def classifly_with_stereoconv_layer(shape=(1, 100, 200), train_custom_layer=True):
    return classifly_with_custom_layer(StereoConv, shape, train_custom_layer=train_custom_layer)


def create_revealing_model(base_model, as_RGB=False, compiled=True):
    
    for layer in base_model.layers:
        layer.trainable = False

    shape = (1, 100, 200)
    i = base_model.input
    overlaps = base_model.layers[0](i)
    probabilities = base_model.layers[-1].output

    permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0,3,2,1)))
    permuted = permute_layer(overlaps)

    weighted_overlaps = multiply([permuted, probabilities])
    weighted_overlaps = permute_layer(weighted_overlaps)

    seen_image = Lambda(lambda x: K.sum(x, axis=1), name='sum')(weighted_overlaps)
    if as_RGB:
        def repeat(x):
            return K.permute_dimensions(K.repeat(x, n=3), (0,2,1))
        seen_image = Lambda(lambda x: K.map_fn(repeat, x))(seen_image)
    
    revealing_model = Model(inputs=base_model.input, outputs=seen_image)
    
    if compiled:
        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=True)
        revealing_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Generators
        training_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=1000,
                                           mode='generator', normalize=True, add_channel=True)
        validation_generator = RandomFiguresDataGenerator(batch_size=32, shape=shape[1:], samples_per_epoch=500,
                                             mode='generator', normalize=True, add_channel=True)


        return revealing_model, training_generator, validation_generator
    
    else:    
        return revealing_model