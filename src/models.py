from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, BatchNormalization, MaxPooling2D
from generators import RandomFiguresDataGenerator, CommonImagesKFoldGenerator
from layers import OverlapingLayer, StereoConv
from keras.layers import Lambda, multiply
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import VGG16
import pickle as pk
from sklearn.metrics import confusion_matrix
import json
import keras

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
    
    
def create_figure_class_model(base_model, use_vgg=True, train_base_model=False):
    
    for layer in base_model.layers:
        layer.trainable = train_base_model


    shape = (1, 100, 200)
    i = base_model.input
    overlaps = base_model.layers[0](i)
    probabilities = base_model.layers[-1].output

    permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0,3,2,1)))
    permuted = permute_layer(overlaps)

    weighted_overlaps = multiply([permuted, probabilities])
    weighted_overlaps = permute_layer(weighted_overlaps)

    seen_image = Lambda(lambda x: K.sum(x, axis=1), name='sum')(weighted_overlaps)
    def repeat(x):
        return K.permute_dimensions(K.repeat(x, n=3), (0,2,1))
    seen_image = Lambda(lambda x: K.map_fn(repeat, x))(seen_image)

    if use_vgg:
        vgg = VGG16(input_shape=(100,200,3), weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False
        output_vgg16_conv = vgg(seen_image)

        x = Flatten(name='flatten_vgg')(output_vgg16_conv)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dense(10, activation='softmax', name='predictions')(x)
    else:
        conv = Conv2D(64, (3,3), activation='relu', name='class_conv1')(seen_image)
        conv = Conv2D(64, (3,3), activation='relu', name='class_conv2')(conv)
        conv = MaxPooling2D((3,3), name='class_pool1')(conv)
        conv = Conv2D(64, (3,3), activation='relu', name='class_conv3')(conv)
        conv = Conv2D(64, (3,3), activation='relu', name='class_conv4')(conv)
        conv = MaxPooling2D((3,3), name='class_pool2')(conv)
        conv = Conv2D(64, (3,3), activation='relu', name='class_conv5')(conv)
        x = Flatten(name='flatten_custom')(conv)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dense(10, activation='softmax', name='predictions')(x)

    class_model = Model(inputs=base_model.input, outputs=x)
    class_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return class_model 

def kfold_training(model, image_path, nfolds=8, epochs=200, fold_callback=None):
    folds = CommonImagesKFoldGenerator(image_path, nfolds=nfolds)
    
    history = []
    for fold_number, (train_iterator, val_iterator) in enumerate(folds):

        h = model.fit_generator(generator=train_iterator, 
                        validation_data=val_iterator, 
                        epochs=epochs, verbose=0
                       )

        
        y_true = []
        y_pred = []
        for i in range(len(val_iterator)):
            y_true += list(val_iterator[i][1].argmax(axis=1))
            y_pred += list(model.predict(val_iterator[i][0]).argmax(axis=1))
        

        info = {
            'fold': fold_number+1,
            'metrics': h.history,
            'val_classes': val_iterator.classes,
            'val_confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'solp_params': [[float(f) for f in w] for w  in model.layers[1].get_weights()]
        }
        history.append(info)
        
        if fold_callback:
            fold_callback(model, history)
        
def train_final_experiment(epochs=200):
    
    # use the stereo_class model pretrained.
    disparity_class_model = keras.models.load_model('../checkpoints/stero_class.h5', 
                                      custom_objects={'StereoConv': StereoConv})

    def fold_callback(model, history, use_vgg, train_base_model):
        vgg_custom = 'vgg' if use_vgg else 'custom'
        fold = history[-1]['fold']
        history[-1]['use_vgg'] = use_vgg
        history[-1]['train_base_model'] = train_base_model
        base_retrained = '_base_retrained' if train_base_model else '' 

        # save history
        json.dump(history, open('../checkpoints/history_%s%s.json' %(vgg_custom, base_retrained)  , 'w'))

        # save model
        model.save('../checkpoints/model_fold%d_%s%s.h5' %(fold, vgg_custom, base_retrained))

        loss = history[-1]['metrics']['val_loss'][-1]
        accuracy = history[-1]['metrics']['val_accuracy'][-1]
        train_base = 'train base' if train_base_model else 'no train base'
        print('Using %s, %s, fold %d, loss: %f, accurracy: %f ' % (vgg_custom, train_base, 
                                                                   fold, loss, accuracy))

    for train_base_model in [True, False]:

        for use_vgg in [True,False]:

            model = create_figure_class_model(disparity_class_model, 
                                              use_vgg=use_vgg, 
                                              train_base_model=train_base_model)

            callback = lambda model, history: fold_callback(model, history, use_vgg, train_base_model)
            kfold_training(model, image_path='../data/fake_heatmaps/', nfolds=8, epochs=epochs, 
                           fold_callback=callback)
    
