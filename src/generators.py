import os
from random import randint
import keras
from utils import create_stereogram, create_random_figure
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, Sequence
from sklearn.model_selection import StratifiedKFold
import pandas as pd

class RandomFiguresDataGenerator(keras.utils.Sequence):
    'Data generator for train models'

    def __init__(self, batch_size=32, shape=(200, 400), samples_per_epoch=1000, mode='multiclass_overlapping',
                 normalize=True, add_channel=False):
        self.shape = shape
        self.batch_size = batch_size
        self.n_classes = shape[1]
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.normalize = normalize
        self.add_channel = add_channel


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.samples_per_epoch / self.batch_size))

    def __getitem__(self, index):
        X, y = [], []

        min_pattern_width = 20
        classes = ['triangle', 'circle', 'clock', 'rectangle', 'random']
        n = 1 if self.mode == 'binary' else self.batch_size
        for _ in range(n):
            _class = randint(0, len(classes) - 1)
            _class = 4
            img = create_random_figure(self.shape, class_=classes[_class], depth=1, minx=10)
            pattern_width = randint(min_pattern_width, self.shape[1] - min_pattern_width)
            st = create_stereogram(img, pattern_width=pattern_width)

            f = 1 / 255 if self.normalize else 1
            if self.mode == 'multiclass_overlapping':
                if self.add_channel:
                    st = st.reshape([1] + list(st.shape))
                X.append(st * f)
                y.append(keras.utils.to_categorical(pattern_width, num_classes=self.n_classes))
            elif self.mode == 'generator':
                if self.add_channel:
                    st = st.reshape([1] + list(st.shape))
                X.append(st * f)
                y.append(img * f)

        return np.array(X), np.array(y)


class CommonImagesGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_path, batch_size=32, shape=(100, 200), samples_per_epoch=1000,
                 mode='overlapping', classes=None, normalize=True):
        self.shape = shape
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.shape = shape
        self.normalize = normalize
        if classes == None:
            self.classes = sorted([d for d in os.listdir(image_path)])
        else:
            self.classes = classes
        self.__images = [os.path.join(image_path, d, file) for d in os.listdir(image_path)
                         for file in os.listdir(os.path.join(image_path, d)) if d in self.classes]
        self.__indexes = np.arange(len(self.__images))
        self.mode = mode
        np.random.shuffle(self.__indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.__images) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.__indexes = np.arange(len(self.__images))
        np.random.shuffle(self.__indexes)

    def __getitem__(self, index):
        X, y = [], []

        min_pattern_width = 20
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            file_name = self.__images[self.__indexes[i]]
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            pattern_width = randint(min_pattern_width, 3 * min_pattern_width)
            st = create_stereogram(img, pattern_width=pattern_width)
            st = st.reshape((1, st.shape[0], st.shape[1]))

            if self.normalize:
                st = st / 255

            X.append(st)

            if self.mode == 'overlapping':
                y.append(keras.utils.to_categorical(pattern_width, num_classes=self.shape[1]))

            if self.mode == 'classification':
                class_ = file_name.split('/')[-2]
                pattern_width, num_classes = self.n_classes
                y.append(keras.utils.to_categorical(self.classes.index(class_), num_classes=len(self.classes)))

        return np.array(X), np.array(y)
  

def preprocess(img):
    
    min_pattern_width = 20
    pattern_width= randint(min_pattern_width, 3*min_pattern_width)
    if len(img.shape) == 3:
        img = img.reshape(img.shape[1:])

    st = img
    st = st.reshape((1, st.shape[0], st.shape[1]))
    st = st / 255
    return st


class CommonImageGeneratorFromFlow(Sequence):
       
    def __init__(self, generator, data, image_shape, seed, batch_size, shuffle=True):
        self.generator = generator
        self.classes = list(np.sort(data['label'].unique()))
        self.iterator = self.generator.flow_from_dataframe(
                                    data, 
                                    x_col='filepath', 
                                    y_col='label', 
                                    target_size=(image_shape[0], image_shape[1]), 
                                    color_mode='grayscale', 
                                    classes=self.classes, 
                                    class_mode='categorical', 
                                    seed=seed,
                                    batch_size=batch_size, 
                                    shuffle=shuffle)
        
    def __len__(self):
        return self.iterator.__len__()
        
    def __getitem__(self, index):
        return self.iterator[index]
        
    def on_eponch_end(self):
        self.iterator.on_eponch_end
    
    def reset(self):
        self.iterator.reset()
            
    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
    
    def next(self):
        return self.iterator.next()

    

class CommonImagesKFoldGenerator:
    'Generates train and validation generators folds '

    def __init__(self, image_path, batch_size=32, shape=(100, 200), nfolds=8):
        self.shape = shape
        self.batch_size = batch_size
        self.shape = shape
        self.nfolds=nfolds
        self.__images = [os.path.join(image_path, d, file) for d in os.listdir(image_path)
                         for file in os.listdir(os.path.join(image_path, d))]
        self.classes = [f.split('/')[-2] for f in self.__images]
        self.__data = pd.DataFrame({'filepath': self.__images, 'label': self.classes})
        
        
        self.__train_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=45,
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=.2,
            shear_range=.2,
            data_format='channels_first',
            preprocessing_function=preprocess,
            fill_mode='nearest')

        self.__val_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            preprocessing_function=preprocess,
            data_format='channels_first',
            fill_mode='reflect')

        
    def __iter__(self):
        skf = StratifiedKFold(n_splits=self.nfolds)

        for i, (train_index, test_index) in enumerate(skf.split(self.__data['filepath'], self.__data['label'])):
            train_iterator = CommonImageGeneratorFromFlow(self.__train_datagen, self.__data.iloc[train_index], self.shape, 
                                                  111, self.batch_size, shuffle=True)
            val_iterator = CommonImageGeneratorFromFlow(self.__val_datagen, self.__data.iloc[test_index], self.shape, 
                                                111, self.batch_size, shuffle=False)

            yield train_iterator, val_iterator