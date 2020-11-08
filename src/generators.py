import os
from random import randint
import keras
from utils import create_stereogram, create_random_figure
import numpy as np

class RandomFiguresDataGenerator(keras.utils.Sequence):
    'Data generator for train models'

    def __init__(self, batch_size=32, shape=(200, 400), samples_per_epoch=1000, mode='binary',
                 normalize=True, add_channel=False, model=None):
        self.shape = shape
        self.batch_size = batch_size
        self.n_classes = shape[1]
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.normalize = normalize
        self.add_channel = add_channel
        self.model = model

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
            if self.mode == 'binary_overlapping':
                r = self.model.predict(np.array([st])) * self.normalize
                X += list(r.reshape(r.shape[1:] + (1,)))
                y += list(keras.utils.to_categorical(pattern_width, num_classes=self.n_classes))
            elif self.mode == 'multiclass_overlapping':
                if self.add_channel:
                    st = st.reshape([1] + list(st.shape))
                X.append(st * f)
                y.append(keras.utils.to_categorical(pattern_width, num_classes=self.n_classes))
            elif self.mode == 'multiclass_classes':
                if self.add_channel:
                    st = st.reshape([1] + list(st.shape))
                X.append(st * self.normalize)
                y.append(keras.utils.to_categorical(_class, num_classes=len(classes)))

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
                y.append(keras.utils.to_categorical(pattern_width, num_classes=shape[1]))

            if self.mode == 'classification':
                class_ = file_name.split('/')[-2]
                pattern_width, num_classes = self.n_classes
                y.append(keras.utils.to_categorical(self.classes.index(class_), num_classes=len(self.classes)))

        return np.array(X), np.array(y)