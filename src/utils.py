import os
import numpy as np
try:
    from IPython import get_ipython
except:
    pass
from skimage.draw import polygon, circle
from math import ceil
from random import randint
import cv2
from matplotlib import pyplot as plt
import random
import keras 
from keras.models import Model
from keras.layers import Input

def create_stereogram(img, pattern_width=50, invert=True):
    '''
    This function create a auto-stereogram

    :param img: image to hide in the auto-stereogram
    :param pattern_width: width of the random pattern
    :param invert:
    :return: the image of the auto-stereogram
    '''
    invert = -1 if invert else 1
    gen_pattern = lambda width, height: np.random.randint(0, 256, (width, height))

    pattern_div = img.shape[1] * pattern_width
    pattern = gen_pattern(img.shape[0], pattern_width)

    out_data = np.zeros(img.shape, dtype=np.uint8)

    # Create stereogram
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if x < pattern_width:
                out_data[y, x] = pattern[y, x]  # Use generated pattern
            else:
                shift = ceil(img[y, x] / pattern_div)  # 255 is closest
                out_data[y, x] = out_data[y, x - pattern_width + (shift * invert)]

    return out_data


def create_fake_depth_map(im_path, shape=(100, 200)):
    '''
    Create a depth map from the image of any size removing the blank pixels (background), centering
    the figure and adding the need borders to fit the input shape.
    :param im_path: file path of the image
    :param shape: shape of the output image
    :return: fake depth map
    '''
    max_w = 80
    max_h = 150

    im_gray = cv2.imread(os.path.join(im_path), cv2.IMREAD_UNCHANGED)

    # create the fake depth map
    if len(im_gray.shape) == 3 and im_gray.shape[2] == 4:
        im_gray = np.rollaxis(im_gray, 2, 0)
        RGB = np.rollaxis(im_gray[:3, ...], 0, 3)
        alpha = im_gray[3, ...]
        RGB[alpha == 0] = 255
        RGB = 255 - RGB
    else:
        RGB = 255 - im_gray

    blur_kelnel_size = int(np.min(RGB.shape[:2]) * 0.05)
    blur_kelnel_size = blur_kelnel_size if blur_kelnel_size % 2 == 1 else blur_kelnel_size + 1
    depth_map = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.GaussianBlur(depth_map, (blur_kelnel_size, blur_kelnel_size), cv2.BORDER_DEFAULT)

    # resize and crop the image keeping the figure in the center

    # get only the figure (discard the blank areas)
    xmin, ymin = np.where(depth_map != 0)[1].min(), np.where(depth_map.T != 0)[1].min()
    xmax, ymax = np.where(depth_map != 0)[1].max(), np.where(depth_map.T != 0)[1].max()
    depth_map = depth_map[ymin:ymax, xmin:xmax]

    h, w = depth_map.shape[:2]

    # resize
    if h > w:
        if h > max_h:
            r = w / h
            h, w = max_h, int(max_h * r)
    else:
        if w > max_w:
            r = h / w
            w, h = max_w, int(max_w * r)
    depth_map = cv2.resize(depth_map, (h, w))

    if h > max_h:
        r = w / h
        h, w = max_h, int(max_h * r)
    if w > max_w:
        r = h / w
        w, h = max_w, int(max_w * r)
    depth_map = cv2.resize(depth_map, (h, w))

    # add borders to fit the shape
    h, w = depth_map.shape
    offset = int((shape[1] - w) / 2)
    depth_map = np.hstack((np.zeros((h, offset)) * 255, depth_map, np.zeros((h, offset)) * 255)).astype('uint8')

    h, w = depth_map.shape
    offset = int((shape[0] - h) / 2)
    depth_map = np.vstack((np.zeros((offset, w)) * 255, depth_map, np.zeros((offset, w)) * 255)).astype('uint8')

    depth_map = cv2.resize(depth_map, (shape[1], shape[0]))

    return depth_map


def create_random_figure(shape, class_, depth=None, minx=0):
    '''
    Create a random figure composed by polygons (triangles, rectangles, ...)
    :param shape: shape of the output image.
    :param class_ (string): class of the polygons: triangle, clock, rectangle, circle and random.
    :param depth: number of figures include in the image.
    :param minx: min position (pixel) to begin to add figures.
    :return:
    '''
    img = np.zeros(shape, dtype=np.uint8)
    if depth is None:
        depth = 1

    if class_ == 'triangle':
        for i in range(depth):
            p1_c, p1_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            p2_c, p2_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            p3_c, p3_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            r = np.array([p1_r + i, p2_r + i, p3_r + i, p1_r + i])
            c = np.array([p1_c + i, p2_c + i, p3_c + i, p1_c + i])
            rr, cc = polygon(r, c)
            img[rr, cc] = 255 - (depth - i) * 10

    elif class_ == 'clock':
        for i in range(depth):
            p1_c, p1_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            p2_c, p2_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            r = np.array([p1_r + i, p1_r + i, p2_r + i, p2_r + i])
            c = np.array([p1_c + i, p2_c + i, p1_c + i, p2_c + i])
            rr, cc = polygon(r, c)
            img[rr, cc] = 255 - (depth - i) * 10

    elif class_ == 'rectangle':
        for i in range(depth):
            p1_c, p1_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            p2_c, p2_r = randint(minx, shape[1] - depth - 1), randint(0, shape[0] - depth - 1)
            r = np.array([p1_r + i, p1_r + i, p2_r + i, p2_r + i])
            c = np.array([p1_c + i, p2_c + i, p2_c + i, p1_c + i])
            rr, cc = polygon(r, c)
            img[rr, cc] = 255 - (depth - i) * 10

    elif class_ == 'circle':
        for i in range(depth):
            r = randint(minx, int(shape[1] * .1))
            p1_c, p1_r = randint(minx, shape[1] - 2 * r - depth - 1), randint(0, shape[0] - 2 * r - depth - 1)
            rr, cc = circle(p1_r, p1_c, r)
            img[rr, cc] = 255 - (depth - i) * 10
    elif class_ == 'random':
        n_points = randint(3, 20)
        points_x, points_y = [randint(minx, shape[0]) for _ in range(n_points)], [randint(minx, shape[1]) for _ in
                                                                                  range(n_points)]
        rr, cc = polygon(points_x, points_y)
        img[rr, cc] = 255

    return img


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except:
        return False



def show_sampledata(generator, samples=10):
    data = generator.__getitem__(0)
    fig = plt.figure(figsize=(30, 50))
    w, h = 20, 10
    columns, rows = 3, min(len(data[0]), samples)

    i = 1
    fig_id = 1
    while i < rows:
        auto_stereogram, stereosis_factor = data[0][i], data[1][i]

        if len(auto_stereogram.shape) == 3:
            shape = auto_stereogram.shape[1:]
            auto_stereogram = np.reshape(auto_stereogram, shape)
        else:
            shape = auto_stereogram.shape

        stereosis_factor = np.where(stereosis_factor == 1)[0][0]

        zeros = np.zeros((shape[0], stereosis_factor))
        sto = .5 * auto_stereogram - .5 * np.hstack((zeros, auto_stereogram[:, :-stereosis_factor]))

        random_stereosis_factor = random.randint(10, 190)
        zeros = np.zeros((shape[0], random_stereosis_factor))
        nosto = .5 * auto_stereogram - .5 * np.hstack((zeros, auto_stereogram[:, :-random_stereosis_factor]))

        fig.add_subplot(rows, columns, fig_id)
        plt.imshow(auto_stereogram, cmap='gray')

        fig.add_subplot(rows, columns, fig_id + 1)
        plt.imshow(sto, cmap='gray')

        fig.add_subplot(rows, columns, fig_id + 2)
        plt.imshow(nosto, cmap='gray')
        i += 1
        fig_id += 3


def show_images(imgs, cmap='gray'):
    fig = plt.figure(figsize=(30, 20))
    columns, rows = 5, imgs.shape[0] / 5

    i = 0
    end = imgs.shape[0]
    while i < end:
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(imgs[i], cmap=cmap)
        ax = fig.gca()
        ax.set_axis_off()
        ax.set_title(str(i+1))
        
        i += 1
    
    plt.axis('off')
    plt.show()        
        
def show_features(features, init=0, end=None):
    fig = plt.figure(figsize=(30, 120))
    columns, rows = 5, features.shape[0] / 5

    i = init
    end = features.shape[0] if end is None else end
    while i < end:
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(features[i], cmap='gray')
        ax = fig.gca()
        ax.set_axis_off()
        ax.set_title(str(i+1))
        
        i += 1
    
    plt.axis('off')
    plt.show()

    

    

    
def get_sample_feature_maps_importance(model, sample, layer_idx, window=0, 
                                       loss=keras.losses.CategoricalCrossentropy()):
    y_true = model.predict(np.array([sample]))[0]

    
    output = model.layers[layer_idx-1].output
    feature_map_getter = Model(model.inputs, output)
    feature_maps = feature_map_getter.predict(np.array([sample])).reshape(output.shape[1:])

    
    # create an auxiliar model where the input will be the layer set as parameter
    idx = layer_idx  # index of desired layer
    input_shape = model.layers[idx].get_input_shape_at(0) # get the input shape of desired layer
    layer_input = Input(shape=input_shape[1:]) # a new input tensor to be able to feed the desired layer

    # create the new nodes for each layer in the path
    x = layer_input
    for layer in model.layers[idx:]:
        x = layer(x)

    # create the model to compute the feature importance
    feature_importance = Model(layer_input, x)
    
    y_true2 = feature_importance.predict(np.array([feature_maps]))[0]
    
    assert (y_true == y_true2).all()
    
    # get the baseline score
    base_score = loss(y_true, feature_importance.predict(np.array([feature_maps]))[0]).numpy()
    
    # introduce a random noise in each feature map
    # and compute the importance as the change 
    # in the prediction
    num_features = feature_maps.shape[0]
    feature_shape = feature_maps.shape[1:]
    scores = []
    feature_batch = []
    preds = []
    for i in range(0, num_features):
        for j in range(3):
            sf = np.copy(feature_maps)

            if window == 0:
                sf[i] = np.random.rand(*feature_shape)
            else:
                for k in range(max(0, i-window), min(num_features, i + window)):
                    sf[k] = np.random.rand(*feature_shape)

            feature_batch.append(sf)
            
        if len(feature_batch) > 32:
            preds.append(feature_importance.predict(np.array(feature_batch)))
            del feature_batch
            feature_batch = []

        
    preds.append(feature_importance.predict(np.array(feature_batch)))
    del feature_batch
    y_pred = np.concatenate(preds)
    
    for i in range(0, y_pred.shape[0], 3):
        score = loss([y_true]*3, y_pred[i:i+3]).numpy()
        scores.append(score - base_score)
    
    scores = np.array(scores) 
    return scores / scores.sum()
 

 
