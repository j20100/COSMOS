from __future__ import absolute_import
from __future__ import print_function
import os
import time
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

print("------------INITIALIZE DEPENDENCIES--------------")

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
K.set_image_data_format("channels_first")

import cv2
import numpy as np

path = './CamVid/'

img_channels = 3
img_rows = 360
img_cols = 480
data_shape = img_rows*img_cols

epochs = 100
batch_size = 6

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []

    with open(path+'train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        train_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
    return np.array(train_data), np.array(train_label)

print("------------FORMATING DATASET--------------")
#train_data, train_label = prep_data()
#train_label = np.reshape(train_label,(367,data_shape,12))

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad), input_shape=(img_channels, img_rows, img_cols)),
        Conv2D(filter_size, kernel, padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel, padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel, padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel, padding='valid'),
        BatchNormalization(),
        Activation('relu')
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel, padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel, padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel, padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size, kernel, padding='valid'),
        BatchNormalization()
    ]

print("------------CREATING NETWORK--------------")
network = models.Sequential()
# Add a noise layer to get a denoising network. This helps avoid overfitting
#network.add(Layer(input_shape=(3, 360, 480)))

#network.add(GaussianNoise(stddev=0.3))
network.encoding_layers = create_encoding_layers()
network.decoding_layers = create_decoding_layers()
for l in network.encoding_layers:
    network.add(l)
for l in network.decoding_layers:
    network.add(l)

network.add(Conv2D(12, 1, padding='valid',))
network.add(Reshape((12, data_shape)))
network.add(Permute((2, 1)))
network.add(Activation('softmax'))
from keras.optimizers import SGD
optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
network.compile(loss="categorical_crossentropy", optimizer=optimizer)




#For training uncomment

#print("------------TRAINING NETWORK--------------")

#history = network.fit(train_data, train_label, batch_size=batch_size, epochs=epochs,
#                    show_accuracy=True, verbose=1, class_weight=class_weighting )#, validation_data=(X_test, X_test))

#network.save_weights('model_weight_ep100.hdf5')

#For deployement

print("------------DEPLOYING NETWORK--------------")

network.load_weights('model_weight_ep100.hdf5')

import matplotlib.pyplot as plt
#matplotlib inline

Sky = [255,255,255]
Building = [255,0,0]
Pole = [255,255,0]
Road_marking = [125,0,255]
Road = [0,255,0]
Pavement = [0,255,255]
Tree = [255,0,255]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [0,0,255]
Pedestrian = [125,125,125]
Bicyclist = [125,0,0]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r)#[:,:,0]
    rgb[:,:,1] = (g)#[:,:,1]
    rgb[:,:,2] = (b)#[:,:,2]
    return rgb

# Code for image analysis
import os
img = cv2.imread(os.getcwd() + '/CamVid/train/0016E5_07650.png')
img_prep = []
img = cv2.resize(img, (480, 360))

img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))
img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))

output = network.predict_proba(np.array(img_prep)[1:2])
pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)))

cv2.imshow('Prediction', pred)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

# Code for video playback analysis
from moviepy.editor import VideoFileClip

def process_image(image):

    vid_img_prep = []
    vid_img = cv2.resize(image, (480, 360))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))

    output = network.predict_proba(np.array(vid_img_prep)[1:2])
    pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)))

    #print(image)
    #print(pred.astype(int))
    #print(pred)

    #cv2.imshow('iage',image)
    #cv2.imshow('pred',pred)
    #cv2.waitKey(0)

    return pred

video = VideoFileClip("01TP_extract.avi")

def invert_red_blue(image):
    return image[:,:,[2,1,0]]

video = video.fl_image(invert_red_blue)
pred_video = video.fl_image(process_image)

pred_video.write_videofile('pred_video.avi', codec='rawvideo', audio=False)

# Code for live stream video analysis

while(True):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    vid_img_prep = []
    vid_img = cv2.resize(frame, (480, 360))

    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    cap.release()
    output = network.predict_proba(np.array(vid_img_prep)[1:2])
    pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)))
    cv2.imshow('Prediction', pred)
    cv2.imshow('Original', vid_img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
'''
