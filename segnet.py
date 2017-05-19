from __future__ import absolute_import
from __future__ import print_function
import os
import time

# Comment to use tensorflow
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

print("------------INITIALIZE DEPENDENCIES--------------")

import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
import random
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

#Variables definitions
"""
Tested configuration : 400x400:bs12, 600x600:bs5 
"""
path = './SYNTHIA_RAND_CVPR16/'
img_channels = 3
img_original_rows=720
img_original_cols=960
img_rows = 600
img_cols = 600
epochs = 10
batch_size = 5
steps_per_epoch = 100

#Model save variables
save_model_name='model_ep10_bs12_st100_res600_ncw.hdf5'
run_model_name='model_ep50_bs5_st100_res600_ncw.hdf5'


#Class wieght for dataset
class_pixel = [1.11849828e+08, 3.62664219e+08, 3.19578306e+09, 2.57847955e+09, 1.21284747e+09, 2.30973570e+07, 1.77853424e+08, 1.08091678e+08, 6.83247417e+08, 2.65943380e+07, 2.77407453e+08, 5.09002610e+08]
dataset_nb = 13407
class_freq = np.zeros(len(class_pixel))

for i in range(len(class_pixel)):
    class_freq[i] = class_pixel[i] / (dataset_nb*img_original_rows*img_original_cols)

class_weight = np.zeros(len(class_pixel))
for i in range(len(class_pixel)):
    class_weight[i] = np.median(class_freq) / class_freq[i]

#class_weighting = {0:class_weight[0] ,1:class_weight[1] ,2:class_weight[2] ,3:class_weight[3] ,4:class_weight[4] ,5:class_weight[5] ,6:class_weight[6] ,7:class_weight[7] ,8:class_weight[8] ,9:class_weight[9] ,10:class_weight[10] ,11:class_weight[11]}
#print(class_weighting)

#Dynamic variables
img_rows_low = (img_original_rows-img_rows)/2
img_rows_high = img_original_rows-(img_original_rows-img_rows)/2
img_cols_low = (img_original_cols-img_cols)/2
img_cols_high = img_original_cols-(img_original_cols-img_cols)/2
data_shape = img_rows*img_cols

#Normaluzing function
def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

#Binary labeling function
def binarylab(labels):
    x = np.zeros([img_original_rows,img_original_cols,12])
    for i in range(img_original_rows):
        for j in range(img_original_cols):
            if labels[i][j] == -1:
                labels[i][j] = 0
            x[i,j,labels[i][j]]=1
    return x

#Calcul the weight of each class in the dataset
def class_weighting():

    with open(path+'ALL.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    dataset = []
    label_weight = np.zeros([12])
    print(label_weight)
    y = 0

    for i in range (len(txt)):
        print(i)
        end_crop=len(txt[i][0])-4
        dest_lab = os.getcwd() + '/SYNTHIA_RAND_CVPR16/GTTXT/' + txt[i][0][:end_crop] + '.txt'

        with open(dest_lab) as f:
            lab = [[int(num) for num in line.split()] for line in f]

        for i in range(img_original_rows):
            for j in range(img_original_cols):
                y = lab[i][j]
                label_weight[y] = label_weight[y]+1
    print(label_weight)
    return label_weight


#Class weighting function call
#label_weight = class_weighting()

#Data generator
def prep_data():
    while 1:
        with open(path+'ALL.txt') as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]
        train_data = []
        train_label = []

        for i in range(batch_size):
            index= random.randint(0, len(txt)-1)
            end_crop=len(txt[index][0])-4
            dest_lab = os.getcwd() + '/SYNTHIA_RAND_CVPR16/GTTXT/' + txt[index][0][:end_crop] + '.txt'
            train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + '/SYNTHIA_RAND_CVPR16/RGB/' + txt[index][0][:])),2))
            with open(dest_lab) as f:
                lab = [[int(num) for num in line.split()] for line in f]
            train_label.append(binarylab(lab))
            train_data_array=np.array(train_data)[:,:,img_rows_low:img_rows_high,img_cols_low:img_cols_high]
            train_label_array=np.array(train_label)[:,img_rows_low:img_rows_high,img_cols_low:img_cols_high,:]
        nb_data=train_data_array.shape[0]
        yield(train_data_array, np.reshape(train_label_array,(nb_data,data_shape,12)))
        f.close()

#Unpooling layer
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
        X = self.get_input(trai40,n)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}

#Encoding architecture
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

#Decoding architecture
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

#Model creation
print("------------CREATING NETWORK--------------")
network = models.Sequential()

# Add a noise layer to get a denoising network. This helps avoid overfitting
#network.add(Layer(input_shape=(3, 960, 720)))

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


print("------------TRAINING NETWORK--------------")
network.fit_generator(prep_data(),epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, class_weight=class_weight)
#history = network.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, class_weight=class_weighting )
#, validation_data=(X_test, X_test))
network.save_weights(save_model_name)


"""
print("------------DEPLOYING NETWORK--------------")

#Deployment variables
network.load_weights(run_model_name)
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

#Visualizing function
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
"""

"""
#Image analysis

import os
img = cv2.imread(os.getcwd() + '/SYNTHIA_RAND_CVPR16/RGB/ap_000_01-11-2015_19-20-57_000003_0_Rand_1.png')
img_prep = []
img = cv2.resize(img, (600, 600))
img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))
img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))
output = network.predict_proba(np.array(img_prep)[1:2])
pred = visualize(np.argmax(output[0],axis=1).reshape((600,600)))
cv2.imshow('Prediction', pred)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#Video playback analysis
"""
from moviepy.editor import VideoFileClip
def process_image(image):
    vid_img_prep = []
    vid_img = cv2.resize(image, (720, 960))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    output = network.predict_proba(np.array(vid_img_prep)[1:2])
    pred = visualize(np.argmax(output[0],axis=1).reshape((960,720)))
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
"""

#Live stream video analysis
"""
while(True):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    #cv2.imshow('frame', frame)
    vid_img_prep = []
    vid_img = cv2.resize(frame, (600, 600))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
    cap.release()
    output = network.predict_proba(np.array(vid_img_prep)[1:2])
    pred = visualize(np.argmax(output[0],axis=1).reshape((600,600)))
    cv2.imshow('Prediction', pred)
    cv2.imshow('Original', vid_img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
"""
