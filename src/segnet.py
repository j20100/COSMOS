#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import os
import time
import rospy
import sys
from sensor_msgs.msg import Image


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
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

#General functoins
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

class Segnet():
    #Class attribute
    """
    Tested configuration : 400x400:bs12, 600x600:bs5, 720x960:bs2
    """
    path = './SYNTHIA_RAND_CVPR16/'
    img_channels = 3
    img_original_rows=720
    img_original_cols=960
    img_rows = 600
    img_cols = 600
    epochs = 1
    batch_size = 2
    steps_per_epoch = 1
    frame = []

    #Model save variables
    save_model_name='model_ep100_bs3_st1000_res900_cw.hdf5'
    run_model_name='model_ep100_bs5_st1000_res600_cw.hdf5'


    #Class wieght for dataset
    class_pixel = [1.11849828e+08, 3.62664219e+08, 3.19578306e+09, 2.57847955e+09, 1.21284747e+09, 2.30973570e+07, 1.77853424e+08, 1.08091678e+08, 6.83247417e+08, 2.65943380e+07, 2.77407453e+08, 5.09002610e+08]
    dataset_nb = 13407
    class_freq = np.zeros(len(class_pixel))

    for i in range(len(class_pixel)):
        class_freq[i] = class_pixel[i] / (dataset_nb*img_original_rows*img_original_cols)

    class_weight = np.zeros(len(class_pixel))
    for i in range(len(class_pixel)):
        class_weight[i] = np.median(class_freq) / class_freq[i]

    #Dynamic variables
    img_rows_low = (img_original_rows-img_rows)/2
    img_rows_high = img_original_rows-(img_original_rows-img_rows)/2
    img_cols_low = (img_original_cols-img_cols)/2
    img_cols_high = img_original_cols-(img_original_cols-img_cols)/2
    data_shape = img_rows*img_cols

    #BGR
    void =	[0,0,0] #Black
    Sky = [255,255,255] # White
    Building = [0,0,255] # Red
    Road = [255,0,0] # Blue
    Sidewalk = [0,255,0] # Green
    Fence = [255,0,255] # Violet
    Vegetation = [255,255,0] # Yellow
    Pole = [0,255,255]
    Car = [128,0,64]
    Sign = [128,128,192]
    Pedestrian = [0,64,64]
    Cyclist = [192,128,0]

    label_colours = np.array([void, Sky, Building, Road, Sidewalk, Fence, Vegetation, Pole, Car, Sign, Pedestrian, Cyclist])

    network = models.Sequential()
    bridge = CvBridge()

    def __init__(self):
        pass


    #Binary labeling function
    def binarylab(self, labels):
        x = np.zeros([self.img_original_rows,self.img_original_cols,12])
        for i in range(self.img_original_rows):
            for j in range(self.img_original_cols):
                if labels[i][j] == -1:
                    labels[i][j] = 0
                x[i,j,labels[i][j]]=1
        return x

    #Calcul the weight of each class in the dataset
    def class_weighting(self):

        with open(self.path+'ALL.txt') as f:
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

            for i in range(self.img_original_rows):
                for j in range(self.img_original_cols):
                    y = lab[i][j]
                    label_weight[y] = label_weight[y]+1
        print(label_weight)
        return label_weight

    #Data generator for the synthia dataset
    def prep_data(self):
        while 1:
            with open(self.path+'ALL.txt') as f:
                txt = f.readlines()
                txt = [line.split(' ') for line in txt]
                train_data = []
                train_label = []

            for i in range(self.batch_size):
                index= random.randint(0, len(txt)-1)
                end_crop=len(txt[index][0])-4
                dest_lab = os.getcwd() + '/SYNTHIA_RAND_CVPR16/GTTXT/' + txt[index][0][:end_crop] + '.txt'
                train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + '/SYNTHIA_RAND_CVPR16/RGB/' + txt[index][0][:])),2))
                with open(dest_lab) as f:
                    lab = [[int(num) for num in line.split()] for line in f]
                train_label.append(binarylab(lab))
                train_data_array=np.array(train_data)[:,:,self.img_rows_low:self.img_rows_high,self.img_cols_low:self.img_cols_high]
                train_label_array=np.array(train_label)[:,self.img_rows_low:self.img_rows_high,self.img_cols_low:self.img_cols_high,:]
            nb_data=train_data_array.shape[0]
            yield(train_data_array, np.reshape(train_label_array,(nb_data,self.data_shape,12)))
            f.close()

    #Prep data for the camvid dataset
    """
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
    """

    #Prep for hybrid dataset
    """

    """

    #Encoding architecture
    def create_encoding_layers(self):
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2
        return [
            ZeroPadding2D(padding=(pad,pad), input_shape=(self.img_channels, self.img_rows, self.img_cols)),
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
    def create_decoding_layers(self):
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

    def create_network(self):
        #Model creation
        print("------------CREATING NETWORK--------------")
        # Add a noise layer to get a denoising network. This helps avoid overfitting
        #network.add(Layer(input_shape=(3, 960, 720)))

        #network.add(GaussianNoise(stddev=0.3))
        self.network.encoding_layers = self.create_encoding_layers()
        self.network.decoding_layers = self.create_decoding_layers()
        for l in self.network.encoding_layers:
            self.network.add(l)
        for l in self.network.decoding_layers:
            self.network.add(l)

        self.network.add(Conv2D(12, 1, padding='valid',))
        self.network.add(Reshape((12, self.data_shape)))
        self.network.add(Permute((2, 1)))
        self.network.add(Activation('softmax'))
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
        self.network.compile(loss="categorical_crossentropy", optimizer=optimizer)

    def train_network(self):
        print("------------TRAINING NETWORK--------------")
        self.network.fit_generator(self.prep_data(),epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, verbose=1, class_weight=self.class_weight)
        #history = network.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, class_weight=class_weighting )
        #, validation_data=(X_test, X_test))
        self.network.save_weights(self.save_model_name)

    def deploy_network(self):

        print("------------DEPLOYING NETWORK--------------")

        #Deployment variables
        self.network.load_weights(self.run_model_name)

    #Visualizing function
    def visualize(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(len(self.label_colours)):
            r[temp==l]=self.label_colours[l,0]
            g[temp==l]=self.label_colours[l,1]
            b[temp==l]=self.label_colours[l,2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:,:,0] = (r)#[:,:,0]
        rgb[:,:,1] = (g)#[:,:,1]
        rgb[:,:,2] = (b)#[:,:,2]
        return rgb

    def image_analysis(self):
        #Image analysis
        import os
        img = cv2.imread(os.getcwd() + '/test.png')
        print(os.getcwd() + '/test.png')
        img_prep = []
        cv2.imshow('Prediction', img)
        img = cv2.resize(img, (self.img_rows,self.img_cols))
        img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))
        img_prep.append(normalized(img).swapaxes(0,2).swapaxes(1,2))
        output = self.network.predict_proba(np.array(img_prep)[1:2])
        pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows,self.img_cols)))
        cv2.imshow('Prediction', pred)
        cv2.imshow('Original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def process_image(self, image):
        vid_img_prep = []
        vid_img = cv2.resize(image, (self.img_rows,self.img_cols))
        vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
        vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
        output = self.network.predict_proba(np.array(vid_img_prep)[1:2])
        pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows,self.img_cols)))
        return pred

    def invert_red_blue(image):
        return image[:,:,[2,1,0]]

    def video_analysis(self):
        #Video playback analysis
        video = VideoFileClip("01TP_extract.avi")
        video = video.fl_image(invert_red_blue)
        pred_video = video.fl_image(process_image)
        pred_video.write_videofile('pred_video.avi', codec='rawvideo', audio=False)

    def live_analysis(self):
        #Live stream video analysis
        while(True):
            vid_img_prep = []
            vid_img = cv2.resize(self.frame, (self.img_rows,self.img_cols))
            vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
            vid_img_prep.append(normalized(vid_img).swapaxes(0,2).swapaxes(1,2))
            output = self.network.predict_proba(np.array(vid_img_prep)[1:2])
            pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows,self.img_cols)))
            cv2.imshow('Prediction', pred)
            cv2.imshow('Original', vid_img)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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





if __name__ == '__main__':
    arg = sys.argv

    sn = Segnet()
    rospy.init_node('segmentation_network_node', anonymous=True)
    rospy.Subscriber("/cv_camera/image_raw", Image, sn.image_callback)

    sn.create_network()
    sn.deploy_network()
    sn.live_analysis()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down artificial neural network")
