#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import os
import time
import rospy
import sys
from sensor_msgs.msg import Image, CompressedImage


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
import scipy.io as sio
import glob
np.random.seed(1337) # for reproducibility

import keras
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
from skimage import color, exposure, transform

np.set_printoptions(threshold=np.nan)
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
    ros_path = '/home/jvincent/ros_ws/src/COSMOS/src/'
    weights_path = '/home/jvincent/ros_ws/src/COSMOS/src/weight/'
    path = '/home/jvincent/ros_ws/src/COSMOS/src/CamVid/'
    path_cityscape = '/home/jvincent/cityscape/'
    path_annotator = '/home/jvincent/Seg_Annotator/static/data/'
    img_channels = 3
    img_original_rows=1024
    img_original_cols=2048
    img_rows = 360
    img_cols = 480
    epochs = 10
    batch_size = 6
    steps_per_epoch = 600
    nb_class = 20
    nb_dim = 3
    frame = []
    start = 0

    #Model save variables
    save_model_name= weights_path + 'cityscape_1.hdf5'
    run_model_name= weights_path + 'weights.00-1.53.hdf5'
    load_model_name= weights_path + 'weights.00-1.53.hdf5'



    #Class wieght for dataset
    class_pixel = [1.11849828e+08, 3.62664219e+08, 3.19578306e+09, 2.57847955e+09, 1.21284747e+09, 2.30973570e+07, 1.77853424e+08, 1.08091678e+08, 6.83247417e+08, 2.65943380e+07, 2.77407453e+08, 5.09002610e+08]
    dataset_nb = 13407
    class_freq = np.zeros(len(class_pixel))

    for i in range(len(class_pixel)):
        class_freq[i] = class_pixel[i] / (dataset_nb*img_original_rows*img_original_cols)

    class_weight = np.zeros(len(class_pixel))
    for i in range(len(class_pixel)):
        class_weight[i] = np.median(class_freq) / class_freq[i]

    #class_weighting_camvid= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826,
    #9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    class_weighting_camvid= [0.01, 0.2595, 0.1826, 0.1417, 0.3826, 6.6823, 9.6446, 4.5640, 6.2478, 1.8418, 3.0, 7.3614]
    class_weighting_courge = [0.01, 10, 0.01]

    #Dynamic variables
    img_rows_low = (img_original_rows-img_rows)/2
    img_rows_high = img_original_rows-(img_original_rows-img_rows)/2
    img_cols_low = (img_original_cols-img_cols)/2
    img_cols_high = img_original_cols-(img_original_cols-img_cols)/2
    data_shape = img_rows*img_cols

    #BGR
    #void =	[0,0,0] #Black 0
    #Sky = [255,255,255] # White 1
    #Building = [0,0,255] # Red 2
    #Road = [255,0,0] # Blue 3
    #Sidewalk = [0,255,0] # Green 4
    #Fence = [255,0,255] # Violet 5
    #Vegetation = [255,255,0] # Yellow 6
    #Pole = [0,255,255] #7
    #Car = [128,0,64] #8
    #Sign = [128,128,192] #9
    #Pedestrian = [0,64,64] #10
    #Cyclist = [192,128,0] #11

    #BGR
    #void =	[0,0,0] #Black 0
    #Sky = [255,255,255] # White 1
    #Building = [0,0,0] # Red 2
    #Road = [0,0,0] # Blue 3
    #Sidewalk = [0,255,0] # Green 4
    #Fence = [0,0,0] # Violet 5
    #Vegetation = [0,0,0] # Yellow 6
    #Pole = [0,0,0] #7
    #Car = [0,0,0] #8
    #Sign = [0,0,0] #9
    #Pedestrian = [0,0,0] #10
    #Cyclist = [0,0,0] #11

    #label_colours = np.array([Fence, Road, Building])
    #label_colours = np.array([void, Sky, Building, Road, Sidewalk, Fence, Vegetation, Pole, Car, Sign, Pedestrian, Cyclist])

    #cityscape dataset
    road = [128,64,128]
    sidewalk = [244,35,232]
    building = [70,70,70]
    wall = [102,102,156]
    fence = [190,153,153]
    pole = [153,153,153]
    trafficlight = [250, 170,30]
    trafficsign = [220,220,0]
    vegetation = [107,142,35]
    terrain = [152,251,152]
    sky = [70,130,180]
    person = [220,20,60]
    rider = [255,0,0]
    car = [0,0,142]
    truck = [0,0,70]
    bus = [0,60,100]
    train = [0,80,100]
    motorcycle = [0,0,230]
    bicycle = [119,11,32]
    void = [0,0,0]

    label_colours = np.array([road, sidewalk, building, wall, fence, pole, \
        trafficlight, trafficsign, vegetation, terrain, sky, person, rider, \
        car, truck, bus, train, motorcycle, bicycle])


    network = models.Sequential()
    bridge = CvBridge()

    def __init__(self):
        self.image_pub = rospy.Publisher("image_seg", Image)

    def resize_input_data(self, input_img):
        x = np.zeros([self.nb_dim, self.img_rows, self.img_cols])
        for i in range(input_img.shape[0]):
            x[i,:,:] = cv2.resize(input_img[i,:,:], (self.img_cols,self.img_rows))
        return x

    #Binary labeling function
    def binarylab(self, labels):
        x = np.zeros([self.img_original_rows, self.img_original_cols, self.nb_class])
        for i in range(self.img_original_rows):
            for j in range(self.img_original_cols):
                #3 dim labels in cityscape dataset
                if labels[i][j][0] > 18:
                    x[i, j, 19] = 1
                else:
                    x[i, j, labels[i][j][0]] = 1
        return x

    def preprocess_img(self, img):
        # Histogram normalization in v channel
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        # rescale to standard size
        img = transform.resize(img, (self.img_rows,self.img_cols))

        return img

    def resize_input_binary_label(self, input_img):
        x = np.zeros([self.img_rows, self.img_cols, self.nb_class])
        for i in range(input_img.shape[2]):
            buff = input_img[:,:,i]
            x[:,:,i] = np.ceil(cv2.resize(buff, (self.img_cols,self.img_rows)))
        return x

    def change_class_id_camvid(self, input_class_id):

        #Camvid class
        #Sky = [128,128,128]
        #Building = [128,0,0]
        #Pole = [192,192,128]
        #Road_marking = [255,69,0]
        #Road = [128,64,128]
        #Pavement = [60,40,222]
        #Tree = [128,128,0]
        #SignSymbol = [192,128,128]
        #Fence = [64,64,128]
        #Car = [64,0,128]
        #Pedestrian = [64,64,0]
        #Bicyclist = [0,128,192]
        #Unlabelled = [0,0,0]

        #Sky
        if input_class_id == 0:
            output_class_id = 1
        #Building
        elif input_class_id == 1:
            output_class_id = 2
        #Pole
        elif input_class_id == 2:
            output_class_id = 7
        #Sign
        elif input_class_id == 3:
            output_class_id = 9
        #Road
        elif input_class_id == 4:
            output_class_id = 3
        #Sidewalk
        elif input_class_id == 5:
            output_class_id = 4
        #Vegetation
        elif input_class_id == 6:
            output_class_id = 6
        #Sign_symbol
        elif input_class_id == 7:
            output_class_id = 9
        #Fence
        elif input_class_id == 8:
            output_class_id = 5
        #Car
        elif input_class_id == 9:
            output_class_id = 8
        #Pedestrian
        elif input_class_id == 10:
            output_class_id = 10
        #Cyclist
        elif input_class_id == 11:
            output_class_id = 11
        #Void
        else:
            output_class_id = 0

        return output_class_id



    #Calcul the weight of each class in the dataset
    def class_weighting(self):

        with open(self.path+'ALL.txt') as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]
        dataset = []
        label_weight = np.zeros([self.nb_class])
        print(label_weight)
        y = 0

        for i in range (len(txt)):
            print(i)
            end_crop=len(txt[i][0])-4
            dest_lab = '/home/jvincent/SYNTHIA_RAND_CVPR16/GTTXT/' + \
                txt[i][0][:end_crop] + '.txt'

            with open(dest_lab) as f:
                lab = [[int(num) for num in line.split()] for line in f]

            for i in range(self.img_original_rows):
                for j in range(self.img_original_cols):
                    y = lab[i][j]
                    label_weight[y] = label_weight[y]+1
        print(label_weight)
        return label_weight

    #Data generator for the cityscape_ dataset
    def prep_data_cityscape(self):

        while 1:
            searchlabel = os.path.join( self.path_cityscape , "*" , "train" , "*"\
                , "*_labelTrainIds.png" )
            fileslabel = glob.glob(searchlabel)
            fileslabel.sort()

            train_data = []
            train_label = []

            for i in range(self.batch_size):
                index= random.randint(0, len(fileslabel)-1)
                t = fileslabel[index].split('/')
                data = os.path.join( self.path_cityscape , "leftImg8bit" , "train" \
                    , t[6] , t[7][0:(len(t[6])+15)]+"leftImg8bit.png" )
                train_data.append(np.rollaxis(self.preprocess_img\
                    (cv2.imread(data)),2))

                train_label.append(self.resize_input_binary_label(self.binarylab\
                    (cv2.imread(fileslabel[index]))))
                #train_data_array=np.array(train_data)
                #train_label_array=np.array(train_label)

            #nb_data=train_data_array.shape[0]
            yield(np.array(train_data), np.reshape(np.array(train_label),\
                (self.batch_size,self.data_shape,self.nb_class)))

    def prep_val_cityscape(self):

        while 1:
            searchlabel = os.path.join( self.path_cityscape , "*" , "val" , "*"\
                , "*_gtCoarse_labelTrainIds.png" )
            fileslabel = glob.glob(searchlabel)
            fileslabel.sort()

            train_data = []
            train_label = []

            for i in range(self.batch_size):
                index= random.randint(0, len(fileslabel)-1)
                t = fileslabel[index].split('/')
                data = os.path.join( self.path_cityscape , "leftImg8bit" , "val"\
                    , t[6] , t[7][0:(len(t[6])+15)]+"leftImg8bit.png" )
                train_data.append(np.rollaxis(self.preprocess_img\
                    (cv2.imread(data)),2))
                train_label.append(self.resize_input_binary_label(self.binarylab\
                    (cv2.imread(fileslabel[index]))))
                #train_data_array=np.array(train_data)
                #train_label_array=np.array(train_label)

            #nb_data=train_data_array.shape[0]
            yield(np.array(train_data), np.reshape(np.array(train_label),\
                (self.batch_size,self.data_shape,self.nb_class)))


    #Data generator for the synthia dataset
    def prep_data_synthia(self):
        while 1:
            with open(self.path+'ALL.txt') as f:
                txt = f.readlines()
                txt = [line.split(' ') for line in txt]
                train_data = []
                train_label = []

            for i in range(self.batch_size):
                index= random.randint(0, len(txt)-1)
                end_crop=len(txt[index][0])-4
                dest_lab = '/home/jvincent/SYNTHIA_RAND_CVPR16/GTTXT/' + \
                    txt[index][0][:end_crop] + '.txt'
                train_data.append(np.rollaxis(normalized(cv2.imread\
                    ('/home/jvincent/SYNTHIA_RAND_CVPR16/RGB/' + txt[index][0][:])),2))
                with open(dest_lab) as f:
                    lab = [[int(num) for num in line.split()] for line in f]
                train_label.append(self.binarylab(lab))
                train_data_array=np.array(train_data)[:,:,self.img_rows_low:\
                    self.img_rows_high,self.img_cols_low:self.img_cols_high]
                train_label_array=np.array(train_label)[:,self.img_rows_low:\
                    self.img_rows_high,self.img_cols_low:self.img_cols_high,:]
            nb_data=train_data_array.shape[0]
            yield(train_data_array, np.reshape(train_label_array,\
                (nb_data,self.data_shape,self.nb_class)))
            f.close()

    #Prep data for the camvid dataset
    def prep_data_camvid(self):
        while 1:
            with open(self.path+'train.txt') as f:
                txt = f.readlines()
                txt = [line.split(' ') for line in txt]
                train_data = []
                train_label = []
            for i in range(self.batch_size):
                index= random.randint(0, len(txt)-1)
                train_data.append(self.resize_input_data(np.rollaxis(\
                    cv2.imread('/home/jvincent/ros_ws/src/COSMOS/src/' \
                        + txt[index][0][7:]),2)))
                train_label.append(self.resize_input_binary_label(\
                    self.binarylab(cv2.imread('/home/jvincent/ros_ws/src/COSMOS/src/' \
                        + txt[index][1][7:][:-1])[:,:,0])))

            yield(np.array(train_data), np.reshape(np.array(train_label),\
                (self.batch_size,self.data_shape,self.nb_class)))
            f.close()

    def prep_val_camvid(self):
        while 1:
            with open(self.path+'val.txt') as f:
                txt = f.readlines()
                txt = [line.split(' ') for line in txt]
                val_data = []
                val_label = []
            for i in range(self.batch_size):
                index= random.randint(0, len(txt)-1)
                val_data.append(self.resize_input_data(np.rollaxis(cv2.imread('/home/jvincent/ros_ws/src/COSMOS/src' + txt[index][0][7:]),2)))
                val_label.append(self.resize_input_binary_label(self.binarylab(cv2.imread('/home/jvincent/ros_ws/src/COSMOS/src' + txt[index][1][7:][:-1])[:,:,0])))

            yield(np.array(val_data), np.reshape(np.array(val_label),(self.batch_size,self.data_shape,self.nb_class)))
            f.close()

    #Prep data for the nuy dataset
    def prep_data_nyu(self):
        while 1:
            data = sio.loadmat('/home/jvincent/ros_ws/src/COSMOS/src/' + 'nyu_dataset.mat')
            labels = data['labels']
            images = np.rollaxis(data['images'],2)
            train_data = []
            train_label = []

            for i in range(self.batch_size):
                index = random.randint(0, images.shape[3]-1)

                train_data.append(np.rollaxis(np.rollaxis(images[:,:,:,index],2),2))
                train_label.append(self.binarylab(labels[:,:,index]))
            yield(np.rollaxis(np.rollaxis(np.array(train_data),3),1), np.reshape(np.array(train_label),(self.batch_size,self.data_shape,self.nb_class)))


    #Prep for segannotator dataset
    def prep_data_annotator(self):
        while 1:
            with open(self.path_annotator+'test.txt') as f:
                txt = f.readlines()
                txt = [line.rstrip() for line in txt]
                train_data = []
                train_label = []
                annot_image = 'none'
            for i in range(self.batch_size):
                index= random.randint(0, len(txt)-1)
                train_data.append(self.resize_input_data(np.rollaxis(cv2.imread(self.path_annotator + 'images/' + txt[index]),2)))

                directory = self.path_annotator + 'annotations/'
                test = os.listdir( directory )
                for item in test:
                    if item.startswith(txt[index]):
                        annot_image = item
                train_label.append(self.resize_input_binary_label(self.binarylab(cv2.imread(self.path_annotator + 'annotations/' + annot_image)[:,:,2])))

            yield(np.array(train_data), np.reshape(np.array(train_label),(self.batch_size,self.data_shape,self.nb_class)))
            f.close()


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

        self.network.add(Conv2D(self.nb_class, 1, padding='valid',))
        self.network.add(Reshape((self.nb_class, self.data_shape)))
        self.network.add(Permute((2, 1)))
        self.network.add(Activation('softmax'))
        from keras.optimizers import SGD, Adam
        #optimizer = SGD(lr=0.01, momentum=0.8, decay=0.1, nesterov=False)
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.network.compile(loss="categorical_crossentropy", optimizer=optimizer)

    def train_network(self):
        print("------------TRAINING NETWORK--------------")

        #Initialise tensorboard
        #tbcallback = keras.callbacks.TensorBoard(log_dir='./logs', \
        #    histogram_freq=1, write_graph=True, \
        #    write_images=True)
        #self.network.load_weights(self.load_model_name)

        logcb = keras.callbacks.ModelCheckpoint(\
        "/home/jvincent/ros_ws/src/COSMOS/src/weight/weights.{epoch:02d}-{val_loss:.2f}.hdf5", \
            monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, \
            mode='auto', period=1)

        escb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, \
            patience=3, verbose=1, mode='auto')

        history = self.network.fit_generator(self.prep_data_cityscape(), \
        epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, \
        validation_data=self.prep_val_cityscape(), validation_steps=100, \
        verbose=1, callbacks=[logcb, escb])
        #, validation_data=self.prep_val_camvid(), validation_steps=10, class_weight=self.class_weighting_camvid)
        #history = network.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, verbose=1, class_weight=class_weighting )
        #, validation_data=(X_test, X_test))
        self.network.save_weights(self.save_model_name)
        print(history.history.keys())
        # summarize history for accuracy
        #plt.plot(history.history['acc'])
        #plt.plot(history.history['val_acc'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def deploy_network(self):

        print("------------DEPLOYING NETWORK--------------")

        #Deployment variables
        d_path = self.run_model_name
        self.network.load_weights(d_path)

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
        return rgb.astype('uint8')

    def visualize_colormap(self, temp):

        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        nb_colors=255^3
        i=0
        j=0
        k=0
        inc=28
        self.color_list=np.zeros((self.nb_class,3))
        print(self.color_list.shape)
        for x in range(self.nb_class):
            self.color_list[x]=[i,j,k]
            k=k+inc
            if k >= 255:
                j=j+inc
                k=0
            if j >= 255:
                i=i+inc
                j=0
        for l in range(self.nb_class):
            r[temp==l]=self.color_list[l,0]
            g[temp==l]=self.color_list[l,1]
            b[temp==l]=self.color_list[l,2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:,:,0] = (r)#[:,:,0]
        rgb[:,:,1] = (g)#[:,:,1]
        rgb[:,:,2] = (b)#[:,:,2]
        return rgb

    def image_analysis(self):
        #Image analysis
        import os

        #data = sio.loadmat('/home/jvincent/ros_ws/src/COSMOS/src/' +'nyu_dataset.mat')
        #labels = data['labels']
        #images = np.rollaxis(data['images'],2)
        #img_label = self.visualize(labels[:,:,55])
        #cv2.imshow('dsa', img_label)
        #img=np.rollaxis(np.rollaxis(images[:,:,:,55],2),2)
        #print(images_data)
        #cv2.imshow('Originali', images_data)

        img = cv2.imread(os.getcwd() + '/ros_ws/src/COSMOS/src/' + 'test.png')
        img_prep = []
        print(img.shape)
        #img = img[:,:,[2,0,1]]
        img = cv2.resize(img, (self.img_cols, self.img_rows))
        img_prep.append(img.swapaxes(0,2).swapaxes(1,2))
        img_prep.append(img.swapaxes(0,2).swapaxes(1,2))
        output = self.network.predict_proba(np.array(img_prep)[1:2])
        pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows, self.img_cols)))
        cv2.imshow('Input', self.preprocess_img(img))
        cv2.imshow('Prediction', pred)
        cv2.imshow('Original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def process_image(self, image):
        vid_img_prep = []
        vid_img = cv2.resize(image, (self.img_rows,self.img_cols))
        vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
        vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
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
            vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
            vid_img_prep.append(vid_img.swapaxes(0,2).swapaxes(1,2))
            output = self.network.predict_proba(np.array(vid_img_prep)[1:2])
            pred = self.visualize(np.argmax(output[0],axis=1).reshape((self.img_rows,self.img_cols)))
            #cv2.imshow('Prediction', pred)
            #cv2.imshow('Original', vid_img)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(pred))
            #cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        #cv2.destroyAllWindows()

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
        X = self.get_input(train)
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
    rospy.init_node('segmentation_network_node_run', anonymous=True)

    #rospy.Subscriber("/center_camera/image_color/raw", Image, sn.image_callback)
    #rospy.Subscriber("/stereo_camera/left/image_rect_color", Image, sn.image_callback)


    sn.create_network()
    #sn.train_network()
    sn.deploy_network()
    sn.image_analysis()
    #sn.live_analysis()
    try:
        rospy.spin()
    except KeyboardInterrupt:image_analysis
    print("Shutting down artificial neural network")
