#!/usr/bin/env python
#from __future__ import absolute_import
from __future__ import print_function
import os
import time
import sys

# Comment to use tensorflow
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'
#import tensorflow as tf

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
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D

from keras.layers.advanced_activations import ELU
from keras.models import Model

from keras import backend as K
K.set_image_dim_ordering('tf')

import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
#from moviepy.editor import VideoFileClip
from skimage import color, exposure, transform


###ADD mask rcnn
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('/home/jvincent/ros_ws/src/COSMOS/src/')
import coco
import utils
import model as modellib
import visualize_cv
import std_msgs.msg
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.set_printoptions(threshold=np.nan)

# Root directory of the project
#ROOT_DIR = os.getcwd()
DEFAULT_DIR = "/home/jvincent/"
ROOT_DIR = "/home/jvincent/ros_ws/src/COSMOS/src/"
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
path_tum = os.path.join(ROOT_DIR, "tum_dataset/")
path_kitti = os.path.join(ROOT_DIR, "kitti_dataset/")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRcnn():

    def __init__(self):

        self.config = InferenceConfig()
        self.config.display()

        #self.image_pub = rospy.Publisher("image_seg_maskrcnn", Image,queue_size=1)
        #self.depth_image_pub = rospy.Publisher("depth_image_seg_maskrcnn", Image,queue_size=1)

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        path_tum = os.path.join(ROOT_DIR, "tum_dataset/")
        self.frame = []
        self.depth_frame = []
        self.msg_header = std_msgs.msg.Header()
        self.depth_msg_header = std_msgs.msg.Header()

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

        self.bridge = CvBridge()

        #image = skimage.io.imread("/home/pmcrivet/catkin_ws/src/Mask_RCNN/script/images/frame0057.jpg")

            #img = numpy.asarray(frame,dtype='uint8')

            #cv2.imshow("image",image[...,::-1])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()



        #results = self.model.detect([image], verbose=1)
        #r = results[0]
            #visualize_cv.display_instances(frame, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
        #visualize_cv.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                    self.class_names, r['scores'])


    def live_analysis(self):
        #Live stream video analysis

        while not rospy.is_shutdown():

            #print("Looking for a frame")

            current_frame = self.frame

            if current_frame != []:
                #print("got frame")
                #image = cv2.imread("/home/pmcrivet/catkin_ws/src/Mask_RCNN/script/images/frame0057.jpg")

                #img = numpy.asarray(frame,dtype='uint8')

                #cv2.imshow("image",image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #print image.shape
                #print image.dtype



                results = self.model.detect([current_frame], verbose=1)
                r = results[0]
                result_image = visualize_cv.cv_img_masked(current_frame, r['rois'], r['masks'], r['class_ids'],
                                                         self.class_names, r['scores'])

                cv2.namedWindow('result_image', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('result_image', 1920, 1080)
                cv2.imshow("result_image", result_image)

                print("Did something")


                #results = self.model.detect([self.frame], verbose=1)
                #r = results[0]
                #visualize_cv.display_instances(image, r['rois'], r['masks'], r['class_ids'],self.class_names, r['scores'])


                ITS = Image()
                ITS = self.bridge.cv2_to_imgmsg(result_image,'bgr8')
                ITS.header = self.msg_header
                self.image_pub.publish(ITS)
                print("Publishing img")
            else:
                print("Waiting for frame")

    def class_selection(self, masks, class_ids):
        x = np.zeros([masks.shape[0], masks.shape[1], class_ids.shape[0]])
        for l in range(class_ids.shape[0]):
            if class_ids[l] == 2 or class_ids[l] == 3 or class_ids[l] == 4 or class_ids[l] == 6 or class_ids[l] == 8 or class_ids[l] == 1:
            #if class_ids[l] == 2:
                x[:, :, l] = masks[:, :, l]
            else:
                x[:, :, l] = 0

        return x

    def live_depth_analysis(self):
        #Live stream video analysis

        while not rospy.is_shutdown():

            #print("Looking for a frame")

            current_frame = self.frame
            current_depth_frame = self.depth_frame

            if (current_frame != [] and current_depth_frame != []):
                #print("got frame")
                #image = cv2.imread("/home/pmcrivet/catkin_ws/src/Mask_RCNN/script/images/frame0057.jpg")

                #img = numpy.asarray(frame,dtype='uint8')

                #cv2.imshow("image",image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #print image.shape
                #print image.dtype

                #Dilatation effect on masks for better feature masking
                dilatation = 20
                threshold = 0.3

                results = self.model.detect([current_frame], dilatation, threshold, verbose=1)
                r = results[0]

                #selected_class = self.class_selection(r['masks'], r['class_ids'])
                selected_class = r['masks']


                result_image = visualize_cv.cv_img_masked(current_frame, r['rois'], selected_class, r['class_ids'],
                                                         self.class_names, r['scores'])
                result_depth_image = visualize_cv.cv_depth_img_masked(current_depth_frame, r['rois'], selected_class, r['class_ids'],
                                                         self.class_names, r['scores'])

                DITS = Image()
                ITS = Image()
                DITS = self.bridge.cv2_to_imgmsg(result_depth_image,'16UC1')
                ITS = self.bridge.cv2_to_imgmsg(result_image,'bgr8')
                DITS.header = self.depth_msg_header
                ITS.header = self.msg_header
                self.image_pub.publish(ITS)
                self.depth_image_pub.publish(DITS)
                print("Publishing img")
            else:
                print("Waiting for frame")

    def tum_dataset_analysis(self):
        searchlabel = os.path.join(path_tum , "rgb_sync" , "*.png" )
        fileslabel = glob.glob(searchlabel)
        fileslabel.sort()

        searchanot = os.path.join(path_tum , "depth_sync" , "*.png" )
        filesanot = glob.glob(searchanot)
        filesanot.sort()

        dilatation = 20
        threshold = 0.3

        for i in range(len(filesanot)):

            img = cv2.imread(fileslabel[i])
            #depth_img = cv2.cvtColor(cv2.imread(filesanot[i]), cv2.COLOR_BGR2GRAY)
            depth_img = cv2.imread(filesanot[i], -1)
            depth_img = depth_img.astype(np.uint16)

            results = self.model.detect([img], dilatation, threshold, verbose=1)
            r = results[0]

            selected_class = self.class_selection(r['masks'], r['class_ids'])
            #selected_class = r['masks']

            result_image = visualize_cv.cv_img_masked(img, r['rois'], selected_class, r['class_ids'],
                                                     self.class_names, r['scores'])
            result_depth_image = visualize_cv.cv_depth_img_masked(depth_img, r['rois'], selected_class, r['class_ids'],
                                                     self.class_names, r['scores'])

            cv2.imwrite(filesanot[i],result_depth_image)
        print("Batch done")


    def kitti_dataset_analysis(self):
        searchlabel = os.path.join(path_kitti , "image_*" , "*.png" )
        fileslabel = glob.glob(searchlabel)
        fileslabel.sort()

        dilatation = 10
        threshold = 0.1

        for i in range(len(fileslabel)):

            img = cv2.imread(fileslabel[i])

            img = cv2.imread(fileslabel[i], -1)
            img = img.astype(np.uint16)
            img = cv2.resize(img, (1024, 320))

            results = self.model.detect([img], dilatation, threshold, verbose=1)
            r = results[0]

            selected_class = self.class_selection(r['masks'], r['class_ids'])
            #selected_class = r['masks']

            result_image = visualize_cv.cv_img_masked(img, r['rois'], selected_class, r['class_ids'],
                                                     self.class_names, r['scores'])


            cv2.imwrite(fileslabel[i],result_image.astype('uint8'))
        print("Batch done")


    def kitti_live_analysis(self):

        search_img_rgb = os.path.join(DEFAULT_DIR , "image.png" )
        search_img_depth = os.path.join(DEFAULT_DIR , "depth.png" )
        path_depth_masked = os.path.join(DEFAULT_DIR , "depth_mask.png" )

        dilatation = 5
        threshold = 0.1


        img = cv2.imread(search_img_rgb, -1)
        img_depth = cv2.imread(search_img_depth, -1)

        img = img.astype(np.uint16)
        img_depth = img_depth.astype(np.uint16)

        img = cv2.resize(img, (1024, 320))
        img_depth = cv2.resize(img_depth, (1024, 320))

        results = self.model.detect([img], dilatation, threshold, verbose=1)
        r = results[0]

        selected_class = self.class_selection(r['masks'], r['class_ids'])
        #selected_class = r['masks']

        result_depth_image = visualize_cv.cv_depth_img_masked(img_depth, r['rois'], selected_class, r['class_ids'],
                                                 self.class_names, r['scores'])

        result_depth_image = cv2.resize(result_depth_image, (1226, 370))
        cv2.imwrite(path_depth_masked,result_depth_image.astype('uint8'))
        print("Depth mask done")


    def image_callback(self, msg):

        self.msg_header = msg.header
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #self.live_analysis(self.frame)

    def depth_image_callback(self, msg):

        self.depth_msg_header = msg.header
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1")

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

if __name__ == '__main__':
    arg = sys.argv

    mr = MaskRcnn()

    #rospy.init_node('segmentation_network_node_run', anonymous=True)
    #rospy.Subscriber("/camera/rgb/image_color", Image, mr.image_callback)
    #rospy.Subscriber("/camera/depth_registered/image_raw", Image, mr.depth_image_callback)

    #rospy.Subscriber("/stereo_camera/left/image_rect_color", Image, sn.image_callback)
    while(1):
        searchlabel = os.path.join("/home/jvincent/1.txt")
        fileslabel = glob.glob(searchlabel)

        if fileslabel != []:
            mr.kitti_live_analysis()
            os.rename("/home/jvincent/1.txt", "/home/jvincent/2.txt")

    #mr.live_analysis()
    #mr.live_depth_analysis()
    #mr.tum_dataset_analysis()
    #mr.kitti_dataset_analysis()

    #sn.create_segnet()
    #sn.train_network()
    #sn.deploy_network()
    #sn.image_batch()
    #sn.image_analysis()
    #sn.live_analysis()
    #try:
    #    rospy.spin()
    #except KeyboardInterrupt:image_analysis
    print("Shutting down artificial neural network")
