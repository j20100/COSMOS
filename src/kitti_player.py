#!/usr/bin/env python
from rtabmap_ros.msg import UserData
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import rosbag
from std_msgs.msg import Int32, String
import std_msgs.msg
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import glob
import numpy as np
import yaml

bridge = CvBridge()

def image_pub():
    rate = rospy.Rate(10)
    right_yaml_file = os.path.join("kitti_dataset" , "rtabmap_calib04_right.yaml" )

    left_yaml_file = os.path.join("kitti_dataset" , "rtabmap_calib04_left.yaml" )

    with open(left_yaml_file, "r") as file_handle:
        calib_data_left = yaml.load(file_handle)

    with open(right_yaml_file, "r") as file_handle:
        calib_data_right = yaml.load(file_handle)

    # Parse
    right_camera_info_msg = CameraInfo()
    right_camera_info_msg.width = calib_data_right["image_width"]
    right_camera_info_msg.height = calib_data_right["image_height"]
    right_camera_info_msg.K = calib_data_right["camera_matrix"]["data"]
    #right_camera_info_msg.D = calib_data_right["distortion_coefficients"]["data"]
    #right_camera_info_msg.R = calib_data_right["rectification_matrix"]["data"]
    right_camera_info_msg.P = calib_data_right["projection_matrix"]["data"]
    #right_camera_info_msg.distortion_model = calib_data_right["distortion_model"]

    left_camera_info_msg = CameraInfo()
    left_camera_info_msg.width = calib_data_left["image_width"]
    left_camera_info_msg.height = calib_data_left["image_height"]
    left_camera_info_msg.K = calib_data_left["camera_matrix"]["data"]
    #left_camera_info_msg.D = calib_data_left["distortion_coefficients"]["data"]
    #left_camera_info_msg.R = calib_data_left["rectification_matrix"]["data"]
    left_camera_info_msg.P = calib_data_left["projection_matrix"]["data"]
    #left_camera_info_msg.distortion_model = calib_data_left["distortion_model"]

    searchlabel = os.path.join("kitti_dataset" , "image_2" , "*.png" )
    fileslabel = glob.glob(searchlabel)
    fileslabel.sort()
    searchlabelr = os.path.join("kitti_dataset" , "image_3" , "*.png" )
    fileslabelr = glob.glob(searchlabelr)
    fileslabelr.sort()

    bag = rosbag.Bag('test.bag', 'w')

    for i in range(len(fileslabel)):
        image_header = std_msgs.msg.Header()
        time_path = os.path.join("kitti_dataset")
        with open(time_path+'/times.txt') as f:
            txt = f.readlines()

        imgl = cv2.imread(fileslabel[i], -1)

        imgr = cv2.imread(fileslabelr[i], -1)

        imgr_msg = Image()
        imgl_msg = Image()
        t = rospy.Time.now()
        image_header.stamp = t
        image_header.stamp = t

        imgl_msg = bridge.cv2_to_imgmsg(imgl,'bgr8')
        imgr_msg = bridge.cv2_to_imgmsg(imgr,'bgr8')
        imgr_msg.header = image_header
        imgl_msg.header = image_header
        right_camera_info_msg.header = image_header
        left_camera_info_msg.header = image_header
        #imgl_msg.data = IL
        #imgr_msg.data = IR

        bag.write("right/camera_info", right_camera_info_msg)
        bag.write("left/camera_info", left_camera_info_msg)
        bag.write("right/image_raw", imgr_msg)
        bag.write("left/image_raw", imgl_msg)
        rate.sleep()


    print("bagdone")
    bag.close()

def image_bag():
    rate = rospy.Rate(10)
    right_yaml_file = os.path.join("kitti_dataset" , "rtabmap_calib04_right.yaml" )

    left_yaml_file = os.path.join("kitti_dataset" , "rtabmap_calib04_left.yaml" )

    with open(left_yaml_file, "r") as file_handle:
        calib_data_left = yaml.load(file_handle)

    with open(right_yaml_file, "r") as file_handle:
        calib_data_right = yaml.load(file_handle)

    # Parse
    right_camera_info_msg = CameraInfo()
    right_camera_info_msg.width = calib_data_right["image_width"]
    right_camera_info_msg.height = calib_data_right["image_height"]
    right_camera_info_msg.K = calib_data_right["camera_matrix"]["data"]
    #right_camera_info_msg.D = calib_data_right["distortion_coefficients"]["data"]
    #right_camera_info_msg.R = calib_data_right["rectification_matrix"]["data"]
    right_camera_info_msg.P = calib_data_right["projection_matrix"]["data"]
    #right_camera_info_msg.distortion_model = calib_data_right["distortion_model"]

    left_camera_info_msg = CameraInfo()
    left_camera_info_msg.width = calib_data_left["image_width"]
    left_camera_info_msg.height = calib_data_left["image_height"]
    left_camera_info_msg.K = calib_data_left["camera_matrix"]["data"]
    #left_camera_info_msg.D = calib_data_left["distortion_coefficients"]["data"]
    #left_camera_info_msg.R = calib_data_left["rectification_matrix"]["data"]
    left_camera_info_msg.P = calib_data_left["projection_matrix"]["data"]
    #left_camera_info_msg.distortion_model = calib_data_left["distortion_model"]

    searchlabel = os.path.join("kitti_dataset" , "image_2" , "*.png" )
    fileslabel = glob.glob(searchlabel)
    fileslabel.sort()
    searchlabelr = os.path.join("kitti_dataset" , "image_3" , "*.png" )
    fileslabelr = glob.glob(searchlabelr)
    fileslabelr.sort()

    bag = rosbag.Bag('test.bag', 'w')

    for i in range(len(fileslabel)):
        image_header = std_msgs.msg.Header()
        time_path = os.path.join("kitti_dataset")
        with open(time_path+'/times.txt') as f:
            txt = f.readlines()

        imgl = cv2.imread(fileslabel[i], -1)

        imgr = cv2.imread(fileslabelr[i], -1)

        imgr_msg = Image()
        imgl_msg = Image()
        t = rospy.Time.now()
        image_header.stamp = t
        image_header.frame_id = "base_link"

        imgl_msg = bridge.cv2_to_imgmsg(imgl,'bgr8')
        imgr_msg = bridge.cv2_to_imgmsg(imgr,'bgr8')
        imgr_msg.header = image_header
        imgl_msg.header = image_header
        right_camera_info_msg.header = image_header
        left_camera_info_msg.header = image_header
        #imgl_msg.data = IL
        #imgr_msg.data = IR

        bag.write("right/camera_info", right_camera_info_msg)
        bag.write("left/camera_info", left_camera_info_msg)
        bag.write("right/image_raw", imgr_msg)
        bag.write("left/image_raw", imgl_msg)
        rate.sleep()


    print("bagdone")
    bag.close()


def image_depth():

    searchlabel = os.path.join("kitti_dataset" , "image_3" , "*.png" )
    fileslabel = glob.glob(searchlabel)
    fileslabel.sort()
    for i in range(len(fileslabel)):

        img = cv2.imread(fileslabel[i], -1)
        #img = img.astype(np.uint8)

        bridge = CvBridge()

        IL = Image()
        IL = bridge.cv2_to_imgmsg(img,'bgr8')
        IL_pub.publish(IL)

if __name__ == '__main__':

    rospy.init_node('image_to_ros_data', anonymous=True)
    IL_pub = rospy.Publisher("/disparity", Image, queue_size=10)
    IL_pub = rospy.Publisher("/disparity", Image, queue_size=10)
    IL_pub = rospy.Publisher("/disparity", Image, queue_size=10)

    image_bag()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
