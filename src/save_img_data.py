#!/usr/bin/env python
from rtabmap_ros.msg import UserData
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import std_msgs.msg
import rospy
import cv2
import os
import message_filters
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

bridge = CvBridge()
image_header = std_msgs.msg.Header()
i = 0

def callback(seg, img):
    frame_seg = bridge.imgmsg_to_cv2(seg)
    frame_img = bridge.imgmsg_to_cv2(img)
    print("saving imgs")
    cv2.imwrite("data_annot_%i.png" % i, frame_seg)
    cv2.imwrite("data_rgb_%i.png" % i, frame_img)
    i=i+1


if __name__ == '__main__':

    rospy.init_node('img_save', anonymous=True)
    image_seg_sub =  message_filters.Subscriber('/image_seg', Image)
    image_rgb_sub =  message_filters.Subscriber('/image_raw', Image)
    ts = message_filters.TimeSynchronizer([image_seg_sub, image_rgb_sub], 1)
    ts.registerCallback(callback)


    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
