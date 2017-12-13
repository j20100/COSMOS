#!/usr/bin/env python
from rtabmap_ros.msg import UserData
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import std_msgs.msg
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
image_header = std_msgs.msg.Header()

def image_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg)
    global image_header
    #UD = UserData()
    #UD.type = 16
    #UD.rows = frame.shape[0]
    #UD.cols = frame.shape[0]
    #UD.data = frame
    ITS = Image()
    ITS = msg
    ITS.encoding = "bgr8"
    ITS.header = image_header

    #UD_pub.publish(UD)
    ITS_pub.publish(ITS)

def image_rgb_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg)
    global image_header
    #UD = UserData()
    #UD.type = 16
    #UD.rows = frame.shape[0]
    #UD.cols = frame.shape[0]
    #UD.data = frame
    IRGBTS = Image()
    IRGBTS = msg
    IRGBTS.encoding = "bgr8"
    IRGBTS.header = image_header

    #UD_pub.publish(UD)
    IRGBTS_pub.publish(IRGBTS)


def cloud_callback(msg):
    global image_header
    CI = CameraInfo()
    CI.header = msg.header
    CI.header.frame_id = "camera"

    image_header = CI.header
    CI.height = 360
    CI.width = 480
    CI.K = [425, 0, 480/2, 0, 425, 360/2, 0, 0, 1]
    CI.P = [425, 0, 480/2, 0, 0, 425, 360/2, 0, 0, 0, 1, 0]

    CI_pub.publish(CI)

if __name__ == '__main__':

    rospy.init_node('image_to_user_data', anonymous=True)
    rospy.Subscriber("/image_seg_maskrcnn", Image, image_callback)
    rospy.Subscriber("/image_raw", Image, image_rgb_callback)
    rospy.Subscriber("/velodyne_points", PointCloud2, cloud_callback)
    ITS_pub = rospy.Publisher("image_seg_ts", Image, queue_size=10)
    IRGBTS_pub = rospy.Publisher("image_rgb_ts", Image, queue_size=10)
    UD_pub = rospy.Publisher("user_data", UserData, queue_size=10)
    CI_pub = rospy.Publisher("camera_info", CameraInfo, queue_size=10)


    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
