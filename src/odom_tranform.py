#!/usr/bin/env python
import rospy
import roslib
import tf
import PyKDL as kdl


from nav_msgs.msg import Odometry

def odometryCb(msg):
    print msg.pose.pose
    odom = Odometry()
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)

    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    print roll
    print pitch
    print yaw
    odom.pose.pose.orientation = quaternion
    print odom.pose.pose

    odom_pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('oodometry_transformed', anonymous=True)
    odom_pub = rospy.Publisher('/integrated_tf', Odometry, queue_size=1)
    rospy.Subscriber('/integrated_to_init',Odometry,odometryCb)
    rospy.spin()
