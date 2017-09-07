#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

ros::Publisher pub_img;
ros::Publisher pub_cloud;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_accumulated(new pcl::PointCloud<pcl::PointXYZRGB>);
//const sensor_msgs::ImageConstPtr img;
sensor_msgs::PointCloud2 cloud_msg;
bool NEW_CLOUD = false;
bool NEW_IMG = false;
namespace enc = sensor_msgs::image_encodings;


void
img_callback (const sensor_msgs::ImageConstPtr& msg)
{
  NEW_IMG = true;
  if(NEW_CLOUD&NEW_IMG){
    // Container for original & filtered data
    pcl::PCLPointCloud2 pcl2;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    // Ros msg
    sensor_msgs::PointCloud2 cloud_out;

    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    // Convert to PCL data type
    pcl_conversions::toPCL(cloud_msg, pcl2);
    pcl::fromPCLPointCloud2(pcl2, cloud);

    //ROS_INFO("TEST in");

    ROS_INFO_STREAM(cloud.points.size());
    for (size_t k = 0; k < cloud.points.size (); k=k+1){

      //ROS_INFO("TEST");
      //ROS_INFO_STREAM(k);
      if (k>500){
        uint8_t r = 0, g = 0, b = 255;    // Example: Red colorConstPtr&dddd
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        cloud.points[k].rgb = *reinterpret_cast<float*>(&rgb);
      }
      else{
        uint8_t r = 255, g = 0, b = 0;    // Example: Red color
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        cloud.points[k].rgb = *reinterpret_cast<float*>(&rgb);
      };

    };

    //ROS_INFO("OUt");
    pcl::toROSMsg(cloud, cloud_out);
    pub_cloud.publish(cloud_out);
    pub_img.publish(msg);
    NEW_IMG = false;
    NEW_CLOUD = false;
  }

}

void
cloud_callback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg_in)
{
  cloud_msg = *cloud_msg_in;
  NEW_CLOUD = true;
}

int
main (int argc, char** argv)
{
  ROS_INFO("INIT");
  ros::init (argc, argv, "seg_3d");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub_cloud = nh.subscribe<sensor_msgs::PointCloud2> ("/rtabmap/cloud_map", 1, cloud_callback);
  ros::Subscriber sub_img = nh.subscribe<sensor_msgs::Image> ("/stereo_camera/left/image_rect_color", 1, img_callback);

  // Create a ROS publisher for the output point cloud
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("seg_cloud", 1);
  pub_img = nh.advertise<sensor_msgs::Image> ("seg_img", 1);


  while (ros::ok())
  {
    ros::spinOnce();
    //loop_rate.sleep();
  };

}
