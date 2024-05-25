#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from send_goals.msg import detect
from tf import TransformListener
import cv2
import numpy as np

class RGBDepthSubscriber:
    def __init__(self):
    
        rospy.init_node('rgb_depth_subscriber', anonymous=True)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.detect_subscriber = rospy.Subscriber("/yolo_info", detect, self.yolo_info_callback)
        
        self.cv_bridge = CvBridge()
        self.listener = TransformListener()
        self.world_point_pub = rospy.Publisher('/yolo_point', PointStamped, queue_size=10)
        self.depth_image = None
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.u = 320
        self.v = 240
    def depth_callback(self, data):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(data)
            # print(self.depth_image)
            # normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # cv2.imshow("Depth Image", normalized_img)
            # cv2.waitKey(1)  # This is necessary for imshow to work properly
        except Exception as e:
            rospy.logerr("Error converting depth image: %s", str(e))

    def yolo_info_callback(self, data):
        try:
            self.u = (data.x1 + data.x2)/2
            self.v = (data.y1 + data.y2)/2
        except Exception as e:
            rospy.logerr("Error converting depth image: %s", str(e))

    def run(self):
        while not rospy.is_shutdown():
            if self.depth_image is not None:
                # print(self.u,self.v)
                # print(self.depth_image.shape)
                if self.u !=0 and self.v !=0:
                    depth_value = self.depth_image[int(self.v), int(self.u)]
                    if depth_value == 0:
                        print("Invalid depth value pixel")
                        pass
                    camera_z = depth_value
                    camera_x = (self.u - self.cx) / self.fx * depth_value
                    camera_y = (self.v - self.cy) / self.fy * depth_value
                    print("Depth value: {:.2f} m".format(depth_value))
                    # 获取当前时间
                    current_time = rospy.Time.now()

                    # 创建一个 PointStamped 消息
                    camera_point_msg = PointStamped()
                    camera_point_msg.header.stamp = current_time
                    camera_point_msg.header.frame_id = "camera_frame_optical"  # 假设相机坐标系为 "camera_frame"
                    camera_point_msg.point.x = camera_x
                    camera_point_msg.point.y = camera_y
                    camera_point_msg.point.z = camera_z
                    try:
                        # 使用 TransformListener 获取相机坐标系到世界坐标系的变换
                        self.listener.waitForTransform("map", "camera_frame_optical", current_time, rospy.Duration(1.0))
                        world_point_msg = self.listener.transformPoint("map", camera_point_msg)

                        # 发布世界坐标系中的点
                        self.world_point_pub.publish(world_point_msg)
                    except Exception as e:
                        rospy.logwarn("Failed to transform point: %s", str(e))
            pass

if __name__ == '__main__':
    
    rgb_depth_subscriber = RGBDepthSubscriber()
    rgb_depth_subscriber.run()

