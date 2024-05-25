#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from send_goals.msg import detect
from tf import TransformListener
from visualization_msgs.msg import Marker, MarkerArray
import cv2
import numpy as np
from PIL import Image as IP
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class RGBDepthSubscriber:
    def __init__(self):
    
        rospy.init_node('rgb_depth_subscriber', anonymous=True)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.detect_subscriber = rospy.Subscriber("/yolo_info", detect, self.yolo_info_callback)
        
        self.cv_bridge = CvBridge()
        self.listener = TransformListener()
        self.world_point_pub = rospy.Publisher('/yolo_point', PointStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/yolo_marker', MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.depth_image = None
        self.roi = None
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.u = 320
        self.v = 240
        self.label = ""
        self.i = 0
    def depth_callback(self, data):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(data)
            # print("depth_image_shape")
            # print(self.depth_image.shape)
            normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.circle(normalized_img, (int(self.u), int(self.v)), 2, (255, 255, 255), 2)
            cv2.imshow("Depth Image", normalized_img)
            cv2.waitKey(1)  # This is necessary for imshow to work properly
        except Exception as e:
            rospy.logerr("depth image: %s", str(e))

    def yolo_info_callback(self, data):
        try:
            self.u = (data.x1 + data.x2)/2
            self.v = (data.y1 + data.y2)/2
            self.label = data.label
            self.roi =  self.cv_bridge.imgmsg_to_cv2(data.roi_image, "bgr8")
            if self.depth_image is not None:
                # print(self.u,self.v)
                # print(self.depth_image.shape)
                if self.u !=0 and self.v !=0:
                    depth_value = self.depth_image[int(self.v), int(self.u)]
                    if depth_value > 0  and depth_value < 4:
                        image_pil = IP.fromarray(cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB))
                        image = preprocess(image_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image).float().cpu()
                            # print(image_features.size)
                            # cv2.imshow('ROI', self.roi)
                            # cv2.waitKey(1)
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

                           # 发布世界坐标系中的sl点
                            self.world_point_pub.publish(world_point_msg)
                            self.marker_append(world_point_msg, self.label)
                            print(self.label)
                            self.marker_pub.publish(self.marker_array)
                        except Exception as e:
                            rospy.logwarn("Failed to transform point: %s", str(e))
        except Exception as e:
            rospy.logerr("depth image: %s", str(e))
            
    def marker_append(self, point, label):

        marker = Marker()
        marker.header.frame_id = point.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.id = self.i
        self.i = self.i + 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = point.point
        marker.text = label 
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.marker_array.markers.append(marker)

if __name__ == '__main__':
    
    rgb_depth_subscriber = RGBDepthSubscriber()
    rospy.spin()
