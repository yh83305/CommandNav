#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from send_goals.msg import detect
from geometry_msgs.msg import PointStamped
from tf import TransformListener
import clip

model = YOLOWorld('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/models/yolov8s-world.pt')  # or choose yolov8m/l-world.pt

class RGBDepthSubscriber:
    def __init__(self):
    
        rospy.init_node('rgb_subscriber', anonymous=True)
        print(torch.cuda.is_available())
        self.cv_bridge = CvBridge()
        # ctypes.CDLL("libX11.so.6").XInitThreads()
        
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.label = ""
        self.conf = 0
        self.depth_image = None
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.u = 320
        self.v = 240
        self.x = 0
        self.y = 0
        self.z = 0
               
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback, queue_size=1, buff_size=52428800)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.listener = TransformListener()
        self.world_point_pub = rospy.Publisher('/yolo_point', PointStamped, queue_size=10)
        self.yolo_info_pub = rospy.Publisher('/yolo_info', detect, queue_size=10)
        
        self.pub1 = rospy.Publisher('/view1', Image, queue_size=10)
        self.pub2 = rospy.Publisher('/view2', Image, queue_size=10)
        
    def depth_callback(self, depth_data):
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_data)
            # print("depth_image_shape")
            # print(self.depth_image.shape)
            normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.circle(normalized_img, (int(self.u), int(self.v)), 2, (255, 255, 255), 2)
            cv2.imshow("Depth Image", normalized_img)
            cv2.waitKey(1)  # This is necessary for imshow to work properly
        except Exception as e:
            rospy.logerr("depth image: %s", str(e))
            
    def rgb_callback(self, rgb_data):
        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
            color_image = rgb_image
            results = model(color_image, agnostic_nms = True)
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    self.label, self.conf = result.names[int(box.cls)], float(box.conf)
                    [[x1, y1, x2, y2]] = box.xyxy.cpu().numpy()
                    self.x1 = int(x1)
                    self.y1 = int(y1)
                    self.x2 = int(x2)
                    self.y2 = int(y2)
                    (self.u, self.v) = (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2,))
                    
                    ispub = self.calculate()
                    print("ispub:")
                    print(ispub)
                    if ispub:
                        roi = color_image[self.y1:self.y2, self.x1:self.x2]
                        roi_imgsmg = self.cv_bridge.cv2_to_imgmsg(roi, encoding="bgr8")
                        detect_msg = detect()
                        detect_msg.x = self.x
                        detect_msg.y = self.y
                        detect_msg.z = self.z
                        detect_msg.x1 = self.x1 
                        detect_msg.y1 = self.y1
                        detect_msg.x2 = self.x2
                        detect_msg.y2 = self.y2
                        detect_msg.roi_image = roi_imgsmg   
                        detect_msg.conf = self.conf
                        detect_msg.label = self.label      
                        self.yolo_info_pub.publish(detect_msg)
                        text = str(self.label) + "{:.2f}".format(self.conf)
                        cv2.rectangle(color_image, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
                        cv2.circle(color_image, (self.u, self.v), 2, (0, 255, 0), 2)
                        cv2.putText(
                            color_image,
                            text,
                            (int(self.x1), int(self.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 200),
                            1
                        )
                        ros_image1 = self.cv_bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                        ros_image2 = self.cv_bridge.cv2_to_imgmsg(roi, encoding="bgr8")
                        self.pub1.publish(ros_image1)
                        self.pub2.publish(ros_image2)
                        #cv2.imshow('yolo', color_image)
                        #cv2.waitKey(10)
                        #cv2.imshow('roi', roi)
                        #cv2.waitKey(10)
                    else:
                        ros_image3 = self.cv_bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
                        self.pub1.publish(ros_image3)
        except Exception as e:
            rospy.logerr("RGB&YOLO: %s", str(e))
    
    def calculate(self):
        try:
            if self.depth_image is not None and self.u !=0 and self.v !=0:
                depth_value = self.depth_image[self.v, self.u]
                print("depth_value:")
                print(depth_value)
                if depth_value > 0  and depth_value < 10:
                    camera_z = depth_value
                    camera_x = (self.u - self.cx) / self.fx * depth_value
                    camera_y = (self.v - self.cy) / self.fy * depth_value
                    print("Depth value: {:.2f} m".format(depth_value))
                    current_time = rospy.Time.now()
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
                        self.x = world_point_msg.point.x
                        self.y = world_point_msg.point.y
                        self.z = world_point_msg.point.z
                        # 发布世界坐标系中的sl点
                        self.world_point_pub.publish(world_point_msg)
                        return True
                    except Exception as e:
                        rospy.logwarn("Failed to transform point: %s", str(e))
                        return False
        except Exception as e:
            rospy.logerr("calculate: %s", str(e))
            return False
        

if __name__ == '__main__':
    
    rgb_depth_subscriber = RGBDepthSubscriber()
    rospy.spin()

