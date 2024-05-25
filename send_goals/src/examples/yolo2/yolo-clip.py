#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from send_goals.msg import detect

model = YOLO('/home/fragerant/Desktop/catkin_turtlebot3/src/send_goals/models/yolov8n.pt')  # or choose yolov8m/l-world.pt

class RGBDepthSubscriber:
    def __init__(self):
    
        rospy.init_node('rgb_subscriber', anonymous=True)
        # ctypes.CDLL("libX11.so.6").XInitThreads()
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback, queue_size=1, buff_size=52428800)
        self.yolo_info_pub = rospy.Publisher('/yolo_info', detect, queue_size=10)
        self.cv_bridge = CvBridge()
        print(torch.cuda.is_available())
        self.yolo_target = []
        
    def rgb_callback(self, data):
        try:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            color_image = rgb_image
            results = model(color_image, conf = 0.3, agnostic_nms = True)
            for result in results:
                boxes = result.boxes
                # print(boxes.cls)
                # print(boxes.conf)
                for i, box in enumerate(boxes):
                    cls, conf = int(box.cls), float(box.conf)
                    # print(box.xyxy.cpu().numpy())
                    [[x1, y1, x2, y2]] = box.xyxy.cpu().numpy()
                    # print(cls, conf)
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    target = (int((x1 + x2) / 2), int((y1 + y2) / 2,))
                    self.yolo_target.append(target)
                    cv2.circle(color_image, target, 2, (0, 255, 0), 2)
                    label = result.names[cls]
                    detect_msg = detect()
                    detect_msg.x1 = int(x1)
                    detect_msg.y1 = int(y1)
                    detect_msg.x2 = int(x2)
                    detect_msg.y2 = int(y2)
                    roi = color_image[int(y1):int(y2), int(x1):int(x2)]

                    roi_imgsmg = self.cv_bridge.cv2_to_imgmsg(roi, encoding="bgr8")
                    detect_msg.roi_image = roi_imgsmg
                    
                    detect_msg.label = label
                    self.yolo_info_pub.publish(detect_msg)
                    text = str(label) + "{:.2f}".format(conf)
                    cv2.putText(
                        color_image,
                        text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 200),
                        1
                   )
            cv2.imshow('yolo', color_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("RGB&YOLO: %s", str(e))

if __name__ == '__main__':
    
    rgb_depth_subscriber = RGBDepthSubscriber()
    rospy.spin()

