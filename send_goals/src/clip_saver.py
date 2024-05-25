#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from send_goals.msg import detect
import cv2
import numpy as np
from PIL import Image as IP
import clip
import torch
from math import sqrt, pi
import csv

# 指定要保存的文件名
filename = "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/data_clip.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class ClipSubscriber:
    def __init__(self):
    
        rospy.init_node('clip_saver', anonymous=True)
        self.cv_bridge = CvBridge()
        self.index = 0
        self.data = {
                "index":0,
                "label":"default",
                "conf":"default",
                "x":0,
                "y":0,
                "z":0,
                 "features":None
                 }
        self.detect_subscriber = rospy.Subscriber("/yolo_info", detect, self.yolo_info_callback)
        
        
    def yolo_info_callback(self, msg):
        self.data["index"] = self.index
        self.index = self.index + 1
        self.data["label"] = msg.label
        self.data["x"] = msg.x
        self.data["y"] = msg.y
        self.data["z"] = msg.z
        self.data["conf"] = msg.conf
        roi =  self.cv_bridge.imgmsg_to_cv2(msg.roi_image, "bgr8")
        image_pil = IP.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(image).cpu().tolist()
            self.data["features"] = ','.join(map(str, features[0]))
            #print(self.data["features"])
        file_empty = False
        try:
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                if len(list(csv_reader)) == 0:
                    file_empty = True
        except FileNotFoundError:
            file_empty = True

        # 将字典保存为csv文件
        with open(filename, 'a+', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.data.keys())
            if file_empty:
                writer.writeheader()
            writer.writerow(self.data)
            print("write an item")
if __name__ == '__main__':
    
    clip_subscriber = ClipSubscriber()
    rospy.spin()

