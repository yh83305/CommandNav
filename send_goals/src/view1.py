#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub1 = rospy.Subscriber('/view1', Image, self.image_callback1)

    def image_callback1(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        cv2.imshow("Image 1", cv_image)
        cv2.waitKey(1)  # 保持图像窗口打开

def main():
    rospy.init_node('image_subscriber')
    subscriber = ImageSubscriber()
    rospy.spin()

if __name__ == '__main__':
    main()

