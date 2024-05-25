#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import torch

class NLU_Node:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('NLU', anonymous=True)

        # 发布文本信息
        self.text_publisher = rospy.Publisher('/text_topic', String, queue_size=10)


if __name__ == '__main__':

    try:
        nlu = NLU_Node()
        while not rospy.is_shutdown():
            command = input()
            nlu.text_publisher.publish(command)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
