#!/usr/bin/env python3
# -*- coding: utf-8 -*
 
import os
import sys
import tty, termios
import rospy
import csv

from std_msgs.msg import String
from nav_msgs.msg import Odometry

data = {
    "label": "default",
    "position_x" : 0,
    "position_y" : 0,
    "position_z" : 0,

    "orientation_x" : 0,
    "orientation_y" : 0,
    "orientation_z" : 0
}

# 指定要保存的文件名
filename = "/home/fragerant/Desktop/catkin_turtlebot3/src/send_goals/src/data.csv"

def callback(odom_msg):
    global data
    data["position_x"] = odom_msg.pose.pose.position.x
    data["position_y"] = odom_msg.pose.pose.position.y
    data["position_z"] = odom_msg.pose.pose.position.z

    # 读取姿态信息
    data["orientation_x"] = odom_msg.pose.pose.orientation.x
    data["orientation_y"] = odom_msg.pose.pose.orientation.y
    data["orientation_z"] = odom_msg.pose.pose.orientation.z
    data["orientation_w"] = odom_msg.pose.pose.orientation.w
    
def save_label():
    label = input("请输入标签字符串并按回车键：")
    # 在这里可以将标签字符串保存到文件中或者进行其他处理
    data["label"] = label
    rospy.loginfo("Position (x, y, z): ({}, {}, {})".format(data["position_x"], data["position_y"], data["position_z"]))
    rospy.loginfo("Orientation (x, y, z, w): ({}, {}, {}, {})".format(data["orientation_x"], data["orientation_y"], data["orientation_z"], data["orientation_w"]))
    rospy.loginfo("保存的标签为：%s", label)
    
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
    	writer = csv.DictWriter(csv_file, fieldnames=data.keys())
    	if file_empty:
            writer.writeheader()  # 如果文件为空，添加表头
    	writer.writerow(data)
def keyboardLoop():
    #初始化
    rospy.init_node('labelling_node')
    rate = rospy.Rate(rospy.get_param('~hz', 1))
    rospy.Subscriber('/odom', Odometry, callback)
    #显示提示信息
    rospy.loginfo ("Reading from keyboard")
    rospy.loginfo ("Use space to control the robot")
 
    #读取按键循环
    while not rospy.is_shutdown(): 
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
	#不产生回显效果
        old_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO
        try :
            tty.setraw( fd )
            ch = sys.stdin.read( 1 )
        finally :
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ch == ' ':
            save_label()
        elif ch == 'q':
            exit()
        rate.sleep()
 
if __name__ == '__main__':
    try:
        keyboardLoop()
    except rospy.ROSInterruptException:
        pass
