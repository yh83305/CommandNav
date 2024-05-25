#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from visualization_msgs.msg import Marker
import csv

filename = "/home/fragerant/Desktop/catkin_turtlebot3/src/send_goals/src/data.csv"


def read_csv_to_dict(filename):
    dict_list = []
    data_dict = {}  # 创建一个空字典用于存储CSV数据

    try:
        with open(filename, 'r') as csv_file:  # 打开CSV文件进行读取
            csv_reader = csv.DictReader(csv_file)  # 创建CSV字典读取器

            for row in csv_reader:  # 逐行读取CSV文件
                for key, value in row.items():  # 遍历每行的键值对
                    if key == "label":
                        data_dict[key] = value  # 将键值对添加到字典中
                    else:
                        data_dict[key] = float(value)  # 将键值对添加到字典中
                dict_list.append(data_dict)
                data_dict = {}

    except IOError:
        rospy.logerr("无法打开文件: %s", filename)

    return dict_list


def send_goal(target_pose):
    # 创建move_base客户端
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    # 创建目标位姿
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = target_pose

    # 发送目标位置和姿态信息
    client.send_goal(goal)
    client.wait_for_result()


def main():
    rospy.init_node('send_goal_node')

    # filename = rospy.get_param("~filename", "data.csv")  # 获取文件名参数，如果未提供则默认为"data.csv"
    rospy.loginfo("正在读取文件: %s", filename)

    dict_list = read_csv_to_dict(filename)  # 调用函数读取CSV文件并转换为字典
    rospy.loginfo("读取的字典数据为: %s", dict_list)

    # 发布目标点用于rviz显示
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    while not rospy.is_shutdown():
        command = input("请输入指令：")
        find_flag = 0
        for data_dict in dict_list:
            print(command)
            if data_dict["label"] == command:
                find_flag = 1
                marker = Marker()  # 创建目标点
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "target"
                marker.id = 0
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                print(type(data_dict["position_x"]))
                marker.pose.position.x = data_dict["position_x"]  # 目标位置x坐标
                marker.pose.position.y = data_dict["position_y"]  # 目标位置y坐标
                marker.pose.position.z = data_dict["position_z"]
                marker.pose.orientation.x = data_dict["orientation_x"]
                marker.pose.orientation.y = data_dict["orientation_y"]
                marker.pose.orientation.z = data_dict["orientation_z"]
                marker.pose.orientation.w = data_dict["orientation_w"]
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                # 发布目标点
                marker_pub.publish(marker)
                rospy.loginfo("发布目标点完毕")
                # 设置目标位置和姿态信息
                target_pose = PoseStamped()
                target_pose.header.frame_id = "map"
                target_pose.pose.position.x = data_dict["position_x"]  # 目标位置x坐标
                target_pose.pose.position.y = data_dict["position_y"]  # 目标位置y坐标
                target_pose.pose.position.z = data_dict["position_z"]
                target_pose.pose.orientation.x = data_dict["orientation_x"]
                target_pose.pose.orientation.y = data_dict["orientation_y"]
                target_pose.pose.orientation.z = data_dict["orientation_z"]
                target_pose.pose.orientation.w = data_dict["orientation_w"]

                # 发送目标位置和姿态信息到move_base
                send_goal(target_pose.pose)
                rospy.loginfo("到达目标位置")
                break
        if(find_flag == 0):
            rospy.loginfo("找不到指定label")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
