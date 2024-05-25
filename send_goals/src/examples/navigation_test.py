#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from visualization_msgs.msg import Marker

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

    # 发布目标点用于rviz显示
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    # 创建目标点
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "target"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = 1.0  # 目标位置x坐标
    marker.pose.position.y = 2.0  # 目标位置y坐标
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # 发布目标点
    marker_pub.publish(marker)

    # 设置目标位置和姿态信息
    target_pose = PoseStamped()
    target_pose.header.frame_id = "map"
    target_pose.pose.position.x = 1.0  # 目标位置x坐标
    target_pose.pose.position.y = 2.0  # 目标位置y坐标
    target_pose.pose.position.z = 0.0
    target_pose.pose.orientation.x = 0.0
    target_pose.pose.orientation.y = 0.0
    target_pose.pose.orientation.z = 0.0
    target_pose.pose.orientation.w = 1.0

    # 发送目标位置和姿态信息到move_base
    send_goal(target_pose.pose)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

